import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


class GCDROLossComputer:
    def __init__(self, dro_args, training_args, n_groups, group_counts, adj=None):
        self.is_robust = dro_args.is_robust
        self.gamma = dro_args.gamma # ema for group loss 
        self.alpha = dro_args.alpha # alpha-cover factor
        self.min_var_weight = dro_args.min_var_weight
        self.step_size = dro_args.step_size
        self.normalize_loss = dro_args.normalize_loss
        self.btl = dro_args.btl
        self.training_args = training_args

        self.beta = dro_args.beta
        self.beta_ema = dro_args.beta_ema
        self.do_instance_reweight = dro_args.do_instance_reweight

        ## Can we pass these arguments, after computing upon reading the data, and then passing it through training args to Trainer.
        self.n_groups = n_groups
        self.group_counts = self._prepare_input(group_counts)
        self.count_cat = self._prepare_input(torch.ones(self.n_groups).float())
        #self.group_frac = self.group_counts/self.group_counts.sum()
        #self.group_str = group_str

        if adj is not None:
            self.adj = self._prepare_input(torch.from_numpy(adj).float())
        else:
            self.adj = self._prepare_input(torch.zeros(self.n_groups).float())

        if dro_args.is_robust:
            assert dro_args.alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = self._prepare_input(torch.ones(self.n_groups)) #/self.n_groups
        self.exp_avg_loss = self._prepare_input(torch.zeros(self.n_groups))
        self.group_loss = self._prepare_input(torch.zeros(self.n_groups))
        self.exp_avg_initialized = self._prepare_input(torch.zeros(self.n_groups).byte())

        # quantities maintained throughout training for instance level G-DRO
        self.accum_losses = None

        self.reset_stats()

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.training_args.device)
            if self.training_args.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.training_args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data

    def loss(self, per_sample_losses, yhat, y, group_idx=None, group_distribution=None, instance_weights=None, is_training=False):
        # compute per-sample and per-group losses
        # per_sample_losses = self.criterion(yhat, y) #TODO: Change, per_sample_loss is already computed.

        """
        GC-DRO loss specifics
        if "weights" in sample:
            ind_loss = ind_loss * sample["weights"]
        """
        if instance_weights is not None and self.do_instance_reweight:
            per_sample_losses = instance_weights*per_sample_losses

        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        #import pdb; pdb.set_trace()
        
        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        #sorted_loss = group_loss[sorted_idx]
        past_frac = self.count_cat / self.count_cat.sum() 
        sorted_frac = past_frac[sorted_idx]

        """
        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        self.adv_probs = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum()
        self.adv_probs[last_idx] = 1 - self.adv_probs.sum()
        self.adv_probs = sorted_frac*self.min_var_weight + self.adv_probs*(1-self.min_var_weight)
        """

        ## Chunting's code verison of greedy alpha-cover upweighting.
        cutoff_count = torch.sum(torch.cumsum(sorted_frac, 0) < self.alpha)
        if cutoff_count == len(sorted_frac):
            cutoff_count = len(sorted_frac) - 1
        self.adv_probs=  self.adv_probs.new_full(self.adv_probs.size(), self.min_var_weight)
        self.adv_probs[sorted_idx[:cutoff_count]] = 1.0 / self.alpha
        leftover_mass = 1.0 - sorted_frac[:cutoff_count].sum().div(self.alpha)
        tiebreak_fraction = leftover_mass / sorted_frac[cutoff_count]  # check!
        self.adv_probs[sorted_idx[cutoff_count]] = tiebreak_fraction

        robust_loss = group_loss @ self.adv_probs

        # sort the weights back
        # _, unsort_idx = sorted_idx.sort()
        # unsorted_weights = weights[unsort_idx]

        # update class objects for logging in trainer_dro
        self.group_loss = group_loss
        #self.adv_probs = weights

        return robust_loss, self.adv_probs

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == self._prepare_input(torch.arange(self.n_groups).unsqueeze(1).long())).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        ## TODO: Chunting's code is doing a different kind of exponential averaging, exp_avf_initialized not used.
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights

        ## TODO: Chunting's code is also doing an exponential averaging of counts (with alpha 0.05)
        self.count_cat = self.count_cat.mul(1 - 0.05).add(group_count, alpha=0.05)
        
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):        
        self.processed_data_counts = self._prepare_input(torch.zeros(self.n_groups))
        self.update_data_counts = self._prepare_input(torch.zeros(self.n_groups))
        self.update_batch_counts = self._prepare_input(torch.zeros(self.n_groups))
        self.avg_group_loss = self._prepare_input(torch.zeros(self.n_groups))
        self.avg_group_acc = self._prepare_input(torch.zeros(self.n_groups))
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.
        
        #TODO: Chunting also sets weights to 1 here, and self.exp_avg_loss to 0
        self.exp_avg_loss.fill_(0.)
        self.adv_probs.fill_(1.)
        

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.info(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.info(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.info(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.info(
                # f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        # logger.flush()

    def compute_beta_cover(self, seed, epoch, dataset, losses=None):
        split_array = np.array([item["group"] for item in dataset])
        total = len(split_array)
        if losses is not None:
            if self.accum_losses is None:
                self.accum_losses = losses
            else:
                self.accum_losses = self.accum_losses * (1 - self.beta_ema) + losses * self.beta_ema
        
            for gidx in range(self.n_groups):
                select_idx = np.where(split_array == gidx)[0]
                count = len(select_idx)
                idx_sorted = np.argsort(self.accum_losses[select_idx])
                idx = select_idx[idx_sorted][::-1]
                cutoff_count = int((total - count) * count * self.beta / (total - count * self.beta))
                self.weight_array[idx] = count / total
                self.weight_array[idx[:cutoff_count]] = 1.0 / self.beta
        else:
            self.weight_array = np.ones(total)
        return self.weight_array


