import math
import numpy as np
from turtle import forward

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from  transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from torch.distributions.kl import kl_divergence
from transformers.modeling_outputs import SequenceClassifierOutput
    

class DominoSlicer(nn.Module):
    def __init__(self, args, training_args, dro_args, task_model):
        super().__init__()
        self.args = args
        self.dro_args = dro_args
        self.training_args = training_args
        self.task_model = task_model
        self.n_slices = args.n_slices
        self.batch_size = training_args.per_device_train_batch_size
        self.n_features = args.n_features

        self.grouper_model = nn.Sequential(
            nn.Linear(self.n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_slices),
            nn.Softmax()
        )


        # gcdro parameters relevant for group-wise maximization
        self.count_cat = self._prepare_input(torch.ones(self.n_slices).float())
        self.gamma = dro_args.gamma 
        self.max_var_weight = dro_args.max_var_weight # GCDRO uses a minimum variance if weight
        self.min_var_weight = dro_args.min_var_weight 
        self.alpha = dro_args.alpha # How many groups to underweigh.

        # Beta cover 
        self.beta = dro_args.beta
        self.beta_ema = dro_args.beta_ema
        self.do_instance_reweight = dro_args.do_instance_reweight
        # quantities maintained throughout training for instance level G-DRO
        self.accum_losses = None

        # running averages
        self.adj = self._prepare_input(torch.zeros(self.n_slices).float())
        self.adv_probs = self._prepare_input(torch.ones(self.n_slices)) #/self.n_groups
        self.reverse_adv_probs = self._prepare_input(torch.ones(self.n_slices)) #/self.n_groups
        self.group_loss = self._prepare_input(torch.zeros(self.n_slices))
        self.count_cat = self._prepare_input(torch.ones(self.n_slices).float())
        self.exp_avg_loss = self._prepare_input(torch.zeros(self.n_slices))
        self.group_distribution = self._prepare_input(torch.full((self.batch_size, self.n_slices), 1/self.n_slices))
        self.exp_avg_initialized = self._prepare_input(torch.zeros(self.n_slices).byte())

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

    def forward(self, input_ids, attention_mask, group, group_features=None, labels=None, adversary=False, **kwargs):
        # inputs has input_ids, attention_mask for task_model, group ids are computed dynamically based on features 
        # which is a vector of size self.n_features 
        
        """
        1. Compute group distributions.
        2. Get instance level losses.
        2. Compute aggregate loss over groups.
        """
        self.group_distribution = self.grouper_model(group_features) # B * G
        task_model_outputs = self.task_model(input_ids, attention_mask, labels=labels) 
        per_sample_losses = task_model_outputs["loss"] # B * 1 [Individual losses]

        # Group wise loss. 
        group_losses = self.compute_soft_group_loss(per_sample_losses, self.group_distribution)
        group_count = self.group_distribution.sum(0)

        # group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        yhat = task_model_outputs[1]
        minibatch_group_acc, minibatch_group_count = self.compute_group_avg((torch.argmax(yhat,1)==labels).float(), self.group_distribution)

        # group_losses = self.compute_group_loss(per_sample_losses, group)
        # group_map = (group == self._prepare_input(torch.arange(self.n_slices).unsqueeze(1).long())).float()
        # group_count = group_map.sum(1)
        
        #dist.all_reduce(group_count, op=ReduceOp.SUM)
        #dist.all_reduce(group_losses, op=ReduceOp.SUM)

        # normalize group_wise loss.
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_losses = (group_losses)/group_denom

        # only update this in primary pass
        if not adversary:
            self.update_exp_avg_loss(group_losses.detach(), group_count.detach())

        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.count_cat)

        if adversary:
            loss, weights = self.compute_adversary_loss_greedy(group_losses, adjusted_loss)
            # Regularizer 1 (Entropy of group distribution should be high)
            cp = Categorical(self.group_distribution)
            reg1 = -cp.entropy()
            loss += reg1.mean()
            # Regularizer 2 (biased estimate of group marginal should be closer to a uniform prior)
            marginal = Categorical(self.group_distribution.sum(0)/self.group_distribution.sum())
            prior = Categorical(self._prepare_input(torch.full((self.n_slices,), 1.0/self.n_slices)))
            reg2 = kl_divergence(marginal, prior)
            loss += reg2
        else:
            loss, weights = self.compute_loss_greedy(group_losses, adjusted_loss)
            # update stats
            self.update_stats(loss, group_losses, minibatch_group_acc, minibatch_group_count, weights)
    
        # the expected return functions should contain same outputs as task_model
        return SequenceClassifierOutput(loss=loss, logits=task_model_outputs["logits"])
        
    def compute_group_loss(self, losses, group_idx):
        group_map = (group_idx == self._prepare_input(torch.arange(self.n_slices).unsqueeze(1).long())).float()
        group_loss = (group_map @ losses.view(-1))
        return group_loss

    def compute_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        past_frac = self.count_cat / self.count_cat.sum() 
        sorted_frac = past_frac[sorted_idx]

        cutoff_count = torch.sum(torch.cumsum(sorted_frac, 0) < self.alpha)
        if cutoff_count == len(sorted_frac):
            cutoff_count = len(sorted_frac) - 1
        self.adv_probs =  self.adv_probs.new_full(self.adv_probs.size(), self.min_var_weight)
        self.adv_probs[sorted_idx[:cutoff_count]] = 1.0 / self.alpha
        leftover_mass = 1.0 - sorted_frac[:cutoff_count].sum().div(self.alpha)
        tiebreak_fraction = leftover_mass / sorted_frac[cutoff_count]  # check!
        self.adv_probs[sorted_idx[cutoff_count]] = tiebreak_fraction

        robust_loss = (group_loss @ self.adv_probs)

        self.group_loss = group_loss
        
        return robust_loss, self.adv_probs

    def compute_soft_group_loss(self, losses, group_prob):
        group_wise_loss = group_prob * losses.unsqueeze(1)
        return group_wise_loss.sum(0)

    def compute_adversary_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        past_frac = self.count_cat / self.count_cat.sum() 
        sorted_frac = past_frac[sorted_idx]

        cutoff_count = torch.sum(torch.cumsum(sorted_frac, 0) < self.alpha)
        if cutoff_count == len(sorted_frac):
            cutoff_count = len(sorted_frac) - 1
        self.reverse_adv_probs =  self.reverse_adv_probs.new_full(self.reverse_adv_probs.size(), self.max_var_weight)
        self.reverse_adv_probs[sorted_idx[:cutoff_count]] = 1.0 * self.alpha
        leftover_mass = 1.0 - sorted_frac[:cutoff_count].sum().mul(self.alpha)
        tiebreak_fraction = leftover_mass * sorted_frac[cutoff_count]  # check!
        self.reverse_adv_probs[sorted_idx[cutoff_count]] = tiebreak_fraction

        robust_loss = -(group_loss @ self.reverse_adv_probs)
        
        return robust_loss, self.reverse_adv_probs

    def update_exp_avg_loss(self, group_loss, group_count):
        ## TODO: Chunting's code is doing a different kind of exponential averaging, exp_avf_initialized not used.
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights

        ## TODO: Chunting's code is also doing an exponential averaging of counts (with alpha 0.05)
        self.count_cat = self.count_cat.mul(1 - 0.05).add(group_count, alpha=0.05)
        
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def compute_group_avg(self, losses, group_distribution):
        # Find argmax for groups
        group_idx = torch.argmax(group_distribution, dim=1)
        group_map = (group_idx == self._prepare_input(torch.arange(self.n_slices).unsqueeze(1).long())).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def reset_stats(self):        
        self.processed_data_counts = self._prepare_input(torch.zeros(self.n_slices))
        self.update_data_counts = self._prepare_input(torch.zeros(self.n_slices))
        self.update_batch_counts = self._prepare_input(torch.zeros(self.n_slices))
        self.avg_group_loss = self._prepare_input(torch.zeros(self.n_slices))
        self.avg_group_acc = self._prepare_input(torch.zeros(self.n_slices))
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
        if self.dro_args.is_robust:
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


    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_slices):
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
        # if model is not None:
        #     assert args is not None
        #     stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.info(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.info(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.info(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_slices):
            logger.info(
                # f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
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
        
            for gidx in range(self.n_slices):
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