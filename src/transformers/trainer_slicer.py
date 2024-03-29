"""
The Trainer class, to train a group classifier using learned features of the model.
"""

import contextlib
import inspect
import math
import os
from attr import dataclass
import pandas as pd
from pandas import DataFrame
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import wandb

from tqdm.auto import tqdm

# Integrations must be imported before ML frameworks:
from .integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
from packaging import version
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from huggingface_hub import Repository

from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .deepspeed import deepspeed_init, deepspeed_reinit, is_deepspeed_zero3_enabled
from .dependency_versions_check import dep_version_check
from .file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from .trainer import Trainer
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from .training_args import OptimizerNames, ParallelMode
from .domino_training_args import DominoTrainingArguments
from .utils import logging

from .dro_loss import LossComputer, DroArguments

_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

class TrainerSlicer(Trainer):
    from .trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: DominoTrainingArguments = None,
            dro_args: DroArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            train_features: Optional[DataFrame]  = None,
            eval_features: Optional[DataFrame] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        self.args = args
        set_seed(self.args.seed)
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        # Setup Sharded DDP training
        # TODO: Add sharded_ddp for multiGPU training.
        self.sharded_ddp = None

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_features = train_features
        self.eval_features = eval_features
        self.tokenizer = tokenizer
        
        self.place_model_on_device = args.place_model_on_device
        if self.place_model_on_device:
            self._move_model_to_device(model, args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = None

        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Optimizers and lr_schedulers
        self.optimizer, self.lr_scheduler = optimizers

        # Callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        
        self._signature_columns = None

        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.do_grad_scaling = False

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.state = TrainerState()
        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = (
            ["start_positions", "end_positions"]
            if type(self.model).__name__ in MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.values()
            else ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)
        # very last
        self._memory_tracker.stop_and_update_metrics()

        self.dro_args = dro_args
        self._add_columns()


    def add_callback(self, callback):
        self.callback_handler.add_callback(callback)
    
    def _move_model_to_device(self, model, device):
        model = model.to(device)
        # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(model, "tie_weights"):
            model.tie_weights()

    def num_examples(self, dataloader: DataLoader) -> int:
        return len(dataloader.dataset)


    def log_training_dynamics(self, output_dir: os.path,
                          epoch: int,
                          train_ids: List[int],
                          train_logits: List[List[float]],
                          train_golds: List[int]):
        """
        Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
        """
        td_df = pd.DataFrame({"guid": train_ids,
                                f"logits_epoch_{epoch}": train_logits,
                                "gold": train_golds})

        logging_dir = os.path.join(output_dir, f"training_dynamics")
        # Create directory for logging training dynamics, if it doesn't already exist.
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
        td_df.to_json(epoch_file_name, lines=True, orient="records")
        logger.info(f"Training Dynamics logged to {epoch_file_name}")

    def log_dro_dynamics(self, output_dir: os.path,
                        epochs: List[int],
                        iterations: List[int],
                        group_probs: List[List[float]],
                        group_losses: List[List[float]],
                        group_counts: List[List[float]],
                        ):
        td_df = pd.DataFrame({"epoch": epochs,
                                f"iteration": iterations,
                                "group_weight": group_probs,
                                "group_loss": group_losses,
                                "group_counts": group_counts})
        logging_dir = os.path.join(output_dir, f"dro_dynamics")
        # Create directory for logging training dynamics, if it doesn't already exist.
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        epoch_file_name = os.path.join(logging_dir, f"dro_dynamics.jsonl")
        td_df.to_json(epoch_file_name, lines=True, orient="records")
        logger.info(f"Training Dynamics logged to {epoch_file_name}")   
    

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args    
        self.is_in_train = True

        # does the model need to be reloaded.

        # Keeping track whether we can len() on the dataset or not
        train_dataset_is_sized = has_length(self.train_dataset)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa        

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss_primary = torch.tensor(0.0).to(args.device)
        tr_loss_adversary = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        # Create dro dynamics variable.
        epoch_list = []
        iteration_list = []
        group_assignment_list = []
        group_loss_list = []
        group_count_list = []

        # Book-keeping for model selection
        worst_valid_acc = None
        valid_acc = None
        bad_counts = 0
        resplit_train_epoch = 0

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1

            # if adversary_warmup is set to X iterations, altrenate between training the adversary and primary for that many number of iterations
            

            for step, inputs in enumerate(epoch_iterator):

                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                
                # Primary
                tr_loss_step_primary = torch.tensor(0.0).to(args.device)
                if epoch >= args.adversary_warmup: 
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step_primary, batch_logits = self.training_step_primary(model, inputs)
                    else:
                        tr_loss_step_primary, batch_logits = self.training_step_primary(model, inputs)

                # Adversary (which is only trained for some X number of epochs before not being used anymore)
                # adversary warmup os for the number of epochs to train the adversary before not using it anymore.
                if (args.adversary_warmup) == -1 or epoch < args.adversary_warmup: # and self.state.global_step < 3300:
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step_adversary, batch_logits = self.training_step_adversary(model, inputs)
                    else:
                        tr_loss_step_adversary, batch_logits = self.training_step_adversary(model, inputs)

                # execute this if not doig parallel training
                # wandb.log({"primary loss": tr_loss_step_primary.item()})
                # wandb.log({"adversary loss": tr_loss_step_adversary.item()})

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step_primary) or torch.isinf(tr_loss_step_primary))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss_primary += tr_loss_primary / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss_primary += tr_loss_step_primary

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step_adversary) or torch.isinf(tr_loss_step_adversary))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss_adversary += tr_loss_adversary / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss_adversary += tr_loss_step_adversary

                self.current_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    # Just log, and save checkpoints, dont evaluate.
                    _ = self._maybe_log_save_evaluate(tr_loss_primary, model, trial, epoch, ignore_keys_for_eval, evaluate=False)
                    if self.dro_args.is_robust and self.state.global_step % self.args.logging_steps == 0:
                        model.module.log_stats(logger, True)
                        self.log(model.module.get_stats(model, args))
                        iteration_list.append(step)
                        epoch_list.append(epoch)
                        group_assignment_list.append(list(model.module.adv_probs.cpu().numpy()))
                        group_loss_list.append(list(model.module.group_loss.detach().cpu().numpy()))
                        group_count_list.append(list(model.module.processed_data_counts.detach().cpu().numpy()))
                        # add group count.
                        # there is a mismatch between Chunting's code where reset happens only after 1 epoch. 
                        # self.train_loss_computer.reset_stats()
                        
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            
            # End of epoch, reset train loss computer.
            if self.dro_args.is_robust and model.module.batch_count > 0:
                model.module.log_stats(logger, True)
                self.log(model.module.get_stats(model, args))
                model.module.reset_stats()

                """
                if self.dro_args.robust_algorithm == "GCDRO":
                    self._update_columns(epoch=epoch) #, dataloader=epoch_iterator)
                    # update epoch iterator, since instance weights are being changed in self.train_dataset
                    train_dataloader = self.get_train_dataloader()
                    if is_torch_tpu_available():
                        parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                        epoch_iterator = parallel_loader
                    else:
                        epoch_iterator = train_dataloader
                """
            
            # End of epoch
            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True
            
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            # this is going to save but only after its worst accuracy has been computed.
            # only start to evaluate, when you are no longer training the adversary model 
            if epoch >= args.adversary_warmup:
                metrics = self._maybe_log_save_evaluate(tr_loss_primary, model, trial, epoch, ignore_keys_for_eval, evaluate=True)

                # Training stopping criterion
                become_better = False
                if self.dro_args.is_robust and args.metric_for_best_model == "eval_worst_accuracy":
                    resplit_train_epoch += 1

                    if self.args.select_predicted_worst_group:
                        valid_group_acc = [(int(key.lstrip("eval_megagroup_accuracy_")), metrics[key]) for key in metrics.keys() if key.startswith("eval_megagroup_accuracy")]
                    else:
                        valid_group_acc = [(int(key.lstrip("eval_group_accuracy_")), metrics[key]) for key in metrics.keys() if key.startswith("eval_group_accuracy")]

                    curr_worst_valid_acc = min([acc for _, acc in valid_group_acc])
                    sorted_by_group_id = sorted(valid_group_acc, key=lambda tup: tup[0])
                    group_acc = " ".join(["%d: %.3f" % (idx, acc if acc > 0 else -acc) for idx, acc in sorted_by_group_id])
                    become_better = (worst_valid_acc is not None and curr_worst_valid_acc > worst_valid_acc) or worst_valid_acc is None
                    worst_valid_acc = curr_worst_valid_acc if worst_valid_acc is None else max(curr_worst_valid_acc, worst_valid_acc)
                    bad_counts = 0 if become_better else bad_counts + 1

                    logger.info("Valid group performance: {}".format(group_acc))
                    logger.info("Better worst valid = {}, bad counts = {}, worst acc = {}".format(become_better, bad_counts, curr_worst_valid_acc))
                    # Update metrics (best_worst_group)
                    metrics["eval_worst_accuracy"] = worst_valid_acc
                else:
                    # Even with robust training, this code will get triggered.
                    current_valid_acc = metrics["eval_accuracy"]
                    become_better = (valid_acc is not None  and current_valid_acc > valid_acc) or valid_acc is None
                    valid_acc = current_valid_acc if valid_acc is None else max(current_valid_acc, valid_acc)
                    bad_counts = 0 if become_better else bad_counts + 1
                    logger.info("Valid performance: {}".format(current_valid_acc))
                    logger.info("Better valid = {}, bad counts = {}, best acc = {}".format(become_better, bad_counts, current_valid_acc))

                # Model selection (save checkpoint with best worst_accuracy as the "best_" checkpoint)
                if become_better:
                    # First time worst_accuracy is computed, or worst accuracy improved.
                    self._save_checkpoint(model, trial, metrics=metrics, save_best=True)

            if self.control.should_training_stop:
                break
        
        # End of training

        # Dump dro group assignments to file.
        self.log_dro_dynamics(output_dir=args.output_dir, epochs=epoch_list, iterations=iteration_list, group_probs=group_assignment_list, group_losses=group_loss_list, group_counts=group_count_list)

        
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                if self.deepspeed:
                    # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
                    deepspeed_engine, optimizer, lr_scheduler = deepspeed_reinit(self)
                    self.model = deepspeed_engine.module
                    self.model_wrapped = deepspeed_engine
                    self.deepspeed = deepspeed_engine
                    self.optimizer = optimizer
                    self.lr_scheduler = lr_scheduler
                    self.deepspeed.load_checkpoint(
                        self.state.best_model_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
                    )
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(best_model_path, map_location="cpu")
                    # If the model is on the GPU, it still works!
                    self._load_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss_primary.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.task_model.load_state_dict(state_dict, strict=False)

        if len(load_result.missing_keys) != 0:
            if self.model.task_model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model.task_model._keys_to_ignore_on_save
            ):
                self.model.task_model.tie_weights()
            else:
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            decay_task_parameters = [name for name in decay_parameters if "grouper_model" not in name]
            decay_grouper_parameters = [name for name in decay_parameters if "grouper_model" in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_task_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_grouper_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.grouper_learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, evaluate=False):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if evaluate: # and self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            # may_log_and_save is called at the end of every epoch or after every iteration, and save_checkpoint is based on save_strategy.
            # setting metrics to none so that metric_to_check is not evaluated.
            self._save_checkpoint(model, trial, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        
        return metrics


    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            if "best" in self.state.best_model_checkpoint:
                # no need to remove any checkpoint from list, since best checkpoint is being explicitly saved.
                return checkpoints_sorted
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted


    def _save_checkpoint(self, model, trial, metrics=None, save_best=False):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if save_best:
            checkpoint_folder = f"best_checkpoint"
        else:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                from ray import tune

                run_id = tune.get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            elif self.hp_search_backend == HPSearchBackend.WANDB:
                import wandb

                run_id = wandb.run.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_fp16_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            if smp.rdp_rank() == 0:
                # Consolidate the state dict on all processed of rdp_rank 0
                opt_state_dict = self.optimizer.state_dict()
                # Save it and the scheduler on the main process
                if self.args.should_save:
                    torch.save(opt_state_dict, os.path.join(output_dir, OPTIMIZER_NAME))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    reissue_pt_warnings(caught_warnings)
                    if self.do_grad_scaling:
                        torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        local_rank = xm.get_local_ordinal() if is_torch_tpu_available() else self.args.local_rank
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{local_rank}.pth"))

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
    

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]
            self._signature_columns += ["guid"]
            self._signature_columns += ["group"]
            self._signature_columns += ["group_distribution"]
            self._signature_columns += ["instance_weight"]
            self._signature_columns += ["group_features"]

        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                f" you can safely ignore this message."
            )

        columns = [k for k in self._signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _add_columns(self):
        seed = self.args.seed
        epoch = 0
        # Check if evaluating.
        if self.train_dataset is not None:
            instance_weights = self.model.compute_beta_cover(seed, epoch, self.train_dataset)
            self.train_dataset = self.train_dataset.add_column("instance_weight", instance_weights)
        # eval datasets also need instance reweight is available, but do not update weight array of the model itself.
        if self.eval_dataset is not None:
            instance_weights = np.ones(len(self.eval_dataset)) # default and does not get udpated.
            self.eval_dataset = self.eval_dataset.add_column("instance_weight", instance_weights)

    def _update_columns(self, epoch):
        # Iterate over training data to compute loss.
        if epoch < self.args.adversary_warmup:
            logger.info(f"---- Skipping Re-Weight in {epoch} due to adversary training -----")
            return
        logger.info(f"---- Re-Weight at the begeinning of epoch {epoch} -----")
        train_losses = None
        train_groups = None
        dataset = self._remove_unused_columns(self.train_dataset, description="evaluation")
        dataloader = DataLoader(
                    dataset,
                    sampler=SequentialSampler(dataset),
                    batch_size=self.args.train_batch_size,
                    collate_fn=self.data_collator,
                    drop_last=False,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                )
        model = self._wrap_model(self.model, training=False)
        model.eval()
        for step, inputs in tqdm(enumerate(dataloader)):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                groups = inputs["group"]
                group_distributions = inputs.get("group_distribution", None)
                instance_weights = inputs.get("instance_weight", None)
                group_features = inputs.get("group_features", None)
                del inputs["group"]
                if group_distributions is not None:
                    del inputs["group_distribution"]
                if instance_weights is not None:
                    del inputs["instance_weight"]
                if group_features is not None:
                    del inputs["group_features"]
                del inputs["guid"]
                outputs = model.task_model(**inputs)
                groups = torch.argmax(model.grouper_model(group_features), dim=1)
                loss, _ = outputs[0], outputs[1]
                if train_losses is None:
                    train_losses = loss.detach().cpu().numpy()
                else:
                    train_losses = np.append(train_losses, loss.detach().cpu().numpy(), axis=0)
                if train_groups is None:
                    train_groups = groups.detach().cpu().numpy()
                else:
                    train_groups = np.append(train_groups, groups.detach().cpu().numpy(), axis=0)
        # Process losses to compute beta cover weights 
        instance_weights = model.compute_beta_cover(self.args.seed, epoch, self.train_dataset, train_losses, train_groups)
        # Update "instance_weights of self.train_dataset in dataloader (in the middle of training)
        # TODO: Check if the dataloader which is consistent, is actually using the updated weights. 
        self.train_dataset = self.train_dataset.remove_columns("instance_weight")
        self.train_dataset = self.train_dataset.add_column("instance_weight", instance_weights)


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(self.train_dataset, generator=generator)
                # TODO: Currently group reweighting is not required.
                return RandomSampler(self.train_dataset)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )        


    def get_train_dataloader(self):
        # add features to the dataset too.
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
                
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        
        """
        train_feature_dict = {}
        for index, row in train_features.iterrows():
            guid = row.guid
            emb = row.emb
            pred_prob = row.pred_probs
            target = np.asarray([row.target])
            feature_vector =  list(np.concatenate([emb, pred_prob, target]))
            train_feature_dict[guid] = feature_vector
        # Add other features : Dynamics, perturbed features etc...
        
        train_feature_list = []
        for id_, ex in enumerate(train_dataset):
            guid = ex["guid"]
            train_feature_list.append(train_feature_dict[guid])

        train_dataset = train_dataset.add_column("group_features", train_feature_list)
        """
        
        logger.info(f'Train features for adversary model loaded')

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    def training_step_primary(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        # Freeze grouper model parameters while training primary model
        for param in model.module.grouper_model.parameters():
            param.requires_grad = False

        inputs = self._prepare_inputs(inputs)

        # Sagemaker not implimented

        with self.autocast_smart_context_manager():
            groups = inputs["group"]
            group_distributions = inputs.get("group_distribution", None)
            instance_weights = inputs.get("instance_weight", None)
            # del inputs["group"]
            if group_distributions is not None:
                del inputs["group_distribution"]
            # Only reweight during task adversary phase.
            # if instance_weights is not None:
            #     del inputs["instance_weight"]
            loss, outputs = self.compute_loss_primary(model, inputs, return_outputs=True)
            # Here is where GCDRO loss was computed.
            loss = loss.mean() # This is not need.
        
        if isinstance(outputs, dict):
            logits = tuple(v for k, v in outputs.items() if k not in ["loss"])
        else:
            logits = outputs[1:]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        #wandb.log({"loss": loss})

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        # Unfreeze grouper model parameters while training primary model
        for param in model.module.grouper_model.parameters():
            param.requires_grad = True

        return loss.detach(), logits


    def training_step_adversary(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        # Freeze task model parameters while training adversary grouper model
        for param in model.module.task_model.parameters():
            param.requires_grad = False
        
        inputs = self._prepare_inputs(inputs)

        # Sagemaker not implimented

        with self.autocast_smart_context_manager():
            groups = inputs["group"]
            group_distributions = inputs.get("group_distribution", None)
            instance_weights = inputs.get("instance_weight", None)
            # del inputs["group"]
            if group_distributions is not None:
                del inputs["group_distribution"]
            if instance_weights is not None:
                del inputs["instance_weight"]
            loss, outputs = self.compute_loss_adversary(model, inputs, return_outputs=True)
            # Here is where GCDRO loss was computed.
            loss = loss.mean() # This is not needed.
        
        if isinstance(outputs, dict):
            logits = tuple(v for k, v in outputs.items() if k not in ["loss"])
        else:
            logits = outputs[1:]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        #wandb.log({"loss": loss})

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        # Unfreeze task model parameters while training adversary grouper model
        for param in model.module.task_model.parameters():
            param.requires_grad = True

        return loss.detach(), logits


    def compute_loss_primary(self, model, inputs, return_outputs=False):
        del inputs["guid"]
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs["adversary"] = False
        outputs = model(**inputs)
        # loss should not be reduced. 
        # handle loss computation across GPUs.

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def compute_loss_adversary(self, model, inputs, return_outputs=False):
        del inputs["guid"]
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs["adversary"] = True
        outputs = model(**inputs)
        # loss should not be reduced. 
        # handle loss computation across GPUs.

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_features = self.eval_features
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        
        """
        eval_feature_list = []
        for id_, ex in enumerate(eval_dataset):
            guid = ex["guid"]
            features = eval_features.loc[eval_features["guid"] == ex["guid"]]
            emb = features["emb"].to_numpy()[0]
            pred_prob = features["pred_probs"].to_numpy()[0]
            target = features["target"].to_numpy()
            eval_feature_list.append(list(np.concatenate([emb, pred_prob, target])))
            # Add other features : Dynamics, perturbed features etc...

        eval_dataset = eval_dataset.add_column("group_features", eval_feature_list)
        """

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        start_time = time.time()

        # Declare an evaluation loss computer object.
        if self.dro_args.is_robust:
            if not self.dro_args.use_group_weights:
                group_list = [ex["group"] for ex in self.eval_dataset]
                unique_groups, group_counts = np.unique(group_list, return_counts=True)
                n_groups = len(unique_groups)
                group_counts = torch.LongTensor(group_counts)
            else:
                group_distributions = np.asarray([ex["group_distribution"] for ex in self.eval_dataset])
                group_list = np.argmax(group_distributions, axis=1)
                unique_groups, group_counts = np.unique(group_list, return_counts=True)
                n_groups = len(unique_groups)
                group_counts = torch.LongTensor(group_counts)
                
            self.val_loss_computer = LossComputer(
                dro_args=self.dro_args,
                training_args=self.args,
                # dataset=dataset['train_data'], ## Why is this needed? to compute n_groups, group_counts which are passed as arguments now.
                n_groups=n_groups,
                group_counts=group_counts)
                # adj=adjustments)


        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Print stats after evaluation loop complete.
        # if self.dro_args.is_robust: 
        #     self.val_loss_computer.log_stats(logger, True)
        #     self.log(self.val_loss_computer.get_stats(self.model, self.args))
        """
        if self.dro_args.is_robust and self.dro_args.automatic_adjustment:
            gen_gap = self.val_loss_computer.avg_group_loss - self.train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(self.train_loss_computer.group_counts)
            self.train_loss_computer.adj = adjustments
            logger.info('Adjustments updated\n')
            for group_idx in range(self.train_loss_computer.n_groups):
                logger.info(
                    f'  {group_idx}:\t'
                    f'adj = {self.train_loss_computer.adj[group_idx]:.3f}\n')
        """

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None

        # also compute current groups so that we are selecting worst group performance based on predicted group performance.
        groups_host = None
        all_groups = None
        
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            # TODO(bparan): Inputs needs to be stripped of non-tensor metadata to be sent to model forward function.
            # Metadata information can be used to either log model performance, or provide group information. 
            # del inputs["guid"]

            # compute argmax group on the evalset
            inputs = self._prepare_inputs(inputs)
            group_distribution = model.grouper_model(inputs["group_features"])
            group = torch.argmax(group_distribution, axis=1)

            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if group is not None:
                groups = self._nested_gather(group)
                groups_host = groups if groups_host is None else nested_concat(groups_host, groups, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            
            
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if groups_host is not None:
                    groups = nested_numpify(groups_host)
                    all_groups = (
                        groups if all_groups is None else nested_concat(all_groups, groups, padding_index=-100)
                    )
                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if groups_host is not None:
            groups = nested_numpify(groups_host)
            all_groups = (
                groups if all_groups is None else nested_concat(all_groups, groups, padding_index=-100)
            )


        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_groups is not None:
            all_groups = nested_truncate(all_groups, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # Compute Worst Group Metrics, if group information is evailable in the evaluation set.
        if hasattr(self, "val_loss_computer"):
            
            key = "accuracy"
            pred = self._prepare_input(torch.tensor((np.argmax(all_preds,1)==all_labels), dtype=torch.float32))

            if self.args.select_predicted_worst_group:
                n_eval_groups = model.n_slices 
                groups = self._prepare_input(torch.tensor(groups)) #TODO: this may have to change for multigpus.
                group_map = (groups == self._prepare_input(torch.arange(model.n_slices).unsqueeze(1).long())).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans
                group_acc = (group_map @ pred.view(-1))/group_denom
            else:
                n_eval_groups = self.val_loss_computer.n_groups
                groups = self._prepare_input(torch.tensor([ex["group"] for ex in self.eval_dataset]))
                group_acc = self.val_loss_computer.compute_group_avg(pred, groups)[0]

            for group_idx in range(n_eval_groups):
                metrics[f"group_{key}_{group_idx}"] = group_acc[group_idx].item()

            if self.args.select_predicted_worst_group:
                # select accuracy of top k worst loss groups and assign a new probability to them.
                # find 50% of groups that have lowest group counts
                # mega_group 0 rest are mega_group 1
                # alpha proportion of the groups ought to be selected
                # top_worst_groups = torch.argsort(group_acc)[:int(len(group_acc)/2)+ 1].cpu().numpy()
                top_worst_groups = torch.argsort(group_acc)[:int(len(group_acc) * self.dro_args.alpha)].cpu().numpy()
                mega_groups = self._prepare_input(torch.tensor([(0 if group.item() in top_worst_groups else 1) for group in groups]))
                mega_group_map = (mega_groups == self._prepare_input(torch.arange(2).unsqueeze(1).long())).float()
                mega_group_count = mega_group_map.sum(1)
                mega_group_denom = mega_group_count + (mega_group_count==0).float() # avoid nans
                mega_group_acc = (mega_group_map @ pred.view(-1))/mega_group_denom
                for group_idx in range(2):
                    metrics[f"megagroup_{key}_{group_idx}"] = mega_group_acc[group_idx].item()

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        groups = inputs["group"]
                        group_distributions = inputs.get("group_distribution", None)
                        instance_weights = inputs.get("instance_weight", None)
                        group_features = inputs.get("group_features", None)
                        if group_distributions is not None:
                            del inputs["group_distribution"]
                        if instance_weights is not None:
                            del inputs["instance_weight"]
                        if group_features is not None:
                            del inputs["group_features"]
                        del inputs["group"]
                        del inputs["guid"]
                        # We are also loading eval features.
                        outputs = model.task_model(**inputs)
                        #loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        loss = outputs[0]
                        # loss on inividual elements of batch
                        loss = loss.mean() # reduce the loss here.
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        #TODO: Remove non-tensorizable elements from inputs.
                        groups = inputs["group"]
                        group_features = inputs.get("group_features", None)
                        del inputs["guid"]
                        del inputs["group"]
                        #del inputs["group"]
                        if self.dro_args.use_group_weights or "group_distribution" in inputs:
                            del inputs["group_distribution"]
                        if "instance_weight" in inputs:
                            del inputs["instance_weight"]
                        if group_features is not None:
                            del inputs["group_features"]
                        outputs = model.task_model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs[0]
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Saving Trainer.model separately into task model and grouper model")
                task_model = self.model.task_model
                grouper_model = self.model.grouper_model
                if state_dict is None:
                    state_dict = task_model.state_dict()
                # torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                task_model.save_pretrained(output_dir, state_dict=state_dict)
                grouper_state_dict = grouper_model.state_dict()
                os.makedirs(os.path.join(output_dir, "grouper"), exist_ok=True)
                torch.save(grouper_state_dict, os.path.join(output_dir, "grouper", WEIGHTS_NAME))   
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))