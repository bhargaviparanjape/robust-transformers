# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .debug_utils import DebugOption
from .file_utils import (
    ExplicitEnum,
    cached_property,
    get_full_repo_name,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    torch_required,
)
from .trainer_utils import EvaluationStrategy, HubStrategy, IntervalStrategy, SchedulerType, ShardedDDPOption
from .utils import logging
from .training_args import TrainingArguments


if is_torch_available():
    import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as sm_dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"


@dataclass
class DominoTrainingArguments(TrainingArguments):
    grouper_learning_rate: float = field(
        default=2e-5, 
        metadata={
            "help": "Learning rate for the grouper model."
        }
    )
    adversary_warmup: int = field(
        default=-1,
        metadata={
            "help": "Number of update steps after which adversary will not be trained."
        }
    )
    select_predicted_worst_group: bool = field(
        default=False,
        metadata={
            "help": "Select worst group based on AGRO predictions."
        }        
    )
    select_mega_worst_group: bool = field(
        default=True,
        metadata={
            "help": "Select based on alpha worst groups. Used in DOMINO baseline and True by default."
        }        
    )
    eiil: bool = field(
        default=False, 
        metadata={
            "help": "Turn on flag for EIIL environment inference.",
        }
    )
    grouper_model_size: int = field(
        default=128,
        metadata={
            "help": "Size of grouper model MLP.",
        }
    )