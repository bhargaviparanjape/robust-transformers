# load for each fold of training data, corresponding model, pretrained model and split out Ypred, Y 
# Combine all that data and create 6 folds. 
# For Dev set, do something similar, but now, the strategy for worst -group selection criterion is different. Recombining some of the low error groups together by class, creates a distribution similar to what was used in clean partitioning.

#!/usr/bin/env python
# coding=utf-8
"""
Code to create feature representations for instances of a text classification task, for automatic spurious-feature slice-discovery.
Supported Feature types:
Given a model path (pretrained or finetuned), return 
    * CLS 
    * Maxpool representations
Given two model representations (pretrained and finetunde, both), return:
    * Difference in CLS representations.
"""

from copyreg import pickle
from fileinput import filename
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from this import d
from typing import Optional
import tqdm
import pickle


import datasets
import numpy as np
from datasets import load_dataset, load_metric
import meerkat as mk
import pandas as pd
from numpy import dot
from numpy.linalg import norm

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    CartographyDataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainerDro,
    TrainingArguments,
    DroArguments,
    cartography_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import nested_numpify, nested_detach
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from domino_slicer import DominoMixture, DominoSlicer

logger = logging.getLogger(__name__)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wanli": ("premise", "hypothesis"),
}


custom_task_to_keys = {
    "mnli_resplit": ("sentence1", "sentence2"),
    "wilds_civil_comments": ("sentence1", None),
    "winogrande": ("sentence", None),
    "sst2": ("sentence", None),
    "fever": ("sentence1", "sentence2"),
    "commonsenseqa": ("sentence1", "sentence2"),
    "wanli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    custom_task_name: Optional[str] = field(
        default="mnli_resplit",
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    create_features: Optional[bool] = field(
        default=False, metadata={"help": "Create training and evaluation data features."}
    )
    find_spurious_features: Optional[bool] = field(
        default=False, metadata={"help": "Use differences between pretrained and finetuned model to find pretrained features."}
    )
    cluster_dev_features: Optional[bool] = field(
        default=False, metadata={"help": "Cluster training and evaluation data features."}
    )
    cluster_train_features: Optional[bool] = field(
        default=False, metadata={"help": "Cluster training and evaluation data features."}
    )
    n_slices: Optional[int] = field(default=10, metadata={"help": "number of error slices to analyze."})
    n_mixture_components: Optional[int] = field(default=50, metadata={"help": "number of mixture components."})
    init_type: Optional[str] = field(default="confusion", metadata={"help": "Type of initialization for Mixture model."})
    include_ypred: Optional[bool] = field(
        default=False, metadata={"help": "Included predicted class for train time filtering"}
    )


    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["tsv", "csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    pretrained_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def create_features(training_args, trainer, dataloader, model, config, split, is_pretraining=False):
    # eval_datalooader = trainer.g
    total_train_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    num_examples = len(dataloader) # Number of batches.

    logger.info("***** Running feature generation for training dataset *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")

    train_representations = np.array([], dtype='float32').reshape(0,model.config.hidden_size)
    mean_train_representations = np.array([], dtype='float32').reshape(0,model.config.hidden_size)
    all_contextual_representations = []
    train_logits = np.array([], dtype='float32').reshape(0, config.num_labels)
    train_labels = np.array([], dtype='float32').reshape(0)
    train_guids = []
    train_lengths = []
    train_groups = np.array([], dtype='float32').reshape(0)
    ## creat a meerkat table.
    for step, inputs in tqdm.tqdm(enumerate(dataloader)):
        inputs = trainer._prepare_inputs(inputs)
        with torch.no_grad():
            with trainer.autocast_smart_context_manager():
                #TODO: Remove non-tensorizable elements from inputs.
                guid = inputs["guid"]
                group = inputs["group"]
                labels = inputs["labels"]
                train_guids += guid
                train_labels = np.concatenate((train_labels, nested_numpify(labels)), axis=0)
                train_groups = np.concatenate((train_groups, nested_numpify(group)), axis=0)
                del inputs["guid"]
                del inputs["group"]
                if "group_distribution" in inputs:
                    del inputs["group_distribution"]
                inputs["output_hidden_states"] = True
                outputs = model(**inputs)

                """
                for batch_index, seq_length in enumerate(inputs["attention_mask"].sum(1)):
                    stacked_output = torch.vstack([outputs["hidden_states"][i][batch_index, :, :].unsqueeze(0) for i in range(model.config.num_hidden_layers + 1)]).cpu().numpy()
                    all_contextual_representations.append(stacked_output)
                    train_lengths.append(seq_length)
                """

                last_hidden_layer = outputs["hidden_states"][-1]
                classifier_representations = last_hidden_layer[:,0,:]
                train_representations = np.concatenate((train_representations, nested_numpify(classifier_representations)), axis=0)
                logits = outputs[1]
                train_logits = np.concatenate((train_logits, nested_numpify(logits)), axis=0)
                # Record predicted class as well.                

    """
    Meerkat DataPanel <https://github.com/robustness-gym/meerkat>`_ with columns
    "emb", "target", and "pred_probs". After loading the DataPanel, you can discover
    underperforming slices of the validation dataset with the following:
    """
    dp = mk.DataPanel({
    'guid': train_guids, 
    'group': train_groups,
    'emb': train_representations,
    # 'mean_pooled': mean_train_representations,
    'target': train_labels,
    'pred_probs': train_logits,
    # 'all_hidden_states': all_contextual_representations,
    # 'sequence_lengths': train_lengths
    })

    pd_df = mk.DataPanel.to_pandas(dp)
    clustering_cache = os.path.join(training_args.output_dir, "clustering") 
    if not os.path.exists(clustering_cache):
        os.mkdir(clustering_cache)
    if not is_pretraining:
        pd_df.to_pickle(os.path.join(training_args.output_dir, "clustering", "{0}.pkl".format(split)))
    else:
        pd_df.to_pickle(os.path.join(training_args.output_dir, "clustering", "{0}_pretrained.pkl".format(split)))

    #return all_contextual_representations, train_lengths

    