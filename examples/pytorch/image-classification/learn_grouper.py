#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
import sys
from xml import dom
import jsonlines
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
from torch.utils.data import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    HfArgumentParser,
    TrainerSlicer,
    TrainingArguments,
    DroArguments,
    DominoTrainingArguments,
    cartography_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from domino_learnt_slicer import DominoSlicer


""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: Optional[str] = field(
        default=None, metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Path of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
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
    k_fold: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Indicate which fold of the training data to use. -1 means use all training data"
        },
    )
    train_metadata_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Group, and group distribution information (indexed by image id/guid) for train dataset"
        },
    )
    validation_metadata_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Group, and group distribution information (indexed by image id/guid) for train dataset"
        },
    )
    train_feature_file: Optional[str] = field(
        default=None, metadata={"help": "A a Meerkat dataframe consisting of train features for group membership.."}
    )
    validation_feature_file: Optional[str] = field(
        default=None, metadata={"help": "A a Meerkat dataframe consisting of validation features for group membership."}
    )

    def __post_init__(self):
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["validation"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    adversary_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained adversary model"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    n_slices: int = field(
        default=9,
        metadata={
            "help": "Number of group assignments to learn."
        }
    )
    n_features: int = field(
        default=772,
        metadata={
            "help": "Number of group assignments to learn."
        }
    )
    entropy_reg: float = field(
        default=0.0,
        metadata={
            "help": "Use Entropy Regularizer"
        }
    )
    marginal_reg: float = field(
        default=0.0,
        metadata={
            "help": "Use Entropy Regularizer"
        }
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    groups = torch.tensor([example["group"] for example in examples])
    instance_weights = torch.tensor([example.get("instance_weight", 1.0) for example in examples])
    group_distribution = torch.stack([torch.tensor(example["group_distribution"]) for example in examples])
    group_features = torch.stack([torch.tensor(example["group_features"]) for example in examples])
    guids = [example["guid"] for example in examples]
    return {"pixel_values": pixel_values, "group" : groups, "group_features": group_features, "labels": labels, "instance_weight": instance_weights, "guid": guids, "group_distribution": group_distribution}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DominoTrainingArguments, DroArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, dro_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, dro_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset and prepare it for the 'image-classification' task.
    task = data_args.task_name

    ds = load_dataset(
        path=data_args.dataset_name,
        # "imagefolder",
        #data_args.dataset_config_name,
        data_files=data_args.data_files,
        cache_dir=model_args.cache_dir,
        task="image-classification",
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds["train"].train_test_split(data_args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.

    is_regression = ds["train"].features["labels"].dtype in ["float32", "float64"]
    labels = ds["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label


    # if doing k fold validation, then train and validation are obtained from splits in train itself.
    if data_args.k_fold>=0:
        # Shuffle training data (but order should be same for all kfold experiments)
        ds["train"] = ds["train"].shuffle(seed=666, load_from_cache_file=False)
        K = 5 # TODO: fixing number of folds to 5 (change this later)
        split_size = int(len(ds["train"])/K) 
        validation_indices = np.arange(data_args.k_fold*split_size, (data_args.k_fold+1)*split_size)
        train_indices = np.asarray(list(set([ex for ex in range(0, len(ds["train"]))]) - set([ex for ex in validation_indices])))
        train_split = ds["train"].select(indices=train_indices)
        test_split = ds["train"].select(indices=validation_indices)
        ds =  DatasetDict({"train": train_split, "validation": test_split})


    # Load the accuracy metric from the datasets package
    metric = datasets.load_metric("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=True, # Adding this argument because most pretrained Image classifiers on ImageNet have hardcoded 1000 classes.
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )


    # ds["train"][0]['image'].filename to get groups
    
    train_guid_array = []
    for ex in ds["train"]:
        filename = ex["image"].filename
        image_id = filename.split("/")[-1]
        train_guid_array.append(image_id)
    validation_guid_array = []
    for ex in ds["validation"]:
        filename = ex["image"].filename
        image_id = filename.split("/")[-1]
        validation_guid_array.append(image_id)

    # get group information to send to trainer_dro.
    train_metadata = json.loads(open(data_args.train_metadata_file).read())
    validation_metadata = json.loads(open(data_args.validation_metadata_file).read())
    # get group, group_distribution id indexed by guid

    # Load external feature files
    train_features = json.loads(open(data_args.train_feature_file).read())
    eval_features = json.loads(open(data_args.validation_feature_file).read())

    ds["train"] = ds["train"].add_column("guid", train_guid_array)
    train_groups = [train_metadata[guid]["group"] for guid in train_guid_array]
    ds["train"] = ds["train"].add_column("group", train_groups)
    ds["train"] = ds["train"].add_column("group_distribution", [train_metadata[guid]["group_distribution"] for guid in train_guid_array])
    ds["train"] = ds["train"].add_column("group_features", [train_features[guid] for guid in train_guid_array])

    ds["validation"] = ds["validation"].add_column("guid", validation_guid_array)
    validation_groups = [validation_metadata[guid]["group"] for guid in validation_guid_array]
    ds["validation"] = ds["validation"].add_column("group", validation_groups)
    ds["validation"] = ds["validation"].add_column("group_distribution", [validation_metadata[guid]["group_distribution"] for guid in validation_guid_array])
    ds["validation"] = ds["validation"].add_column("group_features", [eval_features[guid] for guid in validation_guid_array])

    # group_features 
    
    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Set the training transforms
        ds["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        ds["validation"].set_transform(val_transforms)

    if dro_args.is_robust and training_args.do_train:
        unique_groups, group_counts = np.unique(train_groups, return_counts=True)
        dro_args.n_groups = len(unique_groups)
        dro_args.group_counts = torch.LongTensor(group_counts)
    
    if dro_args.reweight_groups and training_args.do_train:
        # For ERM models, you need group_counts for weighted sampling.
        unique_groups, group_counts = np.unique(train_groups, return_counts=True)
        dro_args.n_groups = len(unique_groups)
        dro_args.group_counts = torch.LongTensor(group_counts)


    # Declare model for group prediction, which is enveloped in a Learned DOMINO model
    domino_model = DominoSlicer(model_args, training_args, dro_args, model)

    # load a pretrained adversarial grouper model
    if model_args.adversary_model_name_or_path:
        state_dict = torch.load(model_args.adversary_model_name_or_path , map_location="cpu")
        state_dict_new = OrderedDict({k.replace("grouper_model.", ""):v for k,v in state_dict.items()})
        domino_model.grouper_model.load_state_dict(state_dict_new)

    # When grouper model is already trained(ie. model_name_or_path is a directory with a grouper model, initialize with it)
    if os.path.exists(model_args.model_name_or_path) and os.path.exists(os.path.join(model_args.model_name_or_path, "grouper")):
        state_dict = torch.load(os.path.join(model_args.model_name_or_path, "grouper", "pytorch_model.bin") , map_location="cpu")
        state_dict_new = OrderedDict({k.replace("grouper_model.", ""):v for k,v in state_dict.items()})
        domino_model.grouper_model.load_state_dict(state_dict_new)

    # Initalize our trainer
    trainer = TrainerSlicer(
        model=domino_model,
        args=training_args,
        dro_args=dro_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = ds["validation"]
        # predict_dataset = predict_dataset.remove_columns("labels")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {task} *****")
                writer.write("index\tprediction\n")
                for index, (guid, item) in enumerate(zip(validation_guid_array, predictions)):
                    if is_regression:
                        writer.write(f"{index}\t{guid}\t{item:3.3f}\n")
                    else:
                        item = labels[item]
                        writer.write(f"{index}\t{guid}\t{item}\n")

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
