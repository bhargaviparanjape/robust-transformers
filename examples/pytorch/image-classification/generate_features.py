#!/usr/bin/env python
# coding=utf-8
# generating features for image classification

import logging
import os
import sys
import jsonlines
import json
from dataclasses import dataclass, field
from typing import Optional
import tqdm
import pickle

import datasets
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
import numpy as np
import meerkat as mk
import pandas as pd
import torch
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
    y_log_likelihood_weight: float = field(
        default=1.0,
        metadata={"help": "Weight on reference class label (Y) of domino slicer."}
    )
    y_hat_log_likelihood_weight: float = field(
        default=1.0,
        metadata={"help": "Weight on predicted class label (Y hat) of domino slicer."}
    )
    cluster_assgn_file: str = field(
        default=None,
        metadata={"help": "Path to error-aware cluster assignment file."}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "output file to store newly re-grouped data."}
    )
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
    num_folds: Optional[int] = field(
        default=5,
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
    create_features: Optional[bool] = field(
        default=False, metadata={"help": "Create training and evaluation data features."}
    )
    cluster_train_features: Optional[bool] = field(
        default=False, metadata={"help": "Cluster training and evaluation data features."}
    )
    cluster_dev_features: Optional[bool] = field(
        default=False, metadata={"help": "Cluster training and evaluation data features."}
    )
    assign_train_groups: Optional[bool] = field(
        default=False, metadata={"help": "Greedily Assign groups based on DOMINO membership."}
    )
    assign_dev_groups: Optional[bool] = field(
        default=False, metadata={"help": "Greedily Assign groups based on DOMINO membership."}
    )
    n_slices: Optional[int] = field(default=10, metadata={"help": "number of error slices to analyze."})
    n_mixture_components: Optional[int] = field(default=50, metadata={"help": "number of mixture components."})
    init_type: Optional[str] = field(default="confusion", metadata={"help": "Type of initialization for Mixture model."})
    include_ypred: Optional[bool] = field(
        default=False, metadata={"help": "Included predicted class for train time filtering"}
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
        metadata={"help": "Path to Trained model or model identifier from huggingface.co/models"},
    )
    pretrained_model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to Pretrained model."},
    )
    kfold_model_path_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
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


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    groups = torch.tensor([example["group"] for example in examples])
    group_distribution = torch.stack([torch.tensor(example["group_distribution"]) for example in examples])
    guids = [example["guid"] for example in examples]
    return {"pixel_values": pixel_values, "labels": labels, "guid": guids, "group" : groups, "group_distribution": group_distribution}

# (model_args, training_args, trainer, train_dataloader, model, config, split="train")
def create_features(model_args, training_args, trainer, dataloader, model, config, split, is_pretraining=False):

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

                """
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
                """
                inputs["output_hidden_states"] = True
                guid, group, group_distribution, labels = [inputs.pop(key) for key in ['guid', 'group', 'group_distribution', 'labels']]
                train_guids += guid
                train_labels = np.concatenate((train_labels, nested_numpify(labels)), axis=0)
                train_groups = np.concatenate((train_groups, nested_numpify(group)), axis=0)
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
                logits = outputs['logits']
                train_logits = np.concatenate((train_logits, nested_numpify(logits)), axis=0)
                # Record predicted class as well.
    
    if is_pretraining:
        dp = mk.DataPanel({
        'guid': train_guids, 
        'emb': train_representations})
    else:
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DroArguments))
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

    is_regression = ds["train"].features["labels"].dtype in ["float32", "float64"]
    labels = ds["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

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
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Pretrained model for feature generation
    pretrained_model = AutoModelForImageClassification.from_pretrained(
        model_args.pretrained_model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
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

    ds["train"] = ds["train"].add_column("guid", train_guid_array)
    train_groups = [train_metadata[guid]["group"] for guid in train_guid_array]
    ds["train"] = ds["train"].add_column("group", train_groups)
    ds["train"] = ds["train"].add_column("group_distribution", [train_metadata[guid]["group_distribution"] for guid in train_guid_array])

    ds["validation"] = ds["validation"].add_column("guid", validation_guid_array)
    validation_groups = [validation_metadata[guid]["group"] for guid in validation_guid_array]
    ds["validation"] = ds["validation"].add_column("group", validation_groups)
    ds["validation"] = ds["validation"].add_column("group_distribution", [validation_metadata[guid]["group_distribution"] for guid in validation_guid_array])



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

    if data_args.create_features:

        # Set train and validation transforms.
        ds["train"].set_transform(train_transforms)
        ds["validation"].set_transform(val_transforms) 

        trainer = TrainerDro(
            model=model,
            args=training_args,
            dro_args=dro_args,
            train_dataset=ds["train"] ,
            eval_dataset=ds["validation"] ,

            compute_metrics=compute_metrics,
            tokenizer=feature_extractor,
            data_collator=collate_fn,
        )

        train_dataloader = trainer.get_train_dataloader()
        eval_dataloader = trainer.get_eval_dataloader(ds["validation"])

        create_features(model_args, training_args, trainer, train_dataloader, model, config, split="train")
        create_features(model_args, training_args, trainer, eval_dataloader, model, config, split="dev")

        trainer = TrainerDro(
            model=pretrained_model,
            args=training_args,
            dro_args=dro_args,
            train_dataset=ds["train"] ,
            eval_dataset=ds["validation"] ,

            compute_metrics=compute_metrics,
            tokenizer=feature_extractor,
            data_collator=collate_fn,
        )

        train_dataloader = trainer.get_train_dataloader()
        eval_dataloader = trainer.get_eval_dataloader(ds["validation"])

        create_features(model_args, training_args, trainer, train_dataloader, pretrained_model, config, split="train", is_pretraining=True)
        create_features(model_args, training_args, trainer, eval_dataloader, pretrained_model, config, split="dev", is_pretraining=True)

    
    if data_args.cluster_dev_features:
        split = "dev"
        logger.info("***** Loading features for {0} dataset *****".format(split))
        pd_df = pd.read_pickle(os.path.join(training_args.output_dir, "clustering", "{0}.pkl".format(split)))
        logits = np.stack(pd_df["pred_probs"].to_numpy())
        dp = mk.DataPanel({
            'guid': pd_df["guid"].to_list(),
            'group': np.stack(pd_df["group"].to_numpy()),
            'emb': np.stack(pd_df["emb"].to_numpy()),
            'target': np.stack(pd_df["target"].to_numpy()),
            'pred_probs': np.asarray(torch.softmax(torch.tensor(logits), dim=-1))
        })
        domino = DominoSlicer(n_slices=data_args.n_slices, n_mixture_components=data_args.n_mixture_components, init_params=data_args.init_type, max_iter=200)
        domino.fit(
            data=dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        # Save the domino class as a pickle
        pickle.dump(domino, open(os.path.join(training_args.output_dir, "clustering", "{0}_dominoclass_{1}_slices.pkl".format(split, data_args.n_slices)), "wb"))
        dp["domino_slices"] = domino.transform(
            data=dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )


        # also load pretrained features in the dataframe
        pretrained_dp = pd.read_pickle(os.path.join(training_args.output_dir, "clustering", "{0}_pretrained.pkl".format(split)))
        # make meerkat dataset
        pretrained_dp = mk.DataPanel({
                'guid': pretrained_dp["guid"].to_list(),
                'pretrained_emb': np.stack(pretrained_dp["emb"].to_numpy()),
            })

        # Merge pretrained_comb_dp and comb_dp along guid column
        merged_dp = dp.merge(pretrained_dp, on="guid")

        # Save the domino object so that it be used to draw comparisons.
        pd_df = mk.DataPanel.to_pandas(merged_dp)
        logger.info("***** Dumping group assingments for {0} to file *****".format(split))
        pd_df.to_pickle(os.path.join(training_args.output_dir, "clustering", "{0}_output_{1}_slices.pkl".format(split, data_args.n_slices)))
        
        logger.info("***** Dumping groups for exploration for {0} to file *****".format(split))
        data_dict = {}
        slices_dict = {i:[] for i in range(-1, data_args.n_slices)}
        if split == "dev":
            for ex in ds["validation"]:
                data_dict[ex["guid"]] = ex
        elif split == "train":
            for ex in ds["train"]:
                data_dict[ex["guid"]] = ex
        for i in range(len(pd_df)):
            ## Usually analysis is done with group assignment only if value is above a threshold.
            slice = int(np.argmax(pd_df.iloc[i]["domino_slices"]))
            slice_val = np.max(pd_df.iloc[i]["domino_slices"])
            if slice_val > 0.90:
                chosen_slice = slice
            else:
                chosen_slice = -1
            guid = pd_df.iloc[i]["guid"]
            ex = data_dict[guid]
            ex["prediction"] = id2label[str(np.argmax(pd_df.iloc[i]["pred_probs"]))]
            # Only look at slices that are actually erroneous
            # if not np.argmax(pd_df.iloc[i]["pred_probs"]) == pd_df.iloc[i]["target"]:
            slices_dict[chosen_slice].append(ex)
        """
        with open(os.path.join(training_args.output_dir, "clustering", "{0}_analysis_{1}_slices.json".format(split, data_args.n_slices)), "w") as fout:
            # ignore all examples that were assigned slice -1
            for i in range(data_args.n_slices):
                for j, ex in enumerate(slices_dict[i]):
                    new_ex = {"slice": i, "label": ex["label"], "predicted": ex["prediction"], "guid": ex["guid"]}
                    fout.write(json.dumps(new_ex) + "\n")
        """
        
    if data_args.cluster_train_features:

        split = "dev"
        # Get json datasets from all split files of the training set.
        # Load training, validation and test data from file paths.
        kfold_model_prefix = model_args.kfold_model_path_prefix
        model_folders = [f"{kfold_model_prefix}{split_no}" for split_no in range(0, data_args.num_folds)]
        fold_dps = []
        for fold_no, model_path in enumerate(model_folders):
            fold_pd_df = pd.read_pickle(os.path.join(model_path, "clustering", "{0}.pkl".format(split)))
            fold_logits = np.stack(fold_pd_df["pred_probs"].to_numpy())
            fold_dps.append(mk.DataPanel({
                'guid': fold_pd_df["guid"].to_list(),
                'group': np.stack(fold_pd_df["group"].to_numpy()),
                'emb': np.stack(fold_pd_df["emb"].to_numpy()),
                'target': np.stack(fold_pd_df["target"].to_numpy()),
                'pred_probs': np.asarray(torch.softmax(torch.tensor(fold_logits), dim=-1))
            }))

        comb_dp = mk.concat(fold_dps)

        # y_hat_log_likelihood_weight may have to be up-played when slices are fewer and far between.
        domino = DominoSlicer(n_slices=data_args.n_slices, 
        n_mixture_components=data_args.n_mixture_components, 
        init_params=data_args.init_type,
        y_log_likelihood_weight=data_args.y_log_likelihood_weight,
        y_hat_log_likelihood_weight=data_args.y_hat_log_likelihood_weight,
        max_iter=200)

        domino.fit(
            data=comb_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        # Save the domino class as a pickle
        pickle.dump(domino, open(os.path.join(training_args.output_dir, "clustering", "error_aware_dominoclass_{0}_slices.pkl".format(data_args.n_slices)), "wb"))
        comb_dp["domino_slices"] = domino.transform(
            data=comb_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )

        # also load pretrained features in the dataframe
        pretrained_fold_dps = []
        for fold_no, model_path in enumerate(model_folders):
            fold_pd_df = pd.read_pickle(os.path.join(model_path, "clustering", "{0}_pretrained.pkl".format(split)))
            pretrained_fold_dps.append(mk.DataPanel({
                'guid': fold_pd_df["guid"].to_list(),
                'pretrained_emb': np.stack(fold_pd_df["emb"].to_numpy()),
            }))

        pretrained_comb_dp = mk.concat(pretrained_fold_dps)

        # Merge pretrained_comb_dp and comb_dp along guid column
        merged_dp = comb_dp.merge(pretrained_comb_dp, on="guid")

        pd_df = mk.DataPanel.to_pandas(merged_dp)
        logger.info("***** Dumping group assingments for {0} to file *****".format(split))
        pd_df.to_pickle(os.path.join(training_args.output_dir, "clustering", "error_aware_output_{0}_slices.pkl".format(data_args.n_slices)))

    if data_args.assign_train_groups:
        pd_df = pd.read_pickle(data_args.cluster_assgn_file)
        slices =  np.stack(pd_df["domino_slices"].to_numpy())
        group_assignment = {}
        group_distributions = {}
        for i in range(len(pd_df)):
            ## Usually analysis is done with group assignment only if value is above a threshold.
            slice = int(np.argmax(pd_df.iloc[i]["domino_slices"]))
            slice_val = np.max(pd_df.iloc[i]["domino_slices"])
            chosen_slice = slice
            guid = pd_df.iloc[i]["guid"]
            group_distributions[guid] = list(pd_df.iloc[i]["domino_slices"])
            group_assignment[guid] = chosen_slice


        slicing_metadata_directory = os.path.join(data_args.dataset_name, "automatic_slicing") 
        if not os.path.exists(slicing_metadata_directory):
            os.mkdir(slicing_metadata_directory)
        output_file = os.path.join(slicing_metadata_directory, data_args.output_file)

        with open(output_file, "w") as fout:
            guid_indexed_dict = {}
            for ex in ds["train"]:
                guid = ex["guid"]
                group = group_assignment[guid]
                group_distribution=[0]*data_args.n_slices
                group_distribution[group] = 1
                dict_ = {"guid":guid, "group":group, "group_distribution":group_distribution}
                guid_indexed_dict[guid] = dict_
            fout.write(json.dumps(guid_indexed_dict))


if __name__ == "__main__":
    main()   
    
