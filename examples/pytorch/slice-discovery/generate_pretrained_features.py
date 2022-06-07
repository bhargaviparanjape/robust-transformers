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
    "wnli": ("sentence1", "sentence2"),
}


custom_task_to_keys = {
    "mnli_resplit": ("sentence1", "sentence2"),
    "wilds_civil_comments": ("sentence1", None),
    "winogrande": ("sentence", None),
    "commonsenseqa": ("sentence1", "sentence2"),
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
    Dump a sample for isotropy measurements. 
    Sample 2*N token positions where i!=j. N=10K
    Compute cosine, dump to file.
    Get baseline consine values
    """
    """
    N = 5000
    X = []
    Y = []
    for i in range(2*N):
        # Generate a sample
        train_id = np.random.randint(len(train_guids))
        # Generate tok_id 
        tok_id = np.random.randint(train_lengths[train_id].item())
        X.append(all_contextual_representations[train_id][:, tok_id, :])
        # Generate a sample
        train_id = np.random.randint(len(train_guids))
        # Generate tok_id 
        tok_id = np.random.randint(train_lengths[train_id].item())
        Y.append(all_contextual_representations[train_id][:, tok_id, :])

    consine_similarities_all_layers = []
    for x, y in zip(X,Y):
        cos_sim_all_layers = []
        for i in range(model.config.num_hidden_layers+1):
            cos_sim_all_layers.append(dot(x[i], y[i])/(norm(x[i])*norm(y[i])))
        consine_similarities_all_layers.append(cos_sim_all_layers)

    dp = mk.DataPanel({
    'cosines': np.asarray(consine_similarities_all_layers)})
    pd_df = mk.DataPanel.to_pandas(dp)
    if not is_pretraining:
        pd_df.to_pickle(os.path.join(training_args.output_dir, "clustering", "{0}_random_cosines.pkl".format(split)))
    else:
        pd_df.to_pickle(os.path.join(training_args.output_dir, "clustering", "{0}_random_cosines_pretrained.pkl".format(split)))
    """

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


def main():
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
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()  

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")  

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load tarining, validation and test data from file paths.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}
    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    # Labels
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.pretrained_model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the raw_datasets
    # TODO:For amazon, sentence1_key is fixed as "text"
    # sentence1_key, sentence2_key = "text", None
    # Custom MNLI with group info
    sentence1_key, sentence2_key = custom_task_to_keys[data_args.custom_task_name]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
    
    # Handling Label to ID mapping
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    # Max sequence length
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        
        result["guid"] = examples["guid"]
        result["group"] = examples["group"]
        return result

    # Preprocess datasets.
    if data_args.create_features:
        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator.
    if data_args.pad_to_max_length:
        data_collator = cartography_data_collator
    elif training_args.fp16:
        data_collator = CartographyDataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Metrics? 
    metric = load_metric("accuracy")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Look at trainer init and init a trainer even here: easiest if initialized? 
    # Initialize our Trainer

    if data_args.create_features:
        trainer = TrainerDro(
        model=model,
        args=training_args,
        dro_args=dro_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        )

        train_dataloader = trainer.get_train_dataloader()
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
        create_features(training_args, trainer, train_dataloader, model, config, split="train")
        create_features(training_args, trainer, eval_dataloader, model, config, split="dev")

        trainer = TrainerDro(
        model=pretrained_model,
        args=training_args,
        dro_args=dro_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        )
        train_dataloader = trainer.get_train_dataloader()
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
        create_features(training_args, trainer, train_dataloader, pretrained_model, config, split="train", is_pretraining=True)
        create_features(training_args, trainer, eval_dataloader, pretrained_model, config, split="dev", is_pretraining=True)

    if data_args.cluster_dev_features:
        id_to_label = {v:k for k,v in label_to_id.items()}
        split = "dev"
        logger.info("***** Loading features for {0} dataset *****".format(split))
        pd_df = pd.read_pickle(os.path.join(training_args.output_dir, "clustering", "{0}.pkl".format(split)))
        logits = np.stack(pd_df["pred_probs"].to_numpy())
        pd_df_pretrained = pd.read_pickle(os.path.join(training_args.output_dir, "clustering", "{0}_pretrained.pkl".format(split)))
        # Reuse current domino object only cluster by both pretrained and finetuned representation? 
        finetuned_emb = np.stack(pd_df["emb"].to_numpy())
        pretrained_emb = np.stack(pd_df_pretrained["emb"].to_numpy())
        concatenated_emb = np.concatenate((finetuned_emb, pretrained_emb), axis=1)
        difference_emb = pretrained_emb - finetuned_emb
        dp = mk.DataPanel({
            'guid': pd_df["guid"].to_list(),
            'group': np.stack(pd_df["group"].to_numpy()),
            'emb': concatenated_emb,
            'target': np.stack(pd_df["target"].to_numpy()),
            'pred_probs': np.asarray(torch.softmax(torch.tensor(logits), dim=-1))
        })
        domino = DominoSlicer(n_slices=data_args.n_slices, n_mixture_components=data_args.n_mixture_components, init_params=data_args.init_type, max_iter=200)
        domino.fit(
            data=dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        # Save the domino class as a pickle
        pickle.dump(domino, open(os.path.join(training_args.output_dir, "clustering", "{0}_dominoclass_{1}_slices_pretrained.pkl".format(split, data_args.n_slices)), "wb"))
        dp["domino_slices"] = domino.transform(
            data=dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        # Save the domino object so that it be used to draw comparisons.
        pd_df = mk.DataPanel.to_pandas(dp)
        logger.info("***** Dumping group assingments for {0} to file *****".format(split))
        pd_df.to_pickle(os.path.join(training_args.output_dir, "clustering", "{0}_output_{1}_slices_pretrained.pkl".format(split, data_args.n_slices)))
        logger.info("***** Dumping groups for exploration for {0} to file *****".format(split))
        data_dict = {}
        slices_dict = {i:[] for i in range(-1, data_args.n_slices)}
        if split == "dev":
            for ex in raw_datasets["validation"]:
                data_dict[ex["guid"]] = ex
        elif split == "train":
            for ex in raw_datasets["train"]:
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
            ex["prediction"] = id_to_label[np.argmax(pd_df.iloc[i]["pred_probs"])]
            # Only look at slices that are actually erroneous
            if not np.argmax(pd_df.iloc[i]["pred_probs"]) == pd_df.iloc[i]["target"]:
                slices_dict[chosen_slice].append(ex)
        with open(os.path.join(training_args.output_dir, "clustering", "{0}_analysis_{1}_slices_pretrained.txt".format(split, data_args.n_slices)), "w") as fout:
            # ignore all examples that were assigned slice -1
            for i in range(data_args.n_slices):
                fout.write("Slice {0}\n".format(i))
                for j, ex in enumerate(slices_dict[i]):
                    if data_args.custom_task_name == "mnli_resplit":
                        fout.write(str(j+1) + "\nPremise:" + ex["sentence1"] + "\nHypothesis:" + ex["sentence2"] + "\nGold:" + ex["label"] +  "\nPrediction:" + ex["prediction"] + "\n")
                    elif data_args.custom_task_name == "wanli":
                        fout.write(str(j+1) + "\nPremise:" + ex["premise"] + "\nHypothesis:" + ex["hypothesis"] + "\nGold:" + ex["label"] +  "\nPrediction:" + ex["prediction"] + "\n")
                    elif data_args.custom_task_name == "wilds_civil_comments":
                        fout.write(str(j+1) + "\Comment:" + ex["sentence1"] + "\nGold:" + ex["label"] +  "\nPrediction:" + ex["prediction"] + "\n")
                    elif data_args.custom_task_name == "commonsenseqa":
                        fout.write(str(j+1) + "\n" + ex["sentence1"] + "\n" + ex["sentence2"] + "\nGold:" + str(ex["label"]) +  "\nPrediction:" + str(ex["prediction"]) + "\n")
                    elif data_args.custom_task_name == "sst2":
                        fout.write(str(j+1) + "\nReview:" + ex["sentence"] + "\nGold:" + str(ex["label"]) +  "\nPrediction:" + str(ex["prediction"]) + "\n")
                    elif data_args.custom_task_name == "qqp":
                        fout.write(str(j+1) + "\nQuestion1:" + ex["question1"] + "\nQuestion2:" + ex["question2"] + "\nGold:" + str(ex["label"]) +  "\nPrediction:" + str(ex["prediction"]) + "\n")
                fout.write("\n\n\n")
        for i in range(-1, data_args.n_slices):
            print(len(slices_dict[i]))



if __name__ == "__main__":
    main()
