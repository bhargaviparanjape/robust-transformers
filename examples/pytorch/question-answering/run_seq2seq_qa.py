#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library's seq2seq models for question answering using the 🤗 Seq2SeqTrainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import copy
import re
import tqdm
import pdb
import json
import random

import datasets
from datasets import load_dataset, load_metric
from datasets import Dataset, DatasetDict

import transformers
from trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
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
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Data folder for newsQA"}
    )
    no_answer_threshold: Optional[float] = field(
        default=1.0, metadata={"help": "Custom no answer threshold for SQuAD 2.0."}
    )
    partial_inputs: Optional[str] = field(
        default=None, metadata={"help": "which kind of partial input perturbation to make"}
    )
    question_only: Optional[bool] = field(
        default=False, metadata={"help": "Question only training"}
    )
    passage_only: Optional[bool] = field(
        default=False, metadata={"help": "Passage only training"}
    )
    augment_data: Optional[bool] = field(
        default=False, metadata={"help": "Augment counterfactual data"}
    )
    partial_inputs_seed: Optional[int] = field(
        default=1234, metadata={"help": "random seed for perturbation"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    context_column: Optional[str] = field(
        default="context",
        metadata={"help": "The name of the column in the datasets containing the contexts (for question answering)."},
    )
    question_column: Optional[str] = field(
        default="question",
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    answer_column: Optional[str] = field(
        default="answers",
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    val_max_answer_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_answer_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
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
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_answer_length is None:
            self.val_max_answer_length = self.max_answer_length


question_answering_column_name_mapping = {
    "squad_v2": ("question", "context", "answer"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.dataset_name == "newsqa_custom":
            data_files = {
                    "train":os.path.join(data_args.data_dir, "train.json"),
                    "validation":os.path.join(data_args.data_dir, "validation.json"),
                    "test":os.path.join(data_args.data_dir, "test.json")}
            raw_datasets = load_dataset("json", cache_dir=model_args.cache_dir, data_files=data_files)
        else:
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=model_args.cache_dir)
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    new_datasets = {}
    if data_args.dataset_name == 'duorc':
        # raw_datasets["train"], raw_datasets["validation"], raw_datasets["test"] should be transformed to squad 2.0 format.
        for split in ["train", "validation", "test"]:
            new_dataset = {'id': [], 'context':[], 'question':[], 'answers': [], "title": []}
            answers = raw_datasets[split]["answers"]
            contexts = raw_datasets[split]["plot"]
            no_answers = raw_datasets[split]["no_answer"]
            for i, (ans, context, imp) in tqdm.tqdm(enumerate(zip(answers, contexts, no_answers))):
                if not imp:
                    # answer exists
                    # for a in ans:
                    #     start = None
                    #     if a in context:
                    #         start = context.find(a) #First occurrence may not be the best occurrence.
                    #         new_dataset['id'].append(raw_datasets[split][i]["question_id"])
                    #         new_dataset['question'].append(raw_datasets[split][i]["question"])
                    #         new_dataset['context'].append(raw_datasets[split][i]["plot"])
                    #         new_dataset['title'].append(raw_datasets[split][i]["title"])
                    #         new_dataset['answers'].append({"text": [a], 'answer_start': [start]})
                    #         # First occurrence of an answer in context.
                    #         break
                    new_dataset['id'].append(raw_datasets[split][i]["question_id"])
                    new_dataset['question'].append(raw_datasets[split][i]["question"])
                    new_dataset['context'].append(raw_datasets[split][i]["plot"])
                    new_dataset['title'].append(raw_datasets[split][i]["title"])
                    new_dataset['answers'].append({"text": ans, 'answer_start': [0]*len(ans)})
                else:
                    # answer does not exist.
                    new_dataset['id'].append(raw_datasets[split][i]["question_id"])
                    new_dataset['question'].append(raw_datasets[split][i]["question"])
                    new_dataset['context'].append(raw_datasets[split][i]["plot"])
                    new_dataset['title'].append(raw_datasets[split][i]["title"])
                    new_dataset['answers'].append({"text": [], 'answer_start': []})
            # Create a new arrow dataset to replace raw_datasets.
            new_datasets[split] = Dataset.from_dict(new_dataset)
        raw_datasets = DatasetDict(new_datasets)

    if data_args.dataset_name == 'newsqa_custom':
        import uuid
        new_datasets = {}
        for split in ["train", "validation", "test"]:
            new_dataset = {'id': [], 'context':[], 'question':[], 'answers': [], "title": [], "story_id": []}
            answers = raw_datasets[split]["answers"]
            contexts = raw_datasets[split]["story_text"]
            questions = raw_datasets[split]["question_text"]
            for i, (question, ans, context) in tqdm.tqdm(enumerate(zip(questions, answers, contexts))):
                new_dataset['id'].append(uuid.uuid4().hex)
                new_dataset['question'].append(raw_datasets[split][i]["question_text"])
                new_dataset['context'].append(raw_datasets[split][i]["story_text"])
                new_dataset['title'].append("")
                new_dataset['story_id'].append(raw_datasets[split][i]["storyId"])
                if "s" in ans and "e" in ans and ans['s'] != None:
                    answer_text = context[ans['s']:ans['e']].strip()
                    new_dataset['answers'].append({"text": [answer_text], 'answer_start': [ans['s']]})
                else:
                    new_dataset['answers'].append({"text": [], 'answer_start': []})
            # Create a new arrow dataset to replace raw_datasets.
            new_datasets[split] = Dataset.from_dict(new_dataset)
        raw_datasets = DatasetDict(new_datasets)

    # Preprocessing the datasets.
    # We need to generate and tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["validation"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = question_answering_column_name_mapping.get(data_args.dataset_name, None)
    if data_args.question_column is None:
        question_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        question_column = data_args.question_column
        if question_column not in column_names:
            raise ValueError(
                f"--question_column' value '{data_args.question_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.context_column is None:
        context_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        context_column = data_args.context_column
        if context_column not in column_names:
            raise ValueError(
                f"--context_column' value '{data_args.context_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.answer_column is None:
        answer_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        answer_column = data_args.answer_column
        if answer_column not in column_names:
            raise ValueError(
                f"--answer_column' value '{data_args.answer_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_answer_length for training.
    max_answer_length = data_args.max_answer_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    if data_args.augment_data:
        train_dataset = raw_datasets["train"]
        new_dataset = {'id': [], 'context':[], 'question':[], 'answers': [], "title": []}
        titles = train_dataset["title"]
        answers = train_dataset["answers"]
        contexts = train_dataset["context"]
        questions = train_dataset["question"]
        for i, (title, question, ans, context) in tqdm.tqdm(enumerate(zip(titles, questions, answers, contexts))):
            if len(ans['text']):
                # This instance has an answer. So add another with a random qiestion.
                while(1):
                    rand_idx = np.random.randint(len(questions))
                    if titles[rand_idx] != titles[i]:
                        new_question = questions[rand_idx]
                        break
                # Add logic such that an additional is addded only with some probability.
                new_dataset['context'].append(context)
                new_dataset['id'].append(train_dataset[i]["id"])
                new_dataset['question'].append(new_question)
                new_dataset['title'].append(title)
                new_dataset['answers'].append({'text': [], 'answer_start': []})
                if 'story_id' in train_dataset[i]:
                    if 'story_id' not in new_dataset:
                        new_dataset['story_id'] = []
                    new_dataset['story_id'].append(train_dataset[i]["story_id"])
            new_dataset['context'].append(context)
            new_dataset['id'].append(train_dataset[i]["id"])
            new_dataset['question'].append(question)
            new_dataset['title'].append(title)
            new_dataset['answers'].append(ans)
            if 'story_id' in train_dataset[i]:
                if 'story_id' not in new_dataset:
                    new_dataset['story_id'] = []
                new_dataset['story_id'].append(train_dataset[i]["story_id"])
        raw_datasets["train"] = Dataset.from_dict(new_dataset)
    
    if data_args.question_only:
        for split_name in ["train", "validation"]:
                dataset = raw_datasets[split_name]
                titles = dataset["title"]
                answers = dataset["answers"]
                contexts = dataset["context"]
                questions = dataset["question"]
                np.random.shuffle(contexts)
                c = list(zip(titles, contexts))
                np.random.shuffle(c)
                titles, contexts = zip(*c)
                new_dataset = {'id': [], 'context':[], 'question':[], 'answers': [], "title": []}
                for i, (title, question, ans, context) in tqdm.tqdm(enumerate(zip(titles, questions, answers, contexts))):
                    if not len(ans['text']):
                        new_dataset['context'].append(context)
                    else:
                        ans_words = ans['text'][0].split()
                        context_words = context.split()
                        if len(context_words)-len(ans_words) <= 0:
                            new_context = ans_words
                        else:
                            random_location = np.random.randint(len(context_words)-len(ans_words))
                            new_context = context_words[:random_location] + ans_words + context_words[random_location + len(ans_words):]
                        new_context = " ".join(new_context)
                        new_dataset['context'].append(new_context)
                    new_dataset['id'].append(dataset[i]["id"])
                    new_dataset['question'].append(question)
                    new_dataset['title'].append(title)
                    new_dataset['answers'].append(ans)
                    if 'story_id' in dataset[i]:
                        if 'story_id' not in new_dataset:
                            new_dataset['story_id'] = []
                        new_dataset['story_id'].append(dataset[i]["story_id"])
                raw_datasets[split_name] = Dataset.from_dict(new_dataset)

    if data_args.passage_only:
        for split_name in ["train", "validation"]:
            dataset = raw_datasets[split_name]
            titles = dataset["title"]
            answers = dataset["answers"]
            contexts = dataset["context"]
            questions = dataset["question"]
            new_dataset = {'id': [], 'context':[], 'question':[], 'answers': [], "title": []}
            for i, (title, question, ans, context) in tqdm.tqdm(enumerate(zip(titles, questions, answers, contexts))):
                new_dataset['context'].append(context)
                new_dataset['id'].append(dataset[i]["id"])
                new_dataset['question'].append("")
                new_dataset['title'].append(title)
                new_dataset['answers'].append(ans)
                if 'story_id' in dataset[i]:
                    if 'story_id' not in new_dataset:
                        new_dataset['story_id'] = []
                    new_dataset['story_id'].append(dataset[i]["story_id"])
            raw_datasets[split_name] = Dataset.from_dict(new_dataset)

    def preprocess_squad_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        contexts = examples[context_column]
        answers = examples[answer_column]

        def generate_input(_question, _context):
            return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

        inputs = [generate_input(question, context) for question, context in zip(questions, contexts)]
        targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
        return inputs, targets

    def preprocess_function(examples):
        inputs, targets = preprocess_squad_batch(examples, question_column, context_column, answer_column)

        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)

        # For longer datasets we should ideally provide that context which contains the answer.

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Validation preprocessing
    def preprocess_validation_function(examples):
        inputs, targets = preprocess_squad_batch(examples, question_column, context_column, answer_column)

        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # # Setup the tokenizer for targets
        # with tokenizer.as_target_tokenizer():
        #     labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)
        
        # Tokenize targets with the `text_target` keyword argument
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore	
        # padding in the loss.	
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:	
            labels["input_ids"] = [	
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]	
            ]

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        model_inputs["example_id"] = []
        # Augment the overflowing tokens to the labels	
        labels_out = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])
            labels_out.append(labels["input_ids"][sample_index])

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.

        # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        #     labels["input_ids"] = [
        #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]

        model_inputs["labels"] = labels_out

        # model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # Define question transformation functions over huggingface datasets


    # Previous question
    def prev_questions(eval_dataset):
        all_questions = [ex["question"] for ex in eval_dataset]
        all_titles = [ex["title"] for ex in eval_dataset]
        new_questions = []
        for i, q in enumerate(range(1, len(all_questions))):
            new_questions.append(all_questions[i-1])
        new_questions.append(all_questions[0])
        
        eval_dataset = eval_dataset.remove_columns(["question"])
        eval_dataset = eval_dataset.add_column("question", new_questions)

        return eval_dataset

    # no Question
    def no_questions(eval_dataset):
        # Replace question with title
        all_questions = [ex["question"] for ex in eval_dataset]
        all_titles = [ex["title"] for ex in eval_dataset]
        new_dataset = {'id': [], 'context':[], 'question':[], 'answers': [], "title": []}
        for i, q in enumerate(range(len(all_questions))):
            ex = eval_dataset[i]
            answer_text = ex["answers"]["text"][0] if len(ex["answers"]["text"]) else ""
            if answer_text != "":
                new_dataset['context'].append(eval_dataset[i]["context"])
                new_dataset['id'].append(eval_dataset[i]["id"])
                new_dataset['question'].append(all_titles[i])
                new_dataset['title'].append(eval_dataset[i]["title"])
                if 'story_id' in eval_dataset[i]:
                    if 'story_id' not in new_dataset:
                        new_dataset['story_id'] = []
                    new_dataset['story_id'].append(eval_dataset[i]["story_id"])
                new_dataset['answers'].append({'text': [], 'answer_start': []})

        eval_dataset = Dataset.from_dict(new_dataset)

        return eval_dataset
        
        
    def same_para_questions(eval_dataset, seed=1234):
        all_questions = [ex["question"] for ex in eval_dataset]
        if "story_id" in eval_dataset[0]:
            all_titles = [ex["story_id"] for ex in eval_dataset]
        else:
            all_titles = [ex["title"] for ex in eval_dataset]
        all_ids = [ex["id"] for ex in eval_dataset]
        id_dict = {ex["id"]:ex for ex in eval_dataset}
        
        new_questions = []
        new_answers = []
        title_question_dict = {}
        title_dict = {}
        random.seed(seed)
        new_dataset = {'id': [], 'context':[], 'question':[], 'answers': [], "title": []}
        for i in range(0, len(all_titles)):
            t = all_titles[i]
            if t in title_dict:
                title_dict[t].append(all_ids[i])
            else:
                title_dict[t] = [all_ids[i]]
        for title in title_dict:
            ids = title_dict[title]
            shuffled = sorted(ids, key=lambda k: random.random())
            for id_, new_id in zip(ids, shuffled):
                ex = id_dict[id_]
                new_ex = id_dict[new_id]

                answer_text = ex["answers"]["text"][0] if len(ex["answers"]["text"]) else ""
                if answer_text != "":
                # This is the subset that needs to be flipped to no answer
                    new_dataset['context'].append(ex["context"])
                    new_dataset['id'].append(ex["id"])
                    new_dataset['question'].append(new_ex["question"])
                    new_dataset['title'].append(ex["title"])
                    if 'story_id' in eval_dataset[i]:
                        if 'story_id' not in new_dataset:
                            new_dataset['story_id'] = []
                        new_dataset['story_id'].append(ex["story_id"])
                    if ex["answers"]['text'] != new_ex["answers"]['text']:
                        new_dataset['answers'].append({'text': [], 'answer_start': []})
                    else:
                        new_dataset['answers'].append(ex["answers"])

            # swap questions with different contexts.
            # context_dict = {}
            # for id_ in title_dict[title]:
            #     ex = id_dict[id_]
            #     if ex["context"] in context_dict:
            #         context_dict[ex["context"]].append(ex)
            #     else:
            #         context_dict[ex["context"]] = [ex]
            # if len(title_dict[title]) > 1 and len(context_dict) > 1:
            #     pdb.set_trace()

        eval_dataset = Dataset.from_dict(new_dataset)

        return eval_dataset

    def random_questions(eval_dataset, seed=1234):
        all_questions = [ex["question"] for ex in eval_dataset]
        all_answer_texts = [ex["answers"]["text"][0] if len(ex["answers"]["text"]) else "" for ex in eval_dataset]
        if "story_id" in eval_dataset[0]:
            all_titles = [ex["story_id"] for ex in eval_dataset]
        else:
            all_titles = [ex["title"] for ex in eval_dataset]
        new_questions = []
        new_answers = []
        np.random.seed(seed)
        new_dataset = {'id': [], 'context':[], 'question':[], 'answers': [], "title": []}
        for i, q in enumerate(range(0, len(all_questions))):
            answer_text = all_answer_texts[i]
            while(1):
                rand_idx = np.random.randint(len(all_questions))
                if all_titles[rand_idx] != all_titles[i]:
                    new_question = all_questions[rand_idx]
                    break
            if answer_text != "":
                # This is the subset that needs to be flipped to no answer
                # new_answers.append({'text': [], 'answer_start': []})
                # new_questions.append(new_question)

                new_dataset['context'].append(eval_dataset[i]["context"])
                new_dataset['id'].append(eval_dataset[i]["id"])
                new_dataset['question'].append(new_question)
                new_dataset['title'].append(eval_dataset[i]["title"])
                if 'story_id' in eval_dataset[i]:
                    if 'story_id' not in new_dataset:
                        new_dataset['story_id'] = []
                    new_dataset['story_id'].append(eval_dataset[i]["story_id"])
                new_dataset['answers'].append({'text': [], 'answer_start': []})

        # eval_dataset = eval_dataset.remove_columns(["question"])
        # eval_dataset = eval_dataset.add_column("question", new_questions)

        # eval_dataset = eval_dataset.remove_columns(["answers"])
        # eval_dataset = eval_dataset.add_column("answers", new_answers)
        eval_dataset = Dataset.from_dict(new_dataset)

        return eval_dataset

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))

        # Partial-input perturbation
        if data_args.partial_inputs == "no":
            eval_examples = no_questions(eval_examples)
        elif data_args.partial_inputs == "previous":
            eval_examples = prev_questions(eval_examples)
        elif data_args.partial_inputs == "random":
            eval_examples = random_questions(eval_examples, data_args.partial_inputs_seed)
        elif data_args.partial_inputs == "same_title":
            eval_examples = same_para_questions(eval_examples, data_args.partial_inputs_seed)

        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "validation" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["validation"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation

        # Partial-input perturbation
        if data_args.partial_inputs == "no":
            predict_examples = no_questions(predict_examples)
        elif data_args.partial_inputs == "previous":
            predict_examples = prev_questions(predict_examples)
        elif data_args.partial_inputs == "random":
            predict_examples = random_questions(predict_examples)
        elif data_args.partial_inputs == "same_title":
            predict_examples = same_para_questions(predict_examples)

        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids, no_answer_threshold=data_args.no_answer_threshold)

    # Post-processing:
    def post_processing_function(
        examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"
    ):
        # Decode the predicted tokens.
        if type(outputs) == np.ndarray:
            preds = outputs
        else:
            preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        # feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        feature_per_example = {}
        for i, feature in enumerate(features):
            if example_id_to_index[feature["example_id"]] not in feature_per_example:
                # The earlist feature is used.
                feature_per_example[example_id_to_index[feature["example_id"]]] = i
        predictions = {}
        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # This is the index of the feature associated to the current example.
            if example_index not in feature_per_example:
                continue
            feature_index = feature_per_example[example_index]
            # predictions over multiple features are getting overwritten here.
            predictions[example["id"]] = decoded_preds[feature_index]

        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references[:len(formatted_predictions)])

    # Initialize our Trainer
    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        post_process_function=post_processing_function,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_answer_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics
        
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        with open(os.path.join(training_args.output_dir, "predictions.jsonl"), "w") as fout:
            predictions = results.predictions
            for example, pred in zip(predict_examples, predictions):
                assert example['id'] == pred['id']
                example['prediction_text'] = pred['prediction_text']
                example['no_answer_probability'] = pred['no_answer_probability']
                fout.write(json.dumps(example) + "\n")


    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
