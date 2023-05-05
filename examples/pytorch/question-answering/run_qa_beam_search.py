#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning XLNet for question answering with beam search using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import copy
import numpy as np
import pdb
import re
import tqdm
import random
import pdb

import datasets
from datasets import load_dataset, load_metric
from datasets import Dataset, DatasetDict

import transformers
# from transformers.utils.dummy_pt_objects import AutoModelForQuestionAnswering
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizerFast,
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions_with_beam_search


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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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
    partial_inputs_seed: Optional[int] = field(
        default=1234, metadata={"help": "random seed for perturbation"}
    )
    question_only: Optional[bool] = field(
        default=False, metadata={"help": "Question only training"}
    )
    passage_only: Optional[bool] = field(
        default=False, metadata={"help": "Passage only training"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to test the perplexity on (a text file)."},
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
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation/test file.")
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
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
        # use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    ## SQUAD 2.0
    # {'id': '56be85543aeaaa14008c9063', 'title': 'Beyoncé', 
    # 'context': 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".', 'question': 'When did Beyonce start becoming popular?', 
    # 'answers': {'text': ['in the late 1990s'], 'answer_start': [269]}}

    ## SQUAD 2.0 (unanswerable)
    # {'id': '5a7e070b70df9f001a87543d', 'title': 'Matter', 
    #'context': 'The term "matter" is used throughout physics in a bewildering variety of contexts: for example, one refers to "condensed matter physics", "elementary matter", "partonic" matter, "dark" matter, "anti"-matter, "strange" matter, and "nuclear" matter. In discussions of matter and antimatter, normal matter has been referred to by Alfvén as koinomatter (Gk. common matter). It is fair to say that in physics, there is no broad consensus as to a general definition of matter, and the term "matter" usually is used in conjunction with a specifying modifier.', 
    # 'question': 'What field of study has a variety of unusual contexts?', 
    # 'answers': {'text': [], 'answer_start': []}}

    # DuoRC
    ## ['plot_id', 'plot', 'title', 'question_id', 'question', 'answers', 'no_answer']

    # NewsQA
    ## ['story_id', 'story_text', 'question', 'answer_token_ranges']

    
    if data_args.dataset_name == 'duorc':
        new_datasets = {}
        # raw_datasets["train"], raw_datasets["validation"], raw_datasets["test"] should be transformed to squad 2.0 format.
        for split in ["train", "validation", "test"]:
            new_dataset = {'id': [], 'context':[], 'question':[], 'answers': [], "title": []}
            answers = raw_datasets[split]["answers"]
            contexts = raw_datasets[split]["plot"]
            no_answers = raw_datasets[split]["no_answer"]
            for i, (ans, context, imp) in tqdm.tqdm(enumerate(zip(answers, contexts, no_answers))):
                if not imp:
                    # answer exists
                    for a in ans:
                        start = None
                        if a in context:
                            start = context.find(a)
                            new_dataset['id'].append(raw_datasets[split][i]["question_id"])
                            new_dataset['question'].append(raw_datasets[split][i]["question"])
                            new_dataset['context'].append(raw_datasets[split][i]["plot"])
                            new_dataset['title'].append(raw_datasets[split][i]["title"])
                            new_dataset['answers'].append({"text": [a], 'answer_start': [start]})
                            # First occurrence of an answer in context.
                            break
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
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

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
                        random_location = 0
                    else:
                        random_location = np.random.randint(len(context_words)-len(ans_words))
                        new_context = context_words[:random_location] + ans_words + context_words[random_location + len(ans_words):]
                    new_context = " ".join(new_context)
                    new_start = new_context.find(ans["text"][0])
                    ans["answer_start"] = [new_start]
                    new_dataset['context'].append(new_context)
                new_dataset['id'].append(dataset[i]["id"])
                new_dataset['question'].append(question)
                new_dataset['title'].append(title)
                if 'story_id' in dataset[i]:
                    if 'story_id' not in new_dataset:
                        new_dataset['story_id'] = []
                    new_dataset['story_id'].append(dataset[i]["story_id"])
                new_dataset['answers'].append(ans)
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
                if 'story_id' in dataset[i]:
                    if 'story_id' not in new_dataset:
                        new_dataset['story_id'] = []
                    new_dataset['story_id'].append(dataset[i]["story_id"])
                new_dataset['answers'].append(ans)
            raw_datasets[split_name] = Dataset.from_dict(new_dataset)

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")
        # The special tokens will help us build the p_mask (which indicates the tokens that can't be in answers).
        special_tokens = tokenized_examples.pop("special_tokens_mask")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["is_impossible"] = []
        tokenized_examples["cls_index"] = []
        tokenized_examples["p_mask"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            tokenized_examples["cls_index"].append(cls_index)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = copy.deepcopy(tokenized_examples["token_type_ids"][i])
            for k, s in enumerate(special_tokens[i]):
                if s:
                    sequence_ids[k] = 3
            
            tokenizer_name = tokenizer.name_or_path
            if "roberta" in tokenizer_name:
                context_idx = 0
            else:
                context_idx = 1 if pad_on_right else 0

            # Build the p_mask: non special tokens and context gets 0.0, the others get 1.0.
            # The cls token gets 1.0 too (for predictions of empty answers).

            # Roberta (token type Id is not provided, so the question tokens are also valid tokens for start and end)
            # BERT (token type Id is provided so question is masked out)
            # DeBERTa (token type Id is provided so question is masked out)

            tokenized_examples["p_mask"].append(
                [
                    0.0 if (not special_tokens[i][k] and s == context_idx) or k == cls_index else 1.0
                    for k, s in enumerate(sequence_ids)
                ]
            )

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["is_impossible"].append(1.0)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != context_idx:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_idx:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["is_impossible"].append(1.0)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["is_impossible"].append(0.0)

        return tokenized_examples

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # Select samples from Dataset, This will help to decrease processing time
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Create Training Features
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Select samples from dataset again since Feature Creation might increase number of features
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # The special tokens will help us build the p_mask (which indicates the tokens that can't be in answers).
        special_tokens = tokenized_examples.pop("special_tokens_mask")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        # We still provide the index of the CLS token and the p_mask to the model, but not the is_impossible label.
        tokenized_examples["cls_index"] = []
        tokenized_examples["p_mask"] = []

        for i, input_ids in enumerate(tokenized_examples["input_ids"]):
            # Find the CLS token in the input ids.
            cls_index = input_ids.index(tokenizer.cls_token_id)
            tokenized_examples["cls_index"].append(cls_index)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = copy.deepcopy(tokenized_examples["token_type_ids"][i])
            for k, s in enumerate(special_tokens[i]):
                if s:
                    sequence_ids[k] = 3
            
            tokenizer_name = tokenizer.name_or_path
            if "roberta" in tokenizer_name:
                context_idx = 0
            else:
                context_idx = 1 if pad_on_right else 0

            # Build the p_mask: non special tokens and context gets 0.0, the others 1.0.
            tokenized_examples["p_mask"].append(
                [
                    0.0 if (not special_tokens[i][k] and s == context_idx) or k == cls_index else 1.0
                    for k, s in enumerate(sequence_ids)
                ]
            )

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_idx else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # Previous question
    def prev_questions(eval_dataset):
        all_questions = [ex["question"] for ex in eval_dataset]
        all_titles = [ex["title"] for ex in eval_dataset]
        new_questions = []
        new_answers = []
        for i, q in enumerate(range(1, len(all_questions))):
            new_questions.append(all_questions[i-1])
            new_answers.append({'text': [], 'answer_start': []})
        new_questions.append(all_questions[0])
        new_answers.append({'text': [], 'answer_start': []})
        
        eval_dataset = eval_dataset.remove_columns(["question"])
        eval_dataset = eval_dataset.add_column("question", new_questions)

        eval_dataset = eval_dataset.remove_columns(["answers"])
        eval_dataset = eval_dataset.add_column("answers", new_answers)

        return eval_dataset

    # no Question
    def no_questions(eval_dataset):
        # Replace question with title
        all_questions = [ex["question"] for ex in eval_dataset]
        all_titles = [ex["title"] for ex in eval_dataset]
        new_questions = []
        new_answers = []
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
        all_answer_texts = [ex["answers"]["text"][0] if len(ex["answers"]["text"]) else "" for ex in eval_dataset]
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

    # Random question with a different title 
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

    def question_same_title(eval_dataset):
        all_questions = [ex["question"] for ex in eval_dataset]
        all_titles = [ex["title"] for ex in eval_dataset]
        all_contexts = [ex["title"] for ex in eval_dataset]
        new_questions = []
        new_answers = []

        # SQUAD : Same title but different paragraph


        eval_dataset = eval_dataset.remove_columns(["question"])
        eval_dataset = eval_dataset.add_column("question", new_questions)

        eval_dataset = eval_dataset.remove_columns(["answers"])
        eval_dataset = eval_dataset.add_column("answers", new_answers)

        return eval_dataset

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # Selecting Eval Samples from Dataset
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

        # eval_examples = no_questions(eval_examples)

        # Create Features from Eval Dataset
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # Selecting Samples from Dataset again since Feature Creation might increase samples size
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Test Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_validation_features,
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
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions, scores_diff_json = postprocess_qa_predictions_with_beam_search(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            start_n_top=5, # model.config.start_n_top,
            end_n_top=5, # model.config.end_n_top,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": scores_diff_json[k]}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids, no_answer_threshold=data_args.no_answer_threshold)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

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

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
