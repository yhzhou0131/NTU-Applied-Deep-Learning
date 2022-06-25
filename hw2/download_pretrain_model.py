import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import datasets
import numpy as np
import torch
from datasets import load_dataset, Dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import PaddingStrategy, check_min_version

import json

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

def main():
  parser = HfArgumentParser((ModelArguments))

  model_args, = parser.parse_args_into_dataclasses()

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
  model = AutoModelForMultipleChoice.from_pretrained(
      model_args.model_name_or_path,
      from_tf=bool(".ckpt" in model_args.model_name_or_path),
      config=config,
      cache_dir=model_args.cache_dir,
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

if __name__ == "__main__":
  main()