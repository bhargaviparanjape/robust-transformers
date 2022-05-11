"""
Utilities for data handling.
"""
import json
import logging
import os
import re
import pandas as pd
import shutil
import tqdm
import random

from typing import Dict

def convert_string_to_unique_number(string: str) -> int:
  """
  Hack to convert SNLI ID into a unique integer ID, for tensorizing.
  """
  id_map = {'e': '0', 'c': '1', 'n': '2'}

  # SNLI-specific hacks.
  if string.startswith('vg_len'):
    code = '555'
  elif string.startswith('vg_verb'):
    code = '444'
  else:
    code = '000'

  try:
    number = int(code + re.sub(r"\D", "", string) + id_map.get(string[-1], '3'))
  except:
    number = random.randint(10000, 99999)
    logger.info(f"Cannot find ID for {string}, using random number {number}.")
  return number

logger = logging.getLogger(__name__)

def read_data(file_path: str,
              task_name: str,
              guid_as_int: bool = False):
  """
  Reads task-specific datasets from corresponding GLUE-style TSV files.
  """
  logger.warning("Data reading only works when data is in TSV format, "
                 " and last column as classification label.")

  # `guid_index`: should be 2 for SNLI, 0 for MNLI and None for any random tsv file.
  if task_name == "MNLI":
    return read_glue_tsv(file_path,
                        guid_index=0,
                        guid_as_int=guid_as_int)
  elif task_name == "MNLI_RESPLIT":
    return read_json(file_path, guid_as_int=guid_as_int)
  elif task_name == "WINOGRANDE":
    return read_glue_tsv(file_path,
                        guid_index=0)
  elif task_name == "QNLI":
    return read_glue_tsv(file_path,
                        guid_index=0)
  elif task_name == "AMAZON":
    return read_amazon_json(file_path)
  else:
    raise NotImplementedError(f"Reader for {task_name} not implemented.")

def read_amazon_json(file_path: str):
  set_type = "train"
  records = {}
  for (i, line) in enumerate(open(file_path)):
    review = json.loads(line.rstrip())
    guid = "%s-%d" % (set_type, i)
    if "reviewText" not in review or "overall" not in review:
      continue
    #assert guid not in records
    records[convert_string_to_unique_number(guid)] = review
  return records

def read_json(file_path: str,
              guid_as_int: bool = False):
  records = {}
  for (i, line) in enumerate(open(file_path)):
    ex = json.loads(line.rstrip())
    guid = ex["guid"]
    #assert guid not in records
    if guid in records:
        logger.info(f"Found clash in IDs ... skipping example {guid}.")
        continue
    records[guid] = ex
  logger.info(f"Read {len(records)} valid examples, with unique IDS, out of {i} from {file_path}")
  if guid_as_int:
    records_numeric = {int(convert_string_to_unique_number(k)): v for k, v in records.items()}
    return records_numeric
  return records

def read_glue_tsv(file_path: str,
                  guid_index: int,
                  label_index: int = -1,
                  guid_as_int: bool = False):
  """
  Reads TSV files for GLUE-style text classification tasks.
  Returns:
    - a mapping between the example ID and the entire line as a string.
    - the header of the TSV file.
  """
  tsv_dict = {}

  i = -1
  with open(file_path, 'r') as tsv_file:
    for line in tqdm.tqdm([line for line in tsv_file]):
      i += 1
      if i == 0:
        header = line.strip()
        field_names = line.strip().split("\t")
        continue

      fields = line.strip().split("\t")
      label = fields[label_index]
      if len(fields) > len(field_names):
        # SNLI / MNLI fields sometimes contain multiple annotator labels.
        # Ignore all except the gold label.
        reformatted_fields = fields[:len(field_names)-1] + [label]
        assert len(reformatted_fields) == len(field_names)
        reformatted_line = "\t".join(reformatted_fields)
      else:
        reformatted_line = line.strip()

      if label == "-" or label == "":
        logger.info(f"Skippping line: {line}")
        continue

      if guid_index is None:
        guid = i
      else:
        guid = fields[guid_index] # PairID.
      if guid in tsv_dict:
        logger.info(f"Found clash in IDs ... skipping example {guid}.")
        continue
      tsv_dict[guid] = reformatted_line.strip()

  logger.info(f"Read {len(tsv_dict)} valid examples, with unique IDS, out of {i} from {file_path}")
  if guid_as_int:
    tsv_numeric = {int(convert_string_to_unique_number(k)): v for k, v in tsv_dict.items()}
    return tsv_numeric, header
  return tsv_dict, header



