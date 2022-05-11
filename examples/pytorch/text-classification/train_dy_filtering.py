"""
Filtering and dataset mapping methods based on training dynamics.
By default, this module reads training dynamics from a given trained model and
computes the metrics---confidence, variability, correctness,
as well as baseline metrics of forgetfulness and threshold closeness
for each instance in the training data.
If specified, data maps can be plotted with respect to confidence and variability.
Moreover, datasets can be filtered with respect any of the other metrics.
"""
import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import tqdm
import csv
from data_utils import read_data

from collections import defaultdict
from typing import List

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def read_training_dynamics(model_dir: os.path,
                           strip_last: bool = False,
                           id_field: str = "guid",
                           burn_out: int = None):
  """
  Given path to logged training dynamics, merge stats across epochs.
  Returns:
  - Dict between ID of a train instances and its gold label, and the list of logits across epochs.
  """
  train_dynamics = {}

  td_dir = os.path.join(model_dir, "training_dynamics")
  num_epochs = len([f for f in os.listdir(td_dir) if os.path.isfile(os.path.join(td_dir, f))])
  if burn_out:
    num_epochs = burn_out

  logger.info(f"Reading {num_epochs} files from {td_dir} ...")
  for epoch_num in tqdm.tqdm(range(num_epochs)):
    epoch_file = os.path.join(td_dir, f"dynamics_epoch_{epoch_num}.jsonl")
    assert os.path.exists(epoch_file)

    with open(epoch_file, "r") as infile:
      for line in infile:
        record = json.loads(line.strip())
        guid = record[id_field] if not strip_last else record[id_field][:-1]
        if guid not in train_dynamics:
          assert epoch_num == 0
          train_dynamics[guid] = {"gold": record["gold"], "logits": []}
        train_dynamics[guid]["logits"].append(record[f"logits_epoch_{epoch_num}"])

  logger.info(f"Read training dynamics for {len(train_dynamics)} train instances.")
  return train_dynamics


def compute_forgetfulness(correctness_trend: List[float]) -> int:
  """
  Given a epoch-wise trend of train predictions, compute frequency with which
  an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
  Based on: https://arxiv.org/abs/1812.05159
  """
  if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
      return 1000
  learnt = False  # Predicted correctly in the current epoch.
  times_forgotten = 0
  for is_correct in correctness_trend:
    if (not learnt and not is_correct) or (learnt and is_correct):
      # nothing changed.
      continue
    elif learnt and not is_correct:
      # Forgot after learning at some point!
      learnt = False
      times_forgotten += 1
    elif not learnt and is_correct:
      # Learnt!
      learnt = True
  return times_forgotten


def compute_correctness(trend: List[float]) -> float:
  """
  Aggregate #times an example is predicted correctly during all training epochs.
  """
  return sum(trend)


def compute_train_dy_metrics(training_dynamics, args):
  """
  Given the training dynamics (logits for each training instance across epochs), compute metrics
  based on it, for data map coorodinates.
  Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
  the last two being baselines from prior work
  (Example Forgetting: https://arxiv.org/abs/1812.05159 and
   Active Bias: https://arxiv.org/abs/1704.07433 respectively).
  Returns:
  - DataFrame with these metrics.
  - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
  """
  confidence_ = {}
  variability_ = {}
  threshold_closeness_ = {}
  correctness_ = {}
  forgetfulness_ = {}

  # Functions to be applied to the data.
  variability_func = lambda conf: np.std(conf)
  if args.include_ci:  # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
    variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
  threshold_closeness_func = lambda conf: conf * (1 - conf)

  loss = torch.nn.CrossEntropyLoss()

  num_tot_epochs = len(list(training_dynamics.values())[0]["logits"])
  if args.burn_out < num_tot_epochs:
    logger.info(f"Computing training dynamics. Burning out at {args.burn_out} of {num_tot_epochs}. ")
  else:
    logger.info(f"Computing training dynamics across {num_tot_epochs} epochs")
  logger.info("Metrics computed: confidence, variability, correctness, forgetfulness, threshold_closeness")

  logits = {i: [] for i in range(num_tot_epochs)}
  targets = {i: [] for i in range(num_tot_epochs)}
  training_accuracy = defaultdict(float)

  for guid in tqdm.tqdm(training_dynamics):
    correctness_trend = []
    true_probs_trend = []

    record = training_dynamics[guid]
    for i, epoch_logits in enumerate(record["logits"]):
      probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
      true_class_prob = float(probs[record["gold"]])
      true_probs_trend.append(true_class_prob)

      prediction = np.argmax(epoch_logits)
      is_correct = (prediction == record["gold"]).item()
      correctness_trend.append(is_correct)

      training_accuracy[i] += is_correct
      logits[i].append(epoch_logits)
      targets[i].append(record["gold"])

      # For PVI
      # Use logits to compute entropy on a null model and a model and compute difference to get PVI for a training examples

      # For Influence:
      # compute train by test example influence and average over testing data.

    if args.burn_out < num_tot_epochs:
      correctness_trend = correctness_trend[:args.burn_out]
      true_probs_trend = true_probs_trend[:args.burn_out]

    correctness_[guid] = compute_correctness(correctness_trend)
    confidence_[guid] = np.mean(true_probs_trend)
    variability_[guid] = variability_func(true_probs_trend)

    forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
    threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])

  # Should not affect ranking, so ignoring.
  epsilon_var = np.mean(list(variability_.values()))

  column_names = ['guid',
                  'index',
                  'threshold_closeness',
                  'confidence',
                  'variability',
                  'correctness',
                  'forgetfulness',]
  df = pd.DataFrame([[guid,
                      i,
                      threshold_closeness_[guid],
                      confidence_[guid],
                      variability_[guid],
                      correctness_[guid],
                      forgetfulness_[guid],
                      ] for i, guid in enumerate(correctness_)], columns=column_names)

  df_train = pd.DataFrame([[i,
                            loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(training_dynamics),
                            training_accuracy[i] / len(training_dynamics)
                            ] for i in range(num_tot_epochs)],
                          columns=['epoch', 'loss', 'train_acc'])
  return df, df_train


def plot_data_map(args, 
                  dataframe: pd.DataFrame,
                  plot_dir: os.path,
                  hue_metric: str = 'correct.',
                  title: str = '',
                  model: str = 'RoBERTa',
                  show_hist: bool = False,
                  max_instances_to_plot = 2000):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')
    logger.info(f"Plotting figure for {title} using the {model} model ...")

    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    if args.task_name == "MNLI_RESPLIT":
      original_train_file = args.train_file
      train_numeric = read_data(original_train_file, task_name=args.task_name, guid_as_int=False)

    hue = hue_metric
    # Check if hue metric is in dataframe, if not extract it from training dataset.
    if hue in dataframe:
      num_hues = len(dataframe[hue].unique().tolist())
      style = hue_metric if num_hues < 8 else None
    else:
      hue_column = []
      selection_iterator = tqdm.tqdm(range(len(dataframe)))
      for idx in selection_iterator:
        selected_id = dataframe.iloc[idx]["guid"]
        hue_column.append(train_numeric[selected_id][hue])
      dataframe[hue] = hue_column
      num_hues = len(dataframe[hue].unique().tolist())
      style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])

    # Make the scatterplot.
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda  text, xyc, bbc : ax0.annotate(text,
                                                          xy=xyc,
                                                          xycoords="axes fraction",
                                                          fontsize=15,
                                                          color='black',
                                                          va="center",
                                                          ha="center",
                                                          rotation=350,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')


    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    else:
        plot.legend(fancybox=True, shadow=True,  ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')

    fig.tight_layout()
    filename = f'{plot_dir}/{title}_{model}.pdf'
    fig.savefig(filename, dpi=300)
    logger.info(f"Plot saved to {filename}")

def tag_data(args, dataframe: pd.DataFrame,
                  tagging_output_dir: os.path,
                  model: str = 'RoBERTa-base'):
  # Normalize correctness to a value between 0 and 1.
  dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
  dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

  # unique correctness values
  correct_values = set(dataframe.loc[:, "correct."])
  correct_groups = {v:i for i,v in enumerate(correct_values)}

  

  if args.task_name == "MNLI_RESPLIT":
    original_train_file = args.train_file
    train_numeric = read_data(original_train_file, task_name=args.task_name, guid_as_int=False)
    original_val_file = args.val_file
    val_numeric = read_data(original_val_file, task_name=args.task_name, guid_as_int=False)
    labels = ["entailment", "neutral", "contradiction"]

  correct_label_groups = {}
  group_no = 0
  for label in labels:
    for correct_group in correct_values:
      correct_label_groups[(label, correct_group)] = group_no
      group_no += 1


  outdir = args.tagging_output_dir
  if not os.path.exists(outdir):
        os.makedirs(outdir)
  
  selection_iterator = tqdm.tqdm(range(len(dataframe)))
  with open(os.path.join(outdir, f"train_resplit_cartography.json"), "w") as outfile:
    for idx in selection_iterator:
      selected_id = dataframe.iloc[idx]["guid"]
      if args.task_name in ["SNLI", "MNLI"]:
        selected_id = int(selected_id)
      record = train_numeric[selected_id]

      confidence = dataframe.iloc[idx]["confidence"]
      variability = dataframe.iloc[idx]["variability"]
      correctness = dataframe.iloc[idx]["correct."]
      label = record["label"]

      # Grouping 1 (based on correctness values):
      # correct_group = correct_label_groups[(label,correctness)]
      correct_group = correct_groups[correctness]

      group = correct_group

      if args.task_name == "MNLI_RESPLIT":
        record["group"] = group
        outfile.write(json.dumps(record) + "\n")
  
  ## For validation file, randomly assign groups, making sure there are same number of groups.
  ## Note that validation_adjustments have to be turned off in case of random assignment of groups.
  with open(os.path.join(outdir, f"dev_resplit_cartography.json"), "w") as outfile:
    for guid, record in val_numeric.items():
      random_group = np.random.randint(0, len(correct_groups))
      record["group"] = random_group
      outfile.write(json.dumps(record) + "\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tag", action="store_true", help="Whether to tag datasets as ambiguous, hard or easy")
  parser.add_argument("--compute", action="store_true", help="Whether to compute td stats or load them")
  parser.add_argument("--filter",
                      action="store_true",
                      help="Whether to filter data subsets based on specified `metric`.")
  parser.add_argument("--plot",
                      action="store_true",
                      help="Whether to plot data maps and save as `pdf`.")
  parser.add_argument("--model_dir",
                      "-o",
                      required=True,
                      type=os.path.abspath,
                      help="Directory where model training dynamics stats reside.")
  parser.add_argument("--data_dir",
                      "-d",
                      default="/Users/swabhas/data/glue/WINOGRANDE/xl/",
                      type=os.path.abspath,
                      help="Directory where data for task resides.")
  parser.add_argument("--train_file",
                    default=None,
                    type=os.path.abspath,
                    help="Full path to train file.")
  parser.add_argument("--val_file",
                    default=None,
                    type=os.path.abspath,
                    help="Full path to validation/test file.")
  parser.add_argument("--plots_dir",
                      default="./cartography/",
                      type=os.path.abspath,
                      help="Directory where plots are to be saved.")
  parser.add_argument("--task_name",
                      "-t",
                      default="WINOGRANDE",
                      choices=("AMAZON", "SNLI", "MNLI", "MNLI_RESPLIT", "QNLI", "WINOGRANDE"),
                      help="Which task are we plotting or filtering for.")
  parser.add_argument('--metric',
                      choices=('threshold_closeness',
                               'confidence',
                               'variability',
                               'correctness',
                               'forgetfulness'),
                      help="Metric to filter data by.",)
  parser.add_argument("--include_ci",
                      action="store_true",
                      help="Compute the confidence interval for variability.")
  parser.add_argument("--filtering_output_dir",
                      "-f",
                      default="./filtered/",
                      type=os.path.abspath,
                      help="Output directory where filtered datasets are to be written.")
  parser.add_argument("--tagging_output_dir",
                      default="./tagged/",
                      type=os.path.abspath,
                      help="Output directory where tagger datasets are to be written.")
  parser.add_argument("--worst",
                      action="store_true",
                      help="Select from the opposite end of the spectrum acc. to metric,"
                           "for baselines")
  parser.add_argument("--both_ends",
                      action="store_true",
                      help="Select from both ends of the spectrum acc. to metric,")
  parser.add_argument("--burn_out",
                      type=int,
                      default=100,
                      help="# Epochs for which to compute train dynamics.")
  parser.add_argument("--model",
                      default="RoBERTa",
                      help="Model for which data map is being plotted")
  parser.add_argument("--artifact", type=str,
                      help="name of hue metric corresponding to artifact")

  args = parser.parse_args()

  training_dynamics = read_training_dynamics(args.model_dir,
                                             strip_last=True if args.task_name in ["QNLI"] else False,
                                             burn_out=args.burn_out if args.burn_out < 100 else None)
  

  total_epochs = len(list(training_dynamics.values())[0]["logits"])

  # Compute or load train_dy_metrics.
  burn_out_str = f"_{args.burn_out}" if args.burn_out != 100 else ""
  train_dy_filename = os.path.join(args.model_dir, f"td_metrics{burn_out_str}.jsonl")

  if os.path.exists(train_dy_filename):
    logger.info(f"Metrics based on Training Dynamics being read from {train_dy_filename}")
    train_dy_metrics = pd.read_json(train_dy_filename, lines=True)
  else:
    train_dy_metrics, _ = compute_train_dy_metrics(training_dynamics, args)
    train_dy_metrics.to_json(train_dy_filename,
                              orient='records',
                              lines=True)  
    logger.info(f"Metrics based on Training Dynamics written to {train_dy_filename}")
  
  print(len(train_dy_metrics))


  if args.tag:
    assert args.tagging_output_dir
    if not os.path.exists(args.tagging_output_dir):
      os.makedirs(args.tagging_output_dir)
    tag_data(args, train_dy_metrics, args.tagging_output_dir, model=args.model)

  if args.plot:
    assert args.plots_dir
    if not os.path.exists(args.plots_dir):
      os.makedirs(args.plots_dir)
    plot_data_map(args, train_dy_metrics, args.plots_dir, title=args.task_name, show_hist=False, model=args.model, hue_metric=args.artifact)

  