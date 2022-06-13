import pickle
import os
import sys
import json
from tokenize import group
import pandas as pd
import argparse
import numpy as np
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_assgn_file", type=str, default=None)
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--split_by_label", action="store_true", default=False)
    parser.add_argument("--split_strategy", type=str, default="domino")
    args = parser.parse_args()

    pd_df = pd.read_pickle(args.cluster_assgn_file)

    slices =  np.stack(pd_df["domino_slices"].to_numpy())

    group_assignment = {}
    group_distributions = {}
    for i in range(len(pd_df)):
        ## Usually analysis is done with group assignment only if value is above a threshold.
        slice = int(np.argmax(pd_df.iloc[i]["domino_slices"]))
        slice_val = np.max(pd_df.iloc[i]["domino_slices"])
        # if slice_val > 0.70:
        #     chosen_slice = slice
        # else:
        #     chosen_slice = -1
        chosen_slice = slice
        guid = pd_df.iloc[i]["guid"]
        group_distributions[guid] = list(pd_df.iloc[i]["domino_slices"])
        group_assignment[guid] = chosen_slice
    
    print(Counter(group_assignment.values()))


    # TODO: CURRENTLY ONLY SUPPORTS MNLI. COMBINE WITH generate_features.py CODE
    label_to_id = {"entailment": 0, "neutral":1, "contradiction":2}

    dataset = [json.loads(line) for line in open(args.dataset_file)]
    with open(args.output_file, "w") as fout:
        for ex in dataset:
            guid = ex["guid"]
            new_ex = ex
            if args.split_by_label:
                new_ex["group"] = 3*((group_assignment[guid])) + label_to_id[ex["label"]]
            else:
                new_ex["group"] = group_assignment[guid]
            new_ex["group_distribution"] = group_distributions[guid]
            fout.write(json.dumps(new_ex) + "\n")

    # Adjust threshold so that more than X>group assignment is made. 
    # Plot Max value to understand how confident the mixture model even is...
