import pickle
import os
import sys
import json
from tokenize import group
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import f1_score
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_assgn_file", type=str, default=None)
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--split_by_label", action="store_true", default=False)
    args = parser.parse_args()

    pd_df = pd.read_pickle(args.cluster_assgn_file)

    slices =  np.stack(pd_df["domino_slices"].to_numpy())

    group_assignment = {}
    predicted_labels = {}
    id_to_label = {0:"contradiction", 1:"entailment", 2:"neutral"}
    label_to_id = {v:k for k,v in id_to_label.items()}
    for i in range(len(pd_df)):
        ## Usually analysis is done with group assignment only if value is above a threshold.
        slice = int(np.argmax(pd_df.iloc[i]["domino_slices"]))
        slice_val = np.max(pd_df.iloc[i]["domino_slices"])
        if slice_val > 0.70:
            chosen_slice = slice
        else:
            chosen_slice = -1
        guid = pd_df.iloc[i]["guid"]
        group_assignment[guid] = chosen_slice
        predicted_labels[guid] = id_to_label[np.argmax(pd_df.iloc[i]["pred_probs"])]

    print(Counter(group_assignment.values()))

    # Find those groups which have a high rate of confusion between predicted and gold labels.
    dataset = [json.loads(line) for line in open(args.dataset_file)]
    groups = {k:[] for k in range(len(set(group_assignment.values())))}
    with open(args.output_file, "w") as fout:
        for ex in dataset:
            guid = ex["guid"]
            new_ex = ex
            group = group_assignment[guid] + 1
            gold_label = ex["label"]
            predicted_label = predicted_labels[guid]
            new_ex["predicted"] = predicted_label
            new_ex["group"] = group
            groups[group].append(new_ex)

    # del the weakly assigned slices?

    group_f1s = []
    for group_no, group in groups.items():
        ypred = []
        ytrue = []
        for ex in group:
            ypred.append(label_to_id[ex["predicted"]])
            ytrue.append(label_to_id[ex["label"]])
        group_f1s.append(f1_score(ytrue, ypred, average="micro"))

    ## Take top 10 slices with lowest F1 and print out slice but spearated by the predicted and gold label.
    with open(args.output_file, "w") as fout:
        for group_no in np.argsort(group_f1s)[:20]:
            fout.write("Slice No. {0}\n\n".format(group_no))
            if group_no == 0:
                continue
            group = groups[group_no]
            
            for ex in group:
                if ex["label"] == "entailment":
                    if ex["predicted"] != "entailment":
                        fout.write("Premise:" + ex["sentence1"] + "\n" + "Hyp:" + ex["sentence2"] + "\n" + "Genre:" +  ex["genre"] + "\n" + "Gold:" + ex["label"] + "\n" +  "Predicted:" + ex["predicted"] + "\n\n")
            fout.write("Slice No. {0}".format(group_no) + "_"*50 + "\n\n")
            for ex in group:
                if ex["label"] == "contradiction":
                    if ex["predicted"] != "contradiction":
                        fout.write("Premise:" + ex["sentence1"] + "\n" + "Hyp:" + ex["sentence2"] + "\n" + "Genre:" +  ex["genre"] + "\n" + "Gold:" + ex["label"] + "\n" +  "Predicted:" + ex["predicted"] + "\n\n")
            fout.write("Slice No. {0}".format(group_no) + "_"*50 + "\n\n")
            for ex in group:
                if ex["label"] == "neutral":
                    if ex["predicted"] != "neutral":
                        fout.write("Premise:" + ex["sentence1"] + "\n" + "Hyp:" + ex["sentence2"] + "\n" + "Genre:" +  ex["genre"] + "\n" + "Gold:" + ex["label"] + "\n" +  "Predicted:" + ex["predicted"] + "\n\n")
            fout.write("*"*50 + "\n\n")