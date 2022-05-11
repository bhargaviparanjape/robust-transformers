import numpy as np
import pickle
import sys
import os
import pdb

domino = pickle.load(open(sys.argv[1], "rb"))
mm = domino.mm
means = mm.means_
cov = mm.covariances_
weights = mm.weights_
y_probs = mm.y_probs
y_hat_probs = mm.y_hat_probs

means_norm = means/np.expand_dims(np.linalg.norm(means, axis=1), axis=1)
cov_norm = cov/np.expand_dims(np.linalg.norm(cov, axis=1), axis=1)
y_probs_norm = y_probs/np.expand_dims(np.linalg.norm(y_probs, axis=1), axis=1)
y_hat_probs_norm = y_hat_probs/np.expand_dims(np.linalg.norm(y_hat_probs, axis=1), axis=1)

mean_sim = np.matmul(means_norm, means_norm.transpose())
cov_sim = np.matmul(cov_norm, cov_norm.transpose())
y_probs_sim = np.matmul(y_probs_norm, y_probs_norm.transpose())
y_hat_probs_sim = np.matmul(y_hat_probs_norm, y_hat_probs_norm.transpose())

n_slices = mean_sim.shape[0]
closest_means = np.dstack(np.unravel_index(np.argsort(mean_sim.ravel())[::-1], (n_slices, n_slices)))[0]
closest_vars = np.dstack(np.unravel_index(np.argsort(cov_sim.ravel())[::-1], (n_slices, n_slices)))[0]
closest_y_hat_probs = np.dstack(np.unravel_index(np.argsort(y_hat_probs_sim.ravel())[::-1], (n_slices, n_slices)))[0]
farthest_y_probs = np.dstack(np.unravel_index(np.argsort(y_probs_sim.ravel()), (n_slices, n_slices)))[0]

closest_means_string = ["_".join([str(elem) for elem in tup]) for tup in closest_means]
closest_vars_string = ["_".join([str(elem) for elem in tup]) for tup in closest_vars]
closest_y_hat_probs_string = ["_".join([str(elem) for elem in tup]) for tup in closest_y_hat_probs]
farthest_y_probs_string = ["_".join([str(elem) for elem in tup]) for tup in farthest_y_probs]

# Find the smallest average position in all:
positions = {}
for i in range(0,n_slices):
    for j in range(0, n_slices):
        if i == j:
            continue
        num_string = str(i) + "_" + str(j)
        # penalize when farthest_y_probs_string is too large compared to the others
        positions[num_string] = (farthest_y_probs_string.index(num_string)/10.0 + 10.0*closest_y_hat_probs_string.index(num_string) + 
                10.0*closest_means_string.index(num_string) + 10.0*closest_vars_string.index(num_string))/4.0


# Find the first 10 elements common to all lists.
# Find average positions for all.

positions_sorted = {k: v for k, v in sorted(positions.items(), key=lambda item: item[1])}
spurious_clusters = list(positions_sorted.keys())[:10]
print(spurious_clusters)

# Read pd file that contains group assignment and print out the groups that satisfy the required condition.
pdb.set_trace()


