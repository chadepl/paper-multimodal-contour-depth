""" Generates figure demonstrating clustering analysis on head and neck data.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from contour_depth.data import han_seg_ensembles as hanseg
from contour_depth.depth import inclusion_depth, band_depth
from contour_depth.clustering import ddclust, inits
from contour_depth.visualization import plot_clustering
from contour_depth.utils import get_masks_matrix, get_sdfs
from contour_depth.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix

# Path to data
data_dir = Path("../data/han_ensembles/")

img, gt, masks = hanseg.get_han_ensemble(data_dir, slice_num=None)

print(img.shape)

# plt.imshow(img[41])
# plt.imshow(ensemble_masks[0][41], alpha=ensemble_masks[0][41])
# plt.show()

slice_masks = [m[41] for m in masks]
#spaghetti_plot(slice_masks, iso_value=0.5)

init_seed = 1
ddclust_seed = 1

# - evaluation of optimal number of clusters
if False:
    all_labs = []
    sils = []
    reds = []
    costs = []
    possible_n_components = [2, 3, 4, 5, 6]
    for n_components in possible_n_components:
        labs = inits.initial_clustering(slice_masks, num_components=n_components, method="random", seed=init_seed)
        pred_labs, sil_i, red_i, cost_i = ddclust.ddclust(slice_masks, labs, cost_lamb=1.0, depth_notion="id", use_modified=True, use_fast=True, output_extra_info=True, seed=ddclust_seed)
        all_labs.append(pred_labs)
        sils.append(sil_i.mean())
        reds.append(red_i.mean())
        costs.append(cost_i.mean())

    plt.plot(possible_n_components, sils, label="sils")
    plt.plot(possible_n_components, reds, label="reds")
    plt.plot(possible_n_components, costs, label="costs")
    plt.legend()
    plt.show()

clustering_seed = 0
init_seed = 0

sdfs = get_sdfs(slice_masks)
sdfs_mat = get_masks_matrix(sdfs)
masks_mat = get_masks_matrix(slice_masks)

strict_inclusion_mat = compute_inclusion_matrix(slice_masks) 
epsilon_inclusion_mat = compute_epsilon_inclusion_matrix(slice_masks)

general_clustering_kwargs = dict(
    feat_mat = [masks_mat, sdfs_mat, strict_inclusion_mat, epsilon_inclusion_mat][1],
    pre_pca = True,
    cost_lamb = 1.0, #1.0,
    beta_init = 1,  # np.inf means no annealing
    beta_mult = 2,
    no_prog_its = 5,
    max_iter=200,
    cost_threshold=0, 
    swap_subset_max_size=5,
    competing_cluster_method=["sil", "red", "inclusion_rel"][2],
    depth_notion = ["id", "cbd"][0],
    use_modified=True, 
    use_fast=False,
    seed = clustering_seed,
    output_extra_info=True
)

clusterings = []
for k in range(2, 6):
    labs = inits.initial_clustering(slice_masks, num_components=k, method="random", seed=init_seed)
    pred_labs, sil_i, red_i, cost_i = ddclust.ddclust(slice_masks, labs, **general_clustering_kwargs)
    clusterings.append(pred_labs)


fig, axs = plt.subplots(ncols=4, figsize=(12, 8))
#spaghetti_plot(slice_masks, iso_value=0.5, arr=pred_labs, is_arr_categorical=True)
for i, clustering in enumerate(clusterings):
    plot_clustering(slice_masks, clustering, ax=axs[i])
plt.show()
