""" This file tests that the ddclust clustering procedure works as expected.
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.contour_depth.data.synthetic_data import magnitude_modes, three_rings, shape_families
from src.contour_depth.depth.utils import get_masks_matrix, get_sdfs
from src.contour_depth.clustering.ddclust import compute_sil, compute_red, compute_cost


from src.contour_depth.clustering.inits import initial_clustering
from src.contour_depth.clustering.ddclust import ddclust
from src.contour_depth.clustering.vis_utils import plot_clustering_results

from src.contour_depth.depth.utils import compute_inclusion_matrix

num_contours = 30
data_seed = 1
clustering_seed = 4
init_seed = 3
#masks, labs = magnitude_modes(num_contours, 521, 512, return_labels=True, seed=seed)
masks, labs = three_rings(num_contours, 512, 512, return_labels=True, seed=data_seed)
# masks, labs = shape_families(num_contours, 512, 512, return_labels=True, seed=seed)
labs = np.array(labs)
num_clusters = np.unique(labs).size

sdfs = get_sdfs(masks)
sdfs_mat = get_masks_matrix(sdfs)
masks_mat = get_masks_matrix(masks)

inclusion_mat = compute_inclusion_matrix(masks)

###################
# Clustering algo #
###################

# Input: data, labels (labels have been assigned by a method or randomly)
# Input: number of clusters K

clustering_kwargs = dict(
    feat_mat = [masks_mat, sdfs_mat, inclusion_mat][2],
    pre_pca = False,
    cost_lamb = 0.5, #1.0,
    beta_init = 1,  # np.inf means no annealing
    beta_mult = 2,
    no_prog_its = 5,
    max_iter=50,
    cost_threshold=0, 
    swap_subset_max_size=5,
    depth_notion = ["id", "cbd"][0],
    use_modified = False,
    use_fast = False,    
    seed = clustering_seed
)

def get_kwargs_str():
    new_kwargs = clustering_kwargs.copy()
    del new_kwargs["feat_mat"]
    kwargs_str = []
    num_kwargs_per_line = 6
    num_kwargs = len(new_kwargs)
    num_lines = int(np.ceil(num_kwargs / num_kwargs_per_line))
    ks = list(new_kwargs.keys())
    vs = list(new_kwargs.values())
    j = 0
    for l in range(num_lines):
        line = []
        for i in range(num_kwargs_per_line):
            if l * num_kwargs_per_line + i <= num_kwargs:
                line.append(f"{ks[j]}: {vs[j]}")
                j += 1
        kwargs_str.append(",".join(line))        
    kwargs_str = "\n".join(kwargs_str)
    return kwargs_str

init_labs = initial_clustering(masks, num_components=num_clusters, feat_mat=clustering_kwargs["feat_mat"], pre_pca=False, method="random", seed=init_seed)

# Initial state
print("Saving initial state ...")
sil_i, sil_a, sil_b, competing_clusters = compute_sil(clustering_kwargs["feat_mat"], init_labs)
red_i, red_w, red_b, medians = compute_red(masks, init_labs, competing_clusters, 
                                           depth_notion=clustering_kwargs["depth_notion"], 
                                           use_modified=clustering_kwargs["use_modified"],
                                           use_fast=clustering_kwargs["use_fast"])
plot_clustering_results(masks, init_labs, sil_i, red_i, fn="/Users/chadepl/Downloads/start_cluster.png", suptitle=get_kwargs_str())

print(get_kwargs_str())

print()
print()
print("Start clustering")
# Run ddclust
new_labs, _, _, _ = ddclust(masks, init_labs, **clustering_kwargs)
print()
print()

# Final state
print("Saving final state ...")
new_sil_i, new_sil_a, new_sil_b, new_competing_clusters = compute_sil(clustering_kwargs["feat_mat"], new_labs)
new_red_i, new_red_w, new_red_b, new_medians = compute_red(masks, new_labs, competing_clusters, 
                                                           depth_notion=clustering_kwargs["depth_notion"], 
                                                           use_modified=clustering_kwargs["use_modified"],
                                                           use_fast=clustering_kwargs["use_fast"])
plot_clustering_results(masks, new_labs, new_sil_i, new_red_i, fn="/Users/chadepl/Downloads/end_cluster.png", suptitle=get_kwargs_str())


# fig, axs = plt.subplots(ncols=2)
# for m in masks:
#     axs[0].contour(m)

# depths = compute_depths(masks, modified=True, fast=True)
# axs[1].hist(depths)
# plt.show()

