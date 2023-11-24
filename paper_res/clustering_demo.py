
from time import time
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from contour_depth.data.synthetic_data import magnitude_modes, three_rings, shape_families, main_shape_with_outliers
from contour_depth.utils import get_masks_matrix, get_sdfs
from contour_depth.clustering.ddclust import compute_sil, compute_red, compute_cost, compute_red_within, compute_red_between


from contour_depth.clustering.inits import initial_clustering
from contour_depth.clustering.ddclust import cdclust, kmeans_cluster_inclusion_matrix, kmeans_cluster_eid
from contour_depth.visualization import plot_clustering, plot_red

from contour_depth.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix

data_seed = 0
clustering_seed = 5
init_seed = 1

################
# Data loading #
################

num_contours = 100
#masks, labs = magnitude_modes(num_contours, 521, 512, return_labels=True, seed=data_seed)
#masks, labs = three_rings(num_contours, 512, 512, return_labels=True, seed=data_seed)
# masks, labs = shape_families(num_contours, 512, 512, return_labels=True, seed=data_seed)
masks, labs = main_shape_with_outliers(num_contours, 512, 512, p_contamination=0.5, return_labels=True, seed=data_seed)
labs = np.array(labs)
num_clusters = np.unique(labs).size

# precompute matrix

# sdfs = get_sdfs(masks)
# sdfs_mat = get_masks_matrix(sdfs)
# masks_mat = get_masks_matrix(masks)

strict_inclusion_mat = compute_inclusion_matrix(masks) 
epsilon_inclusion_mat = compute_epsilon_inclusion_matrix(masks)
#strict_inclusion_mat = epsilon_inclusion_mat = np.zeros((num_contours, num_contours))

###################
# Clustering algo #
###################

# Input: data, labels (labels have been assigned by a method or randomly)
# Input: number of clusters K

clustering_kwargs_kmeans = dict(
    beta_init = np.inf, # np.inf means no annealing
    beta_mult = 2, # will not have any effect
    no_prog_its = 100,
    max_iter = 200,
    cost_threshold = 0.04, # includes every contour in the swapping 
    swap_subset_max_size = num_contours, # all contours are tested
    # sample_swap_size = False,
    #check_cost_function=False,
    # competing_cluster_method=["sil", "red", "inclusion_rel"][2],
       
)

clustering_kwargs_annealing = dict(
    beta_init = 1,  # we do annealing
    beta_mult = 2,
    no_prog_its = 5, # reduce the amount of iterations without progress
    max_iter=200,
    cost_threshold=0,  # we only include contours with ReD below 0 in the swapping 
    swap_subset_max_size=5, # mini-batch
    #sample_swap_size = True,
    #check_cost_function=True,
    # competing_cluster_method=["sil", "red", "inclusion_rel"][2],
)

clustering_kwargs = clustering_kwargs_annealing #clustering_kwargs_annealing
clustering_kwargs.update(dict(
    depth_notion = ["id", "cbd"][0],
    seed = clustering_seed,
    output_extra_info=True,
))

cdclust_kwargs = clustering_kwargs_annealing
cdclust_kwargs["verbose"] = False

init_labs = initial_clustering(masks, num_components=num_clusters, feat_mat=None, pre_pca=False, method="random", seed=init_seed)

# Run kmeans on Contour Band Depth
cdclust_kwargs["use_modified"] = True
cdclust_kwargs["use_fast"] = True
t_start = time()
kmeans_matrix_labs = kmeans_cluster_inclusion_matrix(masks, num_clusters, depth="eid", metric="depth", num_attempts=3)
print(f"Finished inclusion matrix kmeans (eID) in {time() - t_start} seconds")

# epsilon_clustering_kwargs = clustering_kwargs
# epsilon_clustering_kwargs["use_modified"] = True
# epsilon_clustering_kwargs["use_fast"] = True

# Run ddclust
cdclust_kwargs["use_modified"] = True
cdclust_kwargs["use_fast"] = False
t_start = time()
kmeans_ON_labs = kmeans_cluster_eid(masks, num_clusters, num_attempts=3)
print(f"Finished eID O(N) kmeans in {time() - t_start} seconds")

##########
# Figure #
##########

def get_kwargs_str():
    new_kwargs = clustering_kwargs.copy()
    del new_kwargs["feat_mat"]
    del new_kwargs["output_extra_info"]
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


def get_depth_data(masks, labs, n_components, inclusion_mat, use_modified=False):
    red_w = compute_red_within(masks, labs, n_components=n_components, 
                   depth_notion="id", use_modified=use_modified, use_fast=False, inclusion_mat=inclusion_mat)
    red_b, _ = compute_red_between(masks, labs, n_components=n_components, competing_clusters=None,
                                                          depth_notion="id", use_modified=use_modified, use_fast=False, inclusion_mat=inclusion_mat)
    red_i = red_w - red_b
    return red_i, red_w, red_b


fig, axs = plt.subplots(nrows=3, ncols=4, layout="tight")

# p1

axs[0, 0].set_title("Initial labels")
plot_clustering(masks, init_labs, ax=axs[0, 0])
axs[0, 0].set_axis_off()

red_i, red_w, red_b = get_depth_data(masks, init_labs, n_components=num_clusters, inclusion_mat=strict_inclusion_mat, use_modified=False)
axs[1, 0].set_title(f"ID: {red_i.mean():.4f}")
plot_red(red_w, red_b, compute_red=True, labs=init_labs, ax=axs[1, 0])

red_i, red_w, red_b = get_depth_data(masks, init_labs, n_components=num_clusters, inclusion_mat=epsilon_inclusion_mat, use_modified=True)
axs[2, 0].set_title(f"eID: {red_i.mean():.4f}")
plot_red(red_w, red_b, compute_red=True, labs=init_labs, ax=axs[2, 0])


axs[0, 1].set_title("Target labels")
plot_clustering(masks, labs, ax=axs[0, 1])
axs[0, 1].set_axis_off()

red_i, red_w, red_b = get_depth_data(masks, labs, n_components=num_clusters, inclusion_mat=strict_inclusion_mat, use_modified=False)
axs[1, 1].set_title(f"ID: {red_i.mean():.4f}")
plot_red(red_w, red_b, compute_red=True, labs=labs, ax=axs[1, 1])

red_i, red_w, red_b = get_depth_data(masks, labs, n_components=num_clusters, inclusion_mat=epsilon_inclusion_mat, use_modified=True)
axs[2, 1].set_title(f"eID: {red_i.mean():.4f}")
plot_red(red_w, red_b, compute_red=True, labs=labs, ax=axs[2, 1])


axs[0, 2].set_title("kmeans (Inclusion Matrix)")
plot_clustering(masks, kmeans_matrix_labs, ax=axs[0, 2])
axs[0, 2].set_axis_off()

red_i, red_w, red_b = get_depth_data(masks, kmeans_matrix_labs, n_components=num_clusters, inclusion_mat=strict_inclusion_mat, use_modified=False)
axs[1, 2].set_title(f"ID: {red_i.mean():.4f}")
plot_red(red_w, red_b, compute_red=True, labs=kmeans_matrix_labs, ax=axs[1, 2])
axs[1, 2].axhline(y=clustering_kwargs["cost_threshold"], c="green")

red_i, red_w, red_b = get_depth_data(masks, kmeans_matrix_labs, n_components=num_clusters, inclusion_mat=epsilon_inclusion_mat, use_modified=True)
axs[2, 2].set_title(f"eID: {red_i.mean():.4f}")
plot_red(red_w, red_b, compute_red=True, labs=kmeans_matrix_labs, ax=axs[2, 2])


axs[0, 3].set_title("kmeans (O(N) eID)")
plot_clustering(masks, kmeans_ON_labs, ax=axs[0, 3])
axs[0, 3].set_axis_off()

red_i, red_w, red_b = get_depth_data(masks, kmeans_ON_labs, n_components=num_clusters, inclusion_mat=strict_inclusion_mat, use_modified=False)
axs[1, 3].set_title(f"ID: {red_i.mean():.4f}")
plot_red(red_w, red_b, compute_red=True, labs=kmeans_ON_labs, ax=axs[1, 3])

red_i, red_w, red_b = get_depth_data(masks, kmeans_ON_labs, n_components=num_clusters, inclusion_mat=epsilon_inclusion_mat, use_modified=True)
axs[2, 3].set_title(f"eID: {red_i.mean():.4f}")
plot_red(red_w, red_b, compute_red=True, labs=kmeans_ON_labs, ax=axs[2, 3])
axs[2, 3].axhline(y=clustering_kwargs_annealing["cost_threshold"], c="green")

plt.show()



