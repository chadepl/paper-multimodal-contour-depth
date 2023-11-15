# Load dataset
# Try different clusterings (varying num components)
# Compute silhouettes 
# Compute relative depth
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from contour_depth.data.synthetic_data import magnitude_modes, three_rings, shape_families
from contour_depth.utils import get_masks_matrix, get_sdfs


from contour_depth.clustering.inits import initial_clustering
from contour_depth.clustering.ddclust import compute_sil, compute_red
from contour_depth.visualization import plot_clustering, plot_red, plot_clustering_eval

from contour_depth.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix

if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/num_clust_selection")
    assert outputs_dir.exists()

    seed_data = 0
    seed_clustering = 0
    seed_init = 0

    num_contours = 100
    # masks, labs = magnitude_modes(num_contours, 521, 512, return_labels=True, seed=data_seed)
    masks, labs = three_rings(num_contours, 512, 512, return_labels=True, seed=seed_data)
    # masks, labs = shape_families(num_contours, 512, 512, return_labels=True, seed=data_seed)
    labs = np.array(labs)
    num_clusters = np.unique(labs).size

    # precompute matrix

    sdfs = get_sdfs(masks)
    sdfs_mat = get_masks_matrix(sdfs)
    sdfs_mat_red = PCA(n_components=50, random_state=seed_clustering).fit_transform(sdfs_mat)

    inclusion_mat = compute_inclusion_matrix(masks)

    ###################
    # Clustering algo #
    ###################

    # Input: data, labels (labels have been assigned by a method or randomly)
    # Input: number of clusters K

    ks = list(range(2, 10))
    clusterings = []
    sils = []
    reds = []
    costs = []
    for k in ks:
        print(k)
        pred_labs = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=seed_clustering).fit_predict(sdfs_mat_red)
        clusterings.append(pred_labs)
        sil_i, _, _, _ = compute_sil(sdfs_mat_red, pred_labs, n_components=k)
        red_i, _, _, _ = compute_red(masks, pred_labs, n_components=k, competing_clusters=None, depth_notion="id", use_modified=False, use_fast=False, inclusion_mat=inclusion_mat)
        sils.append(sil_i.mean())
        reds.append(red_i.mean())

    #############################
    # Elements for paper figure #
    #############################

    xmin_id = np.argmin(reds)
    xmax_id = np.argmax(reds)
    xmin = ks[xmin_id]
    xmax = ks[xmax_id]

    fig, ax1 = plt.subplots(layout="tight", figsize=(4, 3))

    plot_clustering_eval(ks, reds, metric_b=sils,
                         metric_a_id="ReD", metric_a_lab="Average ReD",
                         metric_b_id="Sil", metric_b_lab="Average Sil", ax=ax1)

    fig.savefig(outputs_dir.joinpath("clust-vs-red-sil.png"), dpi=300)


    labels = ["min", "max"]

    all_labs = [clusterings[xmin_id], clusterings[xmax_id]]
    for i, v in enumerate(all_labs):
        fig, ax = plt.subplots(layout="tight", figsize=(3, 3))
        plot_clustering(masks, v, ax=ax)
        ax.set_axis_off()
        fig.savefig(outputs_dir.joinpath(f"{labels[i]}.png"), dpi=300)


    for i, v in enumerate(clusterings):
        fig, ax = plt.subplots(layout="tight", figsize=(1, 1))
        plot_clustering(masks, v, ax=ax)
        ax.set_axis_off()
        fig.savefig(outputs_dir.joinpath(f"clustering_k{i + 2}.png"), dpi=300)

