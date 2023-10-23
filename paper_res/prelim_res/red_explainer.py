"""This file shows how to use ReD to tell outliers appart
"""

# Load dataset with two types of shapes.
# Compute ReD of optimal allocation.
# Compute ReD of clustering performed by AHC.
# Output: Fig with two rows each with Allocation, ReD and Sil
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")

from src.contour_depth.synthetic_data import shape_families
from src.contour_depth.depth.utils import compute_inclusion_matrix
from src.contour_depth.clustering.ddclust import compute_red, compute_sil, compute_cost
from src.contour_depth.clustering.inits import initial_clustering

if __name__ == "__main__":

    num_masks = 30
    num_rows = num_cols = 512
    masks, labs = shape_families(num_masks, num_rows, num_cols, return_labels=True, seed=2)

    depth_notion = ["id", "cbd"][0]
    use_modified = False
    use_fast = False

    inclusion_mat = compute_inclusion_matrix(masks)

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 8), layout="tight")

    # First row: expected allocation
    sil_i, sil_a, sil_b, competing_clusters = compute_sil(inclusion_mat, labs)
    red_i, red_w, red_b, medians = compute_red(masks, labs, competing_clusters=competing_clusters, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast)

    axs[0, 0].set_title(f"GT labels | ReD={red_i.mean():.4f}")
    axs[0, 0].set_axis_off()
    for m, l in zip(masks, labs):
        c = "red" if l == 0 else "blue"
        axs[0, 0].contour(m, colors=[c, ], linewidths=[1, ], alpha=1)

    axs[0, 1].bar(np.arange(num_masks), red_i, color=["red" if l == 0 else "blue" for l in labs])
    axs[0, 2].axhline(y=0, color="black")
    axs[0, 2].bar(np.arange(num_masks), red_w, color=["red" if l == 0 else "blue" for l in labs])
    axs[0, 2].bar(np.arange(num_masks), -red_b, color=["red" if l == 0 else "blue" for l in labs])
    axs[0, 3].bar(np.arange(num_masks), red_b, color=["red" if l == 0 else "blue" for l in labs])

    # Second row: AHC allocation
    pred_labs = initial_clustering(masks, num_components=2, method="ahc", pre_pca=True)
    sil_i, sil_a, sil_b, competing_clusters = compute_sil(inclusion_mat, pred_labs)
    red_i, red_w, red_b, medians = compute_red(masks, pred_labs, competing_clusters=competing_clusters, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast)
    
    axs[1, 0].set_title(f"AHC labels | ReD={red_i.mean():.4f}")
    axs[1, 0].set_axis_off()
    for m, l in zip(masks, pred_labs):
        c = "red" if l == 0 else "blue"
        axs[1, 0].contour(m, colors=[c, ], linewidths=[1, ], alpha=1)

    axs[1, 1].bar(np.arange(num_masks), red_i, color=["red" if l == 0 else "blue" for l in labs])
    axs[1, 2].axhline(y=0, color="black")
    axs[1, 2].bar(np.arange(num_masks), red_w, color=["red" if l == 0 else "blue" for l in labs])
    axs[1, 2].bar(np.arange(num_masks), -red_b, color=["red" if l == 0 else "blue" for l in labs])


    plt.show()