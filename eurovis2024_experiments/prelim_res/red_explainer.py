"""This file shows how to use ReD to tell outliers appart
"""

# Load dataset with two types of shapes.
# Compute ReD of optimal allocation.
# Compute ReD of clustering performed by AHC.
# Output: Fig with two rows each with Allocation, ReD and Sil
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../..")

from src.contour_depth.data.synthetic_data import magnitude_modes, three_rings, shape_families
from src.contour_depth.depth.utils import compute_inclusion_matrix
from src.contour_depth.clustering.ddclust import compute_red, compute_sil, compute_cost
from src.contour_depth.clustering.inits import initial_clustering

if __name__ == "__main__":

    num_masks = 30
    num_rows = num_cols = 512
    
    depth_notion = ["id", "cbd"][1]
    use_modified = False
    use_fast = False

    seed = 1

    dataset = ["magnitude_modes", "three_rings", "shape_families"][2]
    if dataset == "magnitude_modes":
        masks, labs = magnitude_modes(num_masks, num_rows, num_cols, return_labels=True, seed=seed, modes_proportions=(0.8, 0.2), modes_radius_std=(0.2 * 0.1, 0.16 * 0.05))
    elif dataset == "three_rings":
        masks, labs = three_rings(num_masks, num_rows, num_cols, return_labels=True, seed=seed)
    elif dataset == "shape_families":
        masks, labs = shape_families(num_masks, num_rows, num_cols, return_labels=True, seed=seed)

    num_components = np.unique(labs).size

    inclusion_mat = compute_inclusion_matrix(masks)    

    def plot_row(num_masks, labs, sil_i, sil_a, sil_b, red_i, red_w, red_b, cost_i, axs, row, title=""):

        axs[row, 0].set_title(title)
        axs[row, 0].set_axis_off()
        for m, l in zip(masks, labs):
            axs[row, 0].contour(m, colors=[labs_colors_map[l], ], linewidths=[1, ], alpha=0.1)

        x_range = np.arange(num_masks)

        axs[row, 1].set_title(f"Cost: {cost_i.mean():.4f}")
        axs[row, 1].bar(x_range, cost_i, color=colors)

        axs[row, 2].set_title(f"Sil: {sil_i.mean():.4f}")
        axs[row, 2].set_ylabel("Silhouette width")
        axs[row, 2].set_xlabel("Index")
        axs[row, 2].bar(x_range, sil_i, color=colors)        

        axs[row, 3].set_title("Sil_a, Sil_b")
        axs[row, 3].set_ylabel("Silhouette width")
        axs[row, 3].set_xlabel("Index")
        axs[row, 3].axhline(y=0, color="black")
        axs[row, 3].bar(x_range, sil_a, color=colors)
        axs[row, 3].bar(x_range, -sil_a, color=colors, alpha=0.3)
        axs[row, 3].bar(x_range, -sil_b, color=colors)
    
        axs[row, 4].set_title(f"ReD: {red_i.mean():.4f}")
        axs[row, 4].set_ylabel(f"Depth ({depth_notion})")
        axs[row, 4].set_xlabel("Index")
        axs[row, 4].bar(x_range, red_i, color=colors)

        axs[row, 5].set_title("D_w, D_b")
        axs[row, 5].set_ylabel(f"Depth ({depth_notion})")
        axs[row, 5].set_xlabel("Index")
        axs[row, 5].axhline(y=0, color="black")
        axs[row, 5].bar(x_range, red_w, color=colors)
        axs[row, 5].bar(x_range, -red_w, color=colors, alpha=0.3)
        axs[row, 5].bar(x_range, -red_b, color=colors)


    fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(15, 10), layout="tight")
    
    labs_colors_map = ["red", "blue", "orange"]    

    # First row: expected allocation
    sil_i, sil_a, sil_b, competing_clusters = compute_sil(inclusion_mat, labs)
    red_i, red_w, red_b, medians = compute_red(masks, labs, competing_clusters=competing_clusters, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast)
    cost_i = compute_cost(sil_i, red_i, weight=0.5)
    colors = [labs_colors_map[l] for l in labs]

    plot_row(num_masks, labs, sil_i, sil_a, sil_b, red_i, red_w, red_b, cost_i, axs, 0, title=f"GT labels")

    # Second row: AHC allocation
    pred_labs = initial_clustering(masks, num_components=num_components, method="ahc", pre_pca=True, seed=seed)
    sil_i, sil_a, sil_b, competing_clusters = compute_sil(inclusion_mat, pred_labs)
    red_i, red_w, red_b, medians = compute_red(masks, pred_labs, competing_clusters=competing_clusters, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast)
    colors = [labs_colors_map[l] for l in pred_labs]

    plot_row(num_masks, pred_labs, sil_i, sil_a, sil_b, red_i, red_w, red_b, cost_i, axs, 1, title=f"AHC labels")

    # Third row: random allocation
    pred_labs = initial_clustering(masks, num_components=num_components, method="random", seed=seed)
    sil_i, sil_a, sil_b, competing_clusters = compute_sil(inclusion_mat, pred_labs)
    red_i, red_w, red_b, medians = compute_red(masks, pred_labs, competing_clusters=competing_clusters, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast)
    colors = [labs_colors_map[l] for l in pred_labs]

    plot_row(num_masks, pred_labs, sil_i, sil_a, sil_b, red_i, red_w, red_b, cost_i, axs, 2, title=f"Random labels")

    fig.savefig(f"red_explainer/{dataset}-seed_{seed}-{depth_notion}-mod_{use_modified}-fast_{use_fast}.png")
    plt.show()