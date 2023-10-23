"""Characterizes depth-based clustering for contours.
Where does it help? Where does it not help?"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.contour_depth.synthetic_data import magnitude_modes, three_rings, shape_families
from src.contour_depth.utils import get_masks_matrix, get_sdfs
from src.contour_depth.clustering import ddclust
from src.contour_depth.clustering import inits

if __name__ == "__main__":    
    num_masks = 30
    ROWS = COLS = 512

    color_map = plt.colormaps.get_cmap("rainbow")
    unique_colors = np.linspace(0, 1, num=5)  # we dont have examples with more than 3 modes
    color_dict = {i: color_map(cid) for i, cid in enumerate(unique_colors)}

    # We replicate the grid with different cases as in the paper of streamlines 
    # rows: different datasets. we consider 3 representative ones
    # cols: different methods. we consider the state of the art + the depth-enhanced ones

    masks_d1, labs_d1 = magnitude_modes(num_masks=num_masks, num_rows=ROWS, num_cols=COLS, return_labels=True)
    masks_d2, labs_d2 = three_rings(num_masks=num_masks, num_rows=ROWS, num_cols=COLS, return_labels=True)
    masks_d3, labs_d3 = shape_families(num_masks=num_masks, num_rows=ROWS, num_cols=COLS, return_labels=True)
    # we might want to add a fourth one that is topology or more localized shape differences
    # this should illustrate the advantages of bringing depth

    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(10,8), layout="tight")

    # - ground truth column 
    axs[0, 0].set_title("Ground Truth")
    for m, l in zip(masks_d1, labs_d1):
        axs[0, 0].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    for m, l in zip(masks_d2, labs_d2):
        axs[1, 0].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    for m, l in zip(masks_d3, labs_d3):
        axs[2, 0].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    # - PCA(sdf) + KMeans
    axs[0, 1].set_title("PCA + KMEANS")

    pred_labs_kmeans_1 = inits.initial_clustering(masks_d1, num_components=2, use_sdf=True, pre_pca=True, method="kmeans")
    for m, l in zip(masks_d1, pred_labs_kmeans_1):
        axs[0, 1].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    pred_labs_kmeans_2 = inits.initial_clustering(masks_d2, num_components=3, use_sdf=True, pre_pca=True, method="kmeans")
    for m, l in zip(masks_d2, pred_labs_kmeans_2):
        axs[1, 1].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    pred_labs_kmeans_3 = inits.initial_clustering(masks_d3, num_components=2, use_sdf=True, pre_pca=True, method="kmeans")
    for m, l in zip(masks_d3, pred_labs_kmeans_3):
        axs[2, 1].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    # - PCA(sdf) + AHC column
    axs[0, 2].set_title("PCA + AHC")
    
    pred_labs_ahc_1 = inits.initial_clustering(masks_d1, num_components=2, use_sdf=True, pre_pca=True, method="ahc")
    for m, l in zip(masks_d1, pred_labs_ahc_1):
        axs[0, 2].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    pred_labs_ahc_2 = inits.initial_clustering(masks_d2, num_components=3, use_sdf=True, pre_pca=True, method="ahc")
    for m, l in zip(masks_d2, pred_labs_ahc_2):
        axs[1, 2].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    pred_labs_ahc_3 = inits.initial_clustering(masks_d3, num_components=2, use_sdf=True, pre_pca=True, method="ahc")
    for m, l in zip(masks_d3, pred_labs_ahc_3):
        axs[2, 2].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    # - PCA(sdf) + AHC + Depth-enhanced clustering column (with eID)
    axs[0, 3].set_title("PCA + AHC \n+ DDClust (eID)")

    pred_labs_idahc_1 = ddclust.ddclust(masks_d1, init_labs=pred_labs_ahc_1, depth_notion="id", use_modified=True, output_extra_info=False)
    for m, l in zip(masks_d1, pred_labs_idahc_1):
        axs[0, 3].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    pred_labs_idahc_2 = ddclust.ddclust(masks_d2, init_labs=pred_labs_ahc_2, depth_notion="id", use_modified=True, output_extra_info=False)
    for m, l in zip(masks_d2, pred_labs_idahc_2):
        axs[1, 3].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    pred_labs_idahc_3 = ddclust.ddclust(masks_d3, init_labs=pred_labs_ahc_3, depth_notion="id", use_modified=True, output_extra_info=False)
    for m, l in zip(masks_d3, pred_labs_idahc_3):
        axs[2, 3].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    # - PCA(sdf) + Depth-enhanced clustering column (with mCBD)
    # axs[0, 3].set_title("PCA + DDClust (eID)")

    # pred_labs = ddclust.cluster_contours(masks_d1, num_components=2)
    # for m, l in zip(masks_d1, pred_labs):
    #     axs[0, 3].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    # pred_labs = ddclust.cluster_contours(masks_d2, num_components=3)
    # for m, l in zip(masks_d2, pred_labs):
    #     axs[1, 3].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)

    # pred_labs = ddclust.cluster_contours(masks_d3, num_components=2)
    # for m, l in zip(masks_d3, pred_labs):
    #     axs[2, 3].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)


    for ax in axs.flatten():
        ax.set_axis_off()
        ax.set_ylim(0, ROWS-1)
        ax.set_xlim(0, COLS-1)
        ax.set_box_aspect(ROWS/COLS)  # keep aspect fixed

    plt.savefig("/Users/chadepl/Downloads/clustering_grid.png")
    plt.show()
    