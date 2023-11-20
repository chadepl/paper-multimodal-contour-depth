"""Generates figure demonstrating clustering analysis on meteorological data.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from contour_depth.data import ecmwf_ensembles as ecmwf
from contour_depth.depth import inclusion_depth, band_depth
from contour_depth.clustering import ddclust, inits
from contour_depth.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering, get_cvp_pca_medians, get_cvp_bands, transform_from_pca_to_sdf, get_per_cluster_mean
from contour_depth.visualization import plot_clustering, spaghetti_plot, plot_contour_boxplot


if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/rd_meteo")
    assert outputs_dir.exists()

    # Path to data
    data_dir = Path("../data/cvp-paper-meteo/")

    # Analysis A: ["20121015", "120", 500, 5600],  # time, pressure level, height, contour
    masks = ecmwf.load_data(data_dir, config_id=0)

    # - CVP analysis
    # sdf_mat, pca_mat, transform_mat = get_cvp_sdf_pca_transform(masks, seed=0)
    # pred_labs  = get_cvp_clustering(sdf_mat, num_components=3)
    # pca_medians = get_cvp_pca_medians(pca_mat, pred_labs)
    # sdf_means = get_per_cluster_mean(sdf_mat, pred_labs)
    # medians = transform_from_pca_to_sdf(np.array(pca_medians)*0.1, np.array(sdf_means), transform_mat)
    # bands = get_cvp_bands(sdf_mat, pred_labs)

    

    # - depths
    depths = inclusion_depth.compute_depths(masks, modified=True, fast=True)
    print(depths)

    seed_init = 1
    seed_ddclust = 1


    # masks_shape = masks[0].shape
    # fig, ax = plt.subplots(figsize=(5,5), layout="tight")
    # # ax.imshow(bands[2].reshape(*masks_shape)>0)
    # # plot_clustering(masks, pred_labs, ax=ax)
    # spaghetti_plot(masks, iso_value=0.5, arr=np.ones(len(masks))*0.5, is_arr_categorical=False, ax=ax)
    # ax.contour(medians[0].reshape(*masks_shape), levels=[0,], colors="red")
    # ax.contour(medians[1].reshape(*masks_shape), levels=[0,], colors="green")
    # ax.contour(medians[2].reshape(*masks_shape), levels=[0,], colors="blue")
    # plt.imshow(plt.cm.get_cmap("Purples")(bands[2].reshape(*masks_shape)*0.5))
    # plt.show()
    from skimage.io import imread
    from skimage.transform import resize, EuclideanTransform, warp
    from skimage.filters import gaussian
    from contour_depth.clustering.ddclust import compute_red
    from contour_depth.visualization import get_bp_cvp_elements, get_bp_depth_elements
    from contour_depth.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering

    img = imread(data_dir.joinpath("picking_background.png"), as_gray=True)
    flipped_img = img[::-1, :]
    
    print(flipped_img.shape)

    sdf_mat, pca_mat, transform_mat = get_cvp_sdf_pca_transform(masks)
    pred_labs = get_cvp_clustering(pca_mat, num_components=3)
    cluster_statistics = get_bp_cvp_elements(masks, labs=pred_labs)    
    
    red = compute_red
    cluster_statistics = get_bp_depth_elements(masks, )

    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")    
    # start_point = (145, 72)  # OG
    start_point = (145, 72)
    # target_size = (1292, 1914)  # OG
    target_size = (1292, 1914)  # OG
    for cluster_id in np.unique(pred_labs):
        # resize + flip
        tform = EuclideanTransform(rotation=0, translation=(-start_point[1],-start_point[0]))
        cluster_statistics[cluster_id]["representatives"]["masks"] = [m[::-1, :] for m in cluster_statistics[cluster_id]["representatives"]["masks"]]
        cluster_statistics[cluster_id]["representatives"]["masks"] = [resize(m, target_size, mode="constant", cval=0, order=0) for m in cluster_statistics[cluster_id]["representatives"]["masks"]]
        cluster_statistics[cluster_id]["representatives"]["masks"] = [warp(m, tform, mode="reflect") for m in cluster_statistics[cluster_id]["representatives"]["masks"]]        
        
        cluster_statistics[cluster_id]["bands"]["masks"] = [m[::-1, :] for m in cluster_statistics[cluster_id]["bands"]["masks"]]
        cluster_statistics[cluster_id]["bands"]["masks"] = [resize(m, target_size, mode="constant", cval=0, order=1) for m in cluster_statistics[cluster_id]["bands"]["masks"]]
        cluster_statistics[cluster_id]["bands"]["masks"] = [warp(m, tform, mode="reflect") for m in cluster_statistics[cluster_id]["bands"]["masks"]]

        for smit in range(5):
            cluster_statistics[cluster_id]["representatives"]["masks"] = [gaussian(m, sigma=20) for m in cluster_statistics[cluster_id]["representatives"]["masks"]]
        # for smit in range(20):
        #     cluster_statistics[cluster_id]["bands"]["masks"] = [gaussian(m, sigma=1) for m in cluster_statistics[cluster_id]["bands"]["masks"]]

    # spaghetti_plot(masks, iso_value=0.5, arr=np.ones(len(masks))*0.5, is_arr_categorical=False, ax=ax)spaghetti_plot(masks, iso_value=0.5, arr=np.ones(len(masks))*0.5, is_arr_categorical=False, ax=ax)
    plot_contour_boxplot(masks, pred_labs, cluster_statistics=cluster_statistics, show_out=False, under_mask=img, ax=ax)
    plt.show()

    # compute elements on original resolution
    # resize and reposition 


    # labs = inits.initial_clustering(masks, num_components=2, method="random", seed=seed_init)
    # pred_labs, sil_i, red_i, cost_i = ddclust.ddclust(masks, labs, cost_lamb=1.0, depth_notion="id", use_modified=True, use_fast=False, output_extra_info=True, seed=seed_ddclust)
    # fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
    # ecmwf.plot_contours_with_bg(masks, red_i, data_dir.joinpath("picking_background.png"), is_color_categorical=False, fname=None, ax=axs[0])
    # ecmwf.plot_contours_with_bg(masks, pred_labs, data_dir.joinpath("picking_background.png"), is_color_categorical=True, fname=None, ax=axs[1])
    # plt.show()



# Analysis B: []



