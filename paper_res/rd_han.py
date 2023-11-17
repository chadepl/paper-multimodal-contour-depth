""" Generates figure demonstrating clustering analysis on head and neck data.
"""
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from contour_depth.data import han_seg_ensembles as hanseg
from contour_depth.depth import inclusion_depth, band_depth
from contour_depth.clustering.ddclust import cdclust, compute_red
from contour_depth.clustering.inits import initial_clustering 
from contour_depth.visualization import plot_clustering, spaghetti_plot, plot_red, sort_red, plot_clustering_eval
from contour_depth.utils import get_masks_matrix, get_sdfs
from contour_depth.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix


if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/rd_han")
    assert outputs_dir.exists()

    PATIENT_ID = "HCAI-036"
    OAR, SLICE_NUM = [("BrainStem", 31), ("Parotid_R", 41)][0]
    VERSION = ["strict", "modified"][1]

    # Path to data
    data_dir = Path("../data/han_ensembles/")

    img_vol, gt_vol, masks_vol = hanseg.get_han_ensemble(data_dir, 
                                                         patient_id=PATIENT_ID,
                                                         structure_name=OAR,
                                                         slice_num=None)

    print(img_vol.shape)

    prefix = f"{VERSION}_{PATIENT_ID}_{OAR}_{SLICE_NUM}"

    img = img_vol[SLICE_NUM]
    gt = gt_vol[SLICE_NUM]
    masks = [m[SLICE_NUM] for m in masks_vol]

    seed_init = 2
    seed_cdclust = 1    

    general_clustering_kwargs = dict(
        # feat_mat = [masks_mat, sdfs_mat, strict_inclusion_mat, epsilon_inclusion_mat][1],
        # pre_pca = True,
        beta_init = 1,  # np.inf means no annealing
        beta_mult = 2,
        no_prog_its = 5,
        max_iter=200,
        cost_threshold=0, 
        swap_subset_max_size=5,
        # competing_cluster_method=["sil", "red", "inclusion_rel"][2],
        depth_notion = ["id", "cbd"][0],
        use_modified=False if VERSION == "strict" else True, 
        use_fast=False,
        seed = seed_cdclust,
        output_extra_info=True
    )

    ####################
    # Opt num clusters #
    ####################

    # - evaluation of optimal number of clusters
    if outputs_dir.joinpath(f"{prefix}_k_eval.npy").exists():
        k_eval = np.load(outputs_dir.joinpath(f"{prefix}_k_eval.npy"))
    else:
        with open(outputs_dir.joinpath(f"{prefix}_ensemble.pkl"), "wb") as f:
            pickle.dump(dict(img=img_vol, masks=masks), f)

        possible_n_components = np.arange(2, 20)
        reds = []
        
        for n_components in possible_n_components:
            print(f"k_eval {n_components}")
            labs = initial_clustering(masks, num_components=n_components, method="random", seed=seed_init)
            pred_labs, red_i = cdclust(masks, labs, **general_clustering_kwargs)
            reds.append(red_i.mean())

            np.save(outputs_dir.joinpath(f"{prefix}_k-{n_components}_pred-labs.npy"), pred_labs)

        k_eval = np.array([possible_n_components, reds]).T
        np.save(outputs_dir.joinpath(f"{prefix}_k_eval.npy"), k_eval)

    
    print(k_eval)

    ks = k_eval[:,0].astype(int)
    reds = k_eval[:,1].astype(float)

    with open(outputs_dir.joinpath(f"{prefix}_ensemble.pkl"), "rb") as f:
        masks_dict = pickle.load(f)
        img_vol = masks_dict["img"]
        masks_vol = masks_dict["masks"]    

    #################
    # Figure K eval #
    #################

    fig, ax1 = plt.subplots(layout="tight", figsize=(4, 3))

    plot_clustering_eval(ks, reds,
                         metric_a_id="ReD", metric_a_lab="Average ReD", ax=ax1)

    fig.savefig(outputs_dir.joinpath(f"{prefix}_k-eval.png"), dpi=300)


    #################################
    # Figure band depths clustering #
    #################################  

    best_k_id = np.argsort(k_eval[:, 1])[-1]
    best_k = int(k_eval[best_k_id, 0])
    best_k_val = k_eval[best_k_id, 1]
    print(best_k)
    print(best_k_val) 

    pred_labs = np.load(outputs_dir.joinpath(f"{prefix}_k-{best_k}_pred-labs.npy"))

    full_dephts = inclusion_depth.compute_depths(masks, modified=True, fast=True)
    red_i, red_w, red_b, competing_clusters = compute_red(masks, pred_labs, n_components=best_k) 

    ag1, sorted_labs, sorted_red_w, sorted_red_b, sorted_red = sort_red(pred_labs, red_w, red_b, sort_by="red")
    full_dephts = full_dephts[ag1]

    print(f"full depths vs depth in clusters: {full_dephts.mean()} vs {red_w.mean()}")    

    print(pred_labs)

    CLUSTER_ID = 0
    cluster_members = np.where(sorted_labs == CLUSTER_ID)[0]
    min_depth_id = []#ag1[cluster_members[-10:-1]].tolist()
    max_depth_id = ag1[cluster_members[:10]].tolist()
    to_highlight = min_depth_id + max_depth_id

    # - spaghetti

    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
    spaghetti_plot(masks, 0.5, img, arr=None, smooth_line=False, highlight=None, ax=ax)

    # plt.show()

    fig.savefig(outputs_dir.joinpath(f"{prefix}_spaghetti.png"), dpi=300)

    from contour_depth.visualization import get_bp_depth_elements, plot_contour_boxplot
    
    # - full depth
    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
    depth_method_kwargs = dict(depths=full_dephts, outlier_type="tail", epsilon_out=3)
    plot_contour_boxplot(masks, None, method="depth", method_kwargs=depth_method_kwargs, under_mask=img, ax=ax)
    # plt.show()

    fig.savefig(outputs_dir.joinpath(f"{prefix}_boxplot_full-depth.png"), dpi=300)
    
    # - clusters
    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
    depth_method_kwargs = dict(depths=red_i, outlier_type="tail", epsilon_out=7)
    plot_contour_boxplot(masks, pred_labs, method="depth", method_kwargs=depth_method_kwargs, under_mask=img, ax=ax)
    # plt.show()

    fig.savefig(outputs_dir.joinpath(f"{prefix}_boxplot_clusters.png"), dpi=300)


    # - clusters: focus
    for k in range(best_k):
        fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
        depth_method_kwargs = dict(depths=red_i, outlier_type="tail", epsilon_out=7)
        plot_contour_boxplot(masks, pred_labs, method="depth", focus_clusters=[k,], method_kwargs=depth_method_kwargs, under_mask=img, ax=ax)
        # plt.show()

        fig.savefig(outputs_dir.joinpath(f"{prefix}_boxplot_clusters_focus-{k}.png"), dpi=300)




    # fig, ax = plt.subplots(figsize=(5,5), layout="tight")
    # ax.imshow(img, cmap="gray")
    # plot_clustering(masks, pred_labs, ax=ax)
    # ax.set_axis_off()
    # plt.show()
    # fig.savefig(outputs_dir.joinpath(f"{prefix}_clustering-spaghetti.png"), dpi=300)

    # fig, axs = plt.subplots(ncols=2)
    # plot_red(full_dephts, sort_by=None, ax=axs[0])
    # plot_red(sorted_red_w, sorted_red_b, compute_red=True, labs=sorted_labs, sort_by=None, ax=axs[1])
    # axs[1].set_ylim(0,1)
    # plt.show()
