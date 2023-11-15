from time import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from contour_depth.data.synthetic_data import main_shape_with_outliers
from contour_depth.utils import get_masks_matrix, get_sdfs

from contour_depth.visualization import spaghetti_plot

from contour_depth.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix, compute_inclusion
from contour_depth.depth.inclusion_depth import compute_depths


if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/progressive_demo")
    assert outputs_dir.exists()

    seed_data = 0
    seed_clustering = 0
    seed_init = 0

    N = 100  # fixed
    ROWS = COLS = 512  # fixed
    P_CONTAMINATION = 0.1  # fixed
    BATCH_SIZE = 1
    CHECKPOINTS = [9, 49, 99]

    masks, labs = main_shape_with_outliers(N, ROWS, COLS,
                                           p_contamination=P_CONTAMINATION,
                                           return_labels=True, seed=seed_data)
    
    rng = np.random.default_rng(seed_data)

    masks_ids = np.arange(N)
    rng.shuffle(masks_ids)
    masks = [masks[i] for i in masks_ids]
    labs = [labs[i] for i in masks_ids]

    ###################
    # Data generation #
    ###################
    
    run_prefix = f"batch-{BATCH_SIZE}"
    path_df = outputs_dir.joinpath(f"{run_prefix}_timings.csv")

    if path_df.exists():
        df = pd.read_csv(path_df, index_col=0)
    else:

        rows = [["seed_data", "step_l", "step_r", "method_id", "time_secs"]]

        saved_checkpoints = 0

        in_counts = [] 
        out_counts = []
        inc_mat = np.zeros((0,0)) # 0 x 0
        current_masks = []
        
        for step_l in np.arange(N, step=BATCH_SIZE):  # possible to do more than one at a time
            step_r = step_l + BATCH_SIZE
            subset_masks = masks[:step_r]  # current subset we are dealing with 

            in_counts += [0 for _ in range(BATCH_SIZE)]  # add new element to arrays
            out_counts += [0 for _ in range(BATCH_SIZE)]  # add new element to arrays

            t_start = time()
            sample_ids = np.arange(step_l, step_r)
            masks_to_add = [masks[s] for s in sample_ids]

            # Create new inclusion matrix
            new_inc_mat = np.zeros((inc_mat.shape[0] + BATCH_SIZE, inc_mat.shape[1] + BATCH_SIZE))
            new_inc_mat[0:inc_mat.shape[0], 0:inc_mat.shape[1]] = inc_mat

            # Compute containment of incoming elements in MN'^2 time (if batch is 1 then is O(1))
            new_inc_mat[inc_mat.shape[0]:inc_mat.shape[0]+BATCH_SIZE, inc_mat.shape[1]:inc_mat.shape[1]+BATCH_SIZE] = compute_inclusion_matrix(masks_to_add)

            # Compute rows/cols in 2N'N time
            for i, current_mask in enumerate(current_masks):
                for j, mask_to_add in enumerate(masks_to_add):
                    intersect = ((current_mask + mask_to_add) == 2).astype(float)
                    a_in_b = np.all(current_mask == intersect)
                    b_in_a = np.all(mask_to_add == intersect)
                    if a_in_b:
                        new_inc_mat[i, j + len(current_masks)] = 1
                        in_counts[i] += 1
                        out_counts[j + len(current_masks)] += 1
                    if b_in_a:
                        new_inc_mat[j + len(current_masks), i] = 1
                        in_counts[j + len(current_masks)] += 1
                        out_counts[i] += 1
                        

            # Updates for next iteration
            current_masks += masks_to_add
            inc_mat = new_inc_mat.copy()

            t_common = time() - t_start

            # Depth computation
            # - progressive/faster
            # - print(np.array([in_counts, out_counts]).T/len(contours))
            t_start = time()
            d_progressive_f = np.min(np.array([in_counts, out_counts]).T/len(current_masks), axis=1)# compute_depths(subset_masks, inclusion_mat=im, modified=False, fast=False)        
            t_progressive_f = time() - t_start + t_common

            # - progressive/slower
            t_start = time()
            d_progressive = compute_depths(current_masks, inclusion_mat=inc_mat, modified=False, fast=False)
            t_progressive = time() - t_start + t_common
            
            # - Depth calculation
            t_start = time()        
            batched_im = compute_inclusion_matrix(current_masks)
            d_batched = compute_depths(current_masks, inclusion_mat=batched_im, modified=False, fast=False)
            t_batched = time() - t_start + t_common

            rows.append([seed_data, step_l, step_r, "batched", t_batched])
            rows.append([seed_data, step_l, step_r, "progressive", t_progressive])

            print(step_l, step_r, t_batched, t_progressive, t_progressive_f)

            # print(d_progressive)
            # print(d_progressive_f)
            
            assert np.all(d_progressive == d_batched)
            assert np.all(d_progressive == d_progressive_f)

            if step_l >= CHECKPOINTS[saved_checkpoints] and saved_checkpoints < len(CHECKPOINTS):
                print(f"Saved checkpoint at step_l {step_l}")
                # save depths
                with open(outputs_dir.joinpath(f"{run_prefix}_depths-batched_chkpt-{step_l}.pkl"), "wb") as f:
                    pickle.dump(d_batched, f)
                with open(outputs_dir.joinpath(f"{run_prefix}_depths-progressive_chkpt-{step_l}.pkl"), "wb") as f:
                    pickle.dump(d_progressive, f)
                # save ensemble
                with open(outputs_dir.joinpath(f"{run_prefix}_masks_chkpt-{step_l}.pkl"), "wb") as f:
                    pickle.dump(current_masks, f)
                saved_checkpoints += 1

        df = pd.DataFrame(rows[1:])
        df.columns = rows[0]
        df.to_csv(path_df)  # write to csv

    print(df.head())
    print()

    ############
    # Analysis #
    ############

    import seaborn as sns
    from contour_depth.visualization import spaghetti_plot

    CHECKPOINTS = [9, 49, 99]

    timings_df = df
    timings_df = timings_df.pivot(index=["seed_data", "step_l", "step_r"], columns="method_id", values="time_secs")
    timings_df["batched_cumsum"] = timings_df["batched"].cumsum()
    timings_df["progressive_cumsum"] = timings_df["progressive"].cumsum()
    timings_df = timings_df.drop(["batched", "progressive"], axis=1)
    timings_df = timings_df.reset_index()
    timings_df = timings_df.melt(["seed_data", "step_l", "step_r"], ["batched_cumsum", "progressive_cumsum"], value_name="time")
    timings_df["method_id"] = timings_df["method_id"].apply(lambda d: dict(batched_cumsum="Batched", progressive_cumsum="Progressive")[d]) 
    timings_df = timings_df.rename(lambda d: dict(method_id="Method")[d] if d in ["method_id"] else d, axis=1)

    print(timings_df.head())

    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")

    sns_plt = sns.lineplot(timings_df, x="time", y="step_l", hue="Method", linewidth=2, ax=ax)
    
    sns_plt.set(xscale="log", yscale="log")
    ax.set_title("Runtimes vs Ensemble Size for \n Contour Band Depth and Inclusion Depth ")
    ax.set_ylabel("Log(Number of contours displayed)")
    ax.set_xlabel("Log(Elapsed time (seconds))")

    plt.show()
    
    fig.savefig(outputs_dir.joinpath(f"{run_prefix}_speedcomp.png"), dpi=300)


    for chkpt in CHECKPOINTS:
        with open(outputs_dir.joinpath(f"{run_prefix}_depths-batched_chkpt-{chkpt}.pkl"), "rb") as f:
            d_batched = pickle.load(f)
        with open(outputs_dir.joinpath(f"{run_prefix}_depths-progressive_chkpt-{chkpt}.pkl"), "rb") as f:
            d_progressive = pickle.load(f)
        # save ensemble
        with open(outputs_dir.joinpath(f"{run_prefix}_masks_chkpt-{chkpt}.pkl"), "rb") as f:
            masks = pickle.load(f)
        assert np.all(d_batched == d_progressive)

        fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
        spaghetti_plot(masks, iso_value=0.5, arr=d_progressive, is_arr_categorical=False, ax=ax)
        ax.set_axis_off()
        #plt.show()
        fig.savefig(outputs_dir.joinpath(f"{run_prefix}_chkpt-{chkpt}_spaghetti.png"), dpi=300)
    
    
    
    
    
    # strict_inclusion_mat = compute_inclusion_matrix(masks) 
    # epsilon_inclusion_mat = compute_epsilon_inclusion_matrix(masks)

    ##########
    # Figure #
    ##########



    # to_add_ids = np.random.choice(np.arange(num_contours), 3, replace=False)
    # to_remove_ids = np.random.choice(np.setdiff1d(np.arange(num_contours), to_add_ids), 3, replace=False)
    # remaining_masks = [masks[i] for i in np.arange(num_contours) if i not in np.union1d(to_add_ids, to_remove_ids)]
    # after_adding = masks + [masks[i] for i in np.arange(num_contours) if i in to_add_ids]
    # after_removal = masks + [masks[i] for i in np.arange(num_contours) if i in to_remove_ids]

    # fig, axs = plt.subplots(nrows=2, ncols=3)
    # axs[0,0].set_title("Starting ensemble (N=100)")
    # spaghetti_plot(masks, iso_value=0.5, arr=labs, is_arr_categorical=True, ax=axs[0,0])

    # full_depths = compute_depths(masks, modified=True, fast=False)

    # axs[1,0].set_title("Starting ensemble depths")
    # spaghetti_plot(masks, iso_value=0.5, arr=full_depths, is_arr_categorical=False, ax=axs[1,0])

    # # Adding

    # t_start = time()
    # eid_slow_add = compute_depths(after_adding, modified=True, fast=False)
    # t_add_slow = time() - t_start
    # print(f"Adding time for eID (slow): {t_add_slow} seconds")

    # t_start = time()
    # eid_fast_add = compute_depths(after_adding, modified=True, fast=True)
    # t_add_fast = time() - t_start
    # print(f"Adding time for eID (fast): {t_add_fast} seconds")

    # axs[0,1].set_title(f"Adding (error: {np.sqrt(np.square(eid_fast_add-eid_slow_add).sum()):.4f}) \n eID slow ({t_add_slow:.2f} secs)")
    # spaghetti_plot(after_adding, iso_value=0.5, highlight=to_add_ids, arr=eid_slow_add, is_arr_categorical=False, ax=axs[0,1])

    # axs[1,1].set_title(f"eID fast ({t_add_fast:.2f} secs)")
    # spaghetti_plot(after_adding, iso_value=0.5, highlight=to_add_ids, arr=eid_fast_add, is_arr_categorical=False, ax=axs[1,1])

    # # Removing

    # t_start = time()
    # eid_slow_remove = compute_depths(after_removal, modified=True, fast=False)
    # t_remove_slow = time() - t_start
    # print(f"Removing time for eID (slow): {t_remove_slow} seconds")

    # t_start = time()
    # eid_fast_remove = compute_depths(after_removal, modified=True, fast=True)
    # t_remove_fast = time() - t_start
    # print(f"Removing time for eID (fast): {t_remove_fast} seconds")

    # axs[0,2].set_title(f"Removing (error: {np.sqrt(np.square(eid_fast_remove-eid_slow_remove).sum()):.4f}) \n eID slow ({t_remove_slow:.2f} secs)")
    # spaghetti_plot(after_removal, iso_value=0.5, highlight=to_remove_ids, arr=eid_slow_remove, is_arr_categorical=False, ax=axs[0,2])

    # axs[1,2].set_title(f"eID fast ({t_remove_fast:.2f} secs)")
    # spaghetti_plot(after_removal, iso_value=0.5, highlight=to_remove_ids, arr=eid_fast_remove, is_arr_categorical=False, ax=axs[1,2])


    # plt.show()