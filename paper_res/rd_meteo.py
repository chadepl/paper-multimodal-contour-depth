"""Generates figure demonstrating clustering analysis on meteorological data.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.contour_depth.data import ecmwf_ensembles as ecmwf
from src.contour_depth.depth import inclusion_depth, band_depth
from src.contour_depth.clustering import ddclust, inits

# Path to data
data_dir = Path("../data/cvp-paper-meteo/")

# Analysis A: ["20121015", "120", 500, 5600],  # time, pressure level, height, contour
masks = ecmwf.load_data(data_dir, config_id=1)
depths = band_depth.compute_depths(masks, modified=True, fast=True)
print(depths)

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
        labs = inits.initial_clustering(masks, num_components=n_components, method="random", seed=init_seed)
        pred_labs, sil_i, red_i, cost_i = ddclust.ddclust(masks, labs, cost_lamb=1.0, depth_notion="id", use_modified=True, use_fast=True, output_extra_info=True, seed=ddclust_seed)
        all_labs.append(pred_labs)
        sils.append(sil_i.mean())
        reds.append(red_i.mean())
        costs.append(cost_i.mean())

    plt.plot(possible_n_components, sils, label="sils")
    plt.plot(possible_n_components, reds, label="reds")
    plt.plot(possible_n_components, costs, label="costs")
    plt.legend()
    plt.show()

labs = inits.initial_clustering(masks, num_components=2, method="random", seed=init_seed)
pred_labs, sil_i, red_i, cost_i = ddclust.ddclust(masks, labs, cost_lamb=1.0, depth_notion="id", use_modified=True, use_fast=False, output_extra_info=True, seed=ddclust_seed)
fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
ecmwf.plot_contours_with_bg(masks, red_i, data_dir.joinpath("picking_background.png"), is_color_categorical=False, fname=None, ax=axs[0])
ecmwf.plot_contours_with_bg(masks, pred_labs, data_dir.joinpath("picking_background.png"), is_color_categorical=True, fname=None, ax=axs[1])
plt.show()



# Analysis B: []



