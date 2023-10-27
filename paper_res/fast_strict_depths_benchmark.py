"""In this file we test the fast depth computation methods for
strict depth formulations.
Learnings have been implemented into the contour_depth library.
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.contour_depth.depth.inclusion_depth import compute_depths as inclusion_depth
from src.contour_depth.depth.band_depth import compute_depths as contour_band_depth
from src.contour_depth.data.synthetic_data import circle_ensemble

if __name__ == "__main__":

    depth_fun = [inclusion_depth, contour_band_depth][1]
    
    N = 30
    masks = []
    ROWS = COLS = 512
    CENTER_MEAN = (0.5, 0.5)
    CENTER_STD = (0, 0)
    RADIUS_MEAN = 0.25
    RADIUS_STD = 0.25 * 0.1

    # build ensemble
    masks = circle_ensemble(N, ROWS, COLS, CENTER_MEAN, CENTER_STD, RADIUS_MEAN, RADIUS_STD)

    # depth computation: (slow, fast binary, fast sdf)

    t_tick = time()
    depths_slow = depth_fun(masks, modified=False, fast=False)
    print(f"Slow depths took: {time() - t_tick} seconds")

    t_tick = time()
    depths_fast = depth_fun(masks, modified=False, fast=True)
    print(f"Fast depths took: {time() - t_tick} seconds")

    # depth evaluation
    print()

     # - raw values
    print(f"Slow: {depths_slow[:10]}")
    print(f"Fast: {depths_fast[:10]}")

    # - errors
    print()

    # -- slow vs binary
    print(f"MSE (slow vs fast): {np.mean(np.square(depths_slow - depths_fast))}")

    cm = plt.colormaps.get_cmap("magma")
    fig, axs = plt.subplots(ncols=2)
    for d, m in zip(depths_slow, masks):
        axs[0].contour(m, colors=[cm(d),], linewidths=[0.5, ], alpha=0.2)
    for d, m in zip(depths_fast, masks):
        axs[1].contour(m, colors=[cm(d),], linewidths=[0.5, ], alpha=0.2)
    plt.show()