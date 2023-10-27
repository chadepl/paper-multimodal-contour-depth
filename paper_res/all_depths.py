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
    
    N = 100
    masks = []
    ROWS = COLS = 512
    center = (ROWS//2, COLS//2)
    CENTER_NOISE = (0, 0)
    NOISE_STD = (0, 0)

    # build ensemble
    masks = circle_ensemble(N, ROWS, COLS, CENTER_NOISE, NOISE_STD)

    bd_str = contour_band_depth(masks, modified=False, fast=False)
    bd_mod = contour_band_depth(masks, modified=True, fast=False)
    id_str = inclusion_depth(masks, modified=False, fast=False)
    id_mod = inclusion_depth(masks, modified=True, fast=False)

    print(bd_str)
    print(bd_mod)
    print(id_str)
    print(id_mod)

    plt.scatter(np.arange(N), bd_str, alpha=0.2, label="bd_str")
    plt.scatter(np.arange(N), bd_mod, alpha=0.2, label="bd_mod")
    plt.scatter(np.arange(N), id_str, alpha=0.2, label="id_str")
    plt.scatter(np.arange(N), id_mod, alpha=0.2, label="id_mod")
    plt.legend()
    plt.show()