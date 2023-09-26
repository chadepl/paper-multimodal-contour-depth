from time import time
import numpy as np
from skimage.draw import ellipse
import matplotlib.pyplot as plt
from src.contour_depth.depth.inclusion_depth import compute_depths

if __name__ == "__main__":
    
    N = 1000
    masks = []
    rows = cols = 128
    center = (rows//2, cols//2)
    radius = (20, 20)

    sum = 0

    for i in range(N):
        mask = np.zeros((rows, cols))
        rr, cc = ellipse(center[0], center[1], radius[0] + i*2, radius[1] + i*2, (rows, cols))
        mask[rr, cc] = 1
        masks.append(mask)

        sum += mask

    t_tick = time()
    depths_slow = compute_depths(masks, modified=True, fast=False)
    print(f"Slow depths ran in {time() - t_tick} seconds")

    t_tick = time()
    depths_fast = compute_depths(masks, modified=True, fast=True)
    print(f"Fast depths ran in {time() - t_tick} seconds")

    print(f"Slow depths values (first 10): {depths_slow[:10]}")
    print(f"Fast depths values (first 10): {depths_fast[:10]}")
    print(f"Mean error: {(depths_slow - depths_fast).mean()}")

    plt.imsave("mask.png", sum)

    