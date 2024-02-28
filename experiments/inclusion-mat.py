"""
Computes inclusion matrices for simple ensemble of contours (mock-ensemble)
We then use the values in the plots to construct the depiction in the paper.
"""
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix

masks = [imread(f"mock-ensemble/mock-ensemble-m{i+1}.png") for i in range(6)]
masks = [rgb2gray(m[:, :, 0:3]) for m in masks]
masks = [m.astype(int) for m in masks]

strict_inclusion_mat = compute_inclusion_matrix(masks)
np.fill_diagonal(strict_inclusion_mat, 1)
epsilon_inclusion_mat = compute_epsilon_inclusion_matrix(masks)
np.fill_diagonal(epsilon_inclusion_mat, 1)

fig, axs = plt.subplots(ncols=3, figsize=(10, 4), layout="tight")

axs[0].set_title("Ensemble")
for m in masks:
    axs[0].contour(m, colors=["orange"])
axs[0].set_axis_off()

axs[1].set_title("Strict inclusion mat")
axs[1].matshow(strict_inclusion_mat)

axs[2].set_title("Epsilon inclusion mat")
axs[2].matshow(epsilon_inclusion_mat)

# axs[3].set_title("mat_epsilon x inv(mat_strict)")
# axs[3].matshow(unknown_matrix)

plt.show()


##################################
# THRESHOLDING OF EPSILON MATRIX #
##################################

fig, axs = plt.subplots(ncols=3)
axs[0].set_title("Strict inclusion mat")
axs[0].matshow(strict_inclusion_mat)
axs[1].set_title("Epsilon inclusion mat")
axs[1].matshow(epsilon_inclusion_mat)
axs[2].set_title("Epsilon inclusion mat (thresholded)")
axs[2].matshow(epsilon_inclusion_mat > 0.98)
plt.show()


##################################
# QUANTIZATION OF EPSILON MATRIX #
##################################

num_bins = 7
bin_size = 1 / num_bins
quantized_mat = epsilon_inclusion_mat.copy()
for i in range(num_bins):
    quantized_mat[np.logical_and(quantized_mat > i * bin_size, quantized_mat <= (i+1) * bin_size)] = i
# quantized_mat[np.logical_and(quantized_mat > 1 * bin_size, quantized_mat <= 2 * bin_size)] = 1
# quantized_mat[np.logical_and(quantized_mat > 2 * bin_size, quantized_mat <= 3 * bin_size)] = 2
# quantized_mat[np.logical_and(quantized_mat > 3 * bin_size, quantized_mat <= 4 * bin_size)] = 3
# quantized_mat[np.logical_and(quantized_mat > 4 * bin_size, quantized_mat <= 5 * bin_size)] = 4

fig, axs = plt.subplots(ncols=2)
axs[0].set_title("Epsilon inclusion mat")
axs[0].matshow(epsilon_inclusion_mat)
axs[1].set_title("Epsilon inclusion mat (quantized)")
axs[1].matshow(quantized_mat)
plt.show()