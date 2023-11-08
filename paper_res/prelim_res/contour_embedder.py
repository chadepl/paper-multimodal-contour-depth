"""In this file we try to understand how contours translate to points in a metric space. 
This is important because clustering procedures rely on the relationships between
points in such a space.
We try PCA and tSNE as our embedding methods."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
sys.path.insert(0, "../..")

from src.contour_depth.data.synthetic_data import magnitude_modes, three_rings, shape_families
from src.contour_depth.depth.utils import get_masks_matrix, get_sdfs, compute_inclusion_matrix

num_masks = 30
num_rows = num_cols = 512
seed = 1

dataset = ["magnitude_modes", "three_rings", "shape_families"][2]
if dataset == "magnitude_modes":
    masks, labs = magnitude_modes(num_masks, num_rows, num_cols, return_labels=True, seed=seed, modes_proportions=(0.8, 0.2), modes_radius_std=(0.2 * 0.1, 0.16 * 0.05))
elif dataset == "three_rings":
   masks, labs = three_rings(num_masks, num_rows, num_cols, return_labels=True, seed=seed)
elif dataset == "shape_families":
   masks, labs = shape_families(num_masks, num_rows, num_cols, return_labels=True, seed=seed)


fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(8, 8), layout="tight")

labs_colors_map = ["red", "blue", "orange"]
colors = [labs_colors_map[l] for l in labs]

axs[1, 0].set_title(f"Dataset (seed: {seed})")
for i, (m, l) in enumerate(zip(masks, labs)):
    c = colors[i]
    axs[1, 0].contour(m, colors=[c, ], linewidths=[1, ], alpha=0.1)
for i in range(3):
    axs[i, 0].set_axis_off()

# First row
print("Computing embeddings mask matrix ...")
mat = get_masks_matrix(masks)
emb_pca = PCA(n_components=2).fit_transform(mat)
emb_tsne = TSNE(n_components=2, perplexity=10).fit_transform(mat)
axs[0, 1].set_title("PCA (mat: masks)")
axs[0, 1].scatter(emb_pca[:, 0], emb_pca[:, 1], c="None", edgecolors=colors)
axs[0, 2].set_title("TSNE (mat: masks)")
axs[0, 2].scatter(emb_tsne[:, 0], emb_tsne[:, 1], c="None", edgecolors=colors)

# Second row
print("Computing embeddings sdf matrix ...")
sdfs = get_sdfs(masks)
mat = get_masks_matrix(sdfs)
emb_pca = PCA(n_components=2).fit_transform(mat)
emb_tsne = TSNE(n_components=2, perplexity=10).fit_transform(mat)
axs[1, 1].set_title("PCA (mat: sdfs)")
axs[1, 1].scatter(emb_pca[:, 0], emb_pca[:, 1], c="None", edgecolors=colors)
axs[1, 2].set_title("TSNE (mat: sdfs)")
axs[1, 2].scatter(emb_tsne[:, 0], emb_tsne[:, 1], c="None", edgecolors=colors)

# Third row
print("Computing embeddings inclusion matrix ...")
mat = compute_inclusion_matrix(masks)
emb_pca = PCA(n_components=2).fit_transform(mat)
emb_tsne = TSNE(n_components=2, perplexity=10).fit_transform(mat)
axs[2, 1].set_title("PCA (mat: inclusion)")
axs[2, 1].scatter(emb_pca[:, 0], emb_pca[:, 1], c="None", edgecolors=colors)
axs[2, 2].set_title("TSNE (mat: inclusion)")
axs[2, 2].scatter(emb_tsne[:, 0], emb_tsne[:, 1], c="None", edgecolors=colors)

fig.savefig(f"contour_embedder/{dataset}-seed_{seed}.png")
plt.show()
