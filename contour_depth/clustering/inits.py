"""Cluster contours based on their SDF representation.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from contour_depth.utils import get_sdfs, get_masks_matrix


def initial_clustering(masks, num_components, feat_mat=None, pre_pca=False, method="kmeans", seed=None):
    num_masks = len(masks)

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    if feat_mat is None:
        print("[initial_clustering] Warning: feat_mat is None, using sdfs of the masks as feature matrix ...")
        sdfs = get_sdfs(masks)
        feat_mat = get_masks_matrix(sdfs)
    mat = feat_mat

    if pre_pca:
        pca_embedder = PCA(n_components=30)
        mat = pca_embedder.fit_transform(mat)
    
    if method == "random":
        labs = rng.integers(0, num_components, num_masks)
    elif method == "kmeans":
        labs = KMeans(n_clusters=num_components).fit_predict(mat)
    elif method == "ahc":
        ahc = AgglomerativeClustering(n_clusters=num_components, metric="euclidean", linkage="average").fit(mat)
        labs = ahc.labels_
    else:
        raise ValueError("Only kmeans and ahc supported for now.")

    return labs