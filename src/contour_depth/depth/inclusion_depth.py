"""
This Python file implements the revised version of boundary depth
There are three versions of boundary depth:
 - Vanilla (strict)
 - Modified
 - Linear-time
"""

from time import time
import numpy as np
from skimage.measure import find_contours
from .utils import compute_inclusion_matrix, get_sdfs, get_masks_matrix

def compute_depths(data,                   
                   modified=False,
                   fast=False,
                   inclusion_mat=None,
                   ):
    """Calculate depth of a list of contours using the inclusion depth (ID) method.

    Parameters
    ----------
    data : _type_
        List of contours to calculate the ID from. A contour is assumed to be
        represented as a binary mask with ones denoting inside and zeros outside
        regions.
    modified : bool, optional
        Whether to use or not the epsilon ID (eID). This reduces the sensitivity of
        the method to outliers but yields more informative depth estimates when curves
        cross over a lot, by default False.
    fast : bool, optional
        Whether to use the fast implementation, by default False.
    inclusion_mat : _type_, optional
        Square (N x N) numpy array with the inclusion relationships between 
        contours, by default None.

    Returns
    -------
    ndarray
        Depths of the N contours in data.
    """
    
    num_contours = len(data)

    # Precomputed masks for modified versions
    if modified and fast:
        precompute_in = np.zeros_like(data[0])
        for i in range(num_contours):
            precompute_in += 1 - data[i]

        precompute_out = np.zeros_like(data[0])
        for i in range(num_contours):
            precompute_out += data[i]/data[i].sum()

    if not modified and not fast and inclusion_mat is None:
        print("Warning: pre-computed inclusion matrix not available, computing it ... ")
        inclusion_mat = compute_inclusion_matrix(data)

    if not modified and fast:
        print("Warning: the fast version is based on a paper that potentially has an error")
        depths = inclusion_depth_strict_fast(data)
    else:
        depths = []
        for i in range(num_contours):
            if modified:
                if fast:
                    depth = inclusion_depth_modified_fast(data[i], data, precompute_in=precompute_in, precompute_out=precompute_out)
                else:
                    depth = inclusion_depth_modified(data[i], data)
            else:
                depth = inclusion_depth_strict(i, inclusion_mat)

            depths.append(depth)

    return np.array(depths, dtype=float)


def inclusion_depth_strict(mask_index, inclusion_mat):
    num_masks = inclusion_mat[mask_index].size
    in_count = (inclusion_mat[mask_index] > 0).sum()
    out_count = (inclusion_mat[mask_index] < 0).sum()

    return np.minimum(in_count/num_masks, out_count/num_masks)


def inclusion_depth_strict_fast(data):
    num_contours = len(data)
    data_sdfs = get_sdfs(data)
    
    R = get_masks_matrix(data_sdfs)
    R_p = np.argsort(R, axis=0)
    R_pp = np.argsort(R_p, axis=0) + 1

    # for p in range(R.shape[1]):
    #     for i in range(1, R.shape[0]):
    #         if R[R_p[i, p], p] <= R[R_p[i-1, p], p]: 
    #             R_pp[R_p[i, p], p] = R_pp[R_p[i-1, p], p]

    n_b = np.min(R_pp, axis=1) - 1
    n_a = num_contours - np.max(R_pp, axis=1)

    depths_fast_sdf = np.array([n_b, n_a])/num_contours
    depths_fast_sdf = np.min(depths_fast_sdf, axis=0)
    
    return depths_fast_sdf.tolist()


def inclusion_depth_modified(in_ci, masks):
    num_masks = len(masks)
    in_vals = []
    out_vals = []

    for j in range(num_masks):
        in_cj = masks[j]

        # the smaller eps_out becomes the less outside it is
        # so when it is zero, we now it is inside
        # we add a larger number to in matrix the more inside i is
        eps_out = (in_ci - in_cj)
        eps_out = (eps_out > 0).sum()
        eps_out = eps_out / (in_ci.sum() + np.finfo(float).eps)
        depth_in = 1 - eps_out
        in_vals.append(depth_in)

        # the smaller eps_in becomes, the less j is outside of i
        # so when it is zero, we know i is outside of j
        # we add a larger number to out matrix the more outside i is
        eps_in = (in_cj - in_ci)
        eps_in = (eps_in > 0).sum()
        eps_in = eps_in / (in_cj.sum() + np.finfo(float).eps)
        depth_out = 1 - eps_in
        out_vals.append(depth_out)

    return np.minimum(np.mean(in_vals), np.mean(out_vals))


def inclusion_depth_modified_fast(in_ci, masks, precompute_in=None, precompute_out=None):
    num_masks = len(masks)
    if precompute_in is None:
        precompute_in = np.zeros_like(in_ci)
        for i in range(num_masks):
            precompute_in += 1 - masks[i]
    if precompute_out is None:
        precompute_out = np.zeros_like(in_ci)
        for i in range(num_masks):
            precompute_out += masks[i]/masks[i].sum()

    IN_in = num_masks - ((in_ci / in_ci.sum()) * precompute_in).sum()
    IN_out = num_masks - ((1-in_ci) * precompute_out).sum()

    return np.minimum(IN_in/num_masks, IN_out/num_masks)



