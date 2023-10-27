from math import comb
from time import time
from itertools import combinations
from functools import reduce
import numpy as np
from scipy.optimize import bisect

from .utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix

def compute_depths(data,
                   modified=False,
                   fast=False,
                   inclusion_mat=None,
                   target_mean_depth: float = None, #1 / 6
                   verbose=False
                   ):
    """
    Calculate depth of a list of contours using the contour band depth (CBD) method for J=2.
    :param data: list
        List of contours to calculate the CBD from. A contour is assumed to be
        represented as a binary mask with ones denoting inside and zeros outside
        regions.
    :param band_size: int
        Number of contours to consider when forming the bands.
    :param modified: bool
        Whether to use or not the modified CBD (mCBD). This reduces the sensitivity of
        the method to outliers but yields more informative depth estimates when curves
        cross over a lot.
    :param target_mean_depth: float
        Only applicable if using mCBD. If None, returns the depths as is.
        If a float is specified, finds a threshold of the depth matrix that yields a
        target mean depth of `target_mean_depth` using binary search.
    :param return_data_mat: ndarray
        If true, in addition to the depths, returns the raw depth matrix which is of
        dimensions n x (n combined band_size), where n = len(data).
    :param times_dict: dict
        If a dict is passed as `times_dict`, the times that different stages of the method
        take are logged to this dict.
    :return:
        depths: ndarray
        depth_matrix: ndarray
    """

    num_contours = len(data)
    num_subsets = comb(num_contours, 2)

    # Precomputed masks for modified versions
    if modified and fast:
        precompute_in = np.zeros_like(data[0])
        for i in range(num_contours):
            precompute_in += 1 - data[i]

        precompute_out = np.zeros_like(data[0])
        for i in range(num_contours):
            precompute_out += data[i]/data[i].sum()

    if inclusion_mat is None:
        if verbose:
            print("Warning: pre-computed inclusion matrix not available, computing it ... ")
        if modified:            
            inclusion_mat = compute_epsilon_inclusion_matrix(data)
        else:
            inclusion_mat = compute_inclusion_matrix(data)

    if fast:
        if target_mean_depth is not None:
            raise ValueError("if using modified=True then targer_mean_depth should be None")            

    depths = []
    for i in range(num_contours):
        if modified:
            if fast:
                depth = band_depth_modified_fast(data[i], data, precompute_in=precompute_in, precompute_out=precompute_out) # returns a value
            else:
                depth = band_depth_modified(i, data)  # returns a tuple of arrays (insersect_subset_ci and ci_subset_union), not a point
        else:
            if fast:
                depth = band_depth_strict_fast(i, inclusion_mat)  # returns a value
            else:
                depth = band_depth_strict(i, data)  # returns an array of containment relationships          
                depth = depth.mean()
        depths.append(depth)

    if modified and not fast:
        depth_matrix_left = np.array([a[0] for a in depths])
        depth_matrix_right = np.array([a[1] for a in depths])

        if target_mean_depth is None:  # No threshold  
            print("[cbd] Using modified band depths without threshold (mult agg)")          
            depth_matrix_left = 1 - depth_matrix_left
            depth_matrix_right = 1 - depth_matrix_right
            depth_matrix = np.minimum(depth_matrix_left, depth_matrix_right)  # depth_matrix_left * depth_matrix_right
        else: # automatically determined threshold as in the paper       
            print(f"[cbd] Using modified band depth with specified threshold {target_mean_depth} (max agg)")
            def mean_depth_deviation(mat, threshold, target):
                return target - (((mat < threshold).astype(float)).sum(axis=1) / num_subsets).mean()
        
            depth_matrix = np.maximum(depth_matrix_left, depth_matrix_right)
            try:
                t = bisect(lambda v: mean_depth_deviation(depth_matrix, v, target_mean_depth), depth_matrix.min(),
                        depth_matrix.max())
            except RuntimeError:
                print("[cbd] Binary search failed to converge")
                t = depth_matrix.mean()
            print(f"[cbd] Using t={t}")

            depth_matrix = (depth_matrix < t).astype(float)

        depths = depth_matrix.mean(axis=1)

    return np.array(depths, dtype=float)


def band_depth_strict(contour_index, data):
    num_contours = len(data)
    in_ci = data[contour_index]
    in_band = []
    for i in range(num_contours):
        band_a = data[i]
        for j in range(i, num_contours):
            band_b = data[j]
            if i != j:
                subset_sum = band_a + band_b

                union = (subset_sum > 0).astype(float)
                intersection = (subset_sum == 2).astype(float)

                intersect_in_contour = np.all(((intersection + in_ci) == 2).astype(float) == intersection)
                contour_in_union = np.all(((union + in_ci) == 2).astype(float) == in_ci)
                if intersect_in_contour and contour_in_union:
                    in_band.append(1)
                else:
                    in_band.append(0)

    return np.array(in_band)


def band_depth_strict_fast(contour_index, inclusion_mat):
    num_contours = inclusion_mat[contour_index].size
    num_subsets = comb(num_contours, 2)
    in_count = (inclusion_mat[contour_index, :] > 0).sum()
    out_count = (inclusion_mat[:, contour_index] > 0).sum()

    return (in_count*out_count + num_contours - 1)/num_subsets


def band_depth_modified(contour_index, data):

    num_contours = len(data)
    in_ci = data[contour_index]

    # Compute fractional containment tables
    intersect_subset_ci = []
    ci_subset_union = []

    for i in range(num_contours):
        band_a = data[i]
        for j in range(i, num_contours):
            band_b = data[j]

            if i != j:
                subset_sum = band_a + band_b

                union = (subset_sum > 0).astype(float)
                intersection = (subset_sum == 2).astype(float)
        
                lc_frac = (intersection - in_ci)
                lc_frac = (lc_frac > 0).sum()
                lc_frac = lc_frac / (intersection.sum() + np.finfo(float).eps)

                rc_frac = (in_ci - union)
                rc_frac = (rc_frac > 0).sum()
                rc_frac = rc_frac / (in_ci.sum() + np.finfo(float).eps)

                intersect_subset_ci.append(lc_frac)
                ci_subset_union.append(rc_frac)

    return intersect_subset_ci, ci_subset_union


def band_depth_modified_fast(in_ci, masks, precompute_in=None, precompute_out=None):
    num_contours = len(masks)
    num_subsets = comb(num_contours, 2)

    # if precompute_in is None:
    precompute_in = np.zeros_like(in_ci)
    for i in range(num_contours):
        precompute_in += 1 - masks[i]
    # if precompute_out is None:
    precompute_out = np.zeros_like(in_ci)
    for i in range(num_contours):
        precompute_out += masks[i]/masks[i].sum()

    # precompute_in -= (1 - in_ci)
    # precompute_out -= in_ci/in_ci.sum()

    IN_in = num_contours - ((in_ci / in_ci.sum()) * precompute_in).sum()
    IN_out = num_contours - ((1-in_ci) * precompute_out).sum()

    # print(IN_in, IN_out)

    # return (IN_in * IN_out + num_contours - 1)/(2*num_subsets)
    return (IN_in * IN_out)/(2*num_subsets)



# def band_depth_modified(data):

#     num_contours = len(data)
#     num_subsets = comb(num_contours, 2)

#     # Compute fractional containment tables
#     depth_matrix_left = np.zeros((num_contours, num_subsets))
#     depth_matrix_right = np.zeros((num_contours, num_subsets))

#     for contour_index, in_ci in enumerate(data):
#         subset_id = 0
#         for i in range(num_contours):
#             band_a = data[i]
#             for j in range(i, num_contours):
#                 band_b = data[j]

#                 if i != j:
#                     subset_sum = band_a + band_b

#                     union = (subset_sum > 0).astype(float)
#                     intersection = (subset_sum == 2).astype(float)
            
#                     lc_frac = (intersection - in_ci)
#                     lc_frac = (lc_frac > 0).sum()
#                     lc_frac = lc_frac / (intersection.sum() + np.finfo(float).eps)

#                     rc_frac = (in_ci - union)
#                     rc_frac = (rc_frac > 0).sum()
#                     rc_frac = rc_frac / (in_ci.sum() + np.finfo(float).eps)

#                     depth_matrix_left[contour_index, subset_id] = lc_frac
#                     depth_matrix_right[contour_index, subset_id] = rc_frac

#                     subset_id += 1
        
#     depth_matrix_left = 1 - depth_matrix_left
#     depth_matrix_right = 1 - depth_matrix_right
#     depth_matrix = depth_matrix_left * depth_matrix_right

#     depths = depth_matrix.mean(axis=1)

#     return depths