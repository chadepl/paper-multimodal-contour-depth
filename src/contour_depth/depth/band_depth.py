from math import comb
from time import time
from itertools import combinations
from functools import reduce
import numpy as np
from scipy.optimize import bisect

from .utils import compute_inclusion_matrix

def compute_depths(data,
                   modified=False,
                   fast=False,
                   inclusion_mat=None,
                   target_mean_depth: float = None, #1 / 6
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

    # Precomputed masks for modified versions
    if modified and fast:
        precompute_in = np.zeros_like(data[0])
        for i in range(num_contours):
            precompute_in += 1 - data[i]

        precompute_out = np.zeros_like(data[0])
        for i in range(num_contours):
            precompute_out += data[i]/data[i].sum()

    if not modified and fast and inclusion_mat is None:
        print("Warning: pre-computed inclusion matrix not available, computing it ... ")
        inclusion_mat = compute_inclusion_matrix(data)

    if fast:
        if target_mean_depth is not None:
            raise ValueError("if using modified=True then targer_mean_depth should be None")            

    if modified and not fast:
        depths = band_depth_modified(data, target_mean_depth=target_mean_depth)
    else:
        depths = []
        for i in range(num_contours):
            if modified:
                if fast:
                    depth = band_depth_modified_fast(data[i], data, precompute_in=precompute_in, precompute_out=precompute_out)
            else:
                if fast:
                    depth = band_depth_strict_fast(i, inclusion_mat)
                else:
                    depth = band_depth_strict(i, data)                
            depths.append(depth)

    return np.array(depths, dtype=float)


def band_depth_strict(contour_index, data):
    num_contours = len(data)
    in_ci = data[contour_index]
    in_band = 0    
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
                    in_band += 1

    return in_band/comb(num_contours, 2)


def band_depth_strict_fast(contour_index, inclusion_mat):
    num_contours = inclusion_mat[contour_index].size
    num_subsets = comb(num_contours, 2)
    in_count = (inclusion_mat[contour_index] > 0).sum()
    out_count = (inclusion_mat[contour_index] < 0).sum()

    return (in_count*out_count + num_contours - 1)/num_subsets


def band_depth_modified(data,
                        target_mean_depth: float = None, #1 / 6                        
                        ):

    num_contours = len(data)
    num_subsets = comb(num_contours, 2)

    # Compute fractional containment tables
    depth_matrix_left = np.zeros((num_contours, num_subsets))
    depth_matrix_right = np.zeros((num_contours, num_subsets))

    if target_mean_depth is None:
        print("[cbd] Using modified band depths without threshold (mult agg)")
    else:
        print(f"[cbd] Using modified band depth with specified threshold {target_mean_depth} (max agg)")

    for contour_index, in_ci in enumerate(data):
        subset_id = 0
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

                    depth_matrix_left[contour_index, subset_id] = lc_frac
                    depth_matrix_right[contour_index, subset_id] = rc_frac

                    subset_id += 1

        print(depth_matrix_left[contour_index].sum(), depth_matrix_right[contour_index].sum())
        
    if target_mean_depth is None:  # No threshold            
        depth_matrix_left = 1 - depth_matrix_left
        depth_matrix_right = 1 - depth_matrix_right
        depth_matrix = depth_matrix_left * depth_matrix_right #np.minimum(depth_matrix_left, depth_matrix_left)
    else:            
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

    return depths


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

    print(IN_in, IN_out)

    # return (IN_in * IN_out + num_contours - 1)/(2*num_subsets)
    return (IN_in * IN_out)/(2*num_subsets)



