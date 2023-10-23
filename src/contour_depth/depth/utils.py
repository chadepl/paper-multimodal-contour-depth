import numpy as np

def get_masks_matrix(masks):
    import numpy as np
    mat = np.concatenate([mask.flatten()[np.newaxis] for mask in masks], axis=0)
    return mat

def get_sdfs(masks):
    from scipy.ndimage import distance_transform_edt
    sdfs = [distance_transform_edt(mask) + -1 * distance_transform_edt(1 - mask) for mask in masks]
    return sdfs

def compute_inclusion_matrix(masks):
    """Symmetric matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous. 

    Parameters
    ----------
    masks : _type_
        _description_
    """
    num_masks = len(masks)
    inclusion_mat = np.zeros((num_masks, num_masks))

    for i in range(num_masks):
        inclusion = compute_inclusion(i, masks)
        inclusion_mat[i, :] = inclusion

    return inclusion_mat


def compute_inclusion(contour_index, masks):
    num_masks = len(masks)
    inclusion = np.zeros(num_masks)
    for j in range(num_masks):
        if contour_index != j:
            intersect = ((masks[contour_index] + masks[j]) == 2).astype(float)
            is_in = np.all(masks[contour_index] == intersect)
            is_out = np.all(masks[j] == intersect)
            if is_in:
                inclusion[j] = 1
            if is_out:
                inclusion[j] = -1
    return inclusion