import numpy as np

def compute_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
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
    return inclusion


def compute_epsilon_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous. 

    Parameters
    ----------
    masks : _type_
        _description_
    """
    num_masks = len(masks)
    inclusion_mat = np.zeros((num_masks, num_masks))

    for i in range(num_masks):
        inclusion = compute_epsilon_inclusion(i, masks)
        inclusion_mat[i, :] = inclusion

    return inclusion_mat


def compute_epsilon_inclusion(contour_index, masks):
    num_masks = len(masks)
    inclusion = np.zeros(num_masks)
    for j in range(num_masks):
        in_cj = masks[j]
        if contour_index != j:
            in_ci = masks[contour_index]

            eps_sub = (in_ci * (1 - in_cj)).sum()/in_ci.sum()
            eps_sub = 1 - eps_sub

            inclusion[j] = eps_sub

    return inclusion

