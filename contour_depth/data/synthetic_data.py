
import numpy as np
from skimage.draw import ellipse

############
# Unimodal #
############

def circle_ensemble(num_masks, num_rows, num_cols, center_mean=(0.5, 0.5), center_std=(0, 0), radius_mean=0.25, radius_std=0.25 * 0.1, seed=None):

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    RADIUS_MEAN = np.minimum(num_rows, num_cols) * radius_mean  # radius is 25 percent of diameter
    RADIUS_STD = np.minimum(num_rows, num_cols) * radius_std
    radii = rng.normal(RADIUS_MEAN, RADIUS_STD, num_masks)
    centers_rows = rng.normal(np.floor(num_rows*center_mean[0]), center_std[0], num_masks)
    centers_cols = rng.normal(np.floor(num_cols*center_mean[1]), center_std[1], num_masks)

    # build ensemble
    masks = []
    for i in range(num_masks):
        mask = np.zeros((num_rows, num_cols))
        rr, cc = ellipse(centers_rows[i], centers_cols[i], radii[i], radii[i], (num_rows, num_cols))
        mask[rr, cc] = 1
        masks.append(mask)

    return masks

##############
# Multimodal #
##############

def compute_modes_sizes(num_masks, modes_proportions):
    modes_sizes = []
    running_size = 0
    for mp in modes_proportions[:-1]:
        s = int(np.floor(num_masks * mp))
        modes_sizes.append(s)
        running_size += s
    modes_sizes.append(num_masks - running_size)
    return modes_sizes


def magnitude_modes(num_masks, num_rows, num_cols, 
                    modes_proportions=(0.5, 0.5),
                    modes_radius_mean=(0.2, 0.18),
                    modes_radius_std=(0.2*0.06, 0.16*0.05),
                    modes_center_mean=((0.5, 0.5), (0.5, 0.5)),
                    modes_center_std=((0, 0), (0, 0)),
                    return_labels=False,
                    seed=None):

    num_modes = len(modes_proportions)
    modes_sizes = compute_modes_sizes(num_masks, modes_proportions)
    
    masks = []
    labs = []

    for mode_id in range(num_modes):
        mode_masks = circle_ensemble(modes_sizes[mode_id], num_rows, num_cols, 
                                     center_mean=modes_center_mean[mode_id], center_std=modes_center_std[mode_id],
                                     radius_mean=modes_radius_mean[mode_id], radius_std=modes_radius_std[mode_id], seed=seed)
        mode_labs = [mode_id for _ in range(modes_sizes[mode_id])]
        masks += mode_masks
        labs += mode_labs
    
    if return_labels:
        return masks, labs
    return masks



def three_rings(num_masks, num_rows, num_cols, modes_proportions=(0.5, 0.3, 0.2), return_labels=False, seed=None):
    """
    Three circles with the x, y origin coordinates perturbed.
    This is the example used in the EnConVis paper.
    """
    modes_sizes = compute_modes_sizes(num_masks, modes_proportions)

    masks = []
    labs = []
    
    mode_masks = circle_ensemble(modes_sizes[0], num_rows, num_cols, center_mean=(0.5 + 0.17, 0.5), center_std=(0, 0), radius_mean=0.19, radius_std=0.19*0.1, seed=seed)
    mode_labs = [0 for _ in range(modes_sizes[0])]

    masks += mode_masks
    labs += mode_labs

    mode_masks = circle_ensemble(modes_sizes[1], num_rows, num_cols, center_mean=(0.5, 0.5 - 0.15), center_std=(0, 0), radius_mean=0.2, radius_std=0.2*0.1, seed=seed)
    mode_labs = [1 for _ in range(modes_sizes[1])]

    masks += mode_masks
    labs += mode_labs

    mode_masks = circle_ensemble(modes_sizes[2], num_rows, num_cols, center_mean=(0.5, 0.5 + 0.15), center_std=(0, 0), radius_mean=0.18, radius_std=0.18*0.1, seed=seed)
    mode_labs = [2 for _ in range(modes_sizes[2])]

    masks += mode_masks
    labs += mode_labs

    if return_labels:
        return masks, labs
    return masks


def shape_families(num_masks, num_rows, num_cols, return_labels=False, seed=None):
    """
    Two shape families
    For each member in family A, there is one in family B 
    for for which m_A - m_B = 0 holds even though the two 
    families have clearly visually distinct shapes.
    """
    from skimage.draw import ellipse, polygon

    row_c, col_c = num_rows//2, num_cols//2

    masks = []
    labels = []

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    radii = rng.normal(100, 20, num_masks)

    # family A
    for i in range(num_masks//2):

        radius = radii[i]

        mask = np.zeros((512, 512))
        rr, cc = ellipse(row_c, col_c, radius, radius, shape=(512, 512))
        mask[rr, cc] = 1
        masks.append(mask)
        labels.append(0)

    # family B
    for i in range(num_masks - num_masks//2):
        radius = radii[i]
        
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.ones(100) * radius
        r = r + 15 * np.sin(theta * 10)

        x = r * np.cos(theta) + 256
        y = r * np.sin(theta) + 256

        mask = np.zeros((512, 512))
        rr, cc = polygon(y, x, shape=(512, 512))
        mask[rr, cc] = 1
        masks.append(mask)
        labels.append(1)

    if return_labels:
        return masks, labels
    return masks