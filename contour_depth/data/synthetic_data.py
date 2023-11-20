
import numpy as np
from scipy.spatial.distance import cdist
from skimage.draw import ellipse
from skimage.draw import polygon2mask

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
    centers_rows = rng.normal(np.floor(num_rows*center_mean[0]), num_rows*center_std[0], num_masks)
    centers_cols = rng.normal(np.floor(num_cols*center_mean[1]), num_cols*center_std[1], num_masks)

    # build ensemble
    masks = []
    for i in range(num_masks):
        mask = np.zeros((num_rows, num_cols))
        rr, cc = ellipse(centers_rows[i], centers_cols[i], radii[i], radii[i], (num_rows, num_cols))
        mask[rr, cc] = 1
        masks.append(mask)

    return masks


def get_base_gp(num_masks, domain_points, scale=0.01, sigma=1.0, seed=None):
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    thetas = domain_points.flatten().reshape(-1, 1)
    num_vertices = thetas.size
    gp_mean = np.zeros(num_vertices)

    gp_cov_sin = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.sin(thetas), np.sin(thetas), "sqeuclidean"))
    gp_sample_sin = rng.multivariate_normal(gp_mean, gp_cov_sin, num_masks)
    gp_cov_cos = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.cos(thetas), np.cos(thetas), "sqeuclidean"))
    gp_sample_cos = rng.multivariate_normal(gp_mean, gp_cov_cos, num_masks)

    return gp_sample_sin + gp_sample_cos

def get_xy_coords(angles, radii):
    num_members = radii.shape[0]
    angles = angles.flatten().reshape(1,- 1).repeat(num_members, axis=0)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y

def rasterize_coords(x_coords, y_coords, num_rows, num_cols):    
    masks = []
    for xc, yc in zip(x_coords, y_coords):
        coords_arr = np.concatenate([xc.reshape(-1,1), yc.reshape(-1,1)], axis=1)
        coords_arr *= num_rows//2
        coords_arr += num_cols//2
        mask = polygon2mask((num_rows, num_cols), coords_arr).astype(float)
        masks.append(mask)
    return masks

def main_shape_with_outliers(num_masks, num_rows, num_cols, num_vertices=100, p_contamination=0.1, return_labels=False, seed=None):

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
        
    population_radius=0.5
    normal_scale=0.003
    normal_freq=0.9
    outlier_scale=0.009
    outlier_freq=0.04                                  
    
    thetas = np.linspace(0, 2 * np.pi, num_vertices)
    population_radius = np.ones_like(thetas) * population_radius  # if we want constant radius (for a circle)
    gp_sample_normal = get_base_gp(num_masks, thetas, scale=normal_scale, sigma=normal_freq)
    gp_sample_outliers = get_base_gp(num_masks, thetas, scale=outlier_scale, sigma=outlier_freq)

    should_contaminate = (rng.random(num_masks) > (1 - p_contamination)).astype(float)
    should_contaminate = should_contaminate.reshape(num_masks,-1).repeat(num_vertices, axis=1)

    radii = population_radius + (gp_sample_normal * (1 - should_contaminate)) + (gp_sample_outliers * should_contaminate)

    xs, ys = get_xy_coords(thetas, radii)
    contours = rasterize_coords(xs, ys, num_rows, num_cols)

    labels = should_contaminate[:, 0].astype(int)

    if return_labels:
        return contours, labels
    else:
        return contours

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



def three_rings(num_masks, num_rows, num_cols, modes_proportions=(0.5, 0.3, 0.2), center_stds=(0.009, 0.01, 0.007), return_labels=False, seed=None):
    """
    Three circles with the x, y origin coordinates perturbed.
    This is the example used in the EnConVis paper.
    """
    modes_sizes = compute_modes_sizes(num_masks, modes_proportions)

    masks = []
    labs = []
    
    mode_masks = circle_ensemble(modes_sizes[0], num_rows, num_cols, center_mean=(0.5 + 0.17, 0.5), center_std=(center_stds[0], center_stds[0]), radius_mean=0.19, radius_std=0.19*0.1, seed=seed)
    mode_labs = [0 for _ in range(modes_sizes[0])]

    masks += mode_masks
    labs += mode_labs

    mode_masks = circle_ensemble(modes_sizes[1], num_rows, num_cols, center_mean=(0.5, 0.5 - 0.15), center_std=(center_stds[1], center_stds[1]), radius_mean=0.2, radius_std=0.2*0.1, seed=seed)
    mode_labs = [1 for _ in range(modes_sizes[1])]

    masks += mode_masks
    labs += mode_labs

    mode_masks = circle_ensemble(modes_sizes[2], num_rows, num_cols, center_mean=(0.5, 0.5 + 0.15), center_std=(center_stds[2], center_stds[2]), radius_mean=0.18, radius_std=0.18*0.1, seed=seed)
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
