"""Visualization utilities.
"""

import numpy as np
from skimage.transform import resize
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##########
# CONFIG #
##########

colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3', "red"]

SMOOTHING = 50  # contour smoothing condition
CONTOUR_PERC_POINTS = 0.8

###########
# GENERAL #
###########

def get_smooth_contour(contour, smoothing=SMOOTHING, contour_perc_points=CONTOUR_PERC_POINTS):
    # https://gist.github.com/prerakmody/5454554b63c94304701ed6348c90809c
    # https://gist.github.com/shubhamwagh/b8148e65a8850a974efd37107ce3f2ec
    x = contour[:, 0].tolist()
    y = contour[:, 1].tolist()
    tck, u = splprep([x, y], u=None, s=smoothing, per=0)  # higher the s value, more the smoothing
    u_new = np.linspace(u.min(), u.max(), int(len(x) * contour_perc_points))
    x_new, y_new = splev(u_new, tck, der=0)
    contour_new = np.array([[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)])
    return contour_new.squeeze()


def plot_contour(mask, iso_value=0.5, plot_line=True, line_kwargs=None, plot_markers=False, markers_kwargs=None,
                 smooth_line=True, ax=None):
    if ax is None:
        pobj = plt
    else:
        pobj = ax

    if line_kwargs is None:
        line_kwargs = {"color": "black"}

    if markers_kwargs is None:
        markers_kwargs = {"color": "black"}

    from skimage.measure import find_contours
    contour = find_contours(mask, level=iso_value)
    for c in contour:
        if smooth_line and c.shape[0] > 3:  # smoothing only works if m > k and we use k=3
            c = get_smooth_contour(c, contour_perc_points=0.7, smoothing=1500)
        if plot_line:
            pobj.plot(c[:, 1], c[:, 0], **line_kwargs)
        if plot_markers:
            pobj.scatter(c[:, 1], c[:, 0], **markers_kwargs)


def spaghetti_plot(masks, iso_value, under_mask=None, arr=None, is_arr_categorical=True, vmin=None, vmax=None,
                           highlight=None, ax=None, alpha=0.5, linewidth=1, resolution=None, smooth_line=True):
    num_members = len(masks)
    if resolution is None:
        resolution = masks[0].shape
    masks = [resize(m, resolution, order=1) for m in masks]

    if arr is not None:
        arr = np.array(arr).flatten()
        if is_arr_categorical:
            arr = arr.astype(int)
    else:
        is_arr_categorical = True
        arr = np.random.choice(np.arange(len(colors)), num_members, replace=True)

    if is_arr_categorical:
        cs = [colors[e] for e in arr]
    else:
        arr = np.array(arr)
        if vmin is not None:
            arr = np.clip(arr, a_min=vmin, a_max=arr.max())
        if vmax is not None:
            arr = np.clip(arr, a_min=arr.min(), a_max=vmax)

        if vmin is None and vmax is None:  # scale to fill 0-1 range
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        cs = [cm.magma(e) for e in arr]

    if highlight is None:
        highlight = list()
    elif type(highlight) is int:
        highlight = [highlight, ]

    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(layout="tight", figsize=(10, 10))

    if under_mask is None:
        under_mask_alpha = np.ones(list(resolution) + [3, ])
        under_mask = (under_mask_alpha * 255).astype(int)
        ax.imshow(under_mask, alpha=under_mask_alpha[0])
    else:
        ax.imshow(resize(under_mask, resolution, order=1), cmap="gray")

    for i, mask in enumerate(masks):
        plot_contour(mask, iso_value=iso_value, line_kwargs=dict(linewidth=linewidth, color=cs[i], alpha=alpha, zorder=0),
                     smooth_line=smooth_line, ax=ax)

    for i in highlight:
        plot_contour(masks[i], iso_value=iso_value, plot_line=False, plot_markers=True, markers_kwargs=dict(color="red", s=1, zorder=1),
                     smooth_line=smooth_line, ax=ax)

    ax.set_axis_off()

    if ax_was_none:
        plt.show()
    else:
        return ax

##############
# DEPTH #
##############

def plot_contour_boxplot(masks, depths,
                         outlier_type="tail", epsilon_out=3, show_out=True,
                         under_mask=None,
                         smooth_line=True, axis_off=True,
                         ax=None):
    """
    Renders a contour boxplot using depth data and the provided masks.
    If a list of member_ids is supplied (subset_idx), the contour boxplot
    is constructed only from these elements.
    TODO: implement automatic ways of determining epsilon_th and epsilon_out and set them as default
    """

    depths = np.array(depths).flatten()

    # - classification
    cbp_median = np.array([np.argmax(depths), ])
    if outlier_type == "threshold":
        cbp_outliers = np.where(depths <= epsilon_out)[0]  # should be 0
    elif outlier_type == "tail":
        cbp_outliers = np.argsort(depths)[:int(epsilon_out)]  # should be 0
    sorted_depths = np.argsort(depths)[::-1]
    cbp_band100 = sorted_depths[~np.in1d(sorted_depths, cbp_outliers)]
    cbp_band50 = cbp_band100[:cbp_band100.size // 2]

    cbp_bands = np.setdiff1d(np.arange(depths.size), np.union1d(cbp_outliers, cbp_median))

    cbp_classification = np.zeros_like(depths)
    cbp_classification[cbp_median] = 0
    cbp_classification[cbp_bands] = 1
    cbp_classification[cbp_outliers] = 2

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), layout="tight")
    if axis_off:
        ax.set_axis_off()

    if under_mask is None:
        under_mask_alpha = np.ones(list(masks[0].shape) + [3, ])
        under_mask = (under_mask_alpha * 255).astype(int)
        ax.imshow(under_mask, alpha=under_mask_alpha[0])
    else:
        ax.imshow(under_mask, cmap="gray")

    # Visual encoding
    if len(cbp_band50 >= 2):
        b50 = get_band_components(masks, cbp_band50)["band"]
    if len(cbp_band100 >= 2):
        b100 = get_band_components(masks, cbp_band100)["band"]

    if show_out:
        for outlier_id in cbp_outliers:
            contours = find_contours(masks[outlier_id])
            plot_contour(contours, line_kwargs=dict(c="red", linestyle="dashed", linewidth=1, alpha=0.8),
                         smooth_line=smooth_line, ax=ax)

    if len(cbp_band100 >= 2):
        contours = find_contours(b100)
        plot_contour(contours, line_kwargs=dict(c="purple", linewidth=2), smooth_line=smooth_line, ax=ax)
        c = plt.cm.get_cmap("Purples")(b100)
        c[:, :, -1] = b100 * 0.1
        ax.imshow(c, cmap="Purples")

    if len(cbp_band50 >= 2):
        contours = find_contours(b50)
        plot_contour(contours, line_kwargs=dict(c="plum", linewidth=2), smooth_line=smooth_line, ax=ax)
        c = plt.cm.get_cmap("Purples")(b50)
        c[:, :, -1] = b50 * 0.1
        ax.imshow(c, alpha=b50, cmap="Purples")

    contours = find_contours(masks[cbp_median[0]])
    plot_contour(contours, line_kwargs=dict(c="yellow", linewidth=5), smooth_line=smooth_line, ax=ax)

    # trimmed mean
    masks_arr = np.array([m.flatten() for m in [masks[i] for i in cbp_band100]])
    masks_mean = masks_arr.mean(axis=0)
    contours = find_contours(masks_mean.reshape(masks[0].shape), level=0.5)
    plot_contour(contours, line_kwargs=dict(c="dodgerblue", linewidth=5), smooth_line=smooth_line, ax=ax)

    return ax

##############
# CLUSTERING #
##############

def plot_clustering_eval(k_vals, metric_a, metric_a_id, metric_b=None, metric_b_id=None, metric_a_lab=None, metric_b_lab=None, ax=None):
    
    xmin_id = np.argmin(metric_a)
    xmax_id = np.argmax(metric_a)
    xmin = k_vals[xmin_id]
    xmax = k_vals[xmax_id]

    ax_was_none = False
    if ax is None:
        ax_was_none = True        
        fig, ax = plt.subplots(layout="tight", figsize=(4, 3))

    ax2 = None
    if metric_b is not None:
        ax2 = ax.twinx()
    
    lns1 = ax.plot(k_vals, metric_a, label=metric_a_id, c="orange")
    ax.axvline(x=xmin, c="orange", linestyle="--")
    ax.axvline(x=xmax, c="orange", linestyle="--")
    ax.set_xlabel("Number of clusters (K)")
    if metric_a_lab is not None:
        ax.set_ylabel(metric_a_lab)

    lns2 = None
    if metric_b is not None:
        lns2 = ax2.plot(k_vals, metric_b, label=metric_b_id, c="blue")
        if metric_b_lab is not None:
            ax2.set_ylabel(metric_b_lab)

    # added these three lines
    lns = lns1
    if lns2 is not None:
        lns += lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper right")

    if ax_was_none:
        plt.show()
    else:
        return ax, ax2

def plot_clustering(masks, labs, smooth_line=False, ax=None):

    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(layout="tight", figsize=(10, 10))
    
    cluster_ids = np.unique(labs)
    #colors = ["red", "blue", "orange"]
    for i, cluster_id in enumerate(cluster_ids):
        contour_ids = np.where(labs == cluster_id)[0]
        masks_subset = [masks[contour_id] for contour_id in contour_ids]
        for mask in masks_subset:
            plot_contour(mask, 
                         iso_value=0.5, 
                         plot_line=True, 
                         line_kwargs=dict(c=colors[i], linewidth=1, alpha=0.5), 
                         plot_markers=False, 
                         markers_kwargs=None,
                         smooth_line=smooth_line, ax=ax)
            
    if ax_was_none:
        plt.show()
    else:
        return ax

def sort_red(labs, red_within, red_between=None, sort_by=None):

    num_contours = red_within.size
    clusters_idx = np.unique(labs)
    
    argsort = np.zeros_like(labs)
    sorted_labs = labs.copy()
    sorted_red_within = red_within.copy()
    if red_between is not None:
        sorted_red_between = red_between.copy()
        red = sorted_red_within - sorted_red_between
        sorted_red = red.copy()
    else:
        sorted_red_between = None
        sorted_red = None
        
    if sort_by is None:
        sort_by = "red_within"

    start_id = 0

    sorting_idx = np.zeros(num_contours, dtype=int)      
    for cluster_id in clusters_idx:
        coords = np.where(labs == cluster_id)[0]
        if sort_by == "red_within":
            arr = red_within[coords]
        elif sort_by == "red_between" and red_between is not None:
            arr = red_between[coords]
        elif sort_by == "red" and red_between is not None:
            arr = red[coords]
        else:
            raise ValueError("Make sure the passed values are correct.")
                    
        sorting_idx = np.argsort(arr)[::-1]
        argsort[start_id:start_id + arr.size] = coords[sorting_idx]
        sorted_labs[start_id:start_id + arr.size] = np.ones_like(sorting_idx) * cluster_id
        sorted_red_within[start_id:start_id + arr.size] = red_within[coords[sorting_idx]]
        if red_between is not None:
            sorted_red_between[start_id:start_id + arr.size] = red_between[coords[sorting_idx]]
            sorted_red[start_id:start_id + arr.size] = red[coords[sorting_idx]]
        start_id += arr.size

    return argsort, sorted_labs, sorted_red_within, sorted_red_between, sorted_red


def plot_red(red_within, red_between=None, compute_red=False, labs=None, sort_by=None, ax=None):
    
    num_contours = red_within.size

    if labs is None:
        labs = np.zeros(num_contours, dtype=int)

    if sort_by is not None:
        argsort, labs, red_within, red_between, red = sort_red(labs, red_within, red_between, sort_by=sort_by)
    else:
        red = red_within - red_between if red_between is not None else None
    
    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(layout="tight", figsize=(10, 10))

    #colors = ["red", "blue", "orange"]    
    cs = [colors[l] for l in labs]
    
    ax.bar(np.arange(num_contours), red_within, color=cs)
    if red_between is not None:
        ax.bar(np.arange(num_contours), np.negative(red_between), color=cs)
        if compute_red:
            ax.bar(np.arange(num_contours), red, fill=False, color="black")
        ax.axhline(y=0, c="black")
        
    if ax_was_none:
        plt.show()
    else:
        return ax


def plot_clustering_results(masks, clustering, sil_i, red_i, fn=None, suptitle=None):
    num_contours = len(masks)
    fig, axs = plt.subplots(ncols=3, figsize=(10, 8))
    color_dict = ["red", "blue", "orange"]
    for m, l in zip(masks, clustering):
            axs[0].contour(m, colors=[color_dict[l], ], linewidths=1, alpha=0.1)
    axs[1].set_title("Sil")
    axs[1].bar(np.arange(num_contours), sil_i, color=[color_dict[l] for l in clustering])
    axs[2].set_title("ReD")
    axs[2].bar(np.arange(num_contours), red_i, color=[color_dict[l] for l in clustering])

    if suptitle is not None:
         fig.suptitle(suptitle)

    if fn is not None:
        fig.savefig(fn)
    else:
        plt.show()