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

def get_bp_depth_elements(masks, depths, labs=None, outlier_type="tail", epsilon_out=3) -> dict:
    # returns per cluster: representatives, bands and outliers

    depths = np.array(depths).flatten()
    if labs is None:
        labs = [0 for _ in range(depths.size)]
    labs = np.array(labs)
    clusters_ids = np.unique(labs)

    cluster_statistics = dict()    

    for cluster_id in clusters_ids:
        cluster_statistics[cluster_id] = dict()

        coords = np.where(labs == cluster_id)[0]
        subset_depths = depths[coords]

        # representatives        
        median_id = np.argmax(subset_depths)
        median_coord = coords[median_id]
        median_mask = masks[median_coord]
        cluster_statistics[cluster_id]["representatives"] = dict(idx=[median_coord, ], masks=[median_mask, ])
        
        # outliers
        if outlier_type == "threshold":
            outliers_idx = np.where(subset_depths <= epsilon_out)[0]  # should be 0
        elif outlier_type == "tail":
            outliers_idx = np.argsort(subset_depths)[:int(epsilon_out)]  # should be 0
        outliers_coords = [coords[oid] for oid in outliers_idx]
        cluster_statistics[cluster_id]["outliers"] = dict(idx=[], masks=[])
        for ocoord in outliers_coords:
            cluster_statistics[cluster_id]["outliers"]["idx"].append(ocoord)
            cluster_statistics[cluster_id]["outliers"]["masks"].append(masks[ocoord])

        # bands

        sorted_depths_idx = np.argsort(subset_depths)[::-1]
        band100_idx = sorted_depths_idx[~np.in1d(sorted_depths_idx, outliers_idx)]
        band50_idx = band100_idx[:band100_idx.size // 4]        

        band100_coords = [coords[bid] for bid in band100_idx]
        band50_coords = [coords[bid] for bid in band50_idx]

        if len(band100_coords) >= 2:
            band100_mask = np.array([masks[bcoord] for bcoord in band100_coords]).sum(axis=0)
            band100_mask[band100_mask == len(band100_coords)] = 0
            band100_mask[band100_mask > 0] = 1
        else:
            band100_mask = None

        if len(band50_coords) >= 2:
            band50_mask = np.array([masks[bcoord] for bcoord in band50_coords]).sum(axis=0)
            band50_mask[band50_mask == len(band50_coords)] = 0
            band50_mask[band50_mask > 0] = 1
        else:
            band50_mask = None

        cluster_statistics[cluster_id]["bands"] = dict(idx=["b100", "b50"], masks=[band100_mask, band50_mask], weights=[100, 50])
        

        # trimmed mean
        # masks_arr = np.array([m.flatten() for m in [masks[i] for i in cbp_band100]])
        # masks_mean = masks_arr.mean(axis=0)
        # contours = find_contours(masks_mean.reshape(masks[0].shape), level=0.5)
        # plot_contour(contours, line_kwargs=dict(c="dodgerblue", linewidth=5), smooth_line=smooth_line, ax=ax)

    return cluster_statistics


def get_bp_cvp_elements():
    pass
    # per cluster representatives
    # bands
    # outliers

def plot_contour_boxplot(masks, labs, method="depth", method_kwargs=dict(),
                         focus_clusters=None,
                         show_out=True, under_mask=None,
                         smooth_line=False, axis_off=True,
                         ax=None):
    """
    Renders a contour boxplot using depth data and the provided masks.
    If a list of member_ids is supplied (subset_idx), the contour boxplot
    is constructed only from these elements.
    TODO: implement automatic ways of determining epsilon_th and epsilon_out and set them as default
    """

    import matplotlib

    num_contours = len(masks)
    masks_shape = masks[0].shape  # r, c
    if labs is None:
        labs = [0 for _ in range(len(masks))]
    labs = np.array(labs)

    clusters_ids = np.unique(labs)
    num_contours_cluster = [np.where(labs == cluster_id)[0].size for cluster_id in clusters_ids]
    if method == "depth":
        cluster_statistics = get_bp_depth_elements(masks, labs=labs, **method_kwargs)

    if focus_clusters is None:
        focus_clusters = clusters_ids.tolist()
    focus_clusters = np.array(focus_clusters)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
    if axis_off:
        ax.set_axis_off()

    if under_mask is None:
        under_mask_alpha = np.ones(list(masks_shape) + [3, ])
        under_mask = (under_mask_alpha * 255).astype(int)
        ax.imshow(under_mask, alpha=under_mask_alpha[0])
    else:
        ax.imshow(under_mask, cmap="gray")

    for cluster_id in focus_clusters:
        cluster_color = colors[cluster_id] # plum, purple, yellow, red and teal where used before        

        median_color = cluster_color
        outliers_color = cluster_color
        bands_color = cluster_color        

        if show_out and "outliers" in cluster_statistics[cluster_id]:
            cluster_outliers = cluster_statistics[cluster_id]["outliers"]
            for outlier_id, outlier_mask in zip(cluster_outliers["idx"], cluster_outliers["masks"]):
                plot_contour(outlier_mask, line_kwargs=dict(c=outliers_color, linestyle="dashed", linewidth=1, alpha=0.8),
                            smooth_line=smooth_line, ax=ax)
        
        if "bands" in cluster_statistics[cluster_id]:
            cluster_bands = cluster_statistics[cluster_id]["bands"]

            for i, (bid, bmask, bweight) in enumerate(zip(cluster_bands["idx"], cluster_bands["masks"], cluster_bands["weights"])):                                
                cluster_bands_color_hex = bands_color.lstrip("#")
                cluster_bands_color = list(int(cluster_bands_color_hex[i:i+2], 16) for i in (0, 2, 4)) 
                cluster_bands_color += [int(255 * (bweight/100) * 0.3), ] # full opacity        
                bcolmat = np.array(cluster_bands_color).reshape((1, 1, 4))
                bcolmat = np.repeat(bcolmat, repeats=masks_shape[0], axis=(0))
                bcolmat = np.repeat(bcolmat, repeats=masks_shape[1], axis=(1))
                zero_mask = np.where(bmask == 0)
                bcolmat[zero_mask[0], zero_mask[1], -1] = 0  # set alpha to zero if not in mask
                ax.imshow(bcolmat)#, cmap=["Purples", "Greens"][cluster_id])

                # plot_contour(bmask, line_kwargs=dict(c=bands_color, linewidth=1), smooth_line=smooth_line, ax=ax)

        if "representatives" in cluster_statistics[cluster_id]:
            median_mask = cluster_statistics[cluster_id]["representatives"]["masks"][0]
            plot_contour(median_mask, line_kwargs=dict(c=median_color, linewidth=3), smooth_line=smooth_line, ax=ax)

    # Add legend bar    
    from matplotlib.patches import Rectangle
    RECT_WIDTH = 5
    PADDING_LR = 1
    PADDING_TB = 2
    start = PADDING_TB//2
    for cluster_id in clusters_ids:
        cluster_color = colors[cluster_id]
        if cluster_id not in focus_clusters:
            cluster_color = "lightgray"
        rect_height = masks_shape[0]*(num_contours_cluster[cluster_id]/num_contours) - PADDING_TB//2
        rect = Rectangle((masks_shape[1]-RECT_WIDTH - PADDING_LR, start), RECT_WIDTH, rect_height, color=cluster_color)
        ax.add_patch(rect)
        start += rect_height

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