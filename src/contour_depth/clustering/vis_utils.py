import numpy as np
import matplotlib.pyplot as plt

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