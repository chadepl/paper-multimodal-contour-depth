
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../..")
from contour_depth.data.synthetic_data import three_rings, shape_families, main_shape_with_outliers
from contour_depth.competing.cvp import get_cvp_sdf_pca_transform, get_cvp_clustering, get_cvp_pca_medians, get_cvp_bands, transform_from_pca_to_sdf, get_per_cluster_mean
from contour_depth.visualization import plot_clustering, spaghetti_plot
from contour_depth.data import ecmwf_ensembles as ecmwf


data_dir = Path("../../data/cvp-paper-meteo/")

data_seed = 0
num_contours = 30
# masks, labs = three_rings(num_contours, 512, 512, return_labels=True, seed=data_seed)
# masks, labs = shape_families(num_contours, 512, 512, return_labels=True, seed=data_seed)
# masks, labs = main_shape_with_outliers(num_contours, 512, 512, p_contamination=0.5, return_labels=True, seed=data_seed)
masks = ecmwf.load_data(data_dir, config_id=0)

masks_shape = masks[0].shape

sdf_mat, pca_mat, transform_mat = get_cvp_sdf_pca_transform(masks, seed=0)
pred_labs  = get_cvp_clustering(sdf_mat, num_components=3)
pca_medians = get_cvp_pca_medians(pca_mat, pred_labs)
sdf_means = get_per_cluster_mean(sdf_mat, pred_labs)
medians = transform_from_pca_to_sdf(np.array(pca_medians)*0.1, np.array(sdf_means), transform_mat)
bands = get_cvp_bands(sdf_mat, pred_labs)

print(len(masks))
print(pca_mat.shape)


fig, ax = plt.subplots(figsize=(3,3), layout="tight")
# ax.imshow(bands[2].reshape(*masks_shape)>0)
# plot_clustering(masks, pred_labs, ax=ax)
spaghetti_plot(masks, iso_value=0.5, arr=np.ones(len(masks))*0.5, is_arr_categorical=False, ax=ax)
ax.contour(medians[0].reshape(*masks_shape), levels=[0,], colors="red")
ax.contour(medians[1].reshape(*masks_shape), levels=[0,], colors="green")
ax.contour(medians[2].reshape(*masks_shape), levels=[0,], colors="blue")
plt.imshow(plt.cm.get_cmap("Purples")(bands[2].reshape(*masks_shape)*0.5))
plt.show()