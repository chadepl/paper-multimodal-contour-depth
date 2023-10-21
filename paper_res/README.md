# paper_res

This folder includes the scripts necessary to generate the figures and tables used in the paper. 
Below we include an index of the different files.
In some cases, we don't generate the figures programatically. We still include their description for the sake of completeness.

## Index

- `all_depths.py`: generates a plot that presents a comparison of the four depths we consider in the paper. For a dataset, it compares the slow and fast version of the strict and modified depths. It shows that fast ones are much faster with an MSE of 0.
- `fast_[strict/modified]_depths_benchmark.py`: generates a plot of number of contours vs time and a scatter plot comparing the depth scores, making evident the MSE is 0.
- `fast_strict_explainer` (figma): explainer plot of how we accelerate the strict methods using the inclusion matrix.
- `fast_modified_explainer` (figma): explainer plot of how we accelerate the modified methods using algebraic manipulation.
- `red_explainer` (figma): explainer plot of the different components of the ReD criteria and how it helps identifying outliers that sil might not catch.
- `clustering_demo.py`: generates a plot that shows how changing the parameter alpha increases the effect of depth in the clustering behavior.
- `sil_vs_red.py`: generates a plot that compares and illustrates the usage of the sil and red criteria to select the optimum number of clusters.
- `clustering_grid.py`: generates a plot that compares the results of several clustering algorithms on several multi-modal datasets. We compare KMeans, with the state of the art (PCA + AHC) and the depth-based clustering with different depth notions. For the datasets, we consider several types of multi-modality like spatial, magnitude and shape. We also explore different characteristics of the distributions like spread and number of elements. We want to have a figure simulat to `exp-clustering-streamlines.png`.
- `rd_han.py`: generates a plot that illustrates a multi-modal (clustering) depth-based analysis on the head-and-neck segmentations dataset.
- `rd_meteo.py`: generates a plot that illustrates a multi-modal (clustering) depth-based analysis on the meteorological forecasting dataset.
