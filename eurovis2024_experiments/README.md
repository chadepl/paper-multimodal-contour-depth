# paper_res

This folder includes the scripts necessary to generate the figures and tables used in the paper. 
Below we include an index of the different files.
In some cases, we don't generate the figures programatically. We still include their description for the sake of completeness.

## Index

Main:
- `red_explainer.py`: explainer plot of the different components of the ReD criteria and how it helps identifying outliers that sil might not catch.
- `fast_depth_computation_benchmark.py`: generates a plot of number of contours vs time and a scatter plot comparing the depth scores, making evident the MSE is 0.
- `progressive_demo.py`: generates a plot that shows the time that it takes to compute the depths of an ensemble in a progressive manner. The idea is to show the time that the user would need to wait.
- `num_clust_selection.py`: generates a plot showing how ReD can help determining the optimal number of clusters. It also generates small multiples showing the clusterings obtained with different k's.
- `clustering_grid.py`: generates a plot that compares the results of several clustering algorithms on several multi-modal datasets. We compare KMeans, with the state of the art (PCA + AHC) and the depth-based clustering with different depth notions. For the datasets, we consider several types of multi-modality like spatial, magnitude and shape. We also explore different characteristics of the distributions like spread and number of elements. We want to have a figure simulat to `exp-clustering-streamlines.png`.
- `rd_han.py`: generates a plot that illustrates a multi-modal (clustering) depth-based analysis on the head-and-neck segmentations dataset.
- `rd_meteo.py`: generates a plot that illustrates a multi-modal (clustering) depth-based analysis on the meteorological forecasting dataset.

Misc:
- `all_depths.py`: generates a plot that presents a comparison of the four depths we consider in the paper. For a dataset, it compares the slow and fast version of the strict and modified depths. It shows that fast ones are much faster with an MSE of 0.
- `clustering_demo.py`: permits debugging ddclust on the synthetic datasets.
- `sil_vs_red.py`: generates a plot that compares and illustrates the usage of the sil and red criteria to select the optimum number of clusters.

