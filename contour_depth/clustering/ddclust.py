
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from contour_depth.depth.inclusion_depth import compute_depths as inclusion_depths
from contour_depth.depth.band_depth import compute_depths as band_depths
from contour_depth.depth.utils import get_masks_matrix, get_sdfs, compute_inclusion_matrix, compute_epsilon_inclusion_matrix

# TODO: there is an error when one of the labels dissapears of the clustering, which might happen due to poor initialization
# In this case, sil methods will fail because in correct number of competing clusters will be identified.

def ddclust(masks, init_labs, 
            feat_mat=None, pre_pca=False, 
            cost_lamb = 0.8, beta_init = np.inf, beta_mult=2, no_prog_its=5, max_iter=100,
            cost_threshold=0, swap_subset_max_size=5,
            depth_notion="id", use_modified=False, use_fast=True, output_extra_info=False, verbose=False, seed=42):
    
    print("[ddclust] Initializing ...")
    n_components = np.unique(init_labs).size
    cluster_sizes = np.array([np.where(init_labs == i)[0].size for i in range(n_components)])
    assert not np.any(cluster_sizes < 3)  # no cluster should be smaller than 3 elements

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    
    # Setup matrix
    
    if use_modified:
        inclusion_mat = compute_epsilon_inclusion_matrix(masks)
    else:
        inclusion_mat = compute_inclusion_matrix(masks)

    mat = feat_mat
    if mat is None:
        print("[ddclust] - feat_mat is None, using inclusion_mat as default.")
        mat = inclusion_mat

    if pre_pca:
        pca_embedder = PCA(n_components=30)
        mat = pca_embedder.fit_transform(mat)
    
    # Setup algorithm parameters and preprocessing

    cost_threshold = cost_threshold  # T in the original algorithm
    swap_subset_max_size = swap_subset_max_size  # max size of E in the original algorithm
    
    total_its = 0
    num_its_no_prog = 0
    max_its_no_prog = no_prog_its

    should_finish = False

    pred_labs = init_labs.copy()
    beta = beta_init
    beta_mult = 2 # used to increase the beta

    # Clustering loop
    print("[ddclust] Starting clustering loop ...")
    while not should_finish:
        print(f"Global iter {total_its}")

        # - Compute multivariate medians, sil_i and red_i
        sil_i, _, _, competing_clusters = compute_sil(mat, pred_labs, n_components)
        red_i, _, _, medians = compute_red(masks, pred_labs, n_components,
                                           competing_clusters,
                                           depth_notion=depth_notion,
                                           use_modified=use_modified,
                                           use_fast=use_fast,
                                           inclusion_mat=inclusion_mat)

        # - Compute the cost of the current clustering C(I_1^K)
        cost_i = compute_cost(sil_i, red_i, weight=cost_lamb)
        clustering_cost = cost_i.mean()

        # - Identify observations below acceptance threshold T, take subset S
        working_subset = np.where(cost_i <= cost_threshold)[0]

        if not np.any(cost_i <= cost_threshold):
            print("No more observations with cost below threshold, terminating ...")
            should_finish = True
        
        while working_subset.size > 0:

            swap_subset_size = int(rng.integers(1, swap_subset_max_size, 1)[0])
            if swap_subset_size > working_subset.size:
                swap_subset_size = working_subset.size

            # - Take a random subset E (swap_subset) of S (working_subset)
            swap_subset = rng.choice(working_subset, swap_subset_size, replace=False)

            # - Define new tentative clustering with contours relocated to competing cluster
            new_pred_labs = pred_labs.copy()
            new_pred_labs[swap_subset] = competing_clusters[swap_subset]
            cluster_sizes = np.array([np.where(new_pred_labs == i)[0].size for i in range(n_components)])
            
            accept_clustering = False
            if not np.any(cluster_sizes < 3):  # prevents clusters from getting too small

                # - Compute quantities (sil, red and cost) for tentative clustering
                new_sil_i, _, _, new_competing_clusters = compute_sil(mat, new_pred_labs, n_components)
                new_red_i, _, _, new_medians = compute_red(masks, new_pred_labs, n_components,
                                                        new_competing_clusters,
                                                        depth_notion=depth_notion,
                                                        use_modified=use_modified,
                                                        use_fast=use_fast,
                                                        inclusion_mat=inclusion_mat)
                new_cost_i = compute_cost(new_sil_i, new_red_i, weight=cost_lamb)
                new_clustering_cost = new_cost_i.mean()

                # - Decide whether to accept or not the partition. Criteria to accept:
                # -- if cost(new_partion) > cost(partition)
                # -- if cost(new_partion) <= cost(partition) with Pr(beta, delta cost)            
                delta_cost = clustering_cost - new_clustering_cost
                prob = np.exp(-beta * np.abs(delta_cost))            
                if delta_cost <= 0:                
                    num_its_no_prog = 0
                    accept_clustering = True                      
                else:
                    y = rng.uniform(0, 1, 1)                
                    if y < prob:
                        accept_clustering = True
                    else:                    
                        num_its_no_prog += 1
                        accept_clustering = False

                print(accept_clustering, prob, clustering_cost, new_clustering_cost)

            if accept_clustering:
                pred_labs = new_pred_labs
                sil_i, competing_clusters = new_sil_i, new_competing_clusters
                red_i, medians = new_red_i, new_medians
                cost_i = new_cost_i
                clustering_cost = new_clustering_cost
                print("Updated partition!")
            else:
                print(f"No update (num_its_no_prog: {num_its_no_prog}/{max_its_no_prog})")

            # Update working_subset
            working_subset = np.setdiff1d(working_subset, swap_subset)

            # If no moves have been accepted in the past max_its_no_prog iterations, finish
            if num_its_no_prog > max_its_no_prog:
                if beta < np.inf:
                    print(f"No progress for {num_its_no_prog}, deactivating annealing ...")
                    num_its_no_prog = 0
                    beta = np.inf  # No more annealing
                elif beta == np.inf:
                    print("Should finish")
                    should_finish = True
                break                        
            
            total_its += 1
            if total_its > max_iter:
                should_finish = True
                break

            beta *= beta_mult  # TODO: should be inside or outside the for-loop?


    print(f"ddclust ran for {total_its} iterations")

    if output_extra_info:
        return pred_labs, sil_i, red_i, cost_i
    return pred_labs


###################
# Sil computation #
###################
# Silhouette width: sil(c_i) = (b_i - a_i)/max(a_i, b_i) 
#   with a_i = d(c_i|other members members of c_i's cluster) 
#   and  b_i = min_{other clusters besides the one c_i's in} d(c_i|other members in said cluster)

# first we compute sil_a
def compute_sil_within(contours_mat, clustering, n_components):
    num_contours = contours_mat.shape[0]
    clustering_ids = np.arange(n_components)
    sil_a = np.zeros(num_contours)
    for cluster_id in clustering_ids:
        contour_ids = np.where(clustering == cluster_id)[0]
        dmat = cdist(contours_mat[contour_ids, :], contours_mat[contour_ids, :], metric="sqeuclidean")
        mean_dists = dmat.mean(axis=1)
        sil_a[contour_ids] = mean_dists
    return sil_a

# then we compute sil_b
def compute_sil_between(contours_mat, clustering, n_components):
    num_contours = contours_mat.shape[0]
    clustering_ids = np.arange(n_components)
    sil_b = np.zeros(num_contours)
    competing_cluster_ids = np.zeros(num_contours)
    for cluster_id1 in clustering_ids:
        contour_ids_1 = np.where(clustering == cluster_id1)[0]
        for contour_id in contour_ids_1:
            b_dists = np.zeros(clustering_ids.size)
            b_dists[cluster_id1] = np.inf
            for cluster_id2 in clustering_ids:
                contour_ids_2 = np.where(clustering == cluster_id2)[0]
                if cluster_id1 != cluster_id2:
                    dmat = cdist(contours_mat[contour_id, :].reshape(1, -1), contours_mat[contour_ids_2, :], metric="sqeuclidean")
                    b_dists[cluster_id2] = dmat.mean()
            competing_cid = np.argmin(b_dists)
            competing_cid_dist = b_dists[competing_cid]
            sil_b[contour_id] = competing_cid_dist
            competing_cluster_ids[contour_id] = competing_cid
    return sil_b, competing_cluster_ids

def compute_sil(contours_mat, clustering, n_components):
    sil_a = compute_sil_within(contours_mat, clustering, n_components)
    sil_b, competing_clusters = compute_sil_between(contours_mat, clustering, n_components)
    sil_i = (sil_b - sil_a)/np.maximum(sil_a, sil_b)
    return sil_i, sil_a, sil_b, competing_clusters


#####################
# Depth computation #
#####################
# Relative depth of a point ReD(c_i): D^w(c_i) - D^b(c_i)
#   with D^w(c_i) = ID(c_i|other members of c_i's cluster)
#   and  D^b(c_i) = min_{other clusters besides the one c_i's in} ID(c_i|other members in said cluster)

# first we compute d_w
def compute_red_within(masks, clustering, n_components, depth_notion="id", use_modified=True, use_fast=True, inclusion_mat=None):
    num_contours = len(masks)
    clustering_ids = np.arange(n_components)
    depth_w = np.zeros(num_contours)
    medians = np.empty(clustering_ids.size, int)
    for cluster_id in clustering_ids:
        contour_ids = np.where(clustering == cluster_id)[0]
        mask_subset = [masks[i] for i in contour_ids]
        if inclusion_mat is not None:
            inclusion_mat_subset = inclusion_mat[np.ix_(contour_ids, contour_ids)]
        else:
            inclusion_mat_subset = None
        if depth_notion == "id":
            depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
        elif depth_notion == "cbd":
            depths = band_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
        else:
            raise ValueError("Unsupported depth notion (only id and cbd supported)")
        depth_w[contour_ids] = depths
        median = np.argsort(depths)[-1]
        median = contour_ids[median]
        medians[cluster_id] = median
    return depth_w, medians

# then we compute d_b
def compute_red_between(masks, clustering, n_components, competing_clusters, depth_notion="id", use_modified=True, use_fast=True, inclusion_mat=None):
    num_contours = len(masks)
    clustering_ids = np.arange(n_components)
    depth_b = np.zeros(num_contours)
    for cluster_id1 in clustering_ids:
        contour_ids_1 = np.where(clustering == cluster_id1)[0]
        for contour_id in contour_ids_1:
            competing_cid = competing_clusters[contour_id]
            contour_ids_2 = np.where(clustering == competing_cid)[0].tolist()
            contour_ids_2.append(contour_id)
            mask_subset = [masks[i] for i in contour_ids_2]
            if inclusion_mat is not None:
                inclusion_mat_subset = inclusion_mat[np.ix_(contour_ids_2, contour_ids_2)]
            else:
                inclusion_mat_subset = None
            if depth_notion == "id":
                depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
            elif depth_notion == "cbd":
                depths = band_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
            else:
                raise ValueError("Unsupported depth notion (only id and cbd supported)")
            depth_b[contour_id] = depths[-1]  # we only want the depth of the last contour we appended
    return depth_b

def compute_red(masks, clustering, n_components, competing_clusters, depth_notion="id", use_modified=True, use_fast=True, inclusion_mat=None):
    red_w, medians = compute_red_within(masks, clustering, n_components, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, inclusion_mat=inclusion_mat)
    red_b = compute_red_between(masks, clustering, n_components, competing_clusters, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, inclusion_mat=inclusion_mat)
    red_i = red_w - red_b
    return red_i, red_w, red_b, medians


########
# Cost #
########

def compute_cost(sils, reds, weight=0.5):
    return (1 - weight) * sils + weight * reds
