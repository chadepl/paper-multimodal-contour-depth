
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from contour_depth.depth.inclusion_depth import compute_depths as inclusion_depths
from contour_depth.depth.inclusion_depth import get_precompute_in, get_precompute_out
from contour_depth.depth.band_depth import compute_depths as band_depths
from contour_depth.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix
from contour_depth.utils import get_masks_matrix, get_sdfs

def kmeans_cluster_eid(masks, num_clusters, max_num_iterations=10):
    masks = np.array(masks, dtype=np.float32)
    neg_masks = 1 - masks
    areas = np.sum(masks, axis=(1, 2))
    inv_areas = 1 / areas

    num_masks, height, width = masks.shape
    
    cluster_assignment = np.random.randint(low=0, high=num_clusters, size=num_masks)
    for _ in range(max_num_iterations):
        precompute_in = np.empty((num_clusters, height, width), dtype=np.float32)
        precompute_out = np.empty((num_clusters, height, width), dtype=np.float32)
        
        for c in range(num_clusters):
            selected_masks = masks[cluster_assignment == c]
            selected_areas = areas[cluster_assignment == c]
            selected_inv_masks = neg_masks[cluster_assignment == c]

            precompute_in[c] = np.sum(selected_inv_masks, axis=0)
            precompute_out[c] = np.sum((selected_masks.T / selected_areas.T).T, axis=0)

        depth_in_cluster = np.empty((num_clusters, num_masks), dtype=np.float32)
        for c in range(num_clusters):
            N = np.sum(cluster_assignment == c)
            assert(N > 0)
            IN_in = N - inv_areas * np.sum(masks * precompute_in[c], axis=(1,2))
            IN_out = N - np.sum(neg_masks * precompute_out[c], axis=(1, 2))
            depth_in_cluster[c] = np.minimum(IN_in, IN_out) / N
            
        old_cluster_assignment = cluster_assignment
        cluster_assignment = np.argmax(depth_in_cluster, axis=0)
        if np.all(cluster_assignment == old_cluster_assignment):
            break
    return cluster_assignment


# TODO: there is an error when one of the labels dissapears of the clustering, which might happen due to poor initialization
# In this case, sil methods will fail because in correct number of competing clusters will be identified.

def cdclust(masks, init_labs, 
            beta_init = np.inf, beta_mult=2, no_prog_its=5, max_iter=100,
            cost_threshold=0, swap_subset_max_size=5, sample_swap_size = True, check_cost_function=True,
            competing_cluster_method="red",
            depth_notion="id", use_modified=False, use_fast=True, output_extra_info=False, verbose=False, seed=42):
    
    if verbose:
        print("[cdclust] Initializing ...")
    n_components = np.unique(init_labs).size
    cluster_sizes = np.array([np.where(init_labs == i)[0].size for i in range(n_components)])
    #assert not np.any(cluster_sizes < 1)  # no cluster should be smaller than 3 elements

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)    
    
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

    # Setup depth
    
    # - Setup inclusion matrix
    if use_modified and use_fast:
        assert depth_notion == "id"
        inclusion_mat = None
        epsilon_inclusion_mat = None
    else:
        epsilon_inclusion_mat = compute_epsilon_inclusion_matrix(masks)
        if not use_modified:
            inclusion_mat = (epsilon_inclusion_mat == 1).astype(float)
        else:
            inclusion_mat = epsilon_inclusion_mat

    # - Setup precomputed fields per partition
    precomputed_ins = None
    precomputed_outs = None
    new_precomputed_ins = None
    new_precomputed_outs = None


    # Clustering loop
    if verbose:
        print("[cdclust] Starting clustering loop ...")
    while not should_finish:
        if verbose:
            print(f"Global iter {total_its}")
        
        # - Setup precomputed fields per partition
        if use_fast:
            assert depth_notion == "id"
            precomputed_ins = {k:get_precompute_in([masks[i] for i in np.where(pred_labs == k)[0]]) for k in range(n_components)}
            precomputed_outs = {k:get_precompute_out([masks[i] for i in np.where(pred_labs == k)[0]]) for k in range(n_components)}

        # - Compute multivariate medians, sil_i and red_i
        red_wi = compute_red_within(masks, pred_labs, n_components, 
                                    depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, 
                                    inclusion_mat=inclusion_mat, precompute_ins=precomputed_ins, precompute_outs=precomputed_outs)
        #competing_clusters = get_depth_competing_clusters(pred_labs, n_components, red_wi, inclusion_mat)
        red_bi, competing_clusters = compute_red_between(masks, pred_labs, n_components, competing_clusters=None,
                                     depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, 
                                     inclusion_mat=inclusion_mat, precompute_ins=precomputed_ins, precompute_outs=precomputed_outs)
        red_i = red_wi - red_bi

        # - Compute the cost of the current clustering C(I_1^K)
        clustering_cost = red_i.mean()

        # - Identify observations below acceptance threshold T, take subset S
        working_subset = np.where(red_i <= cost_threshold)[0]

        if not np.any(red_i <= cost_threshold):
            if verbose:
                print("No more observations with cost below threshold, terminating ...")
            should_finish = True
        
        while working_subset.size > 0:

            if sample_swap_size:
                swap_subset_size = int(rng.integers(1, swap_subset_max_size, 1)[0])
            else:
                swap_subset_size = swap_subset_max_size
            if swap_subset_size > working_subset.size:
                swap_subset_size = working_subset.size

            # - Take a random subset E (swap_subset) of S (working_subset)
            swap_subset = rng.choice(working_subset, swap_subset_size, replace=False)

            # - Define new tentative clustering with contours relocated to competing cluster
            new_pred_labs = pred_labs.copy()
            new_pred_labs[swap_subset] = competing_clusters[swap_subset]
            cluster_sizes = np.array([np.where(new_pred_labs == i)[0].size for i in range(n_components)])
            
            accept_clustering = False
            if not np.any(cluster_sizes < 1):  # prevents clusters from getting too small

                # - update precomputed fields per partition
                if use_fast:
                    assert depth_notion == "id"
                    new_precomputed_ins = {k:get_precompute_in([masks[i] for i in np.where(pred_labs == k)[0]]) for k in range(n_components)}
                    new_precomputed_outs = {k:get_precompute_out([masks[i] for i in np.where(pred_labs == k)[0]]) for k in range(n_components)}
                    for i in swap_subset:
                        pre_lab = pred_labs[i]
                        new_lab = new_pred_labs[i]                    
                        new_precomputed_ins[pre_lab] -= 1 - masks[i]
                        new_precomputed_ins[new_lab] += 1 - masks[i]
                        new_precomputed_outs[pre_lab] -= masks[i]/masks[i].sum()
                        new_precomputed_outs[new_lab] += masks[i]/masks[i].sum()

                # - Compute quantities for tentative clustering
                new_red_wi = compute_red_within(masks, new_pred_labs, n_components, 
                                    depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, 
                                    inclusion_mat=inclusion_mat, precompute_ins=new_precomputed_ins, precompute_outs=new_precomputed_outs)
                new_competing_clusters = get_depth_competing_clusters(new_pred_labs, n_components, new_red_wi, inclusion_mat)
                new_red_bi, new_competing_clusters = compute_red_between(masks, new_pred_labs, n_components, competing_clusters=None,
                                            depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast,
                                            inclusion_mat=inclusion_mat, precompute_ins=new_precomputed_ins, precompute_outs=new_precomputed_outs)
                new_red_i = new_red_wi - new_red_bi
                new_clustering_cost = new_red_i.mean()

                # - Decide whether to accept or not the partition. Criteria to accept:
                # -- if cost(new_partion) > cost(partition)
                # -- if cost(new_partion) <= cost(partition) with Pr(beta, delta cost) 
                delta_cost = clustering_cost - new_clustering_cost
                prob = np.exp(-beta * np.abs(delta_cost)) 
                with_annealing = False
                if check_cost_function:                               
                    if delta_cost <= 0:                
                        num_its_no_prog = 0
                        accept_clustering = True                      
                    else:
                        y = rng.uniform(0, 1, 1)                
                        if y < prob:
                            accept_clustering = True
                            with_annealing = True
                        else:                    
                            num_its_no_prog += 1
                            accept_clustering = False
                else:
                    accept_clustering = True

                if verbose:
                    print(accept_clustering, prob, clustering_cost, new_clustering_cost)

            if accept_clustering:                
                pred_labs = new_pred_labs
                red_i = new_red_i
                competing_clusters = new_competing_clusters
                clustering_cost = new_clustering_cost
                precomputed_ins = new_precomputed_ins
                precomputed_outs = new_precomputed_outs
                if delta_cost <= 0 or with_annealing:
                    best_labs = pred_labs
                    best_red_i = red_i
                if verbose:
                    print("Updated partition!")
            else:
                if verbose:
                    print(f"No update (num_its_no_prog: {num_its_no_prog}/{max_its_no_prog})")

            # Update working_subset
            working_subset = np.setdiff1d(working_subset, swap_subset)

            # If no moves have been accepted in the past max_its_no_prog iterations, finish
            if num_its_no_prog > max_its_no_prog:
                if beta < np.inf:
                    if verbose:
                        print(f"No progress for {num_its_no_prog}, deactivating annealing ...")
                    num_its_no_prog = 0
                    beta = np.inf  # No more annealing
                elif beta == np.inf:
                    if verbose:
                        print("Should finish")
                    should_finish = True
                break                        
            
            total_its += 1
            if total_its > max_iter:
                should_finish = True
                break

            beta *= beta_mult  # TODO: should be inside or outside the for-loop?

    if verbose:
        print(f"cdclust ran for {total_its} iterations")

    if output_extra_info:
        return best_labs, best_red_i
    return best_labs


def ddclust(masks, init_labs, 
            feat_mat=None, pre_pca=False, 
            cost_lamb = 0.8, beta_init = np.inf, beta_mult=2, no_prog_its=5, max_iter=100,
            cost_threshold=0, swap_subset_max_size=5,
            competing_cluster_method="sil",
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
        sil_i, _, _, sil_contour_cluster_rels = compute_sil(mat, pred_labs, n_components)
        red_i, _, _, medians, red_contour_cluster_rels = compute_red(masks, pred_labs, n_components,
                                                                     competing_clusters=None,
                                                                     depth_notion=depth_notion,
                                                                     use_modified=use_modified,
                                                                     use_fast=use_fast,
                                                                     inclusion_mat=inclusion_mat)
        competing_clusters = get_competing_clusters(pred_labs, n_components,
                                                    sil_contour_cluster_rels, red_contour_cluster_rels, 
                                                    inclusion_mat=inclusion_mat, depths=red_i, method=competing_cluster_method)

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
                new_sil_i, _, _, new_sil_contour_cluster_rels = compute_sil(mat, new_pred_labs, n_components)
                new_red_i, _, _, new_medians, new_red_contour_cluster_rels = compute_red(masks, new_pred_labs, n_components,
                                                                                         competing_clusters=None,
                                                                                         depth_notion=depth_notion,
                                                                                         use_modified=use_modified,
                                                                                         use_fast=use_fast,
                                                                                         inclusion_mat=inclusion_mat)
                new_competing_clusters = get_competing_clusters(pred_labs, n_components,
                                                    new_sil_contour_cluster_rels, new_red_contour_cluster_rels, 
                                                    inclusion_mat=inclusion_mat, depths=new_red_i, method=competing_cluster_method)
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
                sil_i, sil_contour_cluster_rels = new_sil_i, new_sil_contour_cluster_rels
                red_i, medians, red_contour_cluster_rels = new_red_i, new_medians, new_red_contour_cluster_rels
                competing_clusters = new_competing_clusters
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

#################################
# Competing cluster computation #
#################################

# A good competing cluster is one that
#  a) is nearby (for sil we want to decrease the within term -> increases compactness)
#  b) one that could potentially profit more from having the contour in terms of depth (for red we want to increase depth)
#  b1) alternatively, if we do not want to compute the depth for all cluster, we could calculate the medians and 
#      assess the inclusion relationship
# between depth is potentially more expensive to compute than between sil. 

def get_competing_clusters(clustering_ids, n_components, sil_contour_cluster_rels, red_contour_cluster_rels, inclusion_mat, depths, method="sil"):
    num_contours = clustering_ids.size
    if method=="sil":
        competing_clusters = np.argmin(sil_contour_cluster_rels, axis=1)  # we pick the most compact cluster after transferring the contour.
    elif method=="red":
        competing_clusters = np.argmax(red_contour_cluster_rels, axis=1)  # we pick the deepest cluster after transferring the contour.
    elif method=="inclusion_rel":
        inclusion_rels = np.empty((num_contours, n_components), dtype=float)
        for contour_id, cluster_id_1 in zip(np.arange(num_contours), clustering_ids):
            inclusion_rels[contour_id, cluster_id_1] = 0
            for cluster_id_2 in np.setdiff1d(np.unique(clustering_ids), cluster_id_1):
                cluster_ids = np.where(clustering_ids == cluster_id_2)[0]
                median_id = np.argmax(depths[cluster_ids])
                median_glob_id = cluster_ids[median_id]
                ls = inclusion_mat[contour_id, median_glob_id]
                rs = inclusion_mat[median_glob_id, contour_id]
                inclusion_rels[contour_id, cluster_id_2] = ls + rs  # TODO: add thresholding?
        competing_clusters = np.argmax(inclusion_rels, axis=1) # we only compare the contour against the medians and pick the one with which it has the strongest inclusion relationship
    return competing_clusters


def get_depth_competing_clusters(clustering_ids, n_components, depths, inclusion_mat):
    num_contours = clustering_ids.size
    inclusion_rels = np.empty((num_contours, n_components), dtype=float)
    for contour_id, cluster_id_1 in zip(np.arange(num_contours), clustering_ids):
        inclusion_rels[contour_id, cluster_id_1] = 0
        for cluster_id_2 in np.setdiff1d(np.unique(clustering_ids), cluster_id_1):
            cluster_ids = np.where(clustering_ids == cluster_id_2)[0]
            median_id = np.argmax(depths[cluster_ids])
            median_glob_id = cluster_ids[median_id]
            ls = inclusion_mat[contour_id, median_glob_id]
            rs = inclusion_mat[median_glob_id, contour_id]
            inclusion_rels[contour_id, cluster_id_2] = ls + rs  # TODO: add thresholding?
    competing_clusters = np.argmax(inclusion_rels, axis=1) # we only compare the contour against the medians and pick the one with which it has the strongest inclusion relationship
    return competing_clusters


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


# we need instance to cluster matrix (N, C)

# then we compute sil_b
def compute_sil_between(contours_mat, clustering, n_components):
    num_contours = contours_mat.shape[0]
    clustering_ids = np.arange(n_components)
    contour_cluster_rels = np.empty((num_contours, n_components), dtype=float)
    for cluster_id1 in clustering_ids:
        contour_ids_1 = np.where(clustering == cluster_id1)[0]
        for contour_id in contour_ids_1:
            contour_cluster_rels[contour_id, cluster_id1] = np.inf
            for cluster_id2 in np.setdiff1d(clustering_ids, cluster_id1):
                contour_ids_2 = np.where(clustering == cluster_id2)[0]
                # compute distance from contour_id to all contours in contour_ids_2 (outputs a vector)
                dmat = cdist(contours_mat[contour_id, :].reshape(1, -1), contours_mat[contour_ids_2, :], metric="sqeuclidean")
                contour_cluster_rels[contour_id, cluster_id2] = dmat.mean()
    competing_cluster_ids = np.argmin(contour_cluster_rels, axis=1)
    sil_b = contour_cluster_rels[np.arange(num_contours), competing_cluster_ids]
    return sil_b, contour_cluster_rels

def compute_sil(contours_mat, clustering, n_components):
    sil_a = compute_sil_within(contours_mat, clustering, n_components)
    sil_b, contour_cluster_rels = compute_sil_between(contours_mat, clustering, n_components)
    sil_i = (sil_b - sil_a)/np.maximum(sil_a, sil_b)
    return sil_i, sil_a, sil_b, contour_cluster_rels


#####################
# Depth computation #
#####################
# Relative depth of a point ReD(c_i): D^w(c_i) - D^b(c_i)
#   with D^w(c_i) = ID(c_i|other members of c_i's cluster)
#   and  D^b(c_i) = min_{other clusters besides the one c_i's in} ID(c_i|other members in said cluster)

# first we compute d_w
# for ID, depth is not defined for clusters of size N>1 so we return 0
# for CBD, depth is not defined for clusters of size N>2 so we return 0
def compute_red_within(masks, clustering, n_components, 
                       depth_notion="id", use_modified=True, use_fast=True, 
                       inclusion_mat=None, precompute_ins=None, precompute_outs=None):
    num_contours = len(masks)
    clustering_ids = np.arange(n_components)
    depth_w = np.zeros(num_contours)
    for cluster_id in clustering_ids:
        contour_ids = np.where(clustering == cluster_id)[0]
        mask_subset = [masks[i] for i in contour_ids]
        if (depth_notion == "cbd" and contour_ids.size <= 2) or (depth_notion == "id" and contour_ids.size <= 1):                
            depths = np.zeros_like(contour_ids.size)
        else:
            if use_modified and use_fast:
                assert depth_notion == "id"  # only supported for depth_notion == "id"
                precompute_in = precompute_ins[cluster_id] if precompute_ins is not None and cluster_id in precompute_ins else None
                precompute_out = precompute_outs[cluster_id] if precompute_outs is not None and cluster_id in precompute_outs else None
                depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, precompute_in=precompute_in, precompute_out=precompute_out)
            else:
                inclusion_mat_subset = None
                if inclusion_mat is not None:
                    inclusion_mat_subset = inclusion_mat[np.ix_(contour_ids, contour_ids)]
                    
                if depth_notion == "id":
                    depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
                elif depth_notion == "cbd":
                    depths = band_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
                else:
                    raise ValueError("Unsupported depth notion (only id and cbd supported)")
        depth_w[contour_ids] = depths
    return depth_w

# then we compute d_b
def compute_red_between(masks, clustering, n_components, competing_clusters=None, 
                        depth_notion="id", use_modified=True, use_fast=True, 
                        inclusion_mat=None, precompute_ins=None, precompute_outs=None):
    # if you pass competing clusters, then the same are outputted
    # if you dont pass competing clusters then clusters that maximize depth are outputted
    num_contours = len(masks)
    clustering_ids = np.arange(n_components)
    depth_b_cluster = np.empty((num_contours, n_components), dtype=float)    
    depth_delta_cluster = np.empty((num_contours, n_components), dtype=float)    
    for cluster_id1 in clustering_ids:
        contour_ids_1 = np.where(clustering == cluster_id1)[0]
        for contour_id in contour_ids_1:
            depth_b_cluster[contour_id, cluster_id1] = -np.inf
            depth_delta_cluster[contour_id, cluster_id1] = -np.inf
            if competing_clusters is not None:  # we just want the depth of the competing cluster
                competing_cids = [competing_clusters[contour_id], ]
                other_cids = np.setdiff1d(np.setdiff1d(cluster_id1, cluster_id1), competing_clusters[contour_id])
                for ocid in other_cids:
                    depth_b_cluster[contour_id, ocid] = -np.inf
                    depth_delta_cluster[contour_id, ocid] = -np.inf
            else:
                competing_cids = np.setdiff1d(clustering_ids, np.array([cluster_id1])) # all other clusters
            
            for competing_cid in competing_cids:
                contour_ids_2 = np.where(clustering == competing_cid)[0].tolist()
                contour_ids_2.append(contour_id)
                mask_subset = [masks[i] for i in contour_ids_2]
                if (depth_notion == "cbd" and len(contour_ids_2) <= 2) or (depth_notion == "id" and len(contour_ids_2) <= 1):                
                    depths = np.zeros_like(len(contour_ids_2))
                else:
                    if use_modified and use_fast:
                        assert depth_notion == "id"  # only supported for depth_notion == "id"
                        precompute_in = precompute_ins[competing_cid] if precompute_ins is not None and competing_cid in precompute_ins else None
                        precompute_out = precompute_outs[competing_cid] if precompute_outs is not None and competing_cid in precompute_outs else None
                        if precompute_in is not None:
                            precompute_in += 1 - contour_id
                        if precompute_out is not None:
                            precompute_out += contour_id/contour_id.sum()
                        depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, precompute_in=precompute_in, precompute_out=precompute_out) 
                    else:
                        inclusion_mat_subset = None
                        if inclusion_mat is not None:
                            inclusion_mat_subset = inclusion_mat[np.ix_(contour_ids_2, contour_ids_2)]
                            
                        if depth_notion == "id":
                            depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
                        elif depth_notion == "cbd":
                            depths = band_depths(mask_subset, modified=use_modified, fast=use_fast, inclusion_mat=inclusion_mat_subset)
                        else:
                            raise ValueError("Unsupported depth notion (only id and cbd supported)")
                    
                depth_b_cluster[contour_id, competing_cid] = depths[-1]  # (from ddclust paper): we only want the depth of the last contour we appended
                depth_delta_cluster[contour_id, competing_cid] = depths.mean()  # we also want to keep track to the effect of adding a contour to a cluster
    
    if competing_clusters is None:
        competing_clusters = np.argmax(depth_b_cluster, axis=1) # the competing cluster is the one with the highest depth
        # competing_clusters = np.argmax(depth_delta_cluster, axis=1) # the competing cluster is the one with the highest depth
        # competing_clusters = np.argmax(depth_b_cluster - depth_delta_cluster, axis=1) # the competing cluster is the one with the highest depth
        # from scipy.stats import rankdata
        # ranks = rankdata(depth_b_cluster, method='min', axis=1) - 1
        # ranks = ranks[:, 1:][:, ::-1]  # remove -np.inf column and flip so it is descending
        # is_tie = []
        # for r in ranks:
        #     if np.where(r == r[0])[0].size > 1:
        #         is_tie.append(True)
        #     else:
        #         is_tie.append(False)
        # print(np.any(is_tie == True))

    depth_b = np.array([depth_b_cluster[i, competing_cid] for i, competing_cid in enumerate(competing_clusters)]) # the competing cluster is the one with the highest depth

    return depth_b, competing_clusters

def compute_red(masks, clustering, n_components, competing_clusters=None, depth_notion="id", use_modified=True, use_fast=True, inclusion_mat=None):
    red_w = compute_red_within(masks, clustering, n_components, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, inclusion_mat=inclusion_mat)
    red_b, competing_clusters = compute_red_between(masks, clustering, n_components, competing_clusters, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, inclusion_mat=inclusion_mat)
    red_i = red_w - red_b
    return red_i, red_w, red_b, competing_clusters


########
# Cost #
########

def compute_cost(sils, reds, weight=0.5):
    return (1 - weight) * sils + weight * reds