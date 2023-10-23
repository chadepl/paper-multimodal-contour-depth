
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from src.contour_depth.depth.inclusion_depth import compute_depths as inclusion_depths
from src.contour_depth.depth.band_depth import compute_depths as band_depths
from ..depth.utils import get_masks_matrix, get_sdfs


def ddclust(masks, init_labs, 
            feat_mat=None, pre_pca=False, 
            cost_lamb = 0.8, beta_init = np.inf, no_prog_its=5, max_iter=10,
            E_size=10, swap_subset_max_size=5,
            depth_notion="id", use_modified=False, use_fast=True, output_extra_info=True):
    
    # Setup algorithm parameters and preprocessing

    S_MAX = E_size  # role of T in the original algorithm
    E_MAX = swap_subset_max_size
    M = no_prog_its
    beta = beta_init

    mat = feat_mat
    if mat is None:
        sdfs = get_sdfs(masks)
        mat = get_masks_matrix(sdfs)

    if pre_pca:
        pca_embedder = PCA(n_components=30)
        mat = pca_embedder.fit_transform(mat)

    total_its = 0
    its_no_change = 0

    should_finish = False

    pred_labs = init_labs.copy()

    # Clustering loop

    for it in range(max_iter):

        # - Compute multivariate medians, sil_i and red_i
        sil_i, sil_a, sil_b, competing_clusters = compute_sil(mat, pred_labs)
        red_i, red_w, red_b, medians = compute_red(masks, pred_labs, competing_clusters, 
                                                   depth_notion=depth_notion, 
                                                   use_modified=use_modified,
                                                   use_fast=use_fast)

        # - Compute partition value C(I_1^K)
        cost_i = compute_cost(sil_i, red_i, weight=cost_lamb)
        partition_cost = cost_i.mean()

        # - Identify observations below acceptance threshold T, take subset S
        S = np.argsort(cost_i)[:S_MAX]  # for now ten observations with lower values
        
        while S.size > 0:

            E_size = int(np.random.randint(1, E_MAX, 1)[0])
            if E_size > S.size:
                E_size = S.size

            # - Take a random subset E of S
            E = np.random.choice(S, E_size, replace=False)

            # - Define new partition \tilde I_1^K with points relocated to competing cluster
            tentative_pred_labs = pred_labs.copy()
            tentative_pred_labs[E] = competing_clusters[E]

            # - Compute quantities for \tilde I_1^K, this yields C(\tilde I_1^K)
            new_sil_i, new_sil_a, new_sil_b, new_competing_clusters = compute_sil(mat, tentative_pred_labs)
            new_red_i, new_red_w, new_red_b, new_medians = compute_red(masks, tentative_pred_labs, new_competing_clusters, 
                                                                       depth_notion=depth_notion, 
                                                                       use_modified=use_modified,
                                                                       use_fast=use_fast)
            new_cost_i = compute_cost(new_sil_i, new_red_i, weight=cost_lamb)
            new_partition_cost = new_cost_i.mean()

            # - Decide whether to accept or not the partition. Criteria to accept:
            # -- if cost(new_partion) > cost(partition)
            # -- if cost(new_partion) <= cost(partition) with Pr(beta, delta cost)
            print(partition_cost, new_partition_cost)
            #print(S, E)

            if new_partition_cost > partition_cost:
                its_no_change = 0

                pred_labs = tentative_pred_labs
                sil_i, sil_a, sil_b, competing_clusters = new_sil_i, new_sil_a, new_sil_b, new_competing_clusters
                red_i, red_w, red_b, medians = new_red_i, new_red_w, new_red_b, new_medians
                cost_i = new_cost_i
                partition_cost = new_partition_cost

                print("Updated partition!")                    
            else:
                its_no_change += 1

            # 9. If no moves have been accepted in the past M iterations, finish         
            if its_no_change > M:
                if beta < np.inf:
                    beta = np.inf
                else:
                    print("finish")
                    should_finish = True
                    break
            else:
                beta = 2 * beta

            # 10. Update set S
            S = np.setdiff1d(S, E)

            total_its += 1

        if should_finish:
            break

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
def compute_sil_within(contours_mat, clustering):
    num_contours = contours_mat.shape[0]
    clustering_ids = np.unique(clustering)
    sil_a = np.zeros(num_contours)
    for cluster_id in clustering_ids:
        contour_ids = np.where(clustering == cluster_id)[0]
        dmat = cdist(contours_mat[contour_ids, :], contours_mat[contour_ids, :], metric="sqeuclidean")
        mean_dists = dmat.mean(axis=1)
        sil_a[contour_ids] = mean_dists
    return sil_a

# then we compute sil_b
def compute_sil_between(contours_mat, clustering):
    num_contours = contours_mat.shape[0]
    clustering_ids = np.unique(clustering)
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

def compute_sil(contours_mat, clustering):
    sil_a = compute_sil_within(contours_mat, clustering)
    sil_b, competing_clusters = compute_sil_between(contours_mat, clustering)
    sil_i = (sil_b - sil_a)/np.maximum(sil_a, sil_b)
    return sil_i, sil_a, sil_b, competing_clusters


#####################
# Depth computation #
#####################
# Relative depth of a point ReD(c_i): D^w(c_i) - D^b(c_i)
#   with D^w(c_i) = ID(c_i|other members of c_i's cluster)
#   and  D^b(c_i) = min_{other clusters besides the one c_i's in} ID(c_i|other members in said cluster)

# first we compute d_w
def compute_red_within(masks, clustering, depth_notion="id", use_modified=True, use_fast=True):
    num_contours = len(masks)
    clustering_ids = np.unique(clustering)
    depth_w = np.zeros(num_contours)
    medians = np.empty(clustering_ids.size, int)
    for cluster_id in clustering_ids:
        contour_ids = np.where(clustering == cluster_id)[0]
        mask_subset = [masks[i] for i in contour_ids]
        if depth_notion == "id":
            depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast)
        elif depth_notion == "cbd":
            depths = band_depths(mask_subset, modified=use_modified, fast=use_fast)
        else:
            raise ValueError("Unsupported depth notion (only id and cbd supported)")
        depth_w[contour_ids] = depths
        median = np.argsort(depths)[-1]
        median = contour_ids[median]
        medians[cluster_id] = median
    return depth_w, medians

# then we compute d_b
def compute_red_between(masks, clustering, competing_clusters, depth_notion="id", use_modified=True, use_fast=True):
    num_contours = len(masks)
    clustering_ids = np.unique(clustering)
    depth_b = np.zeros(num_contours)
    for cluster_id1 in clustering_ids:
        contour_ids_1 = np.where(clustering == cluster_id1)[0]
        for contour_id in contour_ids_1:
            competing_cid = competing_clusters[contour_id]
            contour_ids_2 = np.where(clustering == competing_cid)[0]
            mask_subset = [masks[i] for i in contour_ids_2]
            mask_subset.append(masks[contour_id])
            if depth_notion == "id":
                depths = inclusion_depths(mask_subset, modified=use_modified, fast=use_fast)
            elif depth_notion == "cbd":
                depths = band_depths(mask_subset, modified=use_modified, fast=use_fast)
            else:
                raise ValueError("Unsupported depth notion (only id and cbd supported)")
            depth_b[contour_id] = depths[-1]  # we only want the depth of the last contour we appended
    return depth_b

def compute_red(masks, clustering, competing_clusters, depth_notion="id", use_modified=True, use_fast=True):
    red_w, medians = compute_red_within(masks, clustering, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast)
    red_b = compute_red_between(masks, clustering, competing_clusters, depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast)
    red_i = red_w - red_b
    return red_i, red_w, red_b, medians


########
# Cost #
########

def compute_cost(sils, reds, weight=0.5):
    return (1 - weight) * sils + weight * reds
