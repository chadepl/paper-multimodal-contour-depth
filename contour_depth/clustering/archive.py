def cdclust_simple(masks, num_clusters, num_attempts=5, max_num_iterations=10,
                   beta_init = np.inf, beta_mult=2,  
                   depth_notion="id", use_modified=False, use_fast=True, output_extra_info=False, verbose=False, seed=42):
    
    if verbose:
        print("[cdclust] Initializing ...")
    num_masks = len(masks)

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)    

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

    best_red = -np.inf
    best_cluster_assignment = None

    for _ in range(num_attempts):
        init_labs = rng.integers(0, num_clusters, num_masks)
        pred_labs = init_labs.copy()
        beta = beta_init
        beta_mult = 2 # used to increase the beta
        cluster_sizes = np.array([np.where(init_labs == i)[0].size for i in range(num_clusters)])

        # - Setup precomputed fields per partition
        if use_fast:
            assert depth_notion == "id"
            precomputed_ins = {k:get_precompute_in([masks[i] for i in np.where(pred_labs == k)[0]]) for k in range(num_clusters)}
            precomputed_outs = {k:get_precompute_out([masks[i] for i in np.where(pred_labs == k)[0]]) for k in range(num_clusters)}

        # - Compute multivariate medians, sil_i and red_i
        red_wi = compute_red_within(masks, pred_labs, num_clusters, 
                                    depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, 
                                    inclusion_mat=inclusion_mat, precompute_ins=precomputed_ins, precompute_outs=precomputed_outs)
        red_bi, competing_clusters = compute_red_between(masks, pred_labs, num_clusters, competing_clusters=None,
                                    depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, 
                                    inclusion_mat=inclusion_mat, precompute_ins=precomputed_ins, precompute_outs=precomputed_outs)
        red_i = red_wi - red_bi

        # - Compute the cost of the current clustering C(I_1^K)
        clustering_cost = red_i.mean()

        # Clustering loop
        if verbose:
            print("[cdclust] Starting clustering loop ...")
        for it in range(max_num_iterations):
            if verbose:
                print(f"Global iter {it}")                        

            # - Define new tentative clustering with contours relocated to competing cluster
            swap_subset = np.where(red_i<0)[0]
            new_pred_labs = pred_labs.copy()
            new_pred_labs[swap_subset] = competing_clusters[swap_subset]
            cluster_sizes = np.array([np.where(new_pred_labs == i)[0].size for i in range(num_clusters)])

            # print(red_i)
            # print(swap_subset)
            # print(pred_labs)
            # print(competing_clusters)
            #break
                
            accept_clustering = False
            if not np.any(cluster_sizes < 1):  # prevents clusters from getting too small

                # - update precomputed fields per partition
                if use_fast:
                    assert depth_notion == "id"
                    new_precomputed_ins = {k:get_precompute_in([masks[i] for i in np.where(pred_labs == k)[0]]) for k in range(num_clusters)}
                    new_precomputed_outs = {k:get_precompute_out([masks[i] for i in np.where(pred_labs == k)[0]]) for k in range(num_clusters)}
                    for i in swap_subset:
                        pre_lab = pred_labs[i]
                        new_lab = new_pred_labs[i]                    
                        new_precomputed_ins[pre_lab] -= 1 - masks[i]
                        new_precomputed_ins[new_lab] += 1 - masks[i]
                        new_precomputed_outs[pre_lab] -= masks[i]/masks[i].sum()
                        new_precomputed_outs[new_lab] += masks[i]/masks[i].sum()

                # - Compute quantities for tentative clustering
                new_red_wi = compute_red_within(masks, new_pred_labs, num_clusters, 
                                    depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast, 
                                    inclusion_mat=inclusion_mat, precompute_ins=new_precomputed_ins, precompute_outs=new_precomputed_outs)
                new_red_bi, new_competing_clusters = compute_red_between(masks, new_pred_labs, num_clusters, competing_clusters=None,
                                            depth_notion=depth_notion, use_modified=use_modified, use_fast=use_fast,
                                            inclusion_mat=inclusion_mat, precompute_ins=new_precomputed_ins, precompute_outs=new_precomputed_outs)
                new_red_i = new_red_wi - new_red_bi
                new_clustering_cost = new_red_i.mean()

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
                        num_its_no_prog = 0
                        accept_clustering = True
                    else:                    
                        num_its_no_prog += 1
                        accept_clustering = False

                if verbose:
                    print(accept_clustering, prob, clustering_cost, new_clustering_cost)

            if accept_clustering:                
                pred_labs = new_pred_labs
                red_i = new_red_i
                competing_clusters = new_competing_clusters
                clustering_cost = new_clustering_cost
                precomputed_ins = new_precomputed_ins
                precomputed_outs = new_precomputed_outs
                if red_i.mean() > best_red:
                    best_red = red_i.mean()
                    best_cluster_assignment = pred_labs

                if verbose:
                    print("Updated partition!")

            beta *= beta_mult  # TODO: should be inside or outside the for-loop?


    if output_extra_info:
        return best_cluster_assignment, best_red
    return best_cluster_assignment


# TODO: there is an error when one of the labels dissapears of the clustering, which might happen due to poor initialization
# In this case, sil methods will fail because in correct number of competing clusters will be identified.

def cdclust(masks, init_labs, 
            beta_init = np.inf, beta_mult=2, no_prog_its=5, max_iter=100,
            cost_threshold=0, swap_subset_max_size=5,
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
                # new_competing_clusters = get_depth_competing_clusters(new_pred_labs, n_components, new_red_wi, inclusion_mat)
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

                if verbose:
                    print(accept_clustering, prob, clustering_cost, new_clustering_cost)

            if accept_clustering:                
                pred_labs = new_pred_labs
                red_i = new_red_i
                competing_clusters = new_competing_clusters
                clustering_cost = new_clustering_cost
                precomputed_ins = new_precomputed_ins
                precomputed_outs = new_precomputed_outs
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
        return pred_labs, red_i
    return pred_labs


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