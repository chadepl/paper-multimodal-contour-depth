"""
This experiment shows that the fast band depth is wrong
"""

if __name__ == "__main__":
    from math import comb
    import numpy as np
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt

    # model
    # Y = X_i(t) + c_i \sigma_i s
    # X_i(t) = g(t) + e_i(t)
    # g(t) = 4t

    num_curves = 100
    num_points = 50
    domain = np.linspace(0, 1, num_points)

    cs = (np.random.random(num_curves) > 0.9).astype(float)
    s = 6
    sigmas = (np.random.random(num_curves) > 0.5).astype(float) * 2 - 1

    rhs = np.repeat(cs.reshape(num_curves, 1), num_points, axis=1)
    rhs *= s
    rhs *= np.repeat(sigmas.reshape(num_curves, 1), num_points, axis=1)

    gts = 4 * domain

    gp_mean = np.zeros(num_points)
    gp_cov = cdist(domain.reshape(-1, 1), domain.reshape(-1, 1)) 
    gp_cov = 1 / np.exp(gp_cov)

    gp = np.random.multivariate_normal(gp_mean, gp_cov, num_curves)

    print(gp.shape)

    Y = gts + gp + rhs

    import pandas as pd
    pd.DataFrame(Y).to_csv("/Users/chadepl/Downloads/curves.csv", index=False, header=False)

    for line in Y:
        plt.plot(line)
    #plt.show()
    plt.savefig("/Users/chadepl/Downloads/curves.png")

    def n2_banddepth(curves, above_below_counts=True):
        num_curves = curves.shape[0]

        if above_below_counts:
            num_above = np.zeros(curves.shape[0])
            num_below = np.zeros(curves.shape[0])
            for j in range(num_curves):
                for i in range(num_curves):
                    if i != j:
                        num_below[j] += np.all(curves[i] <= curves[j]).astype(float)
                        num_above[j] += np.all(curves[i] >= curves[j]).astype(float)
            depth = (num_above * num_below + (num_curves - 1))/comb(num_curves, 2)
        else:
            contains = np.zeros(num_curves)
            for j in range(num_curves):
                for i in range(j + 1, num_curves):
                    for k in range(num_curves):
                        if (k!= i or k != j):                                                    
                            curves_subset = curves[[i, j, k],:]
                            ranks = np.argsort(curves_subset, axis=0)
                            if np.all(ranks[-1] == 1):
                                contains[k] += 1
            depth = (contains + num_curves - 1)/comb(num_curves, 2)

        return depth

            
                
    from statsmodels.graphics.functional import banddepth
    from time import time

    t_tick = time()
    sbd1 = n2_banddepth(Y, above_below_counts=True)
    print(f"sbd1 time: {time() - t_tick} secs")
    
    t_tick = time()
    sbd2 = n2_banddepth(Y, above_below_counts=False)
    print(f"sbd2 time: {time() - t_tick} secs")
    
    t_tick = time()
    fbd = banddepth(data=Y, method="BD2")
    print(f"fbd time: {time() - t_tick} secs")
    
    print(sbd1[:10])
    print(sbd2[:10])
    print(fbd[:10])

    # plt.plot(sbd[:10])
    # plt.plot(fbd[:10])
    # plt.show()

    print(f"Medians: {np.argsort(sbd1)[-1]} vs {np.argsort(sbd2)[-1]} vs {np.argsort(fbd)[-1]}")
    print(f"Error: {np.square(sbd1 - sbd2).mean()} and {np.square(sbd1 - fbd).mean()}")
    