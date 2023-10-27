
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, "..")
    from synthetic_data import magnitude_modes, three_rings, shape_families    

    num_masks=15
    num_rows = num_cols = 512
    # masks = magnitude_modes(num_masks, num_rows, num_cols, seed=2)
    # masks = three_rings(num_masks, num_rows, num_cols, seed=2)
    masks = shape_families(num_masks, num_rows, num_cols, seed=2)   

    # print(1 - (masks[0] * (1 - masks[1])).sum()/masks[0].sum())  # should be 0
    # print(1 - (masks[1] * (1 - masks[0])).sum()/masks[1].sum())
    # print((masks[0] * masks[1]).sum()/masks[0].sum())  # should be 0
    # print((masks[1] * masks[0]).sum()/masks[1].sum())
    # plt.imshow(masks[0]*3 + masks[1])
    # plt.show()    

    mat1 = compute_inclusion_matrix(masks)
    mat2 = compute_epsilon_inclusion_matrix(masks)
    print(mat1[0, :].flatten())
    print(mat1[:, 0].flatten())
    print(mat1[0, :] + mat1[:, 0])
    print(mat2[0, :] + mat2[:, 0])
    #print(np.corrcoef(mat1.flatten(), mat2.flatten()))

    print(mat1.shape)

    fig, axs = plt.subplots(ncols=3)
    # axs[0].matshow(mat1 + mat1.T)
    # axs[1].matshow(mat2 + mat2.T)
    axs[0].matshow(mat1)
    axs[1].matshow(mat2)
    # axs[1].matshow(np.argsort(np.concatenate([mat2[np.newaxis], mat2.T[np.newaxis]], axis=0), axis=0)[0])
    axs[2].plot(np.arange(num_masks), mat1[:, 0] + mat1[0, :], label="strict")
    axs[2].plot(np.arange(num_masks), (mat2[:, 0] + mat2[0, :])/2, label="epsilon")
    plt.legend()
    plt.show()