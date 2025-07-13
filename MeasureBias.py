# (c) 2025 Anonymous.

import numpy as np

def equibucket_bias(D,k=10,x=0,g=1):
    """
    Measures the bias of equal-size bucketing of the dataset D on attribute x. 
    The biase is mesured based on the attribute g (groups).

    
    Parameters:
    D : np.ndarray
        The dataset, where each row is a data point.
    k : int
        The number of buckets.
    x : int,float, default: the first attribute
        The binning attribute.
    g : int, default: the second attribute
        the group labels.
    
    Returns:
        A tuple containing:
        - the binning bias as the maximum bias for each group.
        - bias: an array of the maximum bias for each group.
    """

    # sort the dataset by the attribute x
    D = D[D[:, x].argsort()]
    n = D.shape[0] 
    ell = len(np.unique(D[:, g])) # set ell as the number of distinct values of g (the number of groups)

    # measure the overall group ratios
    group_ratios = np.zeros(ell)
    for l in range(ell):
        group_ratios[l] = np.sum(D[:, g] == l) / n

    bias = np.zeros(ell) # the max bias for each group
    # for each bucket of n/k elements measure the bias
    for i in range(k):
        # get the bucket
        bucket = D[i * n // k : (i + 1) * n // k]
        # measure the group ratios in the bucket
        bucket_group_ratios = np.zeros(ell)
        for l in range(ell):
            bucket_group_ratios[l] = np.sum(bucket[:, g] == l) / (n // k)
            if abs(bucket_group_ratios[l] - group_ratios[l]) > bias[l]:
                bias[l] = abs(bucket_group_ratios[l] - group_ratios[l])
    return np.max(bias), bias