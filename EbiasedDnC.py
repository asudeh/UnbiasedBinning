# (c) 2025 Anonymous.

from itertools import product
import numpy as np
from math import ceil, floor

infty = None

def ebias_dnc(data, k, x=0, groups=1, eps=0.03, fast=False):
    '''
    Ebiased-binning algorithm based on Divide and Conquer approach.
    
    Parameters:
    data (array-like): Input data to be binned.
    k (int): Number of bins to create.
    x (int): The index of the attribute to be binned.
    groups (int): The index of the attribute used for grouping.
    eps (float): The bias parameter for the bins.

    Returns:
    tuple: Bin edges and binned data.
    '''
    # sort the data based on the specified attribute
    global infty
    sorted_data = data[data[:, x].argsort()]
    n = len(sorted_data)
    if n == 0:
        return [-1]  # no data to bin
    
    infty = n + 100
    
    ell = len(np.unique(sorted_data[:, groups]))  # number of groups
    totalcounts = np.zeros(ell)
    counts = np.zeros((n+1,ell)) # start the counts from index 1 for convenience
    for i in range(1,n+1):
        totalcounts[data[i-1][groups]] +=1
        counts[i] = totalcounts.copy()
    ratios = totalcounts / n

    boundaries = _boundary(0, n, k, ratios, counts, eps)
    if boundaries == [-1]: return [-1] # no valid boundaries found

    boundaries = [0] + boundaries + [n]  # add the start and end boundaries

    if fast: return boundaries  # return boundaries if fast mode is enabled
    # print('initial boundaries:', boundaries)

    w_down, w_up = _buckets_maxmin(boundaries)
    half_w = ceil((w_up-w_down)/2)
    boundary_base = [max(0,((i+1)*n//k)-half_w) for i in range(k-1)] # base boundaries for checking
    tmp_boundary = [0] + boundary_base + [n]
    # generate all possible combinations [i_1, i_2, ..., i_{k-1}] where i_j is in range [0, w]
    for comb in product(range(w_up-w_down), repeat=k-1):
        for i in range(k-1): tmp_boundary[i+1] = boundary_base[i] + comb[i]
        if tmp_boundary[k-1] >=n : continue  # skip if the last boundary exceeds n
        if _ebiasedBinning(tmp_boundary, ratios, counts, eps):
            wp_down, wp_up = _buckets_maxmin(tmp_boundary)
            if wp_down - wp_up < w_up - w_down:
                w_down, w_up = wp_down, wp_up
                boundaries = tmp_boundary.copy()

    return boundaries , w_down, w_up

def _boundary(l,h,k, ratios, counts, eps):
    """
    Helper function to find the boundary for the divide and conquer approach.
    
    Parameters:
    l (int): Lower index.
    h (int): Upper index.
    k (int): Number of bins.
    ratios (array-like): Ratios of counts for each group.
    counts (array-like): group-counts of index i.
    eps (float): Bias parameter.

    Returns:
    array [int]: Boundary indices or [-1] if no valid boundary is found.
    """
    if k == 1: return [] # boundary condition
    b = (l+h)//2 if k % 2 == 0 else l + ((h-l)//k)*ceil(k/2)
    if _ebiased(l,b,h,ratios, counts,eps): 
        found = True
    else: 
        found = False
    i=1
    while i<min(b-l,h-b) and (not found):
        if _ebiased(l,b+i,h,ratios, counts,eps):
            b+=i; found = True; break
        # print('tested',b+i)
        if _ebiased(l,b-i,h,ratios, counts,eps):
            b-=i; found = True; break
        # print('tested',b-i)
        i+=1
    if found:
        S1 = _boundary(l, b, ceil(k/2),ratios, counts,eps)
        S2 = _boundary(b, h, floor(k/2),ratios, counts,eps)
        if S1 != [-1] and S2 != [-1]: return S1 + [b] + S2
    return [-1]  # no valid boundary found

def _ebiased(l,b,h,ratios, counts, eps):
    """ Helper function to check if the two (super-)buckets specified by l,b,h satisfy the eps-biased requirement or not. 
    Parameters: 
        l (int): Lower index of the first bucket.
        b (int): Upper index of the first bucket (also lower index of the second bucket).
        h (int): Upper index of the second bucket.
        eps (float): Bias parameter.
    Returns:   boolean: True if the buckets satisfy the eps-biased requirement, False otherwise.
    """
    for i in range(len(ratios)): # for all groups
        if abs(((counts[b][i]-counts[l][i])*1.0 / (b - l)) - ratios[i])>eps: return False
        if abs(((counts[h][i]-counts[b][i])*1.0 / (h - b)) - ratios[i])>eps: return False
    return True # TBD!

def _buckets_maxmin(boundaries):
    global infty
    w_up = 0; w_down = infty
    for i in range(len(boundaries)-1):
        if boundaries[i+1]-boundaries[i] < w_down:
            w_down = boundaries[i+1] - boundaries[i]
        if boundaries[i+1]-boundaries[i] > w_up:
            w_up = boundaries[i+1] - boundaries[i]
    return w_down,w_up

def _ebiasedBinning(boundaries, ratios, counts, eps):
    for i in range(len(boundaries)-1):
        for l in range(len(ratios)):
            if abs(((counts[boundaries[i+1]][l]-counts[boundaries[i]][l])*1.0 / (boundaries[i+1] - boundaries[i])) - ratios[l])>eps:
                return False
    return True # TBD!


'''Unit Test'''
'''
import os
x = np.array([i+1 for i in range(16)])
g = np.array([1,1,0,1,0,0,1,0,1,1,0,0,0,1,1,0])
os.system('cls' if os.name == 'nt' else 'clear')
D = np.column_stack((x, g))
results = ebias_dnc(D,k=3,fast=False)
print(results)
'''