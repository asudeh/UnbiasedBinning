from itertools import product
import numpy as np
from math import ceil, floor
from MaxGainBinning import __max_gain_boundary

infty = None

# this is the LS (local search) algorithm for e-biased binning, as described in the paper. The DnC (divide and conquer) algorithm is implemented in the "_boundary" function.
def ebias_dnc(data, k, x=0, groups=1, eps=0.03, fast=False, binningMethod=1, y=2):
    '''
    This is the Local Search Algorithm (not the DnC). The DnC (in the paper) is the "_boundary" function.
    Ebiased-binning algorithm based on Divide and Conquer approach.
    
    Parameters:
    data (array-like): Input data to be binned.
    k (int): Number of bins to create.
    x (int): The index of the attribute to be binned.
    groups (int): The index of the attribute used for grouping.
    eps (float): The bias parameter for the bins.
    binningMethod (int): 1 for equal-size binning, 2 for equal-width binning, 3 for entropy_based binning.
    y: the index of the target attribute (not used in this function unless the binning method is 3).

    Returns:
    tuple: Bin edges and binned data.
    '''
    # sort the data based on the specified attribute
    global infty
    sorted_data = data[data[:, x].argsort()]
    n = len(sorted_data)
    if n == 0:
        return [-1],infty  # no data to bin
    
    infty = n + 100
    
    ell = len(np.unique(sorted_data[:, groups]))  # number of groups
    totalcounts = np.zeros(ell)
    counts = np.zeros((n+1,ell)) # start the counts from index 1 for convenience
    for i in range(1,n+1):
        totalcounts[data[i-1][groups]] +=1
        counts[i] = totalcounts.copy()
    ratios = totalcounts / n


    if binningMethod == 1:
        boundaries = _boundary(0, n, k, ratios, counts, eps) # call the D&C function for equal size binning (default)
    else:
        x_vals = sorted_data[:, x]
        y_vals = sorted_data[:, y]
        boundaries = _boundary_others(0, n, k, ratios, counts, eps, method=binningMethod, x_vals=x_vals, y_vals=y_vals) # call the D&C function for other binning methods



    if boundaries == [-1]: return [-1],infty # no valid boundaries found

    boundaries = [0] + boundaries + [n]  # add the start and end boundaries
    obj = _obj(boundaries)

    if fast: return boundaries, obj  # return boundaries if fast mode is enabled
    # print('initial boundaries:', boundaries)

    half_w = ceil(obj/2)
    boundary_base = [max(0,((i+1)*n//k)-half_w) for i in range(k-1)] # base boundaries for checking
    tmp_boundary = [0] + boundary_base + [n]
    # generate all possible combinations [i_1, i_2, ..., i_{k-1}] where i_j is in range [0, w]
    for comb in product(range(obj), repeat=k-1):
        for i in range(k-1): tmp_boundary[i+1] = boundary_base[i] + comb[i]
        if tmp_boundary[k-1] >=n : continue  # skip if the last boundary exceeds n
        if _ebiasedBinning(tmp_boundary, ratios, counts, eps):
            obj_tmp = _obj(tmp_boundary)
            if obj_tmp < obj:
                obj = obj_tmp
                boundaries = tmp_boundary.copy()

    return boundaries , obj

def _boundary(l,h,k, ratios, counts, eps): # D&C function with equal size bins
    """
    This is the DnC function in the paper
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

def _boundary_others(l,h,k, ratios, counts, eps, method=2, x_vals=None, y_vals=None): # D&C function without equal size bins
    """
    This is an alternative DnC function that does not enforce equal size bins.
    Helper function to find the boundary for the divide and conquer approach.
    
    Parameters:
    l (int): Lower index.
    h (int): Upper index.
    k (int): Number of bins.
    ratios (array-like): Ratios of counts for each group.
    counts (array-like): group-counts of index i.
    eps (float): Bias parameter.
    method (int): Binning method (2: equal-width, 3: entropy-based).
    x_vals (array-like): Values of the attribute to be binned.
    y_vals (array-like): Values of the target attribute (to be used in entropy-based binning).

    Returns:
    array [int]: Boundary indices or [-1] if no valid boundary is found.
    """
    if k == 1: return [] # boundary condition
    if h<=l+1: return [-1]  # no valid boundary

    if method == 2:  # equal-width binning
        midpoint = (x_vals[h-1] + x_vals[l])/2
        b = __binary_search(x_vals, l, h, midpoint)
    else:  # entropy-based binning (method == 3)
        b = __max_gain_boundary(l, h, y_vals)
    
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
        S1 = _boundary_others(l, b, ceil(k/2),ratios, counts,eps,method, x_vals, y_vals)
        S2 = _boundary_others(b, h, floor(k/2),ratios, counts,eps,method, x_vals, y_vals)
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

def _obj(boundaries,method=1):
    global infty
    w_up = 0; w_down = infty
    for i in range(len(boundaries)-1):
        if boundaries[i+1]-boundaries[i] < w_down:
            w_down = boundaries[i+1] - boundaries[i]
        if boundaries[i+1]-boundaries[i] > w_up:
            w_up = boundaries[i+1] - boundaries[i]
    return w_up - w_down

def _ebiasedBinning(boundaries, ratios, counts, eps):
    for i in range(len(boundaries)-1):
        for l in range(len(ratios)):
            if abs(((counts[boundaries[i+1]][l]-counts[boundaries[i]][l])*1.0 / (boundaries[i+1] - boundaries[i])) - ratios[l])>eps:
                return False
    return True # TBD!

def __binary_search(x_vals, left, right, midpoint):
    """Helper function used for equal-width binning to find the index closest to midpoint."""
    while left < right:
        mid = (left + right) // 2
        if x_vals[mid] < midpoint:
            left = mid + 1
        else:
            right = mid
    return left

