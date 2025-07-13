# (c) 2025 Anonymous.

import numpy as np

def unbiased_binning(data, k, x=0, groups=1):
    """
    Given as dataset, a value k, and an atttribute x, it creates k bins that are unbiased

    Parameters:
    data (array-like): Input data to be binned.
    k (int): Number of bins to create.
    x (int): The index of the attribute to be binned.
    groups (int): The index of the attribute used for grouping.

    Returns:
    tuple: Bin edges and binned data.
    """
    # sort the data based on the specified attribute
    sorted_data = data[data[:, x].argsort()]
    n = len(sorted_data)
    if n == 0:
        return [-1] # no data to bin
    infty = n+100 # I know the  maximum value of the attribute x is n, so this is a safe value for infinity
    T = boundary_candidates(sorted_data, groups)
    m = len(T)  # number of boundary candidates
    # print('T,m',T,m)
    if m <k:
        return [-2],[-2],m # infeasible: no boundary candidates found
    
    # initialize the matrix M (m by k), with values ((0,maxint),-1)
    M = np.zeros((m, k, 3), dtype=int)
    for j in range(m): # for 1 bucket, k=0
        M[j][0][0] = T[j] # w_down
        M[j][0][1] = T[j] # w_up

    # filling the matrix M
    for kappa in range(1, k): # for 2 to k buckets
        for j in range(kappa): # invalid cases
            M[j][kappa][1] = infty # w_up = invalid
        for j in range(kappa, m):
            w_down = 0; w_up = infty; index = -1
            for i in range(j-1):
                w_down_i = min(M[i][kappa-1][0],T[j]-T[i])
                w_up_i = max(M[i][kappa-1][1],T[j]-T[i])
                if (w_up_i - w_down_i) < (w_up - w_down):
                    w_down = w_down_i
                    w_up = w_up_i
                    index = i
            M[j, kappa][0] = w_down
            M[j, kappa][1] = w_up
            M[j, kappa][2] = index
            
    return Traceback(M,T,sorted_data,x)


def boundary_candidates(data, groups):   
    """
    Given a dataset, it returns the boundary candidates for binning, based on the ratios on the attributes groups

    Parameters:
    data (array-like): Input data (sorted on x) to find boundary candidates.
    groups (int): the index of the attribute used for grouping.

    Returns:
    T: List of boundary candidates.
    """
    n= len(data)
    if n == 0:
        return []
    # ell is the number of distinct groups in the data
    ell = len(set(row[groups] for row in data))
    # First pass: compute the  group counts for each index of the data
    group_counts = np.zeros((n,ell-1),dtype=int)
    counts = np.zeros(ell-1,dtype=int)
    for i in range(n):
        # increment the count of the group at index i
        group_index = data[i][groups]
        if group_index < ell - 1:
            counts[group_index] += 1
        group_counts[i] = counts.copy()
    
    # Second pass: find the boundary candidates based on the ratios
    T = []
    for i in range(n):
        flag = True
        for l in range(ell - 1):
            # if group_counts[i][l] != counts[l]*(i+1)/n:
            if int(group_counts[i][l]*n/(i+1)) != counts[l]: # relax the assumption to make it work in practice
                flag = False
                break
        if flag: 
            if len(T)==0 or T[-1] != (i-1): T.append(int(i)) # relax the assumption to make it work in practice
            # T.append(int(i))
    return T


def Traceback(M,T,sorted_data,x):
    """
    Given the 3-dimensional array M, it returns the bin edges and binned data

    Parameters:
    M (np.ndarray): The 3-dimensional array containing binning information.
    T (list): The boundary candidates.
    sorted_data (array-like): The sorted data based on the attribute x.
    x (int): The index of the attribute to be binned.

    Returns:
    tuple: Bin edges and binned data.
    """
    m = M.shape[0]
    k = M.shape[1]
    j=m-1; # recall the incides in M start from 0, hence we need to subtract 1 from the original values
    S = [] # initialize S 
    selectedIndices = [T[j]] # to store the indices of the selected boundary candidates
    for kappa in range(k-1,0,-1):
        i = M[j][kappa][2] # index of the previous boundary candidate
        S.insert(0, sorted_data[T[i]][x]) # add T[i][x] to the beginning of S
        selectedIndices.insert(0, T[i])
        j = i; kappa -=1
    return S, selectedIndices, m


''' Unit Test'''
'''
import os
x = np.array([i+1 for i in range(16)])
g = np.array([1,1,0,1,0,0,1,0,1,1,0,0,0,1,1,0])
os.system('cls' if os.name == 'nt' else 'clear')
# print(g.transpose())
# stack x and g as a 2D array D
D = np.column_stack((x, g))
# print(D)
# T = boundary_candidates(D, 1)
# print(T) # [5, 7, 11, 13, 15] works fine for complete unbiased binning -- think about practice later
results = unbiased_binning(D,k=3,x=0,groups=1)
print(results)
'''

'''
import os
from MyData import gen_data, load_xg_dataset
os.system('cls' if os.name == 'nt' else 'clear')
D = load_xg_dataset('datasets/compas-scores-raw.csv', x_idx=22, g_idx=8) # x: RawScore, g: Race (White,Black,Hispanic,Other)
results = unbiased_binning(D,k=5,x=0,groups=1)
print(results)
'''
