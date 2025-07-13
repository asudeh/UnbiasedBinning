# (c) 2025 Anonymous.

import numpy as np

def ebias_binning(data, k, x=0, groups=1, eps=0.03):
    """
    Given as dataset, a value k, an atttribute x, and a value eps, it creates k bins that are eps-biased

    Parameters:
    data (array-like): Input data to be binned.
    k (int): Number of bins to create.
    x (int): The index of the attribute to be binned.
    groups (int): The index of the attribute used for grouping.
    eps (float): The bias parameter for the bins.

    Returns:
    tuple: Bin edges and binned data.
    """
    # sort the data based on the specified attribute
    sorted_data = data[data[:, x].argsort()]
    n = len(sorted_data)
    if n == 0:
        return [-1]  # no data to bin
    
    ell = len(np.unique(sorted_data[:, groups]))  # number of groups
    
    totalcounts = np.zeros(ell)
    counts = np.zeros((n+1,ell)) # start the counts from index 1 for convenience
    for i in range(1,n+1):
        totalcounts[data[i-1][groups]] +=1
        counts[i] = totalcounts.copy()
    ratios = totalcounts / n

    # print(data.transpose())
    # print(counts.transpose())

    # initialize the table T as a two dimensional array of size n times (n+1) of zeros of type binary
    T = np.zeros((n, n+1), dtype=int)
    # filling the table T
    # The column 0 of the table T is not used
    # The indexing of T is the same as the paper -- when filling M use look at T[i+1][j+1] for an index i and j
    for i in range(n): 
        for j in range(i + 1, n+1): # the column indices start from 1, not zero here
            flag = True
            for l in range(ell): # for all groups
                if abs(((counts[j][l]-counts[i][l])*1.0 / (j - i)) - ratios[l])>eps:
                    flag = False
                    break
            if flag:
                T[i][j] = 1
    
    infty = n+100 # I know the  maximum value of the attribute x is n, so this is a safe value for infinity

    # initialize the matrix M (n by k), with values ((0,infty),-1)
    M = np.zeros((n, k, 3), dtype=int)
    for j in range(n):
        if T[0][j+1] == 1:
            M[j][0][0] = j+1 # w_down
            M[j][0][1] = j+1 # w_up
        else: # invalid case
            M[j][0][1] = infty # w_up = invalid

    # filling the matrix M
    for kappa in range(1, k): # for 2 to k buckets
        for j in range(kappa): # invalid cases
            M[j][kappa][1] = infty # w_up = invalid
        for j in range(kappa, n):
            w_down = 0; w_up = infty; index = -1
            for i in range(j-1):
                if T[i+1][j+1] == 0: continue # skip the biased bins
                w_down_i = min(M[i][kappa-1][0],j-i)
                w_up_i = max(M[i][kappa-1][1],j-i)
                if (w_up_i - w_down_i) < (w_up - w_down):
                    w_down = w_down_i
                    w_up = w_up_i
                    index = i
            M[j, kappa][0] = w_down
            M[j, kappa][1] = w_up
            M[j, kappa][2] = index
    
    # tracing back the bins
    if M[n-1][k-1][1] == infty:
        return [-2],[-2] # infeasible: no valid binning found
    j=n-1
    S = [] # initialize S 
    selectedIndices = [j] # to store the indices of the selected boundary candidates
    for kappa in range(k-1,0,-1):
        i = M[j][kappa][2] # index of the previous boundary candidate
        S.insert(0, sorted_data[i][x]) # add t_i[x] to the beginning of S
        selectedIndices.insert(0, i)
        j = i; kappa -=1
    return S, selectedIndices


'''
def _test_size_T(data, x=0, groups=1, eps=0.03):
    # this is just a test function to check the size of the table T
    sorted_data = data[data[:, x].argsort()]
    n = len(sorted_data)
    if n == 0:
        return [-1]  # no data to bin
    
    ell = len(np.unique(sorted_data[:, groups]))  # number of groups
    
    totalcounts = np.zeros(ell)
    counts = np.zeros((n+1,ell)) # start the counts from index 1 for convenience
    for i in range(1,n+1):
        totalcounts[data[i-1][groups]] +=1
        counts[i] = totalcounts.copy()
    ratios = totalcounts / n

    NoValid = 0
    j=n-1
    for i in range(n-1): 
        flag = True
        for l in range(ell): # for all groups
            if abs(((counts[j][l]-counts[i][l])*1.0 / (j - i)) - ratios[l])>eps:
                flag = False
                break
        if flag: 
            NoValid += 1
    return NoValid/(n-1)
'''



''' Unit Test
import os
x = np.array([i+1 for i in range(16)])
g = np.array([1,1,0,1,0,0,1,0,1,1,0,0,0,1,1,0])
os.system('cls' if os.name == 'nt' else 'clear')
D = np.column_stack((x, g))
results = ebias_binning(D,k=3)
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
