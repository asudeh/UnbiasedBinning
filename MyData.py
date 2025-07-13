# (c) 2025 Anonymous.

import numpy as np
import pandas as pd

def gen_data(n=1000, m=[1000-50,1000+50], s=[300,300], p=[0.5, 0.5], dist='normal', seed=1):
    """
    Generate a dataset with ell groups based on specified parameters.
    
    Parameters:
    n (int): Number of samples. vary n: [100,1000,10000,100000,1000000]
    m (list): Means for the groups.
    s (list): Standard deviations for the groups. vary number of groups [0.5,0.5], [1/3,1/3,1/3] between: [2,5]
    p (list): Probabilities for each group. [p,1-p]:0.1,...,0.5 
    distribution (str): Distribution type ('normal' or 'uniform').
    seed (int): Random seed for reproducibility.
    
    Returns:
    np.ndarray: Generated dataset with shape (n, 2).
    The first column is the attribute x to be bucketizes; column 2 is the group labels
    """
    D = np.zeros((n, 2), dtype=int)
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Assign group labels based on probabilities
    groups = [i for i in range(len(p))]
    D[:, 1] = np.random.choice(groups, size=n, p=p)
    
    # Generate data based on the specified distribution
    if dist == 'normal':
        for i in range(n):
            D[i, 0] = np.random.normal(m[D[i, 1]], s[D[i, 1]])
    elif dist == 'uniform':
        for i in range(n):
            D[i, 0] = np.random.uniform(m[D[i, 1]] - s[D[i, 1]], m[D[i, 1]] + s[D[i, 1]])
    
    return D


def load_xg_dataset(filepath, x_idx, g_idx, group_labels=None):
    """
    Loads a dataset from a file, extracts columns x and g by index, 
    removes other columns, and encodes text labels as numbers.

    Args:
        filepath (str): Path to the dataset file (CSV/TSV).
        x_idx (int): Index of the x column.
        g_idx (int): Index of the g column.

    Returns:
        np.ndarray: Array with two columns [x, g], where g is numeric.
    """
    
    delimiter = ',' if filepath.endswith('.csv') else '\t'
    df = pd.read_csv(filepath, delimiter=delimiter)
    # remove rows where values of g_idx is not in group_labels
    if group_labels is not None:
        df = df[df.iloc[:, g_idx].isin(group_labels)]
    # Select the x and g columns
    x = df.iloc[:, x_idx].copy()
    g_raw = df.iloc[:, g_idx].copy()
    n =  len(x)

    # if x is not integer, multiply by 100 and convert to int
    if not np.issubdtype(x.dtype, np.integer):
        x = (x * 100).astype(int)
    
    unique_g = g_raw.unique()
    g_mapping = {label: idx for idx, label in enumerate(unique_g)}
    # g = np.zeros(n, dtype=int)
    # for i in range(n):
    #     g[i] = g_mapping[g_raw.iloc[i]]
    g = g_raw.replace(g_mapping).values
    # g = g.astype(int)

    return np.column_stack((x, g))