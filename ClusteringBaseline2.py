import numpy as np
from sklearn.cluster import KMeans

def fairlets_binary_1d(x, g):
    """
    Minimal fairlet decomposition for binary groups in 1D.
    Pairs nearest opposite-group points after sorting.
    
    x: array-like, shape (n,)
    g: array-like, shape (n,) with values in {0,1}
    
    Returns:
        fairlets: list of lists of original indices
        order: sorting order used internally
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=int)
    order = np.argsort(x)
    xs = x[order]
    gs = g[order]

    idx0 = [i for i in range(len(xs)) if gs[i] == 0]
    idx1 = [i for i in range(len(xs)) if gs[i] == 1]

    # Greedy monotone matching in sorted order
    m = min(len(idx0), len(idx1))
    fairlets = []
    for a, b in zip(idx0[:m], idx1[:m]):
        fairlets.append([order[a], order[b]])

    # leftover points become singleton fairlets
    leftovers = idx0[m:] + idx1[m:]
    for i in leftovers:
        fairlets.append([order[i]])

    return fairlets, order


def fairlet_representatives(x, fairlets, rep="mean"):
    """
    Build one representative point per fairlet.
    """
    x = np.asarray(x, dtype=float)
    reps = []
    weights = []
    for fl in fairlets:
        vals = x[fl]
        if rep == "median":
            r = float(np.median(vals))
        else:
            r = float(np.mean(vals))
        reps.append(r)
        weights.append(len(fl))
    return np.array(reps), np.array(weights)


def fairlet_kmeans_bins(x, g, k, random_state=0, rep="mean"):
    """
    Baseline:
      1) build fairlets
      2) run weighted k-means on fairlet representatives
      3) convert centers to 1D bins by midpoints
    """
    x = np.asarray(x, dtype=float)
    fairlets, _ = fairlets_binary_1d(x, g)
    reps, weights = fairlet_representatives(x, fairlets, rep=rep)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(reps.reshape(-1, 1), sample_weight=weights)

    centers = np.sort(km.cluster_centers_.ravel())
    boundaries = (centers[:-1] + centers[1:]) / 2.0

    return {
        "fairlets": fairlets,
        "centers": centers,
        "boundaries": boundaries,
    }


def assign_bins_1d(x, boundaries):
    """
    Given boundaries b1 < ... < b_{k-1}, return bin ids in {0,...,k-1}.
    """
    x = np.asarray(x, dtype=float)
    return np.searchsorted(boundaries, x, side="right")