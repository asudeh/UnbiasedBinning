import numpy as np
from itertools import combinations_with_replacement
from collections import defaultdict

def point_cost_1d(x, centers):
    centers = np.sort(np.asarray(centers))
    return np.sum(np.min(np.abs(x[:, None] - centers[None, :]), axis=1))

def make_pattern_classes(group_membership):
    """
    group_membership[i]: tuple/list of 0/1 membership over t groups for point i
    returns dict: bitvector -> list of point indices
    """
    classes = defaultdict(list)
    for i, g in enumerate(group_membership):
        classes[tuple(g)].append(i)
    return classes

def feasible_patterns(classes, requirements, k):
    """
    Enumerate multisets of class-types whose summed bitvectors satisfy requirements.
    Minimal brute-force version.
    """
    class_keys = list(classes.keys())
    out = []
    for patt in combinations_with_replacement(range(len(class_keys)), k):
        s = np.sum(np.array([class_keys[i] for i in patt]), axis=0)
        if np.all(s >= requirements):
            out.append([class_keys[i] for i in patt])
    return out

def initial_solution_for_pattern(x, classes, pattern):
    """
    Pick one initial point from each class in pattern.
    """
    used = defaultdict(int)
    sol = []
    for cls in pattern:
        idxs = classes[cls]
        j = used[cls] % len(idxs)
        sol.append(idxs[j])
        used[cls] += 1
    return sol

def ls1_same_group_swaps(x, classes, pattern, init_sol):
    """
    1-swap local search:
    slot j can only swap with another point from the same class as pattern[j].
    """
    sol = list(init_sol)
    centers = x[sol].copy()
    best = point_cost_1d(x, centers)

    improved = True
    while improved:
        improved = False
        for j, cls in enumerate(pattern):
            current_idx = sol[j]
            for cand in classes[cls]:
                if cand == current_idx:
                    continue
                new_sol = sol.copy()
                new_sol[j] = cand
                new_centers = x[new_sol]
                cost = point_cost_1d(x, new_centers)
                if cost < best - 1e-12:
                    sol = new_sol
                    centers = new_centers
                    best = cost
                    improved = True
                    break
            if improved:
                break
    return sol, best

def centers_to_bins(x, center_values):
    """
    Convert sorted centers into contiguous 1D bins using midpoints.
    """
    c = np.sort(np.asarray(center_values))
    mids = (c[:-1] + c[1:]) / 2.0
    return mids  # boundaries

def fair_clustering_baseline_1d(x, group_membership, requirements, k):
    """
    Minimal baseline adapted from [3]:
    feasible pattern enumeration + LS1
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    x = x[order]
    group_membership = [group_membership[i] for i in order]

    classes = make_pattern_classes(group_membership)
    patterns = feasible_patterns(classes, np.asarray(requirements), k)

    best_sol, best_cost, best_pattern = None, np.inf, None
    for pattern in patterns:
        init_sol = initial_solution_for_pattern(x, classes, pattern)
        sol, cost = ls1_same_group_swaps(x, classes, pattern, init_sol)
        if cost < best_cost:
            best_sol, best_cost, best_pattern = sol, cost, pattern

    if best_sol is None:
        return None

    centers = x[best_sol]
    boundaries = centers_to_bins(x, centers)
    return {
        "centers": np.sort(centers),
        "boundaries": boundaries,
        "cost": best_cost,
        "pattern": best_pattern,
    }