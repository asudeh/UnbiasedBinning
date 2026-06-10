# Unbiased Binning for Fairness-aware Attribute Representation

## Citation

Abolfazl Asudeh, Zeinab Asoodeh, Bita Asoodeh, and Omid Asudeh.
Unbiased Binning for Fairness-aware Attribute Representation.
Proceedings of the VLDB Endowment, Vol. 19, No. 10, 2026.

## Paper Summary

Discretizing raw features into bucketized attributes is a common step before sharing a dataset. However, this process can inadvertently introduce bias and amplify unfairness in downstream tasks.

In this paper, we address this issue by formulating the unbiased binning problem, which seeks bucketized attributes that satisfy group parity. We develop an efficient dynamic programming algorithm to solve this problem for equal-size binning. 

In practice, however, an unbiased binning may incur a high price of fairness or may not exist at all, particularly when group distributions differ substantially. To accommodate settings in which small deviations from perfect parity are acceptable, we introduce epsilon-biased binning, which restricts group disparities across buckets to at most epsilon. We first present a dynamic programming algorithm, DP, that computes the optimal solution in quadratic time. 

DP does not scale to large datasets. To address this limitation, we propose a practically scalable algorithm based on a local search strategy (LS). A key component of LS is a divide-and-conquer algorithm (D&C) that finds a solution in near-linear time. We prove that D&C always returns a valid solution whenever one exists. The LS algorithm then performs a local search, using the D&C solution as an upper bound, to #nd the optimal solution. Our LS and D&C algorithms are general and are not restricted to equal-size binning.

## Main Files

- `UnbiasedBinning.py`: Includes `unbiased_binning()`, the dynamic programming approach for unbiased binning described in Algorithm 1. This function first calls `boundary_candidates()` to find the boundary candidates.
- `EbiasedDnC.py`: Includes the practical approach for finding an epsilon-biased binning using the local-search algorithm. The divide-and-conquer algorithm, D&C, described in Algorithm 2, is implemented by `_boundary()` and called inside the local-search algorithm `ebiased_dnc()`. To call only D&C without local search, set the `fast` parameter to `True`; otherwise, the function applies local search.
- `EbiasedBinning.py`: Includes the dynamic programming algorithm for epsilon-biased binning, detailed in Appendix A of the technical report. The algorithm has quadratic time complexity and therefore does not scale to very large settings. The function to call is `ebias_binning()`.

## Project

This paper is part of our broader project on fairness-aware data structures. More information is available on the project page: https://www.cs.uic.edu/~indexlab/nsf-iis-2348919/index.html#unbiased-binning
