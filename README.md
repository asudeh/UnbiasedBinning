# Unbiased Binning for Fairness-aware Attribute Representation

## Citation

Abolfazl Asudeh, Zeinab Asoodeh, Bita Asoodeh, and Omid Asudeh.
Unbiased Binning for Fairness-aware Attribute Representation.
Proceedings of the VLDB Endowment, Vol. 19, No. 10, 2026.

## Paper Summary

Discretizing raw features into bucketized attributes is a common preprocessing step before sharing, analyzing, or modeling a dataset. However, this process can inadvertently introduce bias and amplify unfairness in downstream tasks.

To address this problem, we formulate unbiased binning, which seeks bucketized attributes that satisfy group parity. We develop an efficient dynamic programming algorithm for equal-size binning and characterize the practical challenges of exact parity. In practice, however, an unbiased binning may incur a high price of fairness or may not exist at all, particularly when group distributions differ substantially.

To accommodate settings in which small deviations from perfect parity are acceptable, we further introduce epsilon-biased binning, which restricts group disparities across buckets to at most epsilon. We first present a quadratic-time dynamic programming solution. Since this algorithm does not scale to large datasets, we then propose a practical local-search algorithm, LS, built around a divide-and-conquer procedure. D&C runs in near-linear time and is proved to return a valid solution whenever one exists. LS uses the D&C solution as an upper bound to guide the search for the optimum.

The LS and D&C algorithms are general and are not limited to equal-size binning. Complementing the theoretical results, extensive experiments on real-world and synthetic datasets confirm the efficiency of the proposed algorithms and demonstrate the significance of the problem: fairness-unaware binning can generate biased attribute representations, while fairness-aware binning can substantially reduce this bias with a negligible price of fairness.

## Main Files

- `UnbiasedBinning.py`: Includes `unbiased_binning()`, the dynamic programming approach for unbiased binning described in Algorithm 1. This function first calls `boundary_candidates()` to find the boundary candidates.
- `EbiasedDnC.py`: Includes the practical approach for finding an epsilon-biased binning using the local-search algorithm. The divide-and-conquer algorithm, D&C, described in Algorithm 2, is implemented by `_boundary()` and called inside the local-search algorithm `ebiased_dnc()`. To call only D&C without local search, set the `fast` parameter to `True`; otherwise, the function applies local search.
- `EbiasedBinning.py`: Includes the dynamic programming algorithm for epsilon-biased binning, detailed in Appendix A of the technical report. The algorithm has quadratic time complexity and therefore does not scale to very large settings. The function to call is `ebias_binning()`.

## Project

This paper is part of our broader project on fairness-aware data structures. More information is available on the project page: https://www.cs.uic.edu/~indexlab/nsf-iis-2348919/index.html#unbiased-binning
