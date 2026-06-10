import math
from collections import Counter

def _entropy(y):
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def __max_gain_boundary(l, h, y_vals, mid_penalty=1e-3):
    """
    Return split index k in (l, h) maximizing:
        information_gain(y_vals[l:h] -> [l:k], [k:h]) - mid_penalty * |k - mid|
    where mid is the midpoint of [l, h).

    mid_penalty: small coefficient to bias the cut toward the middle.
                (Set to 0 for pure info gain.)
    """
    assert 0 <= l < h <= len(y_vals)
    assert h - l >= 2  # at least one element per side

    parent_entropy = _entropy(y_vals[l:h])
    n = h - l
    mid = (l + h) / 2.0

    best_score = float("-inf")
    best_k = l + 1  # default valid split

    for k in range(l + 1, h):
        left = y_vals[l:k]
        right = y_vals[k:h]

        w_left = len(left) / n
        w_right = len(right) / n

        gain = parent_entropy - w_left * _entropy(left) - w_right * _entropy(right)
        score = gain - mid_penalty * abs(k - mid)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


'''test
y = [0, 0, 0, 1, 1, 1, 1]
print(__max_gain_boundary(0, len(y), y))
'''