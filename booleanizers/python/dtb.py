"""
Decision Tree Binarizer (DTB) v1 — Known Method

A binarizer using well-established decision tree split selection (variance
reduction / information gain) applied in an unsupervised manner.

==========================================================================
CORE PRINCIPLE: UNSUPERVISED RECURSIVE VARIANCE-REDUCTION SPLITS
==========================================================================

Decision tree split selection is one of the most well-studied methods for
finding informative thresholds. DTB applies this WITHOUT labels:

1. For each feature, find the split point that maximizes inter-group
   variance (equivalent to minimizing within-group variance), i.e.,
   the 1D k-means optimal split.
2. Recursively split each resulting partition to get K thresholds.
3. This is equivalent to finding K values that optimally partition the
   feature's marginal distribution into K+1 groups.

This is closely related to:
- Jenks natural breaks optimization
- Fisher's optimal 1D clustering
- Otsu's method (K=1 case)

==========================================================================
HYPERPARAMETERS (3)
==========================================================================

    K : int (default 8)
        Binary bits (thresholds) per feature.

    max_candidates : int (default 200)
        Number of candidate split points to evaluate per feature.
        Higher = better splits but slower.

    method : str (default 'variance')
        'variance' (minimize within-group variance) or
        'density' (split at density valleys using KDE-like approach).

==========================================================================
COMPLEXITY
==========================================================================

    fit:       O(d * K * max_candidates * log(max_candidates)) time
    transform: O(n * d * K) time, O(1) extra memory — fully vectorized
"""

import numpy as np
from .base import ThermometerEncoder


class DecisionTreeBinarizer(ThermometerEncoder):
    """
    Decision Tree Binarizer (DTB) — Known Method.

    Unsupervised threshold placement using variance-reduction splits.

    Parameters
    ----------
    K : int, default=8
        Bits per feature.
    max_candidates : int, default=200
        Candidate split points per feature.
    method : str, default='variance'
        'variance' or 'density'.
    """

    def __init__(self, K: int = 8, max_candidates: int = 200, method: str = "variance"):
        super().__init__(K=K, name="DTB")
        self.max_candidates = max_candidates
        if method not in ("variance", "density"):
            raise ValueError(f"method must be 'variance' or 'density', got {method!r}")
        self.method = method

    def fit(self, X: np.ndarray) -> "DecisionTreeBinarizer":
        """Fit thresholds using recursive variance-reduction splits."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        self.thresholds_ = np.zeros((n_features, self.K), dtype=np.float64)

        for j in range(n_features):
            col = np.sort(X[:, j])

            if self.method == "variance":
                thresholds = self._variance_splits(col)
            else:
                thresholds = self._density_splits(col)

            # Pad if we didn't get enough thresholds
            if len(thresholds) < self.K:
                lo, hi = col[0], col[-1]
                extra = np.linspace(lo, hi, self.K - len(thresholds) + 2)[1:-1]
                thresholds = np.concatenate([thresholds, extra])
            thresholds = np.sort(thresholds[:self.K])

            # Ensure strict monotonicity
            for k in range(1, self.K):
                if thresholds[k] <= thresholds[k - 1]:
                    thresholds[k] = thresholds[k - 1] + 1e-10

            self.thresholds_[j] = thresholds

        self.fitted = True
        return self

    def _variance_splits(self, sorted_col: np.ndarray) -> np.ndarray:
        """Find K thresholds using recursive best-variance-reduction split."""
        n = len(sorted_col)
        if n < 2:
            return np.array([sorted_col[0]])

        # Subsample candidate positions for efficiency
        if n > self.max_candidates:
            indices = np.linspace(1, n - 1, self.max_candidates, dtype=int)
        else:
            indices = np.arange(1, n)

        # Recursive splitting: maintain a list of (lo, hi) intervals
        intervals = [(0, n)]
        thresholds = []

        for _ in range(self.K):
            best_gain = -1
            best_split = None
            best_interval_idx = -1

            for idx, (lo, hi) in enumerate(intervals):
                if hi - lo < 2:
                    continue
                segment = sorted_col[lo:hi]
                seg_n = len(segment)
                seg_var = np.var(segment)
                if seg_var < 1e-20:
                    continue

                # Find best split within this segment
                # Use cumulative sum for O(n) variance computation
                cumsum = np.cumsum(segment)
                cumsq = np.cumsum(segment ** 2)

                # Candidate split positions within segment
                seg_candidates = np.arange(1, min(seg_n, self.max_candidates + 1))
                if seg_n > self.max_candidates:
                    seg_candidates = np.linspace(1, seg_n - 1, self.max_candidates, dtype=int)

                for s in seg_candidates:
                    n_left = s
                    n_right = seg_n - s
                    if n_left < 1 or n_right < 1:
                        continue

                    mean_left = cumsum[s - 1] / n_left
                    mean_right = (cumsum[-1] - cumsum[s - 1]) / n_right

                    var_left = cumsq[s - 1] / n_left - mean_left ** 2
                    var_right = (cumsq[-1] - cumsq[s - 1]) / n_right - mean_right ** 2

                    weighted_var = (n_left * max(var_left, 0) + n_right * max(var_right, 0)) / seg_n
                    gain = seg_var - weighted_var

                    if gain > best_gain:
                        best_gain = gain
                        best_split = (idx, lo + s)
                        best_interval_idx = idx

            if best_split is None:
                break

            interval_idx, split_pos = best_split
            lo, hi = intervals[interval_idx]
            split_val = 0.5 * (sorted_col[split_pos - 1] + sorted_col[min(split_pos, len(sorted_col) - 1)])
            thresholds.append(split_val)

            # Replace interval with two sub-intervals
            intervals[interval_idx] = (lo, split_pos)
            intervals.insert(interval_idx + 1, (split_pos, hi))

        return np.array(thresholds) if thresholds else np.array([sorted_col[n // 2]])

    def _density_splits(self, sorted_col: np.ndarray) -> np.ndarray:
        """Find thresholds at density valleys (local minima of the density)."""
        n = len(sorted_col)
        if n < 2:
            return np.array([sorted_col[0]])

        # Build a histogram for density estimation
        n_bins = min(200, n // 5 + 1)
        if n_bins < 3:
            return np.linspace(sorted_col[0], sorted_col[-1], self.K + 2)[1:-1]

        counts, edges = np.histogram(sorted_col, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Smooth the density
        kernel_size = max(3, n_bins // 20)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(counts.astype(float), kernel, mode='same')

        # Find local minima (density valleys)
        valleys = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                valleys.append((smoothed[i], centers[i]))

        # Sort by density (lowest = deepest valley = best split)
        valleys.sort(key=lambda x: x[0])

        thresholds = [v[1] for v in valleys[:self.K]]

        # If not enough valleys, supplement with quantiles
        if len(thresholds) < self.K:
            q_pos = (np.arange(self.K - len(thresholds)) + 1) / (self.K - len(thresholds) + 1)
            extra = np.quantile(sorted_col, q_pos)
            thresholds.extend(extra.tolist())

        return np.array(thresholds[:self.K])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to binary thermometer codes. Fully vectorized."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        bits = (
            X[:, :, np.newaxis] >= self.thresholds_[np.newaxis, :, :]
        ).astype(np.uint8)
        return bits.reshape(X.shape[0], -1)

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """Encode one sample. O(d * K)."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")
        x = np.asarray(x, dtype=np.float64)
        bits = (x[:, np.newaxis] >= self.thresholds_).astype(np.uint8)
        return bits.ravel()

    def get_n_output_bits(self) -> int:
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K
