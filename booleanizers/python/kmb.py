"""
Known Methods Binarizer (KMB) v1

A binarizer built from well-established quantization methods:
  - Equal-width binning
  - Quantile-based binning
  - Hybrid: quantile for placement + equal-width fallback for low-variance features

==========================================================================
CORE PRINCIPLE: HYBRID QUANTILE + EQUAL-WIDTH
==========================================================================

1. Compute empirical quantiles of the training data for each feature
2. Use quantile-based thresholds as the primary placement strategy
3. For features with very few unique values (or degenerate quantiles),
   fall back to equal-width binning over the value range
4. Optionally clip extreme quantiles to remove outlier influence

This is a standard, well-understood approach — no novel contributions.
Included as a baseline for comparison against novel methods.

==========================================================================
HYPERPARAMETERS (3)
==========================================================================

    K : int (default 8)
        Binary bits per feature.

    method : str (default 'quantile')
        'quantile', 'equal_width', or 'hybrid' (auto-select per feature).

    clip_percentile : float (default 1.0)
        Clip feature range at this percentile from each tail before binning.
        Only for equal_width and hybrid fallback. 0 = no clipping.

==========================================================================
COMPLEXITY
==========================================================================

    fit:       O(n * d * log n) time (sorting for quantiles), O(d * K) memory
    transform: O(n * d * K) time, O(1) extra memory — fully vectorized
"""

import numpy as np
from .base import ThermometerEncoder


class KnownMethodsBinarizer(ThermometerEncoder):
    """
    Known Methods Binarizer (KMB).

    Standard quantile/equal-width/hybrid threshold placement.

    Parameters
    ----------
    K : int, default=8
        Bits per feature.
    method : str, default='hybrid'
        'quantile', 'equal_width', or 'hybrid'.
    clip_percentile : float, default=1.0
        Percentile clipping for equal-width and hybrid fallback.
    """

    def __init__(self, K: int = 8, method: str = "hybrid", clip_percentile: float = 1.0):
        super().__init__(K=K, name="KMB")
        if method not in ("quantile", "equal_width", "hybrid"):
            raise ValueError(f"method must be 'quantile', 'equal_width', or 'hybrid', got {method!r}")
        self.method = method
        self.clip_percentile = np.clip(clip_percentile, 0.0, 10.0)

    def fit(self, X: np.ndarray) -> "KnownMethodsBinarizer":
        """Fit thresholds from training data."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        # Target quantile positions
        q_positions = (np.arange(self.K) + 1) / (self.K + 1)  # e.g., [0.111, 0.222, ..., 0.889]

        self.thresholds_ = np.zeros((n_features, self.K), dtype=np.float64)

        for j in range(n_features):
            col = X[:, j]

            if self.method == "quantile":
                thresholds = np.quantile(col, q_positions)
            elif self.method == "equal_width":
                thresholds = self._equal_width_thresholds(col)
            else:  # hybrid
                thresholds = self._hybrid_thresholds(col, q_positions)

            # Ensure strict monotonicity
            thresholds = np.sort(thresholds)
            for k in range(1, self.K):
                if thresholds[k] <= thresholds[k - 1]:
                    thresholds[k] = thresholds[k - 1] + 1e-10

            self.thresholds_[j] = thresholds

        self.fitted = True
        return self

    def _equal_width_thresholds(self, col: np.ndarray) -> np.ndarray:
        """Equal-width thresholds with optional percentile clipping."""
        if self.clip_percentile > 0:
            lo = np.percentile(col, self.clip_percentile)
            hi = np.percentile(col, 100.0 - self.clip_percentile)
        else:
            lo = col.min()
            hi = col.max()
        if hi - lo < 1e-15:
            lo = col.min()
            hi = col.max()
        if hi - lo < 1e-15:
            return np.linspace(lo - 1e-6, lo + 1e-6, self.K)
        return np.linspace(lo, hi, self.K + 2)[1:-1]

    def _hybrid_thresholds(self, col: np.ndarray, q_positions: np.ndarray) -> np.ndarray:
        """Quantile-based, with equal-width fallback for degenerate features."""
        quantile_thresholds = np.quantile(col, q_positions)
        n_unique = len(np.unique(quantile_thresholds))

        if n_unique < max(2, self.K // 2):
            # Degenerate quantiles — fall back to equal-width
            return self._equal_width_thresholds(col)
        return quantile_thresholds

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
