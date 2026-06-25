"""
Adaptive Quantile Binarizer (AQB)

==========================================================================
CORE PRINCIPLE: ADAPTIVE BITS PER FEATURE + QUANTILE THRESHOLDS
==========================================================================

Key insight from TMU StandardBinarizer: binary/sparse features only need
1-2 bits, not K. Wasting bits on trivial features hurts TM performance.

AQB combines:
  1. Adaptive bit allocation: features with few unique values get fewer bits
  2. Quantile-based threshold placement for continuous features
  3. Unique-value thresholds for low-cardinality features

==========================================================================
HYPERPARAMETERS (2): max_bits_per_feature, method
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class AdaptiveQuantileBinarizer(ThermometerEncoder):
    """Adaptive Quantile Binarizer (AQB).

    Adapts the number of bits per feature based on unique value count.
    Uses quantile thresholds for continuous features, unique-value
    thresholds for low-cardinality features.
    """

    def __init__(self, max_bits_per_feature: int = 10, method: str = "quantile"):
        super().__init__(K=max_bits_per_feature, name="AQB")
        self.max_bits = max_bits_per_feature
        if method not in ("quantile", "unique"):
            raise ValueError(f"method must be 'quantile' or 'unique', got {method!r}")
        self.method = method

    def fit(self, X: np.ndarray) -> "AdaptiveQuantileBinarizer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        # Per-feature: list of threshold arrays (variable length)
        self.feature_thresholds_ = []
        total_bits = 0

        for j in range(n_features):
            col = X[:, j]
            uv = np.unique(col)

            if len(uv) <= 1:
                # Constant feature — 0 bits (but we need at least 1 for the framework)
                thresholds = np.array([uv[0] + 1e-10]) if len(uv) == 1 else np.array([0.0])
            elif len(uv) <= self.max_bits:
                # Few unique values — use unique values as thresholds (skip first/min)
                thresholds = uv[1:]  # threshold at each unique value except the min
            else:
                # Many unique values — use quantile or subsampled unique thresholds
                if self.method == "quantile":
                    n_bits = self.max_bits
                    q_positions = (np.arange(n_bits) + 1) / (n_bits + 1)
                    thresholds = np.quantile(col, q_positions)
                    # Ensure strict monotonicity
                    thresholds = np.sort(thresholds)
                    for k in range(1, len(thresholds)):
                        if thresholds[k] <= thresholds[k - 1]:
                            thresholds[k] = thresholds[k - 1] + 1e-10
                else:  # unique
                    # Subsample unique values evenly (like StandardBinarizer)
                    step = len(uv) / self.max_bits
                    indices = [int(i * step) for i in range(self.max_bits)]
                    thresholds = uv[indices]

            self.feature_thresholds_.append(thresholds)
            total_bits += len(thresholds)

        self.total_bits_ = total_bits
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.total_bits_), dtype=np.uint8)

        pos = 0
        for j, thresholds in enumerate(self.feature_thresholds_):
            n_bits = len(thresholds)
            for k in range(n_bits):
                out[:, pos + k] = (X[:, j] >= thresholds[k]).astype(np.uint8)
            pos += n_bits

        return out

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")
        x = np.asarray(x, dtype=np.float64)
        bits = []
        for j, thresholds in enumerate(self.feature_thresholds_):
            for t in thresholds:
                bits.append(1 if x[j] >= t else 0)
        return np.array(bits, dtype=np.uint8)

    def get_n_output_bits(self) -> int:
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.total_bits_
