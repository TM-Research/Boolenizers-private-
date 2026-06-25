"""
Signal-Quantile Fusion Binarizer (SQF) v2

==========================================================================
CORE PRINCIPLE: FULL-DATA QUANTILE + MULTI-STRATEGY REFINEMENT
==========================================================================

v2: Use full training data for exact quantile computation.
Combine three signal-processing refinements (kurtosis, skewness,
crossing-rate calibration) per feature, skipping sparse/binary features.

==========================================================================
HYPERPARAMETERS (3): K, adapt_strength, sparse_threshold
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class SignalQuantileFusion(ThermometerEncoder):
    """Signal-Quantile Fusion Binarizer (SQF) v2."""

    def __init__(self, K: int = 8, adapt_strength: float = 0.3, sparse_threshold: float = 0.3):
        super().__init__(K=K, name="SQF")
        self.adapt_strength = np.clip(adapt_strength, 0.0, 1.0)
        self.sparse_threshold = sparse_threshold

    def fit(self, X: np.ndarray) -> "SignalQuantileFusion":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        q_positions = (np.arange(self.K) + 1) / (self.K + 1)

        # --- Phase 1: Exact quantiles from FULL training data ---
        thresholds = np.quantile(X, q_positions, axis=0).T  # (n_features, K)

        if self.adapt_strength > 0:
            # --- Phase 2: Statistics on a subsample ---
            stat_size = min(20000, n_samples)
            if n_samples > stat_size:
                rng = np.random.RandomState(42)
                idx = rng.choice(n_samples, stat_size, replace=False)
                X_stat = X[idx]
            else:
                X_stat = X

            mean = np.mean(X_stat, axis=0)
            std = np.std(X_stat, axis=0) + 1e-20
            z = (X_stat - mean) / std
            kurtosis = np.clip(np.mean(z ** 4, axis=0) - 3.0, -2.0, 30.0)
            skewness = np.clip(np.mean(z ** 3, axis=0), -3.0, 3.0)

            for j in range(n_features):
                t = thresholds[j]
                lo, hi = t[0], t[-1]
                if hi - lo < 1e-15:
                    continue

                # Skip sparse/binary features
                n_unique = len(np.unique(X_stat[:, j]))
                if n_unique <= max(3, self.K // 2):
                    continue
                vals, counts = np.unique(np.round(X_stat[:, j], 8), return_counts=True)
                if counts.max() / len(X_stat) > self.sparse_threshold:
                    continue

                # Strategy 1: Kurtosis-adapted spacing
                kurt_factor = np.clip(kurtosis[j] / 15.0, 0.0, 1.0)
                if kurt_factor > 0.05:
                    u = np.linspace(0, 1, self.K)
                    centered = 2.0 * u - 1.0
                    gamma = 2.0 * kurt_factor * self.adapt_strength
                    compressed = np.sign(centered) * np.abs(centered) ** (1.0 + gamma)
                    adapted_u = 0.5 * (compressed + 1.0)
                    adapted_t = lo + (hi - lo) * adapted_u
                    blend = 0.5 * self.adapt_strength * kurt_factor
                    t = (1.0 - blend) * t + blend * adapted_t

                # Strategy 2: Skewness shift
                if abs(skewness[j]) > 0.3:
                    shift = 0.05 * skewness[j] * std[j] * self.adapt_strength
                    t = t + shift

                # Strategy 3: Crossing-rate calibration
                above_rate = np.mean(X_stat[:, j:j+1] >= t[np.newaxis, :], axis=0)
                target_above = 1.0 - q_positions
                imbalance = above_rate - target_above
                correction = imbalance * std[j] * 0.2 * self.adapt_strength
                t = t + correction

                thresholds[j] = t

        # Strict monotonicity
        self.thresholds_ = thresholds
        for j in range(n_features):
            self.thresholds_[j] = np.sort(self.thresholds_[j])
            for k in range(1, self.K):
                if self.thresholds_[j, k] <= self.thresholds_[j, k - 1]:
                    self.thresholds_[j, k] = self.thresholds_[j, k - 1] + 1e-10

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        return (X[:, :, np.newaxis] >= self.thresholds_[np.newaxis, :, :]).astype(np.uint8).reshape(X.shape[0], -1)

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")
        return (np.asarray(x, dtype=np.float64)[:, np.newaxis] >= self.thresholds_).astype(np.uint8).ravel()

    def get_n_output_bits(self) -> int:
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K
