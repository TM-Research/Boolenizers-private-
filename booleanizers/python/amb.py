"""
Adaptive Momentum Binarizer (AMB) v3

==========================================================================
CORE PRINCIPLE: FULL-DATA QUANTILE + KURTOSIS-ADAPTED SPACING
==========================================================================

v3: Use full training data for quantile estimation. Adapt spacing
based on excess kurtosis computed on a subsample.

==========================================================================
HYPERPARAMETERS (3): K, momentum_blend, kurtosis_adapt
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class AdaptiveMomentumBinarizer(ThermometerEncoder):
    """Adaptive Momentum Binarizer (AMB) v3."""

    def __init__(self, K: int = 8, momentum_blend: float = 0.3, kurtosis_adapt: bool = True):
        super().__init__(K=K, name="AMB")
        self.momentum_blend = np.clip(momentum_blend, 0.0, 1.0)
        self.kurtosis_adapt = kurtosis_adapt

    def fit(self, X: np.ndarray) -> "AdaptiveMomentumBinarizer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        q_positions = (np.arange(self.K) + 1) / (self.K + 1)

        # --- Phase 1: Exact quantiles from FULL training data ---
        thresholds = np.quantile(X, q_positions, axis=0).T  # (n_features, K)

        if self.momentum_blend > 0 and self.kurtosis_adapt:
            # --- Phase 2: Kurtosis on subsample ---
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
            kurt_factor = np.clip(kurtosis / 15.0, 0.0, 1.0)

            # --- Phase 3: Kurtosis-adapted spacing ---
            for j in range(n_features):
                if kurt_factor[j] < 0.05:
                    continue

                t = thresholds[j]
                lo, hi = t[0], t[-1]
                if hi - lo < 1e-15:
                    continue

                # Skip sparse/binary features
                n_unique = len(np.unique(X_stat[:, j]))
                if n_unique <= max(3, self.K // 2):
                    continue

                # Power compression toward center for high-kurtosis features
                u = np.linspace(0, 1, self.K)
                centered = 2.0 * u - 1.0
                gamma = 1.5 * kurt_factor[j] * self.momentum_blend
                compressed = np.sign(centered) * np.abs(centered) ** (1.0 + gamma)
                adapted_u = 0.5 * (compressed + 1.0)
                adapted_t = lo + (hi - lo) * adapted_u

                blend = self.momentum_blend * kurt_factor[j]
                thresholds[j] = (1.0 - blend) * t + blend * adapted_t

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
