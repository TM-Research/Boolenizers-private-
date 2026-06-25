"""
Dynamic Pulse Binarizer (DPB) v3

A novel binarization algorithm inspired by digital signal pulse detection.

==========================================================================
CORE PRINCIPLE: FULL-DATA QUANTILE + PULSE-EDGE REFINEMENT
==========================================================================

v3 Fix: Use full training data for quantile estimation (critical for
accuracy on large datasets — reservoir underestimates tail quantiles).

Algorithm:
1. Compute exact quantiles from the FULL training data
2. Compute skewness from a subsample (for speed)
3. Refine thresholds:
   - Shift toward heavy tails based on skewness
   - Shift toward inter-bin transition boundaries (pulse-edge detection)

==========================================================================
HYPERPARAMETERS (3)
==========================================================================
    K : int (default 8)
    pulse_sensitivity : float (default 0.3)  — 0=pure quantile
    skew_adapt : bool (default True)
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class DynamicPulseBinarizer(ThermometerEncoder):
    """Dynamic Pulse Binarizer (DPB) v3."""

    def __init__(self, K: int = 8, pulse_sensitivity: float = 0.3, skew_adapt: bool = True):
        super().__init__(K=K, name="DPB")
        self.pulse_sensitivity = np.clip(pulse_sensitivity, 0.0, 1.0)
        self.skew_adapt = skew_adapt

    def fit(self, X: np.ndarray) -> "DynamicPulseBinarizer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        q_positions = (np.arange(self.K) + 1) / (self.K + 1)

        # --- Phase 1: Exact quantiles from FULL training data ---
        quantile_thresholds = np.quantile(X, q_positions, axis=0).T  # (n_features, K)

        # --- Phase 2: Signal statistics on a subsample (speed) ---
        stat_size = min(20000, n_samples)
        if n_samples > stat_size:
            rng = np.random.RandomState(42)
            stat_idx = rng.choice(n_samples, stat_size, replace=False)
            X_stat = X[stat_idx]
        else:
            X_stat = X

        mean = np.mean(X_stat, axis=0)
        std = np.std(X_stat, axis=0) + 1e-20

        # Skewness
        skewness = np.zeros(n_features)
        if n_samples > 10:
            z = (X_stat - mean) / std
            skewness = np.clip(np.mean(z ** 3, axis=0), -3.0, 3.0)

        # --- Phase 3: Pulse-edge refinement (local std per quantile bin) ---
        if self.pulse_sensitivity > 0:
            for j in range(n_features):
                col = X_stat[:, j]
                t = quantile_thresholds[j]
                edges = np.concatenate([[-np.inf], t, [np.inf]])
                bin_std = np.zeros(self.K + 1)
                for b in range(self.K + 1):
                    mask = (col >= edges[b]) & (col < edges[b + 1])
                    if mask.sum() > 1:
                        bin_std[b] = np.std(col[mask])

                for k in range(self.K):
                    left_d, right_d = bin_std[k], bin_std[k + 1]
                    total = left_d + right_d + 1e-20
                    shift = (right_d - left_d) / total * std[j] * 0.1
                    quantile_thresholds[j, k] += self.pulse_sensitivity * shift

        # --- Phase 4: Skewness shift ---
        if self.skew_adapt:
            skew_shift = 0.05 * skewness * std
            quantile_thresholds += skew_shift[:, np.newaxis]

        # Strict monotonicity
        self.thresholds_ = quantile_thresholds
        for j in range(n_features):
            self.thresholds_[j] = np.sort(self.thresholds_[j])
            for k in range(1, self.K):
                if self.thresholds_[j, k] <= self.thresholds_[j, k - 1]:
                    self.thresholds_[j, k] = self.thresholds_[j, k - 1] + 1e-10

        self.mu_ = mean
        self.std_ = std
        self.skewness_ = skewness
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
