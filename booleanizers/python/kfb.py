"""
Kalman Filter Binarizer (KFB) v3

==========================================================================
CORE PRINCIPLE: FULL-DATA QUANTILE + KALMAN CROSSING-RATE CALIBRATION
==========================================================================

v3: Use full training data for quantile estimation. Apply Kalman-style
crossing-rate correction to fine-tune threshold positions.

A well-placed threshold t_k should satisfy P(X >= t_k) = 1 - q_k.
If the empirical crossing rate deviates, we correct the threshold position
by a fraction of the feature std proportional to the imbalance.

==========================================================================
HYPERPARAMETERS (4): K, process_noise, measurement_noise_init, innovation_scale
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class KalmanFilterBinarizer(ThermometerEncoder):
    """Kalman Filter Binarizer (KFB) v3."""

    def __init__(
        self,
        K: int = 8,
        process_noise: float = 0.01,
        measurement_noise_init: float = 1.0,
        innovation_scale: float = 0.5,
    ):
        super().__init__(K=K, name="KFB")
        self.process_noise = process_noise
        self.measurement_noise_init = measurement_noise_init
        self.innovation_scale = np.clip(innovation_scale, 0.0, 1.0)

    def fit(self, X: np.ndarray) -> "KalmanFilterBinarizer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        q_positions = (np.arange(self.K) + 1) / (self.K + 1)

        # --- Phase 1: Exact quantiles from FULL training data ---
        thresholds = np.quantile(X, q_positions, axis=0).T  # (n_features, K)

        if self.innovation_scale > 0:
            std = np.std(X, axis=0) + 1e-20

            # --- Phase 2: Crossing-rate calibration on a subsample ---
            stat_size = min(20000, n_samples)
            if n_samples > stat_size:
                rng = np.random.RandomState(42)
                idx = rng.choice(n_samples, stat_size, replace=False)
                X_stat = X[idx]
            else:
                X_stat = X

            for j in range(n_features):
                # Empirical above-rate for each threshold
                above_rate = np.mean(X_stat[:, j:j+1] >= thresholds[j][np.newaxis, :], axis=0)
                target_above = 1.0 - q_positions   # P(X >= t_k) = 1 - q_k
                imbalance = above_rate - target_above
                correction = imbalance * std[j] * 0.3 * self.innovation_scale
                thresholds[j] += correction

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
