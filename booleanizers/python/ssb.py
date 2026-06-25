"""
Spectral Stability Binarizer (SSB) v3

==========================================================================
CORE PRINCIPLE: FULL-DATA QUANTILE + SPARSE-AWARE STABILITY REFINEMENT
==========================================================================

v3: Use full training data for quantile estimation. Skip refinement for
sparse/binary features (common after one-hot encoding).

==========================================================================
HYPERPARAMETERS (3): K, n_zones, edge_weight
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class SpectralStabilityBinarizer(ThermometerEncoder):
    """Spectral Stability Binarizer (SSB) v3."""

    def __init__(self, K: int = 8, n_zones: int = 32, edge_weight: float = 0.3):
        super().__init__(K=K, name="SSB")
        self.n_zones = n_zones
        self.edge_weight = np.clip(edge_weight, 0.0, 1.0)

    def fit(self, X: np.ndarray) -> "SpectralStabilityBinarizer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        q_positions = (np.arange(self.K) + 1) / (self.K + 1)

        # --- Phase 1: Exact quantiles from FULL training data ---
        quantile_thresholds = np.quantile(X, q_positions, axis=0).T  # (n_features, K)

        # --- Phase 2: Stability-edge refinement using a subsample ---
        if self.edge_weight > 0:
            stat_size = min(20000, n_samples)
            if n_samples > stat_size:
                rng = np.random.RandomState(42)
                idx = rng.choice(n_samples, stat_size, replace=False)
                X_stat = X[idx]
            else:
                X_stat = X

            fmin = X_stat.min(axis=0)
            fmax = X_stat.max(axis=0)

            for j in range(n_features):
                rng_j = fmax[j] - fmin[j]
                if rng_j < 1e-15:
                    continue

                # Skip sparse/binary features
                n_unique = len(np.unique(X_stat[:, j]))
                if n_unique <= max(3, self.K // 2):
                    continue
                vals, counts = np.unique(np.round(X_stat[:, j], 8), return_counts=True)
                if counts.max() / len(X_stat) > 0.5:
                    continue

                # Density + edge histogram
                margin = 0.02 * rng_j
                zone_edges = np.linspace(fmin[j] - margin, fmax[j] + margin, self.n_zones + 1)
                counts_h, _ = np.histogram(X_stat[:, j], bins=zone_edges)
                density = counts_h.astype(np.float64) / (counts_h.sum() + 1e-10)

                grad = np.abs(np.diff(density))
                edge_strength = np.zeros(self.n_zones)
                edge_strength[:-1] = grad
                edge_strength[-1] = grad[-1] if len(grad) > 0 else 0

                weight = (1.0 - self.edge_weight) * (density + 1e-10) + self.edge_weight * (edge_strength + 1e-10)
                cumw = np.cumsum(weight)
                cumw = cumw / cumw[-1]
                zone_centers = 0.5 * (zone_edges[:-1] + zone_edges[1:])

                edge_thresholds = np.interp(q_positions, cumw, zone_centers)
                quantile_thresholds[j] = (
                    (1.0 - self.edge_weight) * quantile_thresholds[j]
                    + self.edge_weight * edge_thresholds
                )

        # Strict monotonicity
        self.thresholds_ = quantile_thresholds
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
