"""
Drift-Robust Binarizer (DRB) v2

==========================================================================
CORE PRINCIPLE: TEMPORAL-BLENDED QUANTILES + WITHIN-SAMPLE NORMALIZATION
==========================================================================

Gas sensor drift is primarily:
  (a) Multiplicative (sensor gain decreases over time)
  (b) Additive (sensor baseline shifts)

If we normalize each sample BEFORE binarizing, these two drift types
are removed:
  normalized[j] = (x[j] - mean(x)) / std(x)     # removes (a) and (b)
OR
  normalized[j] = x[j] / norm(x)                 # removes (a) multiplicative

After normalization, the distributions are stable across train and test,
and standard quantile thresholds work correctly.

ADDITIONALLY: blend thresholds toward recent training data (end of train
set is closest to test in time) for any residual drift.

==========================================================================
HYPERPARAMETERS (3)
==========================================================================
    K : int (default 8)
    alpha : float (default 0.4)
        Blend toward recent-data quantiles.
    normalize : str (default 'zscore')
        Per-sample normalization: 'zscore', 'l2', 'range', or 'none'.
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class DriftRobustBinarizer(ThermometerEncoder):
    """Drift-Robust Binarizer (DRB) v2."""

    def __init__(self, K: int = 8, alpha: float = 0.4, normalize: str = "zscore"):
        super().__init__(K=K, name="DRB")
        self.alpha = np.clip(alpha, 0.0, 0.9)
        self.normalize = normalize
        if normalize not in ("zscore", "l2", "range", "none"):
            raise ValueError(f"normalize must be 'zscore','l2','range','none', got {normalize!r}")

    def _normalize_samples(self, X: np.ndarray) -> np.ndarray:
        """Normalize each sample (row) to remove gain/offset drift."""
        if self.normalize == "none":
            return X
        X = X.copy()
        if self.normalize == "zscore":
            mu = X.mean(axis=1, keepdims=True)
            sig = X.std(axis=1, keepdims=True) + 1e-20
            return (X - mu) / sig
        elif self.normalize == "l2":
            nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-20
            return X / nrm
        elif self.normalize == "range":
            lo = X.min(axis=1, keepdims=True)
            hi = X.max(axis=1, keepdims=True)
            rng = (hi - lo) + 1e-20
            return (X - lo) / rng
        return X

    def fit(self, X: np.ndarray) -> "DriftRobustBinarizer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        # Per-sample normalization first (drift removal)
        X_norm = self._normalize_samples(X)

        q_positions = (np.arange(self.K) + 1) / (self.K + 1)

        # Full-set quantiles from normalized data
        q_full = np.quantile(X_norm, q_positions, axis=0).T   # (n_features, K)

        # Recent-data quantiles (last 20% of training)
        n_recent = max(int(n_samples * 0.2), self.K + 2)
        X_recent = X_norm[-n_recent:]
        q_recent = np.quantile(X_recent, q_positions, axis=0).T

        # Blend
        thresholds = (1.0 - self.alpha) * q_full + self.alpha * q_recent

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
        X_norm = self._normalize_samples(X)
        return (X_norm[:, :, np.newaxis] >= self.thresholds_[np.newaxis, :, :]).astype(np.uint8).reshape(X.shape[0], -1)

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        x_norm = self._normalize_samples(x)[0]
        return (x_norm[:, np.newaxis] >= self.thresholds_).astype(np.uint8).ravel()

    def get_n_output_bits(self) -> int:
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K
