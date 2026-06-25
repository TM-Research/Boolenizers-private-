"""
Online Quantile-Tracker Binarizer (OQTB)
========================================
True sample-by-sample online thermometer using per-feature P² streaming
quantile estimators (Jain & Chlamtac, 1985). Each feature maintains K+2
markers that track equally-spaced quantiles without storing data.

This is the *purest* online quantile binarizer: thresholds are the
quantile estimates themselves, so the K bits emitted per feature form a
true thermometer of the empirical distribution as observed so far.

HYPERPARAMETERS (3)
-------------------
    K     : int   (default 8)   thermometer bits / feature
    speed : float (default 1.0) P² marker update speed
    warmup_uniform : int (default 0)
        Number of leading samples to encode against a uniform [-1,+1] grid
        before P² has gathered K+2 observations. ``0`` means the encoder
        emits all-zero bits during the warm-up period.
"""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import P2Quantile


class OnlineQuantileTrackerBinarizer(ThermometerEncoder):
    def __init__(self, K: int = 8, speed: float = 1.0, warmup_uniform: int = 0):
        super().__init__(K=K, name="OQTB")
        self.speed = float(speed)
        self.warmup_uniform = int(warmup_uniform)

    def _init_state(self):
        n = self.n_features
        self.trackers_ = [P2Quantile(K=self.K, speed=self.speed) for _ in range(n)]
        self.n_seen_ = 0
        self._warmup_grid = np.linspace(-1.0, 1.0, self.K + 2)[1:-1]

    def _cold_start_init(self, x: np.ndarray):
        self.n_features = len(x)
        self._init_state()
        self.fitted = True

    def fit(self, X: np.ndarray) -> "OnlineQuantileTrackerBinarizer":
        X = np.asarray(X, dtype=np.float64)
        self.n_features = X.shape[1]
        self._init_state()
        self.fitted = True
        for i in range(X.shape[0]):
            self._encode_single(X[i])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], self.n_features * self.K), dtype=np.uint8)
        for i in range(X.shape[0]):
            out[i] = self._encode_single(X[i])
        return out

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not self.fitted or getattr(self, "n_features", None) is None:
            self._cold_start_init(x)

        self.n_seen_ += 1
        K = self.K
        n = self.n_features
        bits = np.zeros(n * K, dtype=np.uint8)

        for j in range(n):
            tracker = self.trackers_[j]
            tracker.update(float(x[j]))
            if tracker.count < K + 2:
                if self.warmup_uniform > 0:
                    for k in range(K):
                        if x[j] >= self._warmup_grid[k]:
                            bits[j * K + k] = 1
                continue
            # Read inner markers (skip outer min/max)
            thresholds = tracker.q[1:-1]
            for k in range(K):
                if x[j] >= thresholds[k]:
                    bits[j * K + k] = 1
        return bits

    def get_n_output_bits(self) -> int:
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K
