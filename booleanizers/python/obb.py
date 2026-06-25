"""
Online Bollinger Binarizer (OBB)
================================
True sample-by-sample online thermometer encoder using Bollinger-band style
adaptive thresholds.

Per-feature state (updated incrementally via EMA — no batch lookback):
    ema_mu     : running mean
    ema_var    : running variance (Welford-EMA, EWMA of squared deviations)
    bit_state  : last emitted bits (for optional hysteresis)
    n_seen     : observation count (for EMA bias-correction warm-up)

Each feature emits ``K`` thermometer bits with thresholds placed at
``mu + z_k * sigma`` where ``z_k`` is evenly spaced in ``[-band, +band]``.

Inputs of any dtype are coerced to float64. Categorical-looking integer columns
still receive useful bits because z-score thermometers degenerate gracefully
when ``sigma`` collapses to ~0.

HYPERPARAMETERS (4)
-------------------
    K          : int    (default 8)   thermometer bits per feature
    alpha      : float  (default 0.02) EMA decay factor (≈ half-life 35 samples)
    band       : float  (default 2.5)  z-score range covered by the K thresholds
    hysteresis : float  (default 0.0)  Schmitt-trigger margin in units of sigma
"""

import numpy as np
from .base import ThermometerEncoder


class OnlineBollingerBinarizer(ThermometerEncoder):
    def __init__(self, K: int = 8, alpha: float = 0.02, band: float = 2.5,
                 hysteresis: float = 0.0):
        super().__init__(K=K, name="OBB")
        self.alpha = float(alpha)
        self.band = float(band)
        self.hysteresis = float(hysteresis)
        # K thermometer z-levels evenly spread inside [-band, +band] (excluding endpoints)
        self.z_levels_ = np.linspace(-self.band, self.band, self.K + 2)[1:-1]

    # ---- state management -------------------------------------------------
    def _init_state(self):
        n = self.n_features
        self.n_seen_ = 0
        self.ema_mu_ = np.zeros(n, dtype=np.float64)
        self.ema_var_ = np.full(n, 1e-6, dtype=np.float64)
        self.bit_state_ = np.zeros(n * self.K, dtype=np.uint8)

    def _cold_start_init(self, x: np.ndarray):
        self.n_features = len(x)
        self._init_state()
        self.fitted = True

    def _update_stats(self, x: np.ndarray):
        self.n_seen_ += 1
        # Bias-corrected effective alpha during warm-up (Adam-style)
        if self.n_seen_ < 200:
            a = self.alpha / max(1.0 - (1.0 - self.alpha) ** self.n_seen_, 1e-12)
            a = min(a, 1.0)
        else:
            a = self.alpha
        delta = x - self.ema_mu_
        self.ema_mu_ += a * delta
        # EWMA variance (West, 1979 — running EWMA of squared centered deltas)
        self.ema_var_ = (1.0 - a) * (self.ema_var_ + a * delta * delta)

    # ---- public API -------------------------------------------------------
    def fit(self, X: np.ndarray) -> "OnlineBollingerBinarizer":
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

        self._update_stats(x)
        sigma = np.sqrt(np.maximum(self.ema_var_, 1e-20))
        # (n_features, K) threshold matrix
        thresholds = self.ema_mu_[:, None] + self.z_levels_[None, :] * sigma[:, None]

        if self.hysteresis > 0.0:
            bits = self.bit_state_.copy()
            margins = self.hysteresis * sigma
            for j in range(self.n_features):
                m = margins[j]
                base = j * self.K
                for k in range(self.K):
                    th = thresholds[j, k]
                    bi = base + k
                    if bits[bi] == 0:
                        bits[bi] = 1 if x[j] >= th + m else 0
                    else:
                        bits[bi] = 0 if x[j] <= th - m else 1
            self.bit_state_ = bits
            return bits

        bits = (x[:, None] >= thresholds).astype(np.uint8).ravel()
        return bits

    def get_n_output_bits(self) -> int:
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K
