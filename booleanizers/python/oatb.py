"""
Online ATR Binarizer (OATB)
===========================
True sample-by-sample online thermometer encoder using ATR-style
(volatility-of-deltas) spacing rather than standard-deviation spacing.

Motivation
----------
Bollinger-style spacing uses the EWMA standard deviation of values; ATR-style
spacing uses the EWMA of |x_t - x_{t-1}| ("true range"). On stationary
distributions the two are linearly related, but ATR is far more responsive
to regime shifts and clipped-extreme values that inflate variance disproportionately.

Per-feature state:
    ema_mu     : running mean (slow)
    ema_atr    : EWMA of |delta|
    x_prev     : last observation
    bit_state  : last emitted bits (for hysteresis)

Thresholds per emission: ``mu + s_k * atr`` for K evenly-spaced s_k in
``[-band, +band]``.

HYPERPARAMETERS (5)
-------------------
    K          : int    (default 8)
    alpha_mu   : float  (default 0.01)  EMA decay for mean (slow)
    alpha_atr  : float  (default 0.05)  EMA decay for ATR (faster)
    band       : float  (default 2.0)   range covered by K levels (in ATR units)
    hysteresis : float  (default 0.0)   Schmitt margin in ATR units
"""

import numpy as np
from .base import ThermometerEncoder


class OnlineATRBinarizer(ThermometerEncoder):
    def __init__(self, K: int = 8, alpha_mu: float = 0.01, alpha_atr: float = 0.05,
                 band: float = 2.0, hysteresis: float = 0.0):
        super().__init__(K=K, name="OATB")
        self.alpha_mu = float(alpha_mu)
        self.alpha_atr = float(alpha_atr)
        self.band = float(band)
        self.hysteresis = float(hysteresis)
        self.s_levels_ = np.linspace(-self.band, self.band, self.K + 2)[1:-1]

    def _init_state(self):
        n = self.n_features
        self.n_seen_ = 0
        self.ema_mu_ = np.zeros(n, dtype=np.float64)
        self.ema_atr_ = np.full(n, 1e-6, dtype=np.float64)
        self.x_prev_ = None
        self.bit_state_ = np.zeros(n * self.K, dtype=np.uint8)

    def _cold_start_init(self, x: np.ndarray):
        self.n_features = len(x)
        self._init_state()
        self.fitted = True

    def _update_stats(self, x: np.ndarray):
        self.n_seen_ += 1
        if self.x_prev_ is None:
            self.x_prev_ = x.copy()
            self.ema_mu_ = x.copy()
            return
        # Bias-corrected effective alpha during warm-up
        t = self.n_seen_
        if t < 200:
            a_mu = min(self.alpha_mu / max(1.0 - (1.0 - self.alpha_mu) ** t, 1e-12), 1.0)
            a_atr = min(self.alpha_atr / max(1.0 - (1.0 - self.alpha_atr) ** t, 1e-12), 1.0)
        else:
            a_mu, a_atr = self.alpha_mu, self.alpha_atr
        self.ema_mu_ += a_mu * (x - self.ema_mu_)
        tr = np.abs(x - self.x_prev_)
        self.ema_atr_ = (1.0 - a_atr) * self.ema_atr_ + a_atr * tr
        self.x_prev_ = x.copy()

    def fit(self, X: np.ndarray) -> "OnlineATRBinarizer":
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

        atr = np.maximum(self.ema_atr_, 1e-12)
        thresholds = self.ema_mu_[:, None] + self.s_levels_[None, :] * atr[:, None]

        if self.hysteresis > 0.0:
            bits = self.bit_state_.copy()
            margins = self.hysteresis * atr
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

        return (x[:, None] >= thresholds).astype(np.uint8).ravel()

    def get_n_output_bits(self) -> int:
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K
