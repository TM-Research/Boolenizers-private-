"""
Online Delta-Momentum Binarizer (ODMB)
======================================
True sample-by-sample online binarizer focused on temporal dynamics —
the delta, the momentum, the rate-of-change. Useful for streaming
sensor data, intrusion-detection traces and time-series anomaly
detection where the *change* often matters more than the level.

For each feature, K bits are emitted per sample. Layout (default K=8):

    bit 0:   x > x_{t-1}                   (delta sign)
    bit 1:   |delta| > ATR / 2             (small move)
    bit 2:   |delta| > ATR                 (medium move)
    bit 3:   |delta| > 2 * ATR             (large move / pulse)
    bit 4:   momentum > 0                  (EMA of delta is positive)
    bit 5:   |momentum| > 0.5 * ATR        (strong directional drift)
    bit 6:   rolling extremum: x == EMA-max within +-ATR
    bit 7:   accel sign                    ((x - x_{t-1}) - (x_{t-1} - x_{t-2}))

If ``K > 8`` extra bits form a |delta|/ATR thermometer.
If ``K < 8`` the layout is truncated.

HYPERPARAMETERS (3)
-------------------
    K           : int   (default 8)
    alpha_atr   : float (default 0.05)
    alpha_mom   : float (default 0.10)   EMA decay for delta-momentum
"""

import numpy as np
from .base import ThermometerEncoder


class OnlineDeltaMomentumBinarizer(ThermometerEncoder):
    _BASE_BITS = 8

    def __init__(self, K: int = 8, alpha_atr: float = 0.05, alpha_mom: float = 0.10):
        super().__init__(K=K, name="ODMB")
        self.alpha_atr = float(alpha_atr)
        self.alpha_mom = float(alpha_mom)
        self._n_extra = max(0, K - self._BASE_BITS)
        if self._n_extra > 0:
            self._levels = np.linspace(0.0, 4.0, self._n_extra + 1)[1:]
        else:
            self._levels = np.empty(0, dtype=np.float64)

    def _init_state(self):
        n = self.n_features
        self.n_seen_ = 0
        self.x_prev_ = None
        self.x_prev2_ = None
        self.atr_ = np.full(n, 1e-6, dtype=np.float64)
        self.momentum_ = np.zeros(n, dtype=np.float64)
        self.ema_max_ = np.full(n, -np.inf, dtype=np.float64)
        self.ema_max_decay_ = 0.999  # gentle decay so old extrema fade

    def _cold_start_init(self, x: np.ndarray):
        self.n_features = len(x)
        self._init_state()
        self.fitted = True

    def fit(self, X: np.ndarray) -> "OnlineDeltaMomentumBinarizer":
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
        if self.x_prev_ is None:
            self.x_prev_ = x.copy()
            self.ema_max_ = x.copy()
            return np.zeros(self.n_features * self.K, dtype=np.uint8)

        t = self.n_seen_
        a_atr = min(self.alpha_atr / max(1.0 - (1.0 - self.alpha_atr) ** t, 1e-12), 1.0) if t < 200 else self.alpha_atr
        a_mom = min(self.alpha_mom / max(1.0 - (1.0 - self.alpha_mom) ** t, 1e-12), 1.0) if t < 200 else self.alpha_mom

        delta = x - self.x_prev_
        self.atr_ = (1.0 - a_atr) * self.atr_ + a_atr * np.abs(delta)
        self.momentum_ = (1.0 - a_mom) * self.momentum_ + a_mom * delta

        # Decaying rolling maximum
        self.ema_max_ = np.maximum(self.ema_max_ * self.ema_max_decay_, x)

        atr = np.maximum(self.atr_, 1e-12)
        n = self.n_features
        K = self.K
        bits = np.zeros(n * K, dtype=np.uint8)

        delta_sign = (delta > 0).astype(np.uint8)
        abs_d = np.abs(delta)
        small_move = (abs_d > 0.5 * atr).astype(np.uint8)
        med_move = (abs_d > atr).astype(np.uint8)
        large_move = (abs_d > 2.0 * atr).astype(np.uint8)
        mom_pos = (self.momentum_ > 0).astype(np.uint8)
        mom_strong = (np.abs(self.momentum_) > 0.5 * atr).astype(np.uint8)
        near_max = (x >= self.ema_max_ - atr).astype(np.uint8)

        if self.x_prev2_ is not None:
            prev_delta = self.x_prev_ - self.x_prev2_
            accel_sign = (delta > prev_delta).astype(np.uint8)
        else:
            accel_sign = np.zeros(n, dtype=np.uint8)

        base_bits = [delta_sign, small_move, med_move, large_move,
                     mom_pos, mom_strong, near_max, accel_sign]
        n_base = min(self._BASE_BITS, K)
        for k in range(n_base):
            bits[k::K] = base_bits[k]

        # Extra |delta|/ATR thermometer
        if self._n_extra > 0:
            ratio = abs_d / atr
            for k in range(self._n_extra):
                slot = n_base + k
                if slot >= K:
                    break
                bits[slot::K] = (ratio >= self._levels[k]).astype(np.uint8)

        self.x_prev2_ = self.x_prev_
        self.x_prev_ = x.copy()
        return bits

    def get_n_output_bits(self) -> int:
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K
