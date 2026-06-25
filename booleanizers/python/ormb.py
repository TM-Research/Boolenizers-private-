"""
Online RSI/MACD Binarizer (ORMB)
================================
True sample-by-sample online binarizer using trading-indicator adaptations.

Each feature emits K bits per sample. The bit layout (for K=8 default):

    bit 0:  delta sign         (x_t > x_{t-1})
    bit 1:  zero-crossing      (sign(x_t - mu) != sign(x_{t-1} - mu))
    bit 2:  RSI > 0.5          (relative-strength index above midline)
    bit 3:  RSI > 0.7          (RSI overbought zone)
    bit 4:  MACD sign          (ema_short > ema_long)
    bit 5:  MACD vs ATR        (|ema_short - ema_long| > ATR)
    bit 6:  vol regime         (vol_short > vol_long, "expansion")
    bit 7:  above baseline     (x > ema_mu)

If ``K > 8``, additional bits form an RSI thermometer (RSI >= τ_k for
K-8 evenly-spaced τ_k in (0,1)).
If ``K < 8``, the layout is truncated from the tail.

HYPERPARAMETERS (4)
-------------------
    K               : int   (default 8)
    alpha_short     : float (default 0.10)   short EMA decay (~9 sample HL)
    alpha_long      : float (default 0.02)   long EMA decay (~50 sample HL)
    alpha_rsi       : float (default 0.07)   RSI gain/loss EMA decay
"""

import numpy as np
from .base import ThermometerEncoder


class OnlineRSIMACDBinarizer(ThermometerEncoder):
    _BASE_BITS = 8

    def __init__(self, K: int = 8, alpha_short: float = 0.10,
                 alpha_long: float = 0.02, alpha_rsi: float = 0.07):
        super().__init__(K=K, name="ORMB")
        self.alpha_short = float(alpha_short)
        self.alpha_long = float(alpha_long)
        self.alpha_rsi = float(alpha_rsi)
        self._n_extra_rsi = max(0, K - self._BASE_BITS)
        if self._n_extra_rsi > 0:
            self._rsi_levels = np.linspace(0.0, 1.0, self._n_extra_rsi + 2)[1:-1]
        else:
            self._rsi_levels = np.empty(0, dtype=np.float64)

    def _init_state(self):
        n = self.n_features
        self.n_seen_ = 0
        self.x_prev_ = None
        self.ema_mu_ = np.zeros(n, dtype=np.float64)
        self.ema_short_ = np.zeros(n, dtype=np.float64)
        self.ema_long_ = np.zeros(n, dtype=np.float64)
        self.rsi_gain_ = np.full(n, 1e-6, dtype=np.float64)
        self.rsi_loss_ = np.full(n, 1e-6, dtype=np.float64)
        self.atr_ = np.full(n, 1e-6, dtype=np.float64)
        self.vol_short_ = np.full(n, 1e-6, dtype=np.float64)
        self.vol_long_ = np.full(n, 1e-6, dtype=np.float64)

    def _cold_start_init(self, x: np.ndarray):
        self.n_features = len(x)
        self._init_state()
        self.fitted = True

    def fit(self, X: np.ndarray) -> "OnlineRSIMACDBinarizer":
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
        t = self.n_seen_

        if self.x_prev_ is None:
            self.x_prev_ = x.copy()
            self.ema_mu_ = x.copy()
            self.ema_short_ = x.copy()
            self.ema_long_ = x.copy()
            # First sample: emit all-zero bits (no useful comparison yet)
            return np.zeros(self.n_features * self.K, dtype=np.uint8)

        # Bias-corrected alphas during warm-up
        def _eff(a, t):
            return min(a / max(1.0 - (1.0 - a) ** t, 1e-12), 1.0) if t < 200 else a
        a_s = _eff(self.alpha_short, t)
        a_l = _eff(self.alpha_long, t)
        a_r = _eff(self.alpha_rsi, t)

        delta = x - self.x_prev_

        # Mean & MACD EMAs
        self.ema_mu_ += a_l * (x - self.ema_mu_)
        self.ema_short_ += a_s * (x - self.ema_short_)
        self.ema_long_ += a_l * (x - self.ema_long_)

        # ATR (EWMA of |delta|)
        self.atr_ = (1.0 - a_s) * self.atr_ + a_s * np.abs(delta)

        # RSI gain / loss
        gain = np.maximum(delta, 0.0)
        loss = np.maximum(-delta, 0.0)
        self.rsi_gain_ = (1.0 - a_r) * self.rsi_gain_ + a_r * gain
        self.rsi_loss_ = (1.0 - a_r) * self.rsi_loss_ + a_r * loss
        rs = self.rsi_gain_ / (self.rsi_loss_ + 1e-12)
        rsi = 1.0 - 1.0 / (1.0 + rs)   # in [0, 1]

        # Volatility regime (short vs long EWMA of squared delta)
        d2 = delta * delta
        self.vol_short_ = (1.0 - a_s) * self.vol_short_ + a_s * d2
        self.vol_long_ = (1.0 - a_l) * self.vol_long_ + a_l * d2

        # ---- bit emission ------------------------------------------------
        n = self.n_features
        K = self.K
        bits = np.zeros(n * K, dtype=np.uint8)

        # Common per-feature signals
        delta_sign = (delta > 0).astype(np.uint8)
        prev_above = (self.x_prev_ > self.ema_mu_)
        curr_above = (x > self.ema_mu_)
        zero_cross = (prev_above != curr_above).astype(np.uint8)
        rsi_gt_50 = (rsi > 0.5).astype(np.uint8)
        rsi_gt_70 = (rsi > 0.7).astype(np.uint8)
        macd = self.ema_short_ - self.ema_long_
        macd_sign = (macd > 0).astype(np.uint8)
        macd_strong = (np.abs(macd) > self.atr_).astype(np.uint8)
        vol_expand = (self.vol_short_ > self.vol_long_).astype(np.uint8)
        above_baseline = curr_above.astype(np.uint8)

        base_bits = [delta_sign, zero_cross, rsi_gt_50, rsi_gt_70,
                     macd_sign, macd_strong, vol_expand, above_baseline]
        n_base = min(self._BASE_BITS, K)
        for k in range(n_base):
            bits[k::K] = base_bits[k]

        # Extra RSI thermometer bits
        for k in range(self._n_extra_rsi):
            slot = n_base + k
            if slot >= K:
                break
            bits[slot::K] = (rsi >= self._rsi_levels[k]).astype(np.uint8)

        self.x_prev_ = x.copy()
        return bits

    def get_n_output_bits(self) -> int:
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K
