"""
Online Quantile-Signal Binarizer (OQSB)
=======================================
True sample-by-sample online binarizer that fuses:

  * **Quantile thermometer** — per-feature P² streaming quantile estimator
    (Jain & Chlamtac, 1985), producing ``K_q`` thermometer bits that match
    the empirical CDF observed so far. This is the workhorse — it is the
    bit family that wins on most IDS datasets in our own benchmarks
    (see codes/results/online_binarizer_bench).
  * **Trading-indicator signal bits** — ``K_s`` bits per feature derived from
    online EMAs of delta, MACD, RSI, volatility regime and ATR-relative
    moves. These augment the quantile thermometer with *temporal* and
    *anomaly* information that pure quantiles miss.
  * **Zero-inflation + categorical detection** — IDS counters that are
    zero most of the time emit an ``is_zero`` bit; small-cardinality
    integer features fall back to a stable universal-hash signature.

Each feature emits ``K = K_q + K_s`` bits per sample. The default
``K_q = 8`` matches the strongest single-family result from our
benchmarks; ``K_s = 4`` covers the four most consistently informative
signal bits (sign of delta, ATR-relative pulse, MACD sign, is_zero).

Per-feature online state (no batch lookback):
    P² tracker (markers, no value storage)
    ema_short, ema_long  — MACD-style fast/slow EMAs
    atr                  — EMA of |delta|
    rsi_gain / rsi_loss  — RSI EMAs
    x_prev               — last sample
    n_seen               — count

Cold-start: the very first ``K_q + 2`` samples populate the P² markers;
during this warm-up the quantile bits are emitted against a uniform
``[-1, +1]`` reference grid so they aren't all dead.

Hyperparameters (5)
-------------------
    K_q             : int   (default 8)   quantile thermometer bits / feature
    K_s             : int   (default 4)   signal bits / feature
    alpha_fast      : float (default 0.10)
    alpha_slow      : float (default 0.01)
    zero_inflation_threshold : float (default 0.30)
"""

from __future__ import annotations

import numpy as np

from .base import ThermometerEncoder
from .p2_algorithm import P2Quantile


class OnlineQuantileSignalBinarizer(ThermometerEncoder):
    def __init__(self, K_q: int = 8, K_s: int = 4,
                 alpha_fast: float = 0.10, alpha_slow: float = 0.01,
                 alpha_rsi: float = 0.07,
                 zero_inflation_threshold: float = 0.30,
                 max_cat_cardinality: int = 16):
        K = int(K_q) + int(K_s)
        super().__init__(K=K, name="OQSB")
        if K_q < 2:
            raise ValueError("K_q must be >= 2")
        if K_s < 0:
            raise ValueError("K_s must be >= 0")
        self.K_q = int(K_q)
        self.K_s = int(K_s)
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)
        self.alpha_rsi = float(alpha_rsi)
        self.zero_inflation_threshold = float(zero_inflation_threshold)
        self.max_cat_cardinality = int(max_cat_cardinality)
        self._warmup_grid = np.linspace(-1.0, 1.0, self.K_q + 2)[1:-1]

        rng = np.random.RandomState(0xC0FFEE)
        self._hash_a = rng.randint(1, 1 << 31, size=K).astype(np.int64)
        self._hash_b = rng.randint(0, 1 << 31, size=K).astype(np.int64)
        self._hash_mod = np.int64(2147483647)

    def _init_state(self):
        n = self.n_features
        self.trackers_ = [P2Quantile(K=self.K_q, speed=1.0) for _ in range(n)]
        self.n_seen_ = 0
        self.x_prev_ = None
        self.ema_short_ = np.zeros(n, dtype=np.float64)
        self.ema_long_ = np.zeros(n, dtype=np.float64)
        self.atr_ = np.full(n, 1e-6, dtype=np.float64)
        self.rsi_gain_ = np.full(n, 1e-6, dtype=np.float64)
        self.rsi_loss_ = np.full(n, 1e-6, dtype=np.float64)
        self.zero_freq_ = np.zeros(n, dtype=np.float64)

        # Categorical detection (sticky integer + low cardinality)
        self._cat_counts: list = [{} for _ in range(n)]
        self._is_categorical = np.ones(n, dtype=bool)
        self._top_values = [np.empty(0, dtype=np.float64) for _ in range(n)]

    def _cold_start_init(self, x: np.ndarray):
        self.n_features = len(x)
        self._init_state()
        self.fitted = True

    @staticmethod
    def _eff_alpha(alpha: float, t: int) -> float:
        if t >= 200:
            return alpha
        return min(alpha / max(1.0 - (1.0 - alpha) ** t, 1e-12), 1.0)

    def _maybe_demote(self, j: int, value: float) -> None:
        if not self._is_categorical[j]:
            return
        if self.max_cat_cardinality == 0:
            self._is_categorical[j] = False
            self._cat_counts[j] = None
            return
        if not np.isfinite(value):
            return
        if abs(value - round(value)) > 1e-9:
            self._is_categorical[j] = False
            self._cat_counts[j] = None
            self._top_values[j] = np.empty(0, dtype=np.float64)
            return
        counts = self._cat_counts[j]
        k = int(round(value))
        counts[k] = counts.get(k, 0) + 1
        if len(counts) > self.max_cat_cardinality:
            self._is_categorical[j] = False
            self._cat_counts[j] = None
            self._top_values[j] = np.empty(0, dtype=np.float64)

    def _refresh_top(self, j: int) -> None:
        if not self._is_categorical[j]:
            return
        counts = self._cat_counts[j]
        if not counts:
            self._top_values[j] = np.empty(0, dtype=np.float64)
            return
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        # Use up to K (= K_q + K_s) rank slots, but reserve room for hash bits later.
        rank_cap = max(1, min(self.K, len(items)))
        self._top_values[j] = np.array([float(k) for k, _ in items[:rank_cap]], dtype=np.float64)

    def _hash_bits(self, value: float) -> np.ndarray:
        key = np.int64(np.frombuffer(np.float64(value).tobytes(), dtype=np.int64)[0])
        out = ((self._hash_a * key + self._hash_b) % self._hash_mod) & np.int64(1)
        return out.astype(np.uint8)

    def fit(self, X: np.ndarray) -> "OnlineQuantileSignalBinarizer":
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
        n = self.n_features
        K = self.K
        K_q = self.K_q
        out = np.zeros(n * K, dtype=np.uint8)

        first_sample = self.x_prev_ is None
        if first_sample:
            self.ema_short_ = x.copy()
            self.ema_long_ = x.copy()

        a_s = self._eff_alpha(self.alpha_fast, t)
        a_l = self._eff_alpha(self.alpha_slow, t)
        a_r = self._eff_alpha(self.alpha_rsi, t)

        for j in range(n):
            x_j = float(x[j])
            self._maybe_demote(j, x_j)

            if self._is_categorical[j]:
                # Categorical path: emit hash + rank bits in the K-bit slot.
                self._refresh_top(j)
                top = self._top_values[j]
                R = min(K // 2, len(top))
                for k in range(R):
                    if x_j == top[k]:
                        out[j * K + k] = 1
                        break
                if K - R > 0:
                    out[j * K + R: (j + 1) * K] = self._hash_bits(x_j)[R:K]
                # We still want to update temporal state so a later demotion
                # finds the temporal EMAs warm.
                prev_j = float(self.x_prev_[j]) if not first_sample else x_j
                delta_j = x_j - prev_j
                self.ema_short_[j] += a_s * (x_j - self.ema_short_[j])
                self.ema_long_[j] += a_l * (x_j - self.ema_long_[j])
                self.atr_[j] = (1.0 - a_s) * self.atr_[j] + a_s * abs(delta_j)
                self.zero_freq_[j] = (1.0 - a_l) * self.zero_freq_[j] + a_l * (
                    1.0 if abs(x_j) < 1e-12 else 0.0
                )
                continue

            # Continuous path -------------------------------------------------
            tracker = self.trackers_[j]
            # 1) Update quantile tracker FIRST so its markers reflect this sample.
            tracker.update(x_j)
            # 2) Quantile thermometer bits.
            if tracker.count >= K_q + 2:
                thresholds = tracker.q[1:-1]
                for k in range(K_q):
                    if x_j >= thresholds[k]:
                        out[j * K + k] = 1
            else:
                for k in range(K_q):
                    if x_j >= self._warmup_grid[k]:
                        out[j * K + k] = 1

            # 3) Temporal EMAs
            prev_j = float(self.x_prev_[j]) if not first_sample else x_j
            delta_j = x_j - prev_j
            self.ema_short_[j] += a_s * (x_j - self.ema_short_[j])
            self.ema_long_[j] += a_l * (x_j - self.ema_long_[j])
            self.atr_[j] = (1.0 - a_s) * self.atr_[j] + a_s * abs(delta_j)
            self.rsi_gain_[j] = (1.0 - a_r) * self.rsi_gain_[j] + a_r * max(delta_j, 0.0)
            self.rsi_loss_[j] = (1.0 - a_r) * self.rsi_loss_[j] + a_r * max(-delta_j, 0.0)
            self.zero_freq_[j] = (1.0 - a_l) * self.zero_freq_[j] + a_l * (
                1.0 if abs(x_j) < 1e-12 else 0.0
            )

            # 4) Signal bits (up to K_s)
            atr = max(float(self.atr_[j]), 1e-12)
            rs = self.rsi_gain_[j] / (self.rsi_loss_[j] + 1e-12)
            rsi = 1.0 - 1.0 / (1.0 + rs)
            macd = self.ema_short_[j] - self.ema_long_[j]
            zero_inflated = self.zero_freq_[j] > self.zero_inflation_threshold

            signals = [
                1 if delta_j > 0 else 0,                          # delta sign
                1 if abs(delta_j) > atr else 0,                   # ATR-relative pulse
                1 if macd > 0 else 0,                             # MACD sign
                # IDS-friendly: is_zero when zero-inflated, else RSI > 0.5
                (1 if abs(x_j) < 1e-12 else 0) if zero_inflated
                else (1 if rsi > 0.5 else 0),
                1 if rsi > 0.7 else 0,                            # RSI overbought
                1 if abs(macd) > atr else 0,                      # MACD strength
                1 if rsi < 0.3 else 0,                            # RSI oversold
                1 if delta_j * (prev_j - self.ema_long_[j]) < 0   # zero-cross of (x - long)
                else 0,
            ]
            for k in range(min(self.K_s, len(signals))):
                out[j * K + K_q + k] = signals[k]

        self.x_prev_ = x.copy()
        return out

    def get_n_output_bits(self) -> int:
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K

    def get_config(self) -> dict:
        return {
            "K_q": self.K_q,
            "K_s": self.K_s,
            "K": self.K,
            "alpha_fast": self.alpha_fast,
            "alpha_slow": self.alpha_slow,
            "alpha_rsi": self.alpha_rsi,
            "zero_inflation_threshold": self.zero_inflation_threshold,
            "max_cat_cardinality": self.max_cat_cardinality,
        }
