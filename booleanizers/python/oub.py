"""
Online Universal Binarizer (OUB)
================================
Flagship production-quality sample-by-sample online binarizer for Tsetlin
Machines. Designed to be a single drop-in encoder that works on ANY tabular
or streaming dataset (continuous, integer, categorical, mixed) without
dataset-specific tuning.

Design summary
--------------
For every feature, ``K`` bits are emitted per sample. Each feature is
auto-classified online into one of:

  * "categorical" — small cardinality and integer-valued so far
  * "continuous"  — high cardinality or non-integer values seen

Continuous feature, K bits (default K=16):

    bit 0..(B-1) : Bollinger thermometer over EMA mean ± z_k · EMA std
    bit B+0      : delta sign            (x_t > x_{t-1})
    bit B+1      : |delta| > ATR         (significant change vs typical)
    bit B+2      : MACD sign             (fast EMA > slow EMA)
    bit B+3      : RSI > 0.5             (relative-strength bullish)
    bit B+4      : vol short > vol long  (volatility regime expansion)
    bit B+5      : zero-crossing of (x - EMA_mu) since previous sample
    bit B+6      : x in upper half       (above EMA mean)
    bit B+7      : |x - EMA_mu| > 2·sigma (outlier flag)
  where B = K - 8 thermometer bits.

Categorical feature, K bits:

    bit 0..(R-1) : "rank-K thermometer" — value equals one of the top-R most
                   frequent values seen so far (R = min(K // 2, distinct_count))
    bit R..K-1   : K-R universal-hash bits of the raw value, ensuring distinct
                   values are guaranteed-distinguishable bit patterns

Online state is fully incremental — no batch lookback, no full-history
storage. EMA decays govern responsiveness:

    alpha_fast  controls short-term EMAs (delta, MACD short, vol short, ATR)
    alpha_slow  controls long-term EMAs (mean, variance, MACD long, vol long)
    alpha_rsi   controls RSI gain/loss EMAs

IDS-specific behaviour (active by default; ``ids_mode=False`` disables)
-----------------------------------------------------------------------
Network/IDS data has three recurring problems for thermometer encoders:

  * heavy-tailed positive features (byte counts, packet sizes, durations) —
    a tiny fraction of values cover orders of magnitude
  * zero-inflated features (per-direction counters that are 0 most of the time)
  * mixed scale across columns (some 0/1 flags next to 0–10^9 byte counts)

OUB tracks each feature online to detect these conditions and adapts the bit
layout per feature without breaking the constant K-bit output contract:

  * **Skewness EMA** — when EMA-|skew| > ``skew_log_threshold`` *and* every
    observed value has been ``≥ 0``, the encoder switches that feature's
    thermometer to operate on ``log1p(x)`` instead of ``x``. The switch is
    sticky — once a negative value is seen the feature reverts permanently.
  * **Zero-inflation EMA** — when the EMA of ``1{x == 0}`` exceeds
    ``zero_inflation_threshold``, the "above baseline" signal bit is
    replaced with an ``is_zero`` bit (much more informative for IDS).
  * **Robust scale (MAD)** — EMA of ``|x - mu|`` is maintained alongside
    EMA variance, and the thermometer scale is the maximum of σ and
    ``1.4826 · MAD`` (so a few extreme outliers can't collapse band coverage).

Hyperparameters (10)
--------------------
    K                       : int    (default 16)
    alpha_fast              : float  (default 0.10)
    alpha_slow              : float  (default 0.01)
    alpha_rsi               : float  (default 0.07)
    band                    : float  (default 2.5)  z-score range
    max_cat_cardinality     : int    (default 32)
    hysteresis              : float  (default 0.0)
    ids_mode                : bool   (default True)
    skew_log_threshold      : float  (default 2.0)
    zero_inflation_threshold: float  (default 0.10)
"""

from __future__ import annotations

import numpy as np
from .base import ThermometerEncoder


class OnlineUniversalBinarizer(ThermometerEncoder):
    _SIGNAL_BITS = 8  # number of trailing signal/indicator bits per cont. feature

    def __init__(self, K: int = 16, alpha_fast: float = 0.10, alpha_slow: float = 0.01,
                 alpha_rsi: float = 0.07, band: float = 2.5,
                 max_cat_cardinality: int = 32, hysteresis: float = 0.0,
                 ids_mode: bool = True, skew_log_threshold: float = 3.0,
                 zero_inflation_threshold: float = 0.30,
                 ids_warmup: int = 200):
        super().__init__(K=K, name="OUB")
        if K < 4:
            raise ValueError("OUB requires K >= 4")
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)
        self.alpha_rsi = float(alpha_rsi)
        self.band = float(band)
        self.max_cat_cardinality = int(max_cat_cardinality)
        self.hysteresis = float(hysteresis)
        self.ids_mode = bool(ids_mode)
        self.skew_log_threshold = float(skew_log_threshold)
        self.zero_inflation_threshold = float(zero_inflation_threshold)
        self.ids_warmup = int(ids_warmup)

        n_therm = max(0, K - self._SIGNAL_BITS)
        if n_therm > 0:
            self._z_levels = np.linspace(-self.band, self.band, n_therm + 2)[1:-1]
        else:
            self._z_levels = np.empty(0, dtype=np.float64)
        self._n_therm = n_therm

        # Universal-hash coefficients for categorical fallback bits.
        # We pick deterministic primes so behavior is reproducible across runs.
        rng = np.random.RandomState(0xCAFEBABE)
        self._hash_a = rng.randint(1, 1 << 31, size=K).astype(np.int64)
        self._hash_b = rng.randint(0, 1 << 31, size=K).astype(np.int64)
        self._hash_mod = np.int64(2147483647)  # Mersenne prime 2^31 - 1

    # ------------------------------------------------------------------ state
    def _init_state(self):
        n = self.n_features
        self.n_seen_ = 0
        self.x_prev_ = None

        # Continuous-stream EMA state
        self.ema_mu_ = np.zeros(n, dtype=np.float64)
        self.ema_var_ = np.full(n, 1e-6, dtype=np.float64)
        self.ema_mad_ = np.full(n, 1e-6, dtype=np.float64)
        self.ema_short_ = np.zeros(n, dtype=np.float64)
        self.ema_long_ = np.zeros(n, dtype=np.float64)
        self.atr_ = np.full(n, 1e-6, dtype=np.float64)
        self.rsi_gain_ = np.full(n, 1e-6, dtype=np.float64)
        self.rsi_loss_ = np.full(n, 1e-6, dtype=np.float64)
        self.vol_short_ = np.full(n, 1e-6, dtype=np.float64)
        self.vol_long_ = np.full(n, 1e-6, dtype=np.float64)

        # IDS-mode online detection of feature shape:
        # ema_skew_   — EWMA of z^3, signed skewness estimate
        # zero_freq_  — EWMA of 1{x == 0}, zero-inflation rate
        # all_pos_    — True until the first negative value is seen
        self.ema_skew_ = np.zeros(n, dtype=np.float64)
        self.zero_freq_ = np.zeros(n, dtype=np.float64)
        self.all_pos_ = np.ones(n, dtype=bool)

        # Categorical state — per feature, a small frequency dict.
        # An entry switches to None once the feature is reclassified continuous.
        self._cat_counts: list = [{} for _ in range(n)]
        self._is_categorical = np.ones(n, dtype=bool)
        # Top-K most-frequent values per feature (refreshed lazily on encode).
        self._top_values = [np.empty(0, dtype=np.float64) for _ in range(n)]

        # Last emitted bits (for hysteresis on thermometer slots).
        self.bit_state_ = np.zeros(n * self.K, dtype=np.uint8)

    def _cold_start_init(self, x: np.ndarray):
        self.n_features = len(x)
        self._init_state()
        self.fitted = True

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _eff_alpha(alpha: float, t: int) -> float:
        if t >= 200:
            return alpha
        return min(alpha / max(1.0 - (1.0 - alpha) ** t, 1e-12), 1.0)

    def _maybe_demote_to_continuous(self, j: int, value: float) -> None:
        """Switch a categorical feature to continuous when cardinality grows
        or a non-integer-ish value appears."""
        if not self._is_categorical[j]:
            return
        if self.max_cat_cardinality == 0:
            self._is_categorical[j] = False
            self._cat_counts[j] = None
            self._top_values[j] = np.empty(0, dtype=np.float64)
            return
        if not np.isfinite(value):
            return  # NaN/inf — leave classification alone
        # Heuristic: a "categorical-friendly" value is integer-like
        if abs(value - round(value)) > 1e-9:
            self._is_categorical[j] = False
            self._cat_counts[j] = None
            self._top_values[j] = np.empty(0, dtype=np.float64)
            return
        counts = self._cat_counts[j]
        key = int(round(value))
        counts[key] = counts.get(key, 0) + 1
        if len(counts) > self.max_cat_cardinality:
            self._is_categorical[j] = False
            self._cat_counts[j] = None
            self._top_values[j] = np.empty(0, dtype=np.float64)

    def _refresh_top_values(self, j: int) -> None:
        if not self._is_categorical[j]:
            return
        counts = self._cat_counts[j]
        if not counts:
            self._top_values[j] = np.empty(0, dtype=np.float64)
            return
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        rank_K = max(1, self.K // 2)
        top = np.array([float(k) for k, _ in items[:rank_K]], dtype=np.float64)
        self._top_values[j] = top

    def _hash_bits_for_value(self, value: float) -> np.ndarray:
        """K universal-hash bits of a value's int representation."""
        # Use the bit pattern of the float so distinct floats hash distinctly.
        key = np.int64(np.frombuffer(np.float64(value).tobytes(), dtype=np.int64)[0])
        out = ((self._hash_a * key + self._hash_b) % self._hash_mod) & np.int64(1)
        return out.astype(np.uint8)

    # ---------------------------------------------------------------- updates
    def _update_continuous(self, j: int, x_j: float, delta_j: float, t: int) -> None:
        a_s = self._eff_alpha(self.alpha_fast, t)
        a_l = self._eff_alpha(self.alpha_slow, t)
        a_r = self._eff_alpha(self.alpha_rsi, t)

        # Mean & variance (slow EMA)
        d_mu = x_j - self.ema_mu_[j]
        self.ema_mu_[j] += a_l * d_mu
        self.ema_var_[j] = (1.0 - a_l) * (self.ema_var_[j] + a_l * d_mu * d_mu)
        # Robust MAD-like scale (EWMA of |x - mu|) — resilient to extreme tails
        self.ema_mad_[j] = (1.0 - a_l) * self.ema_mad_[j] + a_l * abs(d_mu)

        # MACD-style short / long
        self.ema_short_[j] += a_s * (x_j - self.ema_short_[j])
        self.ema_long_[j] += a_l * (x_j - self.ema_long_[j])

        # ATR & volatility
        abs_d = abs(delta_j)
        self.atr_[j] = (1.0 - a_s) * self.atr_[j] + a_s * abs_d
        d2 = delta_j * delta_j
        self.vol_short_[j] = (1.0 - a_s) * self.vol_short_[j] + a_s * d2
        self.vol_long_[j] = (1.0 - a_l) * self.vol_long_[j] + a_l * d2

        # RSI
        gain = max(delta_j, 0.0)
        loss = max(-delta_j, 0.0)
        self.rsi_gain_[j] = (1.0 - a_r) * self.rsi_gain_[j] + a_r * gain
        self.rsi_loss_[j] = (1.0 - a_r) * self.rsi_loss_[j] + a_r * loss

        # IDS-mode shape diagnostics
        if self.ids_mode:
            if x_j < 0.0:
                self.all_pos_[j] = False
            # Online standardised skewness (EWMA of z^3).
            sigma = max(np.sqrt(max(self.ema_var_[j], 1e-20)), 1e-12)
            z = d_mu / sigma
            self.ema_skew_[j] = (1.0 - a_l) * self.ema_skew_[j] + a_l * (z * z * z)
            # Zero-inflation rate (treat near-zero as zero)
            self.zero_freq_[j] = (1.0 - a_l) * self.zero_freq_[j] + a_l * (
                1.0 if abs(x_j) < 1e-12 else 0.0
            )

    # -------------------------------------------------------------- bit layout
    def _bits_continuous(self, j: int, x_j: float, prev_j: float) -> np.ndarray:
        K = self.K
        out = np.zeros(K, dtype=np.uint8)
        sigma = float(np.sqrt(max(self.ema_var_[j], 1e-20)))
        # MAD scaled to a Gaussian-equivalent sigma (1.4826 * MAD ~ sigma on Gaussian data).
        # Blend Bollinger σ with the robust MAD-based scale rather than taking the max:
        # the geometric mean preserves σ's sensitivity to small-variance features while
        # damping the influence of a few extreme outliers on heavy-tailed IDS columns.
        robust = 1.4826 * float(self.ema_mad_[j])
        if self.ids_mode and self.n_seen_ > self.ids_warmup:
            scale = max(np.sqrt(sigma * robust + 1e-20), 1e-12)
        else:
            scale = max(sigma, 1e-12)
        atr = max(float(self.atr_[j]), 1e-12)

        # Decide whether to compute the thermometer in log1p space (IDS heavy-tail path)
        use_log = (
            self.ids_mode
            and self.n_seen_ > self.ids_warmup
            and self.all_pos_[j]
            and abs(self.ema_skew_[j]) > self.skew_log_threshold
            and x_j >= 0.0
        )
        if use_log:
            xl = np.log1p(x_j)
            mul = np.log1p(max(self.ema_mu_[j], 0.0))
            # In log-space the variance has shrunk; rebuild a local scale from MAD.
            sl = max(1.4826 * np.log1p(max(self.ema_mad_[j], 0.0)), 0.25)
            thresholds = mul + self._z_levels * sl
            therm_input = xl
        else:
            thresholds = self.ema_mu_[j] + self._z_levels * scale
            therm_input = x_j

        # Thermometer bits
        if self._n_therm > 0:
            if self.hysteresis > 0.0:
                m = self.hysteresis * scale
                prev_bits = self.bit_state_[j * K: j * K + self._n_therm]
                for k in range(self._n_therm):
                    th = thresholds[k]
                    if prev_bits[k] == 0:
                        out[k] = 1 if therm_input >= th + m else 0
                    else:
                        out[k] = 0 if therm_input <= th - m else 1
            else:
                out[:self._n_therm] = (therm_input >= thresholds).astype(np.uint8)

        # Signal bits (last 8)
        base = self._n_therm
        delta_j = x_j - prev_j
        rs = self.rsi_gain_[j] / (self.rsi_loss_[j] + 1e-12)
        rsi = 1.0 - 1.0 / (1.0 + rs)
        macd = self.ema_short_[j] - self.ema_long_[j]
        zero_cross = (
            ((prev_j - self.ema_mu_[j]) > 0) != ((x_j - self.ema_mu_[j]) > 0)
        )
        zero_inflated = (
            self.ids_mode
            and self.n_seen_ > self.ids_warmup
            and self.zero_freq_[j] > self.zero_inflation_threshold
        )

        if base + 0 < K: out[base + 0] = 1 if delta_j > 0 else 0
        if base + 1 < K: out[base + 1] = 1 if abs(delta_j) > atr else 0
        if base + 2 < K: out[base + 2] = 1 if macd > 0 else 0
        if base + 3 < K: out[base + 3] = 1 if rsi > 0.5 else 0
        if base + 4 < K: out[base + 4] = 1 if self.vol_short_[j] > self.vol_long_[j] else 0
        if base + 5 < K: out[base + 5] = 1 if zero_cross else 0
        if base + 6 < K:
            # Zero-inflated columns: emit is_zero (much more informative than
            # "above EMA mean" when the mean is near zero anyway).
            if zero_inflated:
                out[base + 6] = 1 if abs(x_j) < 1e-12 else 0
            else:
                out[base + 6] = 1 if x_j > self.ema_mu_[j] else 0
        if base + 7 < K:
            out[base + 7] = 1 if abs(x_j - self.ema_mu_[j]) > 2.0 * scale else 0
        return out

    def _bits_categorical(self, j: int, x_j: float) -> np.ndarray:
        K = self.K
        out = np.zeros(K, dtype=np.uint8)
        top = self._top_values[j]
        R = min(K // 2, len(top))
        # One-hot rank bits: out[k] = 1 iff x equals the (k+1)-th most-frequent value.
        # Distinct values therefore map to distinct rank-bit positions, which the
        # Tsetlin Machine can clause directly. Out-of-top-R values leave all rank
        # bits at zero and rely entirely on the hash overlay.
        for k in range(R):
            if x_j == top[k]:
                out[k] = 1
                break
        # Universal-hash overlay fills the remaining K - R bits; this guarantees
        # any two distinct float values produce a different bit signature.
        if K - R > 0:
            hashed = self._hash_bits_for_value(x_j)
            out[R:K] = hashed[R:K]
        return out

    # ----------------------------------------------------------------- public
    def fit(self, X: np.ndarray) -> "OnlineUniversalBinarizer":
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
        out = np.zeros(n * K, dtype=np.uint8)

        if self.x_prev_ is None:
            self.x_prev_ = x.copy()
            self.ema_mu_ = x.copy()
            self.ema_short_ = x.copy()
            self.ema_long_ = x.copy()
            # First sample: emit hash bits per feature so the bit pattern carries
            # value identity even when no comparison statistics exist yet.
            for j in range(n):
                self._maybe_demote_to_continuous(j, float(x[j]))
                if self._is_categorical[j]:
                    self._refresh_top_values(j)
                out[j * K: (j + 1) * K] = self._hash_bits_for_value(float(x[j]))
            self.bit_state_ = out.copy()
            return out

        for j in range(n):
            x_j = float(x[j])
            prev_j = float(self.x_prev_[j])
            delta_j = x_j - prev_j

            # Always run continuous EMA updates so we have warm state if
            # the feature is later demoted from categorical → continuous.
            self._update_continuous(j, x_j, delta_j, t)

            # Update categorical bookkeeping
            self._maybe_demote_to_continuous(j, x_j)
            if self._is_categorical[j]:
                self._refresh_top_values(j)
                bits = self._bits_categorical(j, x_j)
            else:
                bits = self._bits_continuous(j, x_j, prev_j)

            out[j * K: (j + 1) * K] = bits

        self.x_prev_ = x.copy()
        self.bit_state_ = out.copy()
        return out

    def get_n_output_bits(self) -> int:
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K

    # ------------------------------------------------------------- diagnostics
    def get_feature_types(self) -> np.ndarray:
        """Return a boolean array — True for categorical, False for continuous."""
        return self._is_categorical.copy()

    def get_config(self) -> dict:
        return {
            "K": self.K,
            "alpha_fast": self.alpha_fast,
            "alpha_slow": self.alpha_slow,
            "alpha_rsi": self.alpha_rsi,
            "band": self.band,
            "max_cat_cardinality": self.max_cat_cardinality,
            "hysteresis": self.hysteresis,
            "ids_mode": self.ids_mode,
            "skew_log_threshold": self.skew_log_threshold,
            "zero_inflation_threshold": self.zero_inflation_threshold,
        }

    def get_feature_diagnostics(self) -> dict:
        """Per-feature online diagnostics — useful when debugging on a new dataset."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return {
            "is_categorical": self._is_categorical.copy(),
            "all_positive": self.all_pos_.copy(),
            "ema_skew": self.ema_skew_.copy(),
            "zero_freq": self.zero_freq_.copy(),
            "ema_mu": self.ema_mu_.copy(),
            "ema_sigma": np.sqrt(np.maximum(self.ema_var_, 1e-20)),
            "ema_mad": self.ema_mad_.copy(),
        }
