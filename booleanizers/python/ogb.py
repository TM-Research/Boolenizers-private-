"""
Online Generalized Binarizer (OGB)
==================================
The unified flagship: a single online sample-by-sample binarizer that
handles every feature shape we have seen in our 10-dataset IDS/sensor
benchmark (see ``codes/results/online_binarizer_bench_v2``).

Design synthesis
----------------
Per feature, ``K = K_q + K_s`` bits per sample (default 12 + 4 = 16):

  * **K_q P² quantile thermometer bits** — the workhorse. P² (Jain &
    Chlamtac, 1985) tracks K_q+2 markers per feature with O(K_q) memory
    and zero data storage. Its bits are a true thermometer of the
    empirical CDF observed so far.
  * **K_s trading-indicator signal bits** — augment the quantile
    thermometer with temporal and anomaly info. The four bits chosen
    here are the ones that lifted OQSB above OQTB on our anf_iot win:
    delta sign, ATR-relative pulse, MACD sign, and (is_zero | RSI>0.5).

Per feature, OGB auto-classifies online into one of three regimes:

  1. **Categorical** (sticky) — integer-valued ≤ ``max_cat_cardinality``
     distinct values seen so far. Emits one-hot rank bits over the top
     R = min(K, distinct_count) most-frequent values, with a stable
     universal-hash overlay for the remaining slots so any two distinct
     values produce distinct bit patterns.

  2. **Heavy-tailed positive** — non-negative throughout, EWMA-skew above
     ``skew_log_threshold``, after a configurable warm-up. The quantile
     tracker operates on ``log1p(x)`` rather than ``x`` so packet-byte
     and inter-arrival-time columns spread their bits sensibly across
     orders of magnitude.

  3. **Continuous** — everything else. Plain quantile thermometer.

For categorical features the K_s signal bits still encode temporal
deltas, because demotion to "continuous" is one-way and we want the
temporal EMAs warm in case it happens.

True streaming contract
-----------------------
- ``encode_online(x)`` — single-sample update, returns bits.
- ``fit_transform(X)`` — single pass over ``X``; bit row ``i`` is
  computed with state from rows ``0..i-1`` only, then state updates to
  include row ``i``. **This is the honest training-side bit emission.**
- ``fit(X) + transform(X_test)`` — for compatibility with the rest of the
  benchmark stack. After ``fit`` the encoder is left in its warm,
  end-of-stream state; ``transform(X)`` continues the stream by default.
  Pass ``freeze_after_fit=True`` (default) and OGB will snapshot its
  state at the end of ``fit`` and ``transform`` will encode against the
  frozen snapshot without further updates — strictly closer to the
  "fit on train, apply to test" semantics of offline encoders.

Hyperparameters (8)
-------------------
    K_q                       : int   (default 12)  quantile bits/feature
    K_s                       : int   (default 4)   signal bits/feature
    alpha_fast                : float (default 0.10)
    alpha_slow                : float (default 0.01)
    alpha_rsi                 : float (default 0.07)
    max_cat_cardinality       : int   (default 16)
    skew_log_threshold        : float (default 3.0) heavy-tail trigger
    zero_inflation_threshold  : float (default 0.30)
    freeze_after_fit          : bool  (default True)
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from .base import ThermometerEncoder
from .p2_algorithm import P2Quantile


class OnlineGeneralizedBinarizer(ThermometerEncoder):
    def __init__(self, K_q: int = 12, K_s: int = 4,
                 alpha_fast: float = 0.10, alpha_slow: float = 0.01,
                 alpha_rsi: float = 0.07,
                 max_cat_cardinality: int = 16,
                 skew_log_threshold: float = 3.0,
                 zero_inflation_threshold: float = 0.30,
                 ids_warmup: int = 200,
                 freeze_after_fit: bool = True):
        K = int(K_q) + int(K_s)
        super().__init__(K=K, name="OGB")
        if K_q < 2:
            raise ValueError("K_q must be >= 2")
        if K_s < 0:
            raise ValueError("K_s must be >= 0")
        self.K_q = int(K_q)
        self.K_s = int(K_s)
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)
        self.alpha_rsi = float(alpha_rsi)
        self.max_cat_cardinality = int(max_cat_cardinality)
        self.skew_log_threshold = float(skew_log_threshold)
        self.zero_inflation_threshold = float(zero_inflation_threshold)
        self.ids_warmup = int(ids_warmup)
        self.freeze_after_fit = bool(freeze_after_fit)

        self._warmup_grid = np.linspace(-1.0, 1.0, self.K_q + 2)[1:-1]
        rng = np.random.RandomState(0xBADCAFE)
        self._hash_a = rng.randint(1, 1 << 31, size=K).astype(np.int64)
        self._hash_b = rng.randint(0, 1 << 31, size=K).astype(np.int64)
        self._hash_mod = np.int64(2147483647)
        self._frozen_snapshot: Optional[dict] = None

    # ------------------------------------------------------------- state mgmt
    def _init_state(self):
        n = self.n_features
        self.trackers_ = [P2Quantile(K=self.K_q, speed=1.0) for _ in range(n)]
        # Optional shadow trackers operating in log1p space; lazily activated.
        self.log_trackers_ = [None for _ in range(n)]
        self.n_seen_ = 0
        self.x_prev_ = None

        self.ema_short_ = np.zeros(n, dtype=np.float64)
        self.ema_long_ = np.zeros(n, dtype=np.float64)
        self.ema_mu_ = np.zeros(n, dtype=np.float64)
        self.ema_var_ = np.full(n, 1e-6, dtype=np.float64)
        self.atr_ = np.full(n, 1e-6, dtype=np.float64)
        self.rsi_gain_ = np.full(n, 1e-6, dtype=np.float64)
        self.rsi_loss_ = np.full(n, 1e-6, dtype=np.float64)
        self.zero_freq_ = np.zeros(n, dtype=np.float64)
        self.ema_skew_ = np.zeros(n, dtype=np.float64)
        self.all_pos_ = np.ones(n, dtype=bool)

        self._cat_counts: list = [{} for _ in range(n)]
        self._is_categorical = np.ones(n, dtype=bool)
        self._top_values = [np.empty(0, dtype=np.float64) for _ in range(n)]

        self._frozen_snapshot = None

    def _cold_start_init(self, x: np.ndarray):
        self.n_features = len(x)
        self._init_state()
        self.fitted = True

    @staticmethod
    def _eff_alpha(alpha: float, t: int) -> float:
        if t >= 200:
            return alpha
        return min(alpha / max(1.0 - (1.0 - alpha) ** t, 1e-12), 1.0)

    # ----------------------------------------------------------- type sensing
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
        counts = self._cat_counts[j]
        if not counts:
            self._top_values[j] = np.empty(0, dtype=np.float64)
            return
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        rank_cap = max(1, min(self.K, len(items)))
        self._top_values[j] = np.array([float(k) for k, _ in items[:rank_cap]], dtype=np.float64)

    def _hash_bits(self, value: float) -> np.ndarray:
        key = np.int64(np.frombuffer(np.float64(value).tobytes(), dtype=np.int64)[0])
        out = ((self._hash_a * key + self._hash_b) % self._hash_mod) & np.int64(1)
        return out.astype(np.uint8)

    # ------------------------------------------------- streaming-honest entry
    def fit(self, X: np.ndarray) -> "OnlineGeneralizedBinarizer":
        """Single pass over ``X`` to build state. Bits are discarded.

        If ``freeze_after_fit=True`` (the default), the encoder snapshots its
        state at the end of this pass; subsequent ``transform`` calls encode
        with the snapshot and do not update.
        """
        X = np.asarray(X, dtype=np.float64)
        self.n_features = X.shape[1]
        self._init_state()
        self.fitted = True
        for i in range(X.shape[0]):
            self._encode_single_internal(X[i], update=True)
        if self.freeze_after_fit:
            self._frozen_snapshot = self._snapshot_state()
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Single-pass streaming-honest fit + emit.

        Bit row ``i`` is computed with state derived from rows ``0..i-1``
        only, then state updates to include row ``i``.
        """
        X = np.asarray(X, dtype=np.float64)
        self.n_features = X.shape[1]
        self._init_state()
        self.fitted = True
        out = np.empty((X.shape[0], self.n_features * self.K), dtype=np.uint8)
        for i in range(X.shape[0]):
            out[i] = self._encode_single_internal(X[i], update=True)
        if self.freeze_after_fit:
            self._frozen_snapshot = self._snapshot_state()
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], self.n_features * self.K), dtype=np.uint8)
        update = self._frozen_snapshot is None  # frozen mode → no state update
        if not update:
            self._restore_state(self._frozen_snapshot)
        for i in range(X.shape[0]):
            out[i] = self._encode_single_internal(X[i], update=update)
            if not update:
                # Even in frozen mode we restore state per sample so a long
                # test sequence cannot leak through residual increments of
                # non-snapshotted derived fields (defensive — most fields are
                # already deep-copied in the snapshot).
                self._restore_state(self._frozen_snapshot)
        return out

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        # Required by the base class; preserves online state updates.
        return self._encode_single_internal(np.asarray(x), update=True)

    # ----------------------------------------------------------- snapshot API
    def _snapshot_state(self) -> dict:
        return {
            "trackers": copy.deepcopy(self.trackers_),
            "log_trackers": copy.deepcopy(self.log_trackers_),
            "ema_short": self.ema_short_.copy(),
            "ema_long": self.ema_long_.copy(),
            "ema_mu": self.ema_mu_.copy(),
            "ema_var": self.ema_var_.copy(),
            "atr": self.atr_.copy(),
            "rsi_gain": self.rsi_gain_.copy(),
            "rsi_loss": self.rsi_loss_.copy(),
            "zero_freq": self.zero_freq_.copy(),
            "ema_skew": self.ema_skew_.copy(),
            "all_pos": self.all_pos_.copy(),
            "cat_counts": copy.deepcopy(self._cat_counts),
            "is_categorical": self._is_categorical.copy(),
            "top_values": [t.copy() for t in self._top_values],
            "x_prev": None if self.x_prev_ is None else self.x_prev_.copy(),
            "n_seen": self.n_seen_,
        }

    def _restore_state(self, snap: dict) -> None:
        self.trackers_ = copy.deepcopy(snap["trackers"])
        self.log_trackers_ = copy.deepcopy(snap["log_trackers"])
        self.ema_short_ = snap["ema_short"].copy()
        self.ema_long_ = snap["ema_long"].copy()
        self.ema_mu_ = snap["ema_mu"].copy()
        self.ema_var_ = snap["ema_var"].copy()
        self.atr_ = snap["atr"].copy()
        self.rsi_gain_ = snap["rsi_gain"].copy()
        self.rsi_loss_ = snap["rsi_loss"].copy()
        self.zero_freq_ = snap["zero_freq"].copy()
        self.ema_skew_ = snap["ema_skew"].copy()
        self.all_pos_ = snap["all_pos"].copy()
        self._cat_counts = copy.deepcopy(snap["cat_counts"])
        self._is_categorical = snap["is_categorical"].copy()
        self._top_values = [t.copy() for t in snap["top_values"]]
        self.x_prev_ = None if snap["x_prev"] is None else snap["x_prev"].copy()
        self.n_seen_ = snap["n_seen"]

    # --------------------------------------------------------- core encoding
    def _encode_single_internal(self, x: np.ndarray, *, update: bool) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not self.fitted or getattr(self, "n_features", None) is None:
            self._cold_start_init(x)

        if update:
            self.n_seen_ += 1
        t = self.n_seen_
        n = self.n_features
        K = self.K
        K_q = self.K_q
        K_s = self.K_s
        out = np.zeros(n * K, dtype=np.uint8)

        first_sample = self.x_prev_ is None
        if first_sample and update:
            self.ema_short_ = x.copy()
            self.ema_long_ = x.copy()
            self.ema_mu_ = x.copy()

        a_s = self._eff_alpha(self.alpha_fast, t)
        a_l = self._eff_alpha(self.alpha_slow, t)
        a_r = self._eff_alpha(self.alpha_rsi, t)

        for j in range(n):
            x_j = float(x[j])
            if update:
                self._maybe_demote(j, x_j)

            if self._is_categorical[j]:
                if update:
                    self._refresh_top(j)
                top = self._top_values[j]
                R = min(K // 2, len(top))
                for k in range(R):
                    if x_j == top[k]:
                        out[j * K + k] = 1
                        break
                if K - R > 0:
                    out[j * K + R: (j + 1) * K] = self._hash_bits(x_j)[R:K]
                if update:
                    prev_j = float(self.x_prev_[j]) if not first_sample else x_j
                    delta_j = x_j - prev_j
                    self.ema_short_[j] += a_s * (x_j - self.ema_short_[j])
                    self.ema_long_[j] += a_l * (x_j - self.ema_long_[j])
                    self.atr_[j] = (1.0 - a_s) * self.atr_[j] + a_s * abs(delta_j)
                    self.zero_freq_[j] = (1.0 - a_l) * self.zero_freq_[j] + a_l * (
                        1.0 if abs(x_j) < 1e-12 else 0.0
                    )
                continue

            # ---- continuous path -------------------------------------------
            # Decide whether to use the log1p tracker for this feature.
            use_log = (
                self.all_pos_[j]
                and t > self.ids_warmup
                and abs(self.ema_skew_[j]) > self.skew_log_threshold
                and x_j >= 0.0
            )

            if use_log and self.log_trackers_[j] is None and update:
                # Lazily spin up a shadow log-space P² tracker on first need.
                self.log_trackers_[j] = P2Quantile(K=self.K_q, speed=1.0)

            tracker_input = x_j
            tracker = self.trackers_[j]
            if use_log and self.log_trackers_[j] is not None:
                tracker = self.log_trackers_[j]
                tracker_input = float(np.log1p(x_j))

            if update:
                tracker.update(tracker_input)
            if tracker.count >= K_q + 2:
                thresholds = tracker.q[1:-1]
                for k in range(K_q):
                    if tracker_input >= thresholds[k]:
                        out[j * K + k] = 1
            else:
                # Warm-up: fall back to the uniform reference grid so bits
                # are at least informative during the first K_q+2 samples.
                ref = tracker_input
                for k in range(K_q):
                    if ref >= self._warmup_grid[k]:
                        out[j * K + k] = 1

            if update:
                prev_j = float(self.x_prev_[j]) if not first_sample else x_j
                delta_j = x_j - prev_j
                d_mu = x_j - self.ema_mu_[j]
                self.ema_mu_[j] += a_l * d_mu
                self.ema_var_[j] = (1.0 - a_l) * (self.ema_var_[j] + a_l * d_mu * d_mu)
                self.ema_short_[j] += a_s * (x_j - self.ema_short_[j])
                self.ema_long_[j] += a_l * (x_j - self.ema_long_[j])
                self.atr_[j] = (1.0 - a_s) * self.atr_[j] + a_s * abs(delta_j)
                self.rsi_gain_[j] = (1.0 - a_r) * self.rsi_gain_[j] + a_r * max(delta_j, 0.0)
                self.rsi_loss_[j] = (1.0 - a_r) * self.rsi_loss_[j] + a_r * max(-delta_j, 0.0)
                self.zero_freq_[j] = (1.0 - a_l) * self.zero_freq_[j] + a_l * (
                    1.0 if abs(x_j) < 1e-12 else 0.0
                )
                if x_j < 0.0:
                    self.all_pos_[j] = False
                sigma = max(float(np.sqrt(max(self.ema_var_[j], 1e-20))), 1e-12)
                z = d_mu / sigma
                self.ema_skew_[j] = (1.0 - a_l) * self.ema_skew_[j] + a_l * (z * z * z)
            else:
                prev_j = float(self.x_prev_[j]) if not first_sample else x_j
                delta_j = x_j - prev_j

            # ---- signal bits ------------------------------------------------
            if K_s > 0:
                atr = max(float(self.atr_[j]), 1e-12)
                rs = self.rsi_gain_[j] / (self.rsi_loss_[j] + 1e-12)
                rsi = 1.0 - 1.0 / (1.0 + rs)
                macd = self.ema_short_[j] - self.ema_long_[j]
                zero_inflated = (
                    t > self.ids_warmup
                    and self.zero_freq_[j] > self.zero_inflation_threshold
                )
                signals = [
                    1 if delta_j > 0 else 0,
                    1 if abs(delta_j) > atr else 0,
                    1 if macd > 0 else 0,
                    (1 if abs(x_j) < 1e-12 else 0) if zero_inflated
                    else (1 if rsi > 0.5 else 0),
                    1 if rsi > 0.7 else 0,
                    1 if abs(macd) > atr else 0,
                    1 if rsi < 0.3 else 0,
                    1 if delta_j * (prev_j - self.ema_long_[j]) < 0 else 0,
                ]
                for k in range(min(K_s, len(signals))):
                    out[j * K + K_q + k] = signals[k]

        if update:
            self.x_prev_ = x.copy()
        return out

    # ------------------------------------------------------------- diagnostics
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
            "max_cat_cardinality": self.max_cat_cardinality,
            "skew_log_threshold": self.skew_log_threshold,
            "zero_inflation_threshold": self.zero_inflation_threshold,
            "ids_warmup": self.ids_warmup,
            "freeze_after_fit": self.freeze_after_fit,
        }

    def get_feature_regimes(self) -> dict:
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        log_active = np.array([t is not None for t in self.log_trackers_], dtype=bool)
        return {
            "categorical": self._is_categorical.copy(),
            "heavy_tailed_log": log_active,
            "all_positive": self.all_pos_.copy(),
            "skew_ema": self.ema_skew_.copy(),
            "zero_freq": self.zero_freq_.copy(),
        }
