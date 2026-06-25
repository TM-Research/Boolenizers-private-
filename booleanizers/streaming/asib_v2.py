"""
ASIB-v2 — Warmup-Anchored Adaptive-Quantile Streaming IDS Booleanizer
====================================================================

Iteration 2. ASIB-v1 was drift-robust but lagged batch quantile encoders on
stationary Macro-F1 (its fixed normal-z levels assume normality; its P² variant
collapses during warmup on wide datasets). ASIB-v2 fixes both by combining the
*batch-quality* threshold placement that won our 148-dataset study (SAQT/AQB:
cardinality-adaptive bit budget + empirical-quantile knots + categorical
detection + dead/duplicate pruning) with *online drift adaptation*:

  1. **Warmup (first W samples):** buffer, then fit per-feature thresholds —
     low-cardinality → unique-value (categorical) knots; else K_j quantile knots
     with K_j from a Freedman–Diaconis / Sturges budget, pruned of dead/duplicate
     bits. This gives batch-level stationary accuracy and a FIXED bit schema
     (so the Tsetlin Machine input width never changes).

  2. **Stream:** emit the thermometer against the current thresholds plus compact
     domain literals (`is_zero` for zero-inflated flow features, `burst` for
     |Δx|>scale, `drift` while Page–Hinkley fires). Keep a small recent ring
     buffer; run an O(1) Page–Hinkley drift test per feature.

  3. **Drift adaptation:** on a Page–Hinkley trigger for feature j, RE-FIT its
     threshold *values* from the recent window at the SAME fixed probabilities —
     width unchanged, positions tracked to the new regime (ADWIN-style fresh
     window, Gama et al. architecture: heavy re-fit off the hot path, only on drift).

O(1) amortized per sample (re-fit only on drift), low fixed memory.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from math import log2, ceil


def _fd_bins(col, kmax):
    n = len(col)
    if n < 4: return 1
    lo, hi = np.min(col), np.max(col)
    if hi <= lo: return 0
    q1, q3 = np.percentile(col, [25, 75]); iqr = q3 - q1
    sturges = max(1, ceil(log2(max(n, 2))) + 1)
    cap = min(kmax, sturges, max(1, len(np.unique(col)) - 1))
    if iqr <= 0: return cap
    h = 2 * iqr * n ** (-1 / 3)
    return int(np.clip(ceil((hi - lo) / h), 1, cap))


class ASIBv2:
    def __init__(self, warmup: int = 600, kmax: int = 12, max_cat: int = 12,
                 recent: int = 400, ph_delta: float = 0.25, ph_lambda: float = 10.0,
                 zero_inflation: float = 0.30, temporal: bool = True):
        self.warmup = int(warmup); self.kmax = int(kmax); self.max_cat = int(max_cat)
        self.recent = int(recent); self.ph_delta = float(ph_delta); self.ph_lambda = float(ph_lambda)
        self.zero_inflation = float(zero_inflation); self.temporal = bool(temporal)
        self.fitted = False; self.schema = False

    # ---- warmup schema fit (batch-quality thresholds) ------------------------
    def _fit_schema(self, B):
        d = B.shape[1]
        self.thr = []         # per-feature threshold array (sorted)
        self.probs = []       # per-feature quantile probabilities (None if categorical)
        self.is_zero = np.zeros(d, dtype=bool)
        for j in range(d):
            col = B[:, j]; u = np.unique(col)
            if len(u) <= self.max_cat + 1:                 # categorical / low-card
                t = u[1:].astype(np.float64) if len(u) > 1 else np.array([], np.float64)
                p = None
            else:
                K = _fd_bins(col, self.kmax)
                if K <= 0: t = np.array([], np.float64); p = None
                else:
                    p = (np.arange(1, K + 1)) / (K + 1)
                    t = np.quantile(col, p)
                    # prune dead/duplicate bits on the warmup window
                    keep = []; seen = []
                    for k, tv in enumerate(t):
                        s = int((col >= tv).sum())
                        if s == 0 or s == len(col): continue
                        if any(np.array_equal(col >= tv, c) for c in seen): continue
                        seen.append(col >= tv); keep.append(k)
                    t = t[keep]; p = p[keep] if len(keep) else None
            self.thr.append(np.sort(t)); self.probs.append(p)
            self.is_zero[j] = (np.mean(col == 0) > self.zero_inflation)
        self.kj = np.array([len(t) for t in self.thr])
        extra = (1 if self.temporal else 0)  # placeholder; computed below
        self.extra_bits = np.zeros(d, dtype=int)
        for j in range(d):
            e = 0
            if self.is_zero[j]: e += 1
            if self.temporal: e += 2          # burst + drift
            self.extra_bits[j] = e
        self.offsets = np.cumsum([0] + [int(self.kj[j] + self.extra_bits[j]) for j in range(d)])
        self.width = int(self.offsets[-1])
        # streaming state
        self.mu = B.mean(0); self.scale = np.maximum(np.abs(B - self.mu).mean(0), 1e-6)
        self.x_prev = B[-1].astype(np.float64).copy()
        self.ph_m = np.zeros(d); self.ph_min = np.zeros(d); self.ph_mean = np.zeros(d); self.ph_n = 0
        self.boost_left = np.zeros(d, dtype=np.int32)
        self.buf = deque(maxlen=self.recent)
        for i in range(max(0, len(B) - self.recent), len(B)):
            self.buf.append(B[i])
        self.d = d; self.schema = True

    def _encode_row(self, x):
        out = np.zeros(self.width, dtype=np.uint8)
        s = np.maximum(self.scale, 1e-9)
        for j in range(self.d):
            o = self.offsets[j]; t = self.thr[j]; k = len(t)
            if k:
                # thermometer: count of thresholds <= x  -> first c bits set (sorted)
                c = int(np.searchsorted(t, x[j], side="right"))
                if c: out[o:o + c] = 1
            p = o + k
            if self.is_zero[j]:
                out[p] = 1 if x[j] == 0 else 0; p += 1
            if self.temporal:
                out[p] = 1 if abs(x[j] - self.x_prev[j]) > s[j] else 0; p += 1     # burst
                out[p] = 1 if self.boost_left[j] > 0 else 0; p += 1                 # drift
        return out

    def _update(self, x):
        z = (x - self.mu) / np.maximum(self.scale, 1e-9)
        self.mu += 0.02 * (x - self.mu)
        self.scale = 0.98 * self.scale + 0.02 * np.abs(x - self.mu)
        az = np.abs(z); self.ph_n += 1
        self.ph_mean += (az - self.ph_mean) / self.ph_n
        self.ph_m += az - self.ph_mean - self.ph_delta
        self.ph_min = np.minimum(self.ph_min, self.ph_m)
        drift = (self.ph_m - self.ph_min) > self.ph_lambda
        self.buf.append(x.copy())
        if drift.any() and len(self.buf) >= 20:
            W = np.array(self.buf)
            for j in np.nonzero(drift)[0]:
                if self.probs[j] is not None and len(self.probs[j]):   # re-fit values, same probs
                    self.thr[j] = np.sort(np.quantile(W[:, j], self.probs[j]))
                self.boost_left[j] = 60
                self.ph_m[j] = 0.0; self.ph_min[j] = 0.0
        self.boost_left = np.maximum(self.boost_left - 1, 0)
        self.x_prev = x

    # ---- streaming / batch API ----------------------------------------------
    def update_transform_row(self, x):
        x = np.asarray(x, dtype=np.float64)
        row = self._encode_row(x); self._update(x)
        return row

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if not self.schema:
            w = min(self.warmup, len(X))
            self._fit_schema(X[:w])
        out = np.empty((len(X), self.width), dtype=np.uint8)
        for i in range(len(X)):
            out[i] = self.update_transform_row(X[i])
        return out

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self._fit_schema(X[:min(self.warmup, len(X))]); self.fitted = True
        return self

    @property
    def n_literals(self): return int(self.width) if self.schema else 0

    def state_bytes(self):
        if not self.schema: return 0
        b = sum(t.nbytes for t in self.thr)
        b += self.mu.nbytes + self.scale.nbytes + self.x_prev.nbytes
        b += self.ph_m.nbytes * 3 + self.boost_left.nbytes
        b += self.recent * self.d * 8
        return int(b)
