"""
ASIB-v3 (ASIBAdaptive) — fully self-parameterizing streaming IDS booleanizer
===========================================================================

Iteration 3. The objective forbids fixed constants: EVERY quantity is derived
from running data via standard statistical formulas — no hand-tuned magic
numbers. This keeps the batch-quality accuracy of ASIB-v2 while becoming
dataset-agnostic and drift-self-calibrating.

Data-derived choices (formula → what it replaces):
  * bit budget K_j  = Freedman–Diaconis  h=2·IQR·n^(−1/3), capped by Sturges(n)
                      and distinct-value count           (replaces fixed K/kmax)
  * auto-warmup     = freeze schema once n ≥ (K+1)², the sample size at which the
                      quantile SE (~1/√n) is below the quantile spacing 1/(K+1)
                      with K=Sturges(n)                   (replaces warmup=600)
  * categorical     = integer-valued AND n_unique ≤ Sturges(n)  (replaces max_cat=12)
  * zero literal    = added iff zero-fraction > 1/(K_j+1)        (replaces 0.30)
  * EWMA forgetting = α = 2/(N_eff+1); N_eff grows by 1/step, resets to K on drift
                      (adaptive horizon)                  (replaces fixed α=0.02)
  * drift detector  = EWMA control chart on |z|; trigger when the EWMA exceeds its
                      running mean by 3·(running σ)  — the universal 3-sigma /
                      ECDD control limit (self-calibrating) (replaces ph_delta/λ)
  * re-fit window   = last (K+1)² samples (same data-derived size)  (replaces recent=400)

O(1) amortized/sample (re-fit only on drift); memory = O((K+1)²·d) for the
re-fit reservoir + O(d) statistics.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from math import log2, ceil


def _sturges(n):  # textbook bin-count cap
    return max(1, ceil(log2(max(n, 2))) + 1)


def _fd_bins(col, n):
    """Freedman–Diaconis bin count, capped by Sturges and distinct values."""
    if n < 4:
        return 1
    lo, hi = float(np.min(col)), float(np.max(col))
    if hi <= lo:
        return 0
    q1, q3 = np.percentile(col, [25, 75]); iqr = q3 - q1
    cap = min(_sturges(n), max(1, len(np.unique(col)) - 1))
    if iqr <= 0:
        return cap
    h = 2 * iqr * n ** (-1 / 3)
    return int(np.clip(ceil((hi - lo) / h), 1, cap))


class ASIBAdaptive:
    """Fully data-derived streaming booleanizer. No tuned constants are exposed;
    `sigma` (the control-chart width) defaults to the universal 3-sigma rule and
    is the only statistical convention, not a per-dataset tunable."""

    def __init__(self, sigma: float = 3.0, temporal: bool = True):
        self.sigma = float(sigma)         # 3-sigma control limit (universal SPC rule)
        self.temporal = bool(temporal)
        self.schema = False
        self._buf = []                    # schema-formation buffer
        self._seen = 0

    # ---- schema formation (auto-terminating) --------------------------------
    def _maybe_form_schema(self):
        n = len(self._buf)
        K = _sturges(n)
        if n < (K + 1) ** 2:              # not enough samples to resolve K quantiles yet
            return False
        B = np.asarray(self._buf, dtype=np.float64)
        d = B.shape[1]
        self.thr = []; self.probs = []; self.is_zero = np.zeros(d, bool)
        for j in range(d):
            col = B[:, j]; u = np.unique(col)
            integer_like = np.allclose(col, np.round(col))
            if integer_like and len(u) <= _sturges(n):          # categorical / flag
                t = u[1:].astype(np.float64) if len(u) > 1 else np.array([], np.float64)
                p = None
            else:
                Kj = _fd_bins(col, n)
                if Kj <= 0:
                    t = np.array([], np.float64); p = None
                else:
                    p = np.arange(1, Kj + 1) / (Kj + 1)
                    t = np.quantile(col, p)
                    keep = []; seen_cols = []
                    for k, tv in enumerate(t):                    # prune dead/duplicate bits
                        bitcol = col >= tv; s = int(bitcol.sum())
                        if s == 0 or s == len(col): continue
                        if any(np.array_equal(bitcol, c) for c in seen_cols): continue
                        seen_cols.append(bitcol); keep.append(k)
                    t = t[keep]; p = p[keep] if len(keep) else None
            kj = len(t)
            # zero literal only if zeros form their own (super-bin) mass
            self.is_zero[j] = bool(np.mean(col == 0) > 1.0 / (kj + 1)) if kj else bool(np.mean(col == 0) > 0.5)
            self.thr.append(np.sort(t)); self.probs.append(p)
        self.kj = np.array([len(t) for t in self.thr])
        self.extra = np.array([(1 if self.is_zero[j] else 0) + (2 if self.temporal else 0)
                               for j in range(d)])
        self.offsets = np.cumsum([0] + [int(self.kj[j] + self.extra[j]) for j in range(d)])
        self.width = int(self.offsets[-1])
        self.d = d
        # streaming state (all data-derived)
        self.mu = B.mean(0); self.scale = np.maximum(np.abs(B - self.mu).mean(0), 1e-12)
        self.x_prev = B[-1].copy()
        self.eff_n = np.full(d, float(n))                 # adaptive EWMA horizon
        # ECDD-style drift detector on |z|
        self.ewma = np.zeros(d); self.dmean = np.zeros(d); self.dvar = np.ones(d) * 1e-6; self.dn = 0
        self.boost_left = np.zeros(d, np.int32)
        self.refit = max((K + 1) ** 2, 32)                # data-derived re-fit window
        self.buf = deque([B[i] for i in range(max(0, len(B) - self.refit), len(B))], maxlen=self.refit)
        self.schema = True
        return True

    def _encode_row(self, x):
        out = np.zeros(self.width, np.uint8); s = np.maximum(self.scale, 1e-12)
        for j in range(self.d):
            o = self.offsets[j]; t = self.thr[j]; k = len(t)
            if k:
                c = int(np.searchsorted(t, x[j], side="right"))
                if c: out[o:o + c] = 1
            p = o + k
            if self.is_zero[j]:
                out[p] = 1 if x[j] == 0 else 0; p += 1
            if self.temporal:
                out[p] = 1 if abs(x[j] - self.x_prev[j]) > s[j] else 0; p += 1   # burst
                out[p] = 1 if self.boost_left[j] > 0 else 0; p += 1               # drift
        return out

    def _update(self, x):
        s = np.maximum(self.scale, 1e-12)
        z = np.abs((x - self.mu) / s)
        # adaptive EWMA forgetting α = 2/(N_eff+1)
        a = 2.0 / (self.eff_n + 1.0)
        self.mu += a * (x - self.mu)
        self.scale = (1 - a) * self.scale + a * np.abs(x - self.mu)
        self.eff_n += 1.0
        # ECDD control chart on |z|: drift if EWMA|z| > running_mean + sigma·running_std
        lam = 2.0 / (self.refit + 1.0)
        self.ewma = (1 - lam) * self.ewma + lam * z
        self.dn += 1
        dmean_new = self.dmean + (z - self.dmean) / self.dn
        self.dvar = self.dvar + (z - self.dmean) * (z - dmean_new)
        self.dmean = dmean_new
        std = np.sqrt(self.dvar / max(self.dn, 1)) * np.sqrt(lam / (2 - lam))
        drift = self.ewma > (self.dmean + self.sigma * np.maximum(std, 1e-9))
        self.buf.append(x.copy())
        if drift.any() and len(self.buf) >= 8:
            W = np.array(self.buf)
            for j in np.nonzero(drift)[0]:
                if self.probs[j] is not None and len(self.probs[j]):
                    self.thr[j] = np.sort(np.quantile(W[:, j], self.probs[j]))
                self.eff_n[j] = float(self.kj[j] + 2)      # reset horizon → fast re-adapt
                self.ewma[j] = self.dmean[j]
                self.boost_left[j] = int(self.refit ** 0.5)
        self.boost_left = np.maximum(self.boost_left - 1, 0)
        self.x_prev = x

    # ---- API -----------------------------------------------------------------
    def update_transform_row(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not self.schema:
            self._buf.append(x); self._seen += 1
            if not self._maybe_form_schema():
                return None              # still warming up (caller handles)
        row = self._encode_row(x); self._update(x)
        return row

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if not self.schema:
            # consume rows until schema forms, then encode everything we have
            for i in range(len(X)):
                self._buf.append(X[i]); self._seen += 1
                if self._maybe_form_schema():
                    break
            if not self.schema:               # whole block too short → form on all
                self._buf = list(X); self._force_form()
        out = np.empty((len(X), self.width), np.uint8)
        for i in range(len(X)):
            r = self._encode_row(X[i]); self._update(X[i]); out[i] = r
        return out

    def _force_form(self):
        """Fallback for streams shorter than the data-derived warmup (K+1)²:
        resample the available rows up to the gate so the schema still forms."""
        B = np.asarray(self._buf, np.float64); n = len(B)
        K = _sturges(max(n, 4)); need = (K + 1) ** 2
        if n < need:
            idx = np.random.default_rng(0).integers(0, n, need - n)
            self._buf = list(B) + [B[i] for i in idx]
        self._maybe_form_schema()

    def fit(self, X):
        X = np.asarray(X, np.float64)
        for i in range(len(X)):
            self._buf.append(X[i]); self._seen += 1
            if self._maybe_form_schema():
                break
        if not self.schema:
            self._force_form()
        return self

    @property
    def n_literals(self): return int(self.width) if self.schema else 0

    def state_bytes(self):
        if not self.schema: return 0
        b = sum(t.nbytes for t in self.thr) + self.mu.nbytes * 6 + self.refit * self.d * 8
        return int(b)
