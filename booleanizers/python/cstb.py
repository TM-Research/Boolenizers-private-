"""cstb.py — Cross-Scale Transient Binarizer (CSTB)
===================================================
Novel binarization method for IoT time-series attack/tampering detection.

Design Motivation
-----------------
IoT attack patterns violate *temporal coherence* — the property that a normal
signal behaves consistently across multiple timescales.  Three canonical cases:

  1. Sudden spikes (packet floods, sensor spoofing):
         The instantaneous value diverges sharply from even a fast-moving average.
         Captured by the **level term**:  |x(t) − μ_slow(t)|  is large.

  2. Sustained attacks (DDoS, replay, jamming):
         The fast-scale mean shifts to an abnormal level while the slow-scale
         baseline lags behind, creating a persistent cross-scale gap.
         Captured by the **cross-scale term**:  |μ_fast(t) − μ_slow(t)|  is large.

  3. Tampering / abrupt level shifts:
         Both timescales shift together, but diverge from the pre-attack baseline.
         The adaptive variance σ (anchored to the slow baseline) stays small while
         the residual grows, inflating TIS above the threshold.

Temporal Incoherence Score (TIS)
---------------------------------
    μ_fast(t)  =  α · x(t) + (1 − α) · μ_fast(t−1)         [fast EMA, decay α]
    μ_slow(t)  =  α² · x(t) + (1 − α²) · μ_slow(t−1)       [slow EMA, derived — no extra param]
    σ²(t)      =  α · r²(t) + (1 − α) · σ²(t−1),  r = x(t) − μ_slow(t)

    TIS(t)  =  ( |x(t) − μ_slow(t)|  +  |μ_fast(t) − μ_slow(t)| ) / (σ(t) + ε)

Binary output  (K = 1):  bit  = 1  iff  TIS(t) > k
Thermometer    (K > 1):  bit_j = 1  iff  TIS(t) > j · k/K   for j = 1 … K

Hyperparameters (exactly 2)
-----------------------------
    alpha  (0 < α < 1)  — timescale decay factor for the fast EMA.
                          Slow EMA uses α² automatically (no second param).
    k      (k > 0)      — detection threshold in units of adaptive σ.

Structural parameter (output width, not a tuning knob):
    K      (int ≥ 1)   — thermometer bits per numerical feature (K=1 → pure binary).

Special-value handling (automatic, no extra parameters)
--------------------------------------------------------
    NaN / missing  →  impute with μ_slow (model's best estimate of "normal").
                       State is NOT updated for that feature (avoids circular bias).
                       Result: NaN is treated as neutral (not an attack signal).

    ±Inf           →  always flagged as maximally anomalous (all bits = 1).
                       State is NOT updated for that feature (protects baseline).
                       Result: corrupt / overflow readings always trigger detection.

Categorical columns (cat_cols + cat_vocab)
------------------------------------------
    Two modes depending on whether cat_vocab is supplied:

    Mode A — cat_vocab provided (recommended for online / ESP32 use):
        Categories are declared upfront.  No fit() scan needed for categoricals.
        transform_one() works from the very first sample.

        cat_vocab = {col_idx: [val0, val1, val2, …]}
        Values are assigned codes in the order given.
        n_bits = ceil(log2(len(vocab))) — fixed forever.

        Example:
            cat_vocab={3: ["icmp", "tcp", "udp"]}
            icmp → 0 → [0, 0]
            tcp  → 1 → [0, 1]
            udp  → 2 → [1, 0]

    Mode B — no cat_vocab (batch / offline use):
        fit() scans the full batch, collects unique values alphabetically,
        then assigns codes.  transform_one() works only after fit().

    In both modes:
        Unknown category at inference → all-zero bits (safe fallback).
        NaN / None in a categorical column → all-zero bits (neutral).

Computational complexity
-------------------------
    Time:  O(n_features) per sample  —  ~10 float32 ops per numerical feature.
    Space: O(n_features)             —  4 state scalars per numerical feature.

    ESP32 pseudo-C (per feature, float arithmetic):
    ─────────────────────────────────────────────────────────
        if (!isfinite(x)) { bit = 1; goto next; }   // Inf guard
        if (isnan(x))     { x = mu_slow; }           // NaN imputation
        mu_fast = alpha  * x + inv_alpha  * mu_fast;
        mu_slow = alpha2 * x + inv_alpha2 * mu_slow;
        r       = x - mu_slow;
        sigma2  = alpha * r * r + inv_alpha * sigma2;
        sigma   = sqrtf(sigma2 + EPS);
        tis     = (fabsf(r) + fabsf(mu_fast - mu_slow)) / sigma;
        bit     = (tis > k) ? 1 : 0;
    ─────────────────────────────────────────────────────────
"""

import math
import numpy as np
from typing import Optional, List, Dict

_EPS: float     = 1e-8   # variance floor / division guard
_INF_TIS: float = 1e9   # TIS assigned to ±Inf values (always exceeds any threshold)
_CAT_BITS: int  = 4     # fixed bits per categorical column  (2^4 = 16 slots)
_CAT_MAX:  int  = 16    # maximum unique categories per column (9–16 range)


class CSTBinarizer:
    """Cross-Scale Transient Binarizer (CSTB).

    Handles numerical, categorical, NaN, and ±Inf features in a single pass.

    Parameters
    ----------
    alpha : float, default=0.1
        Fast-timescale EMA decay (0 < alpha < 1).
        Slow timescale = alpha**2 (derived automatically).

    k : float, default=2.5
        Detection threshold in units of adaptive σ.

    K : int, default=1
        Thermometer bits per *numerical* feature.
        K=1 → pure binary.  K>1 → graded anomaly severity.

    warmup : int, default=20
        Samples to suppress output while statistics stabilise.
        State IS updated throughout; only output is suppressed.

    cat_cols : list of int, optional
        Column indices that carry categorical data (strings or ints).
        These columns are binary-encoded rather than fed through TIS.
        If None, all columns are treated as numerical.

    cat_vocab : dict {col_idx: [val, val, …]}, optional
        Pre-declared vocabulary for each categorical column.
        Recommended for online / ESP32 use — avoids needing a batch fit()
        to discover unique values.  Values are assigned codes in the order
        given; n_bits = ceil(log2(len(vocab))) and never changes.

        Example:
            cat_vocab={3: ["icmp", "tcp", "udp"]}
            → icmp=0=[0,0], tcp=1=[0,1], udp=2=[1,0], unknown=[0,0]

        If omitted, unique values are discovered automatically during fit().

    name : str
        Identifier shown in __repr__.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        k: float = 2.5,
        K: int = 1,
        warmup: int = 20,
        cat_cols: Optional[List[int]] = None,
        cat_vocab: Optional[Dict[int, List]] = None,
        name: str = "CSTBinarizer",
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if k <= 0.0:
            raise ValueError(f"k must be positive; got {k}")
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")

        self.alpha: float      = float(alpha)
        self.alpha_slow: float = float(alpha ** 2)   # derived — not a free parameter
        self.k: float          = float(k)
        self.K: int            = int(K)
        self.warmup: int       = int(warmup)
        self.cat_cols: List[int] = sorted(cat_cols) if cat_cols else []
        self.cat_vocab: Dict[int, List] = cat_vocab or {}
        self.name: str         = name

        # Pre-computed complements (avoid recomputing in the hot loop)
        self._inv_alpha: float      = 1.0 - self.alpha
        self._inv_alpha_slow: float = 1.0 - self.alpha_slow

        # ── Numerical state (shape (n_num,) after fit) ──────────────────────
        self._n_num: int = 0                          # number of numerical cols
        self._num_cols: List[int] = []                # numerical column indices
        self._mu_fast: Optional[np.ndarray] = None   # fast EMA
        self._mu_slow: Optional[np.ndarray] = None   # slow EMA
        self._sigma2:  Optional[np.ndarray] = None   # adaptive variance
        self._n_seen:  int = 0                        # total samples processed

        # ── Categorical state ────────────────────────────────────────────────
        # _cat_maps[col_idx] = {str_value: int_code}
        self._cat_maps:  Dict[int, Dict[str, int]] = {}
        # _cat_nbits[col_idx] = number of binary bits for that column
        self._cat_nbits: Dict[int, int] = {}

        self.n_features: int  = 0     # total columns in the fitted data
        self.is_fitted: bool  = False

        # Build categorical maps immediately from cat_vocab (Mode A).
        # This makes transform_one() work without any fit() call.
        if self.cat_vocab:
            self._build_maps_from_vocab()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "CSTBinarizer":
        """Fit running statistics and categorical maps on a batch.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            May contain numerical values (float/int), string categories,
            NaN, or ±Inf.  Mixed-type columns require cat_cols to be set.
        """
        X = self._to2d_obj(X)
        self._setup_column_layout(X.shape[1])
        self._fit_categorical(X)
        if self._n_num > 0:
            X_num = self._extract_num(X)
            self._init_num_state(self._n_num)
            for row in X_num:
                self._update(row)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode a batch using current fitted state (state NOT updated).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        bits : np.ndarray of uint8, shape (n_samples, n_output_bits)
        """
        if not self.is_fitted:
            raise ValueError("Call fit() before transform().")
        X = self._to2d_obj(X)
        n = len(X)
        out = np.zeros((n, self.get_n_output_bits()), dtype=np.uint8)
        for i, row in enumerate(X):
            out[i] = self._encode_row(row)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Single-pass: update state AND encode each sample in order.

        Each sample is encoded using statistics gathered *before* it, then
        the state is updated.  Outputs are zero during the warmup window.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)

        Returns
        -------
        bits : np.ndarray of uint8, shape (n_samples, n_output_bits)
        """
        X = self._to2d_obj(X)
        self._setup_column_layout(X.shape[1])
        self._fit_categorical(X)
        out = np.zeros((len(X), self.get_n_output_bits()), dtype=np.uint8)
        if self._n_num > 0:
            X_num = self._extract_num(X)
            self._init_num_state(self._n_num)
            for i, (row_obj, row_num) in enumerate(zip(X, X_num)):
                self._update(row_num)
                if self._n_seen > self.warmup:
                    out[i] = self._encode_row(row_obj)
        else:
            # All categorical — no warmup needed
            for i, row in enumerate(X):
                out[i] = self._encode_row(row)
        self.is_fitted = True
        return out

    def transform_one(self, x) -> np.ndarray:
        """Online single-sample encode + state update.  O(n_features).

        Works without any prior fit() call when cat_vocab is supplied for all
        categorical columns — the very first sample already gets encoded.

        Parameters
        ----------
        x : array-like, shape (n_features,) — may contain strings, NaN, ±Inf.

        Returns
        -------
        bits : np.ndarray of uint8, shape (n_output_bits,)
            All-zero during the numerical warmup window.
            Categorical bits are always valid (vocab is pre-set).
        """
        x = np.asarray(x, dtype=object).ravel()

        # Auto-init column layout on the very first call
        if self._mu_fast is None and self.n_features == 0:
            self._setup_column_layout(len(x))
            if self._n_num > 0:
                self._init_num_state(self._n_num)

        if self._n_num > 0:
            x_num = self._extract_num_row(x)
            self._update(x_num)

        # During numerical warmup: return zeros for numerical bits,
        # but still encode categorical bits correctly (vocab is known).
        if self._n_seen <= self.warmup and self._n_num > 0:
            bits = np.zeros(self.get_n_output_bits(), dtype=np.uint8)
            # Fill in the categorical portion (offset = n_num * K)
            cat_offset = self._n_num * self.K
            for c in self.cat_cols:
                cat_bits = self._encode_one_cat(x[c], c)
                bits[cat_offset: cat_offset + len(cat_bits)] = cat_bits
                cat_offset += len(cat_bits)
            return bits

        self.is_fitted = True
        return self._encode_row(x)

    def get_tis(self, X: np.ndarray) -> np.ndarray:
        """Return raw Temporal Incoherence Scores for numerical columns only.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        tis : np.ndarray of float32, shape (n_samples, n_num_features)
              ±Inf positions get TIS = _INF_TIS.  NaN positions get TIS = 0.
        """
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        X = self._to2d_obj(X)
        X_num = self._extract_num(X)
        sigma = np.sqrt(self._sigma2 + _EPS)                             # (F,)
        level = np.abs(X_num - self._mu_slow[np.newaxis, :]) / sigma    # (N, F)
        cross = np.abs(self._mu_fast - self._mu_slow)[np.newaxis, :] / sigma
        tis = (level + cross).astype(np.float32)

        # Override ±Inf cells
        inf_mask = ~np.isfinite(X_num)
        tis[inf_mask] = _INF_TIS
        # NaN cells: set to 0 (neutral, imputed = baseline)
        nan_mask = np.isnan(X_num)
        tis[nan_mask] = 0.0
        return tis

    def get_n_output_bits(self) -> int:
        """Total number of output bits = K * n_num_cols + sum(cat_bits)."""
        num_bits = self._n_num * self.K
        cat_bits = sum(self._cat_nbits.get(c, 0) for c in self.cat_cols)
        return num_bits + cat_bits

    def reset(self) -> "CSTBinarizer":
        """Clear all state so the encoder can be re-fitted from scratch."""
        self._mu_fast = self._mu_slow = self._sigma2 = None
        self._n_seen = self._n_num = self.n_features = 0
        self._num_cols.clear()
        self._cat_maps.clear()
        self._cat_nbits.clear()
        self.is_fitted = False
        return self

    def get_state_summary(self) -> dict:
        vocab_mode = {c: list(self._cat_maps[c].keys())
                      for c in self.cat_cols if c in self.cat_vocab}
        scan_mode  = {c: list(self._cat_maps[c].keys())
                      for c in self.cat_cols if c not in self.cat_vocab
                      and c in self._cat_maps}
        return {
            "name":              self.name,
            "alpha":             self.alpha,
            "alpha_slow":        self.alpha_slow,
            "k":                 self.k,
            "K":                 self.K,
            "warmup":            self.warmup,
            "n_features":        self.n_features,
            "n_num_cols":        self._n_num,
            "num_cols":          self._num_cols,
            "cat_cols":          self.cat_cols,
            "cat_vocab_cols":    vocab_mode,   # pre-declared (Mode A)
            "cat_scanned_cols":  scan_mode,    # discovered via fit() (Mode B)
            "cat_nbits":         dict(self._cat_nbits),
            "n_output_bits":     self.get_n_output_bits(),
            "n_seen":            self._n_seen,
            "is_fitted":         self.is_fitted,
        }

    def get_threshold_levels(self) -> np.ndarray:
        """Thermometer thresholds in TIS units: [k/K, 2k/K, …, k]."""
        return np.arange(1, self.K + 1) * (self.k / self.K)

    # ──────────────────────────────────────────────────────────────────────────
    # Column layout helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_column_layout(self, n_total: int) -> None:
        """Determine which columns are numerical vs categorical."""
        self.n_features = n_total
        cat_set = set(self.cat_cols)
        self._num_cols = [c for c in range(n_total) if c not in cat_set]
        self._n_num = len(self._num_cols)

    def _build_maps_from_vocab(self) -> None:
        """Mode A: build cat maps directly from cat_vocab (no batch scan needed)."""
        for c, vals in self.cat_vocab.items():
            unique = [str(v) for v in vals]
            if len(unique) > _CAT_MAX:
                raise ValueError(
                    f"cat_vocab col {c}: {len(unique)} values exceed "
                    f"the {_CAT_MAX}-value limit ({_CAT_BITS} bits)."
                )
            self._cat_maps[c]  = {v: i for i, v in enumerate(unique)}
            self._cat_nbits[c] = _CAT_BITS   # always 4 bits → up to 16 categories

    def _fit_categorical(self, X: np.ndarray) -> None:
        """Mode B: scan batch to discover unique values.  Skips vocab columns."""
        for c in self.cat_cols:
            if c in self.cat_vocab:
                continue   # already built from vocab in __init__
            col = X[:, c]
            unique = sorted({str(v) for v in col if _is_valid_cat(v)})
            if not unique:
                unique = ["__unknown__"]
            if len(unique) > _CAT_MAX:
                raise ValueError(
                    f"Column {c} has {len(unique)} unique categories; "
                    f"maximum supported is {_CAT_MAX} ({_CAT_BITS} bits). "
                    f"Reduce cardinality or group rare values."
                )
            self._cat_maps[c]  = {v: i for i, v in enumerate(unique)}
            self._cat_nbits[c] = _CAT_BITS   # always 4 bits → up to 16 categories

    # ──────────────────────────────────────────────────────────────────────────
    # Numerical state helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _init_num_state(self, n_num: int) -> None:
        self._mu_fast = np.zeros(n_num, dtype=np.float32)
        self._mu_slow = np.zeros(n_num, dtype=np.float32)
        self._sigma2  = np.full(n_num, _EPS, dtype=np.float32)
        self._n_seen  = 0

    def _update(self, x_num: np.ndarray) -> None:
        """Update dual-timescale EMA for numerical features.

        NaN and ±Inf are imputed with μ_slow for the state update so they
        cannot corrupt the running baseline.  The actual anomalous values
        are used only in _encode, where ±Inf triggers a guaranteed detection.
        """
        n = self._n_seen

        # Identify special-value positions
        finite_mask = np.isfinite(x_num)   # True = safe to use
        x_safe = x_num.copy()
        x_safe[~finite_mask] = self._mu_slow[~finite_mask]   # impute

        if n == 0:
            self._mu_fast[:] = x_safe
            self._mu_slow[:] = x_safe
            # sigma² stays at _EPS
            self._n_seen = 1
            return

        # Update only finite positions (protect baseline from NaN / Inf)
        m = finite_mask
        self._mu_fast[m] = (
            self.alpha * x_safe[m] + self._inv_alpha * self._mu_fast[m]
        )
        self._mu_slow[m] = (
            self.alpha_slow * x_safe[m] + self._inv_alpha_slow * self._mu_slow[m]
        )
        r = x_safe[m] - self._mu_slow[m]
        self._sigma2[m] = (
            self.alpha * (r * r) + self._inv_alpha * self._sigma2[m]
        )
        self._n_seen += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Encoding helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _encode_row(self, row: np.ndarray) -> np.ndarray:
        """Encode one sample row (object dtype, mixed types) → flat uint8 bits."""
        parts: List[np.ndarray] = []

        # ── Numerical columns ────────────────────────────────────────────────
        if self._n_num > 0:
            x_num = self._extract_num_row(row)
            parts.append(self._encode_numerical(x_num))

        # ── Categorical columns ──────────────────────────────────────────────
        for c in self.cat_cols:
            parts.append(self._encode_one_cat(row[c], c))

        return np.concatenate(parts).astype(np.uint8)

    def _encode_numerical(self, x: np.ndarray) -> np.ndarray:
        """Compute TIS for numerical features and apply thermometer thresholds.

        TIS formula
        -----------
            σ        =  sqrt(σ² + ε)
            level[f] =  |x[f] − μ_slow[f]| / σ[f]        (point anomaly)
            cross[f] =  |μ_fast[f] − μ_slow[f]| / σ[f]   (persistent anomaly)
            TIS[f]   =  level[f] + cross[f]

        Special values
        --------------
            NaN  →  TIS = 0  (neutral; imputed to baseline, no anomaly flagged)
            ±Inf →  TIS = _INF_TIS  (always exceeds k; all bits = 1)
        """
        sigma = np.sqrt(self._sigma2 + _EPS)

        nan_mask  = np.isnan(x)
        inf_mask  = np.isinf(x)
        bad_mask  = nan_mask | inf_mask

        # Safe proxy for arithmetic (NaN/Inf → μ_slow so maths doesn't blow up)
        x_safe = x.copy()
        x_safe[bad_mask] = self._mu_slow[bad_mask]

        level = np.abs(x_safe - self._mu_slow) / sigma
        cross = np.abs(self._mu_fast - self._mu_slow) / sigma
        tis   = level + cross

        # Override TIS for special values
        tis[nan_mask] = 0.0       # NaN → neutral
        tis[inf_mask] = _INF_TIS  # Inf → maximally anomalous

        if self.K == 1:
            return (tis > self.k).astype(np.uint8)

        thresholds = np.arange(1, self.K + 1, dtype=np.float32) * (self.k / self.K)
        bits = (tis[:, np.newaxis] > thresholds[np.newaxis, :]).astype(np.uint8)
        return bits.ravel()  # (n_num * K,), order: f0_b1…f0_bK, f1_b1…f1_bK, …

    def _encode_one_cat(self, value, col_idx: int) -> np.ndarray:
        """Map one categorical value to its binary bit-pattern.

        Encoding
        --------
            categories sorted alphabetically → codes 0, 1, 2, …
            code i → ceil(log2(n_cats)) bits, MSB first.

            Example (3 categories):  n_bits = 2
                icmp → 0 → [0, 0]
                tcp  → 1 → [0, 1]
                udp  → 2 → [1, 0]

        Unknown / NaN category → all-zero bits.
        """
        n_bits = self._cat_nbits.get(col_idx, 1)
        bits   = np.zeros(n_bits, dtype=np.uint8)

        if not _is_valid_cat(value):
            return bits   # NaN / None → all zeros (unknown)

        code = self._cat_maps.get(col_idx, {}).get(str(value), None)
        if code is None:
            return bits   # unseen category → all zeros

        # MSB first: bit 0 = most significant
        for i in range(n_bits):
            bits[n_bits - 1 - i] = (code >> i) & 1
        return bits

    # ──────────────────────────────────────────────────────────────────────────
    # Data extraction helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_num(self, X: np.ndarray) -> np.ndarray:
        """Extract numerical columns as float32 array, shape (N, n_num)."""
        if not self._num_cols:
            return np.empty((len(X), 0), dtype=np.float32)
        cols = X[:, self._num_cols]
        # Convert to float32, propagating NaN / ±Inf naturally
        return np.array(cols, dtype=np.float32)

    def _extract_num_row(self, row: np.ndarray) -> np.ndarray:
        """Extract numerical values from one object-dtype row as float32."""
        if not self._num_cols:
            return np.empty(0, dtype=np.float32)
        vals = row[self._num_cols]
        return np.array([_to_float(v) for v in vals], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Static helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to2d_obj(X) -> np.ndarray:
        """Convert input to 2-D object array (preserves strings and NaN)."""
        X = np.asarray(X, dtype=object)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    # ──────────────────────────────────────────────────────────────────────────
    # Dunder
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CSTBinarizer(alpha={self.alpha}, k={self.k}, K={self.K}, "
            f"warmup={self.warmup}, cat_cols={self.cat_cols}, "
            f"fitted={self.is_fitted}, n_features={self.n_features})"
        )


# ────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ────────────────────────────────────────────────────────────────────────────

def _is_valid_cat(value) -> bool:
    """Return True if value is a usable category (not None, not float NaN)."""
    if value is None:
        return False
    try:
        if math.isnan(float(value)):
            return False
    except (TypeError, ValueError):
        pass   # non-numeric string → valid category
    return True


def _to_float(value) -> float:
    """Convert a mixed-type cell to float, returning NaN for non-numeric."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
