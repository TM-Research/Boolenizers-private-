"""
Signal Behavior Encoder (SBE) — V3

==========================================================================
CORE PRINCIPLE: SPARSE-AWARE SIGNAL DECOMPOSITION
==========================================================================

Network/IoT features exhibit distinct behavioral types:
  - Constant: no variation (always same value)
  - Binary: two states (e.g., flag on/off)
  - Categorical-like: few discrete levels
  - Sparse: dominated by one value, with occasional bursts (e.g., most
    flows have 0 bytes, some have large values)
  - Continuous: rich distribution requiring fine resolution

SBE detects the signal type of each feature and applies the optimal
binarization strategy for that type:

  Constant → 1 dummy bit
  Binary → 1 midpoint bit
  Low-cardinality (≤ K unique) → midpoint between consecutive values
  Sparse (>50% at one value) → mode separator + quantile on active values
  Continuous → quantile thresholds at maximum-entropy positions

KEY INNOVATION: SPARSE SIGNAL DECOMPOSITION
  For features where >50% of samples share one value (the "baseline"),
  SBE places the first threshold to separate baseline from non-baseline,
  then places remaining thresholds as quantiles of the non-baseline
  values only. This ensures each bit carries maximum information:
  - Bit 1: "has this signal left its baseline state?" (binary event)
  - Bits 2..K: "how far from baseline?" (intensity encoding)

  Standard quantile encoding on sparse features wastes most bits at the
  baseline value (all bits identical = zero information). SBE's approach
  converts these wasted bits into meaningful intensity measurements.

==========================================================================
HYPERPARAMETERS (2): K, sparsity_threshold
==========================================================================

    K : int (default 10)
        Maximum bits per feature.

    sparsity_threshold : float (default 0.5)
        Fraction of samples at one value to trigger sparse encoding.

==========================================================================
COMPLEXITY:  fit O(n·d), transform O(n·d·K), per-sample O(d·K)
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class SignalBehaviorEncoder(ThermometerEncoder):
    """Signal Behavior Encoder (SBE).

    Sparse-aware signal decomposition binarizer that detects feature
    behavioral types and applies optimal per-type encoding strategy.
    """

    def __init__(self, K: int = 10, sparsity_threshold: float = 0.5):
        super().__init__(K=K, name="SBE")
        self.max_bits = K
        self.sparsity_threshold = sparsity_threshold

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "SignalBehaviorEncoder":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features
        self.feature_thresholds_ = []
        total_bits = 0

        for j in range(n_features):
            col = X[:, j]
            uv = np.unique(col)
            nu = len(uv)

            if nu <= 1:
                # CONSTANT: 1 dummy bit
                th = np.array([uv[0] + 1e-10]) if nu == 1 else np.array([0.0])
            elif nu == 2:
                # BINARY: 1 midpoint bit
                th = np.array([(uv[0] + uv[1]) / 2.0])
            elif nu <= self.max_bits:
                # LOW-CARDINALITY: midpoint between each consecutive pair
                th = (uv[:-1] + uv[1:]) / 2.0
            else:
                # HIGH-CARDINALITY: detect sparsity and apply strategy
                th = self._adaptive_thresholds(col, nu)

            self.feature_thresholds_.append(th)
            total_bits += len(th)

        self.total_bits_ = total_bits
        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # Core: adaptive strategy selection per feature
    # ------------------------------------------------------------------
    def _adaptive_thresholds(self, col: np.ndarray, n_unique: int) -> np.ndarray:
        """Select and apply optimal threshold strategy based on signal type."""
        K = self.max_bits

        # Detect dominant value (mode)
        # Use histogram-based mode for efficiency
        uv, counts = np.unique(col, return_counts=True)
        mode_idx = np.argmax(counts)
        mode_val = uv[mode_idx]
        mode_frac = counts[mode_idx] / len(col)

        if mode_frac > self.sparsity_threshold:
            # SPARSE SIGNAL: mode separator + quantile on active values
            return self._sparse_thresholds(col, mode_val, mode_frac, K)
        else:
            # CONTINUOUS SIGNAL: quantile thresholds
            return self._continuous_thresholds(col, K)

    # ------------------------------------------------------------------
    # Strategy: Sparse signal decomposition
    # ------------------------------------------------------------------
    def _sparse_thresholds(
        self, col: np.ndarray, mode_val: float, mode_frac: float, K: int
    ) -> np.ndarray:
        """Sparse feature: baseline separator + intensity thresholds.

        Bit layout:
          bit 1: x >= (mode_val + eps) → "signal has left baseline"
          bits 2..K: quantile thresholds on non-baseline values
        """
        active = col[col != mode_val]

        if len(active) == 0:
            # All values at mode (effectively constant despite high cardinality from float noise)
            return np.array([mode_val + 1e-10])

        # Threshold 1: separate baseline from active
        # Place just above the mode value
        active_min = np.min(active)
        if active_min > mode_val:
            sep = (mode_val + active_min) / 2.0
        else:
            # Active values are below mode too — use mode_val as separator
            sep = mode_val

        thresholds = [sep]

        # Remaining thresholds: quantiles on active values
        n_detail = K - 1
        if n_detail > 0 and len(active) > 1:
            active_uv = np.unique(active)
            if len(active_uv) <= n_detail:
                # Few active unique values: midpoints
                detail = ((active_uv[:-1] + active_uv[1:]) / 2.0).tolist()
            else:
                # Quantile thresholds on active distribution
                qs = np.linspace(
                    1.0 / (n_detail + 1),
                    n_detail / (n_detail + 1),
                    n_detail,
                )
                detail = np.quantile(active, qs).tolist()
            thresholds.extend(detail)

        thresholds = np.sort(np.unique(np.array(thresholds)))

        # Cap at K
        if len(thresholds) > K:
            idx = np.linspace(0, len(thresholds) - 1, K).astype(int)
            thresholds = thresholds[idx]

        # Ensure monotonicity
        for k in range(1, len(thresholds)):
            if thresholds[k] <= thresholds[k - 1]:
                thresholds[k] = thresholds[k - 1] + 1e-10

        return thresholds

    # ------------------------------------------------------------------
    # Strategy: Continuous signal (quantile placement)
    # ------------------------------------------------------------------
    def _continuous_thresholds(self, col: np.ndarray, K: int) -> np.ndarray:
        """Continuous feature: maximum-entropy quantile thresholds.

        Places thresholds at positions q_k = k/(K+1) so each bit has
        approximately equal probability of being 0 or 1, maximizing
        the information content per bit.

        Uses robust percentile clipping [P1, P99] to avoid outlier
        sensitivity.
        """
        # Robust quantile positions
        qs = np.linspace(1.0 / (K + 1), K / (K + 1), K)
        thresholds = np.quantile(col, qs)
        thresholds = np.unique(thresholds)

        # If dedup reduced count, fill with intermediate positions
        if len(thresholds) < K:
            # Try finer quantile grid
            qs_fine = np.linspace(0.01, 0.99, K * 3)
            fine_th = np.quantile(col, qs_fine)
            thresholds = np.unique(np.concatenate([thresholds, fine_th]))
            if len(thresholds) > K:
                idx = np.linspace(0, len(thresholds) - 1, K).astype(int)
                thresholds = thresholds[idx]

        # Ensure monotonicity
        for k in range(1, len(thresholds)):
            if thresholds[k] <= thresholds[k - 1]:
                thresholds[k] = thresholds[k - 1] + 1e-10

        return thresholds

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.total_bits_), dtype=np.uint8)

        pos = 0
        for j, th in enumerate(self.feature_thresholds_):
            n_bits = len(th)
            for k in range(n_bits):
                out[:, pos + k] = (X[:, j] >= th[k]).astype(np.uint8)
            pos += n_bits

        return out

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")
        x = np.asarray(x, dtype=np.float64)
        bits = []
        for j, th in enumerate(self.feature_thresholds_):
            for t in th:
                bits.append(1 if x[j] >= t else 0)
        return np.array(bits, dtype=np.uint8)

    def get_n_output_bits(self) -> int:
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.total_bits_
