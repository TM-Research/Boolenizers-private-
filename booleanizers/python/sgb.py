"""
Signal Gradient Binarizer (SGB) — Iteration 3 Final Design

==========================================================================
CORE PRINCIPLE: FUSED GRADIENT-SEGMENT ADAPTIVE BINARIZATION
==========================================================================

Combines the winning elements from Iterations 1-2:
1. RGB-G50's gradient/density-aware threshold placement (91.24% WUSTL)
2. MWB2-K8's segment-aware drift robustness (90.90% WUSTL, 131 bits)
3. AQB's adaptive bit allocation for low-cardinality features

KEY DESIGN INSIGHT from iteration analysis:
- Gradient-focused placement captures WHERE data concentrates
- Segment-awareness captures HOW data SHIFTS across time
- Adaptive bit allocation avoids wasting bits on trivial features
- The sweet spot is max_bits=10-12 with midpoint thresholds for low-card

ALGORITHM:
1. For each feature:
   a. Compute unique values → adaptive bit count
   b. If low-cardinality: use midpoint thresholds (proven optimal)
   c. If high-cardinality:
      - Split data into segments
      - If drift detected: use per-segment quantiles + boundary thresholds
      - If stable: use density-gradient thresholds (CDF gradient peaks)
2. All thresholds are pre-sorted and strictly monotonic

This creates a binarization that is both drift-robust AND density-sensitive.

==========================================================================
HYPERPARAMETERS (2): max_bits_per_feature, n_segments
==========================================================================

    max_bits_per_feature : int (default 10)
        Maximum bits per feature.

    n_segments : int (default 4)
        Segments for drift detection. More = finer drift detection.

==========================================================================
COMPLEXITY: O(n*d) fit, O(n*d*K) transform, O(d*K) memory
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class SignalGradientBinarizer(ThermometerEncoder):
    """Signal Gradient Binarizer (SGB).

    Fuses gradient-density placement with segment-aware drift robustness
    and adaptive bit allocation. Best-of-breed from iterations 1-2.
    """

    def __init__(self, max_bits_per_feature: int = 10, n_segments: int = 4):
        super().__init__(K=max_bits_per_feature, name="SGB")
        self.max_bits = max_bits_per_feature
        self.n_segments = max(2, n_segments)
        self._B = 128  # histogram resolution

    def fit(self, X: np.ndarray) -> "SignalGradientBinarizer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        # Split into segments
        seg_size = max(10, n_samples // self.n_segments)
        segments = []
        for i in range(0, n_samples, seg_size):
            seg = X[i:i + seg_size]
            if len(seg) >= 5:
                segments.append(seg)
        if not segments:
            segments = [X]

        self.feature_thresholds_ = []
        total_bits = 0

        for j in range(n_features):
            col = X[:, j]
            uv = np.unique(col)
            n_unique = len(uv)

            if n_unique <= 1:
                th = np.array([uv[0] + 1e-10]) if n_unique == 1 else np.array([0.0])
            elif n_unique <= 2:
                th = np.array([(uv[0] + uv[1]) / 2.0])
            elif n_unique <= self.max_bits:
                # Midpoint thresholds — proven optimal for low-cardinality
                th = (uv[:-1] + uv[1:]) / 2.0
            else:
                seg_cols = [s[:, j] for s in segments]
                th = self._adaptive_thresholds(col, seg_cols)

            self.feature_thresholds_.append(th)
            total_bits += len(th)

        self.total_bits_ = total_bits
        self.fitted = True
        return self

    def _adaptive_thresholds(self, col: np.ndarray, seg_cols: list) -> np.ndarray:
        """Route to gradient or segment-aware thresholds based on drift."""
        # Detect drift via segment mean variation
        seg_means = np.array([np.mean(s) for s in seg_cols])
        global_std = np.std(col)
        mean_spread = np.std(seg_means)
        drift_ratio = mean_spread / (global_std + 1e-15)

        if drift_ratio > 0.15:
            return self._segment_thresholds(col, seg_cols, seg_means)
        else:
            return self._gradient_thresholds(col)

    def _gradient_thresholds(self, col: np.ndarray) -> np.ndarray:
        """Density-gradient aware placement (from RGB-G50 approach)."""
        B = min(self._B, len(np.unique(col[:min(5000, len(col))])))
        if B < 4:
            B = 4

        # Quantile-based bins
        q_positions = np.linspace(0, 1, B + 1)
        bin_edges = np.quantile(col, q_positions)

        # CDF gradient = density proxy
        bin_widths = np.diff(bin_edges)
        bin_widths = np.maximum(bin_widths, 1e-15)
        density = (1.0 / B) / bin_widths
        density /= density.sum() + 1e-20

        n_bits = self.max_bits
        # 50% gradient-focused, 50% uniform coverage (the sweet spot from RGB-G50)
        n_grad = max(1, n_bits // 2)
        n_unif = n_bits - n_grad

        thresholds = []

        # High-density positions
        if n_grad > 0:
            top_bins = np.argsort(density)[-min(n_grad * 3, B):]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            candidates = np.sort(bin_centers[top_bins])
            if len(candidates) <= n_grad:
                thresholds.extend(candidates)
            else:
                idx = np.linspace(0, len(candidates) - 1, n_grad).astype(int)
                thresholds.extend(candidates[idx])

        # Uniform coverage
        if n_unif > 0:
            unif_q = np.linspace(0.02, 0.98, n_unif + 2)[1:-1]
            thresholds.extend(np.quantile(col, unif_q))

        thresholds = np.sort(np.unique(np.array(thresholds)))
        if len(thresholds) > n_bits:
            idx = np.linspace(0, len(thresholds) - 1, n_bits).astype(int)
            thresholds = thresholds[idx]

        # Strict monotonicity
        for k in range(1, len(thresholds)):
            if thresholds[k] <= thresholds[k - 1]:
                thresholds[k] = thresholds[k - 1] + 1e-10

        return thresholds

    def _segment_thresholds(self, col: np.ndarray, seg_cols: list, seg_means: np.ndarray) -> np.ndarray:
        """Segment-aware thresholds for drifting features (from MWB2 approach)."""
        n_segs = len(seg_cols)
        n_bits = self.max_bits

        n_per_seg = max(2, n_bits // n_segs)

        candidates = []
        for sc in seg_cols:
            q_pos = np.linspace(0.05, 0.95, n_per_seg + 2)[1:-1]
            candidates.extend(np.quantile(sc, q_pos))

        # Inter-segment boundary thresholds
        sorted_means = np.sort(seg_means)
        for i in range(len(sorted_means) - 1):
            boundary = (sorted_means[i] + sorted_means[i + 1]) / 2.0
            candidates.append(boundary)

        candidates = np.sort(np.unique(np.array(candidates)))

        if len(candidates) > n_bits:
            idx = np.linspace(0, len(candidates) - 1, n_bits).astype(int)
            candidates = candidates[idx]

        # Ensure minimum bits
        if len(candidates) < 4:
            fill_q = np.linspace(0.1, 0.9, max(4, n_bits))
            candidates = np.sort(np.unique(np.concatenate([candidates, np.quantile(col, fill_q)])))
            if len(candidates) > n_bits:
                idx = np.linspace(0, len(candidates) - 1, n_bits).astype(int)
                candidates = candidates[idx]

        for k in range(1, len(candidates)):
            if candidates[k] <= candidates[k - 1]:
                candidates[k] = candidates[k - 1] + 1e-10

        return candidates

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
