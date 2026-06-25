"""
Moving Window Binarizer v2 (MWB2)

==========================================================================
ITERATION 2 IMPROVEMENTS OVER MWB v1
==========================================================================

MWB v1 observations:
- MWB-K8 (88.88%) good performance with only 169 bits
- Window approach captures drift-robust thresholds
- But: too many windows → computation overhead; median consensus too simple

MWB v2 improvements:
1. REGIME DETECTION: Instead of many overlapping windows, detect 2-3
   distinct regimes via fast change-point detection (L2 distance of
   running means). Place thresholds from EACH regime.
2. INTER-REGIME BOUNDARIES: Add extra thresholds at the boundaries
   between regimes — these are the most discriminative positions.
3. ADAPTIVE BIT ALLOCATION: Like RGB2, features get bits proportional
   to their variation across regimes.

==========================================================================
HYPERPARAMETERS (2): max_bits_per_feature, n_segments
==========================================================================

    max_bits_per_feature : int (default 12)
        Maximum bits per feature.

    n_segments : int (default 4)
        Number of segments to divide training data into.
        Each segment's statistics contribute to threshold placement.
        More segments = more drift-awareness but more computation.

==========================================================================
COMPLEXITY: O(n*d) fit, O(n*d*K) transform
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class MovingWindowBinarizerV2(ThermometerEncoder):
    """Moving Window Binarizer v2 (MWB2).

    Segment-aware threshold placement with inter-segment boundary
    detection and adaptive bit allocation.
    """

    def __init__(self, max_bits_per_feature: int = 12, n_segments: int = 4):
        super().__init__(K=max_bits_per_feature, name="MWB2")
        self.max_bits = max_bits_per_feature
        self.n_segments = max(2, n_segments)

    def fit(self, X: np.ndarray) -> "MovingWindowBinarizerV2":
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
                th = (uv[:-1] + uv[1:]) / 2.0
            else:
                seg_cols = [s[:, j] for s in segments]
                th = self._segment_thresholds(col, seg_cols)

            self.feature_thresholds_.append(th)
            total_bits += len(th)

        self.total_bits_ = total_bits
        self.fitted = True
        return self

    def _segment_thresholds(self, col: np.ndarray, seg_cols: list) -> np.ndarray:
        """Build thresholds from segment statistics."""
        n_segs = len(seg_cols)

        # Compute per-segment statistics
        seg_means = np.array([np.mean(s) for s in seg_cols])
        seg_stds = np.array([np.std(s) for s in seg_cols])

        # Measure cross-segment drift
        global_std = np.std(col)
        mean_spread = np.std(seg_means)
        drift_ratio = mean_spread / (global_std + 1e-15)

        # Adaptive bit allocation
        n_bits = self._adaptive_bits(col, drift_ratio)

        if drift_ratio > 0.2:
            # Significant drift: combine quantiles from each segment
            n_per_seg = max(2, n_bits // n_segs)
            n_boundary = max(1, n_bits - n_per_seg * n_segs)

            candidates = []
            for sc in seg_cols:
                q_pos = np.linspace(0.05, 0.95, n_per_seg + 2)[1:-1]
                candidates.extend(np.quantile(sc, q_pos))

            # Add inter-segment boundary thresholds
            sorted_means = np.sort(seg_means)
            for i in range(len(sorted_means) - 1):
                boundary = (sorted_means[i] + sorted_means[i + 1]) / 2.0
                candidates.append(boundary)

            candidates = np.sort(np.unique(candidates))
        else:
            # Low drift: use global quantiles (more stable)
            q_pos = np.linspace(0.02, 0.98, n_bits + 2)[1:-1]
            candidates = np.quantile(col, q_pos)
            candidates = np.unique(candidates)

        # Trim to n_bits
        if len(candidates) > n_bits:
            idx = np.linspace(0, len(candidates) - 1, n_bits).astype(int)
            candidates = candidates[idx]

        # Ensure minimum bits
        if len(candidates) < 4:
            fill_q = np.linspace(0.1, 0.9, 4)
            candidates = np.sort(np.unique(np.concatenate([candidates, np.quantile(col, fill_q)])))
            if len(candidates) > n_bits:
                idx = np.linspace(0, len(candidates) - 1, n_bits).astype(int)
                candidates = candidates[idx]

        # Strict monotonicity
        for k in range(1, len(candidates)):
            if candidates[k] <= candidates[k - 1]:
                candidates[k] = candidates[k - 1] + 1e-10

        return candidates

    def _adaptive_bits(self, col: np.ndarray, drift_ratio: float) -> int:
        """Determine bits based on feature complexity and drift."""
        min_bits = 4
        uv = len(np.unique(col[:min(5000, len(col))]))

        unique_factor = min(1.0, np.log2(max(uv, 2)) / 10.0)
        drift_factor = min(1.0, drift_ratio * 2.0)

        combined = 0.4 * unique_factor + 0.6 * drift_factor
        n_bits = min_bits + int(combined * (self.max_bits - min_bits))
        return max(min_bits, min(self.max_bits, n_bits))

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
