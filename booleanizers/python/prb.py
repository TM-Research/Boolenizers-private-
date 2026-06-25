"""
Pulse Resonance Binarizer (PRB)

==========================================================================
CORE PRINCIPLE: SIGNAL PULSE DETECTION + RESONANCE PLACEMENT
==========================================================================

Inspired by how digital signal processors detect meaningful transitions:

A feature's value distribution can be viewed as a "pulse train" where
each unique value region represents a "pulse". The PRB places thresholds
at the natural pulse boundaries — transitions between value clusters.

ALGORITHM:
1. Compute sorted unique values with their frequencies
2. Identify "gaps" in the value space (regions with no data)
3. Place thresholds at the midpoints of the largest gaps
4. Fill remaining bits with density-proportional quantiles

This naturally captures:
- Categorical-like features: gaps between distinct groups
- Continuous features: density modes separated by low-density valleys
- Mixed features: combination of both

The "resonance" comes from matching the binarization resolution to the
natural frequency (cluster spacing) of the data.

==========================================================================
HYPERPARAMETERS (2): max_bits_per_feature, gap_weight
==========================================================================

    max_bits_per_feature : int (default 12)
        Maximum bits per feature.

    gap_weight : float (default 0.5)
        Weight for gap-based vs quantile-based threshold placement.
        0 = pure quantile, 1 = pure gap-based.
        0.5 = balanced between gap and density-aware placement.

==========================================================================
COMPLEXITY: O(n*d + d*U*log(U)) where U = unique values per feature
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class PulseResonanceBinarizer(ThermometerEncoder):
    """Pulse Resonance Binarizer (PRB).

    Places thresholds at natural cluster boundaries (gaps) and
    density-proportional positions, with adaptive bit allocation.
    """

    def __init__(self, max_bits_per_feature: int = 12, gap_weight: float = 0.5):
        super().__init__(K=max_bits_per_feature, name="PRB")
        self.max_bits = max_bits_per_feature
        self.gap_weight = np.clip(gap_weight, 0.0, 1.0)

    def fit(self, X: np.ndarray) -> "PulseResonanceBinarizer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

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
                th = self._pulse_thresholds(col, uv)

            self.feature_thresholds_.append(th)
            total_bits += len(th)

        self.total_bits_ = total_bits
        self.fitted = True
        return self

    def _pulse_thresholds(self, col: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """Place thresholds using gap detection + density quantiles."""
        # Subsample unique values for efficiency (max 1000)
        if len(uv) > 1000:
            idx = np.linspace(0, len(uv) - 1, 1000).astype(int)
            uv_sub = uv[idx]
        else:
            uv_sub = uv

        # Adaptive bit count
        n_bits = self._adaptive_bits(col, uv)

        # === Component 1: Gap-based thresholds ===
        n_gap = max(1, int(n_bits * self.gap_weight))
        n_density = n_bits - n_gap

        # Find gaps between consecutive unique values
        gaps = np.diff(uv_sub)
        if len(gaps) > 0:
            # Threshold at midpoint of the largest gaps
            gap_order = np.argsort(gaps)[::-1]
            gap_th = []
            for i in gap_order[:n_gap * 2]:  # take more candidates, filter later
                midpoint = (uv_sub[i] + uv_sub[i + 1]) / 2.0
                gap_th.append(midpoint)
            gap_th = np.sort(np.unique(gap_th))
            if len(gap_th) > n_gap:
                idx = np.linspace(0, len(gap_th) - 1, n_gap).astype(int)
                gap_th = gap_th[idx]
        else:
            gap_th = np.array([])

        # === Component 2: Density quantile thresholds ===
        if n_density > 0:
            density_q = np.linspace(0.02, 0.98, n_density + 2)[1:-1]
            density_th = np.quantile(col, density_q)
        else:
            density_th = np.array([])

        # === Merge ===
        all_th = np.sort(np.unique(np.concatenate([gap_th, density_th])))

        if len(all_th) > n_bits:
            idx = np.linspace(0, len(all_th) - 1, n_bits).astype(int)
            all_th = all_th[idx]

        # Fill if needed
        if len(all_th) < 4:
            fill_q = np.linspace(0.1, 0.9, max(4, n_bits))
            all_th = np.sort(np.unique(np.concatenate([all_th, np.quantile(col, fill_q)])))
            if len(all_th) > n_bits:
                idx = np.linspace(0, len(all_th) - 1, n_bits).astype(int)
                all_th = all_th[idx]

        # Strict monotonicity
        for k in range(1, len(all_th)):
            if all_th[k] <= all_th[k - 1]:
                all_th[k] = all_th[k - 1] + 1e-10

        return all_th

    def _adaptive_bits(self, col: np.ndarray, uv: np.ndarray) -> int:
        """Determine bits based on unique value distribution."""
        min_bits = 4
        n_unique = len(uv)

        # Features with many clusters need more bits
        # Estimate number of clusters via gap analysis
        if n_unique > 2:
            sub = uv if len(uv) <= 1000 else uv[np.linspace(0, len(uv)-1, 1000).astype(int)]
            gaps = np.diff(sub)
            median_gap = np.median(gaps)
            large_gaps = np.sum(gaps > 3 * median_gap) if median_gap > 0 else 0
            cluster_factor = min(1.0, large_gaps / 10.0)
        else:
            cluster_factor = 0.0

        unique_factor = min(1.0, np.log2(max(n_unique, 2)) / 10.0)
        combined = 0.5 * unique_factor + 0.5 * cluster_factor
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
