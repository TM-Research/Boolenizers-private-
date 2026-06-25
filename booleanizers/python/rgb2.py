"""
Resonant Gradient Binarizer v2 (RGB2)

==========================================================================
ITERATION 2 IMPROVEMENTS OVER RGB v1
==========================================================================

RGB v1 observations:
- Gradient-focused placement (RGB-G50 @ 91.24%) beats pure quantile (AQB @ 84.83%)
- Fewer bits (189 vs 245) with higher accuracy
- K8 variant strong at 90.29% with only 162 bits

Weaknesses identified:
1. DDB's dual-dynamics approach failed because edge thresholds at extreme
   quantiles (1%, 99%) are fragile — they capture noise, not signal
2. MWB's window approach is good for drift but adds computation cost
3. Missing: combining gradient focus with RANGE coverage in a single pass

RGB v2 design:
1. GRADIENT-DENSITY FUSION: Instead of separate core/edge, fuse density
   gradient with range-proportional spacing in a single threshold set
2. PERCENTILE CLIPPING: Robust range from [2nd, 98th] percentile instead
   of min/max — removes outlier sensitivity
3. DYNAMIC BIT SCALING: Features with high density contrast get more bits,
   uniform features get fewer (automatic, not fixed K)

==========================================================================
HYPERPARAMETERS (2): max_bits_per_feature, density_weight
==========================================================================

    max_bits_per_feature : int (default 12)
        Maximum bits per feature.

    density_weight : float (default 0.6)
        Weight for density-proportional placement vs uniform spacing.
        0 = pure uniform, 1 = pure density-proportional.

==========================================================================
COMPLEXITY: O(n*d) fit, O(n*d*K) transform, O(d*K) memory
==========================================================================
"""

import numpy as np
from .base import ThermometerEncoder


class ResonantGradientBinarizerV2(ThermometerEncoder):
    """Resonant Gradient Binarizer v2 (RGB2).

    Fuses density-proportional and uniform threshold placement with
    adaptive bit allocation per feature.
    """

    def __init__(self, max_bits_per_feature: int = 12, density_weight: float = 0.6):
        super().__init__(K=max_bits_per_feature, name="RGB2")
        self.max_bits = max_bits_per_feature
        self.density_weight = np.clip(density_weight, 0.0, 1.0)

    def fit(self, X: np.ndarray) -> "ResonantGradientBinarizerV2":
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
                # Midpoint thresholds between unique values
                th = (uv[:-1] + uv[1:]) / 2.0
            else:
                # Adaptive bit count based on "information content"
                n_bits = self._adaptive_bits(col, n_unique)
                th = self._fused_thresholds(col, n_bits)

            self.feature_thresholds_.append(th)
            total_bits += len(th)

        self.total_bits_ = total_bits
        self.fitted = True
        return self

    def _adaptive_bits(self, col: np.ndarray, n_unique: int) -> int:
        """Determine bits based on feature information content."""
        # Coefficient of variation (CV) — measures how spread out the data is
        mu = np.mean(col)
        std = np.std(col)
        if abs(mu) > 1e-10:
            cv = std / abs(mu)
        else:
            cv = std

        # High CV = more variation = needs more bits
        # Scale from 4 bits (boring/uniform) to max_bits (high variation)
        min_bits = 4
        # Use log-scaled unique count as well
        unique_factor = min(1.0, np.log2(n_unique) / 10.0)
        cv_factor = min(1.0, cv / 2.0)

        combined = 0.5 * unique_factor + 0.5 * cv_factor
        n_bits = min_bits + int(combined * (self.max_bits - min_bits))
        return max(min_bits, min(self.max_bits, n_bits))

    def _fused_thresholds(self, col: np.ndarray, n_bits: int) -> np.ndarray:
        """Fused density-proportional + uniform threshold placement."""
        # Robust range: clip to [2nd, 98th] percentile
        p2, p98 = np.percentile(col, [2, 98])
        if p98 <= p2:
            p2, p98 = np.min(col), np.max(col)
        if p98 <= p2:
            return np.array([p2 + 1e-10])

        # === Component 1: Density-proportional (quantile) thresholds ===
        dw = self.density_weight
        n_density = max(1, int(n_bits * dw))
        n_uniform = n_bits - n_density

        density_q = np.linspace(0.02, 0.98, n_density + 2)[1:-1]
        density_th = np.quantile(col, density_q)

        # === Component 2: Uniform spacing within robust range ===
        if n_uniform > 0:
            uniform_th = np.linspace(p2, p98, n_uniform + 2)[1:-1]
        else:
            uniform_th = np.array([])

        # === Fuse and deduplicate ===
        all_th = np.sort(np.unique(np.concatenate([density_th, uniform_th])))

        # Trim to target
        if len(all_th) > n_bits:
            idx = np.linspace(0, len(all_th) - 1, n_bits).astype(int)
            all_th = all_th[idx]

        # Fill if dedup reduced count
        while len(all_th) < n_bits:
            fill_q = np.random.uniform(0.05, 0.95)
            fill_th = np.quantile(col, fill_q)
            all_th = np.sort(np.unique(np.append(all_th, fill_th)))
            if len(all_th) >= n_bits:
                break
            # Add jitter to fill remaining
            all_th = np.sort(np.unique(np.append(all_th, fill_th + 1e-10 * len(all_th))))

        if len(all_th) > n_bits:
            idx = np.linspace(0, len(all_th) - 1, n_bits).astype(int)
            all_th = all_th[idx]

        # Ensure strict monotonicity
        for k in range(1, len(all_th)):
            if all_th[k] <= all_th[k - 1]:
                all_th[k] = all_th[k - 1] + 1e-10

        return all_th

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
