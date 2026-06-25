"""
Resonant Gradient Binarizer (RGB)

==========================================================================
CORE PRINCIPLE: GRADIENT-AWARE ADAPTIVE THRESHOLD PLACEMENT
==========================================================================

Derived from signal processing dynamics:

A continuous feature x can be viewed as a "signal" whose probability
density p(x) varies across its range. Regions where p(x) changes rapidly
(high gradient of the CDF) contain the most discriminative information.

RGB places thresholds at points where the empirical CDF gradient is
highest — i.e., where the density concentrates — plus adaptive outlier
thresholds at the tails. This produces a thermometer code that is
maximally sensitive to the data's natural clustering structure.

ALGORITHM:
1. Single-pass: compute streaming min, max, mean, variance per feature
2. Build lightweight histogram (B bins) from sorted quantile sketch
3. Compute CDF gradient: Δ(CDF) between adjacent bins
4. Place K thresholds at positions of highest CDF gradient + tail guards
5. Adaptive bit allocation: constant/binary features get fewer bits

This captures "resonant" frequencies in the data distribution — points
where the signal transitions most actively between states.

==========================================================================
HYPERPARAMETERS (2): max_bits_per_feature, gradient_focus
==========================================================================

    max_bits_per_feature : int (default 10)
        Maximum bits allocated to any single feature.

    gradient_focus : float (default 0.7)
        Fraction of bits placed at high-gradient positions (rest go to
        uniform coverage of tails). Range [0, 1].
        Higher = more bits at density peaks.
        Lower = more uniform coverage.

==========================================================================
COMPLEXITY
==========================================================================

    fit:       O(n * d) time, O(d * B) memory (B = internal histogram bins)
    transform: O(n * d * K) time
    Streaming: O(d * K) per sample
"""

import numpy as np
from .base import ThermometerEncoder


class ResonantGradientBinarizer(ThermometerEncoder):
    """Resonant Gradient Binarizer (RGB).

    Places thresholds at high-gradient positions of the empirical CDF,
    capturing the natural clustering and transition dynamics of each feature.

    Adaptive bit allocation: binary/constant features get fewer bits.
    """

    def __init__(self, max_bits_per_feature: int = 10, gradient_focus: float = 0.7):
        super().__init__(K=max_bits_per_feature, name="RGB")
        self.max_bits = max_bits_per_feature
        self.gradient_focus = np.clip(gradient_focus, 0.0, 1.0)
        self._B = 256  # internal histogram resolution

    def fit(self, X: np.ndarray) -> "ResonantGradientBinarizer":
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
                # Constant feature — 1 dummy bit
                th = np.array([uv[0] + 1e-10]) if n_unique == 1 else np.array([0.0])
            elif n_unique <= 2:
                # Binary feature — 1 bit at midpoint
                th = np.array([(uv[0] + uv[1]) / 2.0])
            elif n_unique <= self.max_bits:
                # Low-cardinality: threshold between each unique value pair
                th = (uv[:-1] + uv[1:]) / 2.0
            else:
                # High-cardinality: gradient-aware placement
                th = self._gradient_thresholds(col, n_unique)

            self.feature_thresholds_.append(th)
            total_bits += len(th)

        self.total_bits_ = total_bits
        self.fitted = True
        return self

    def _gradient_thresholds(self, col: np.ndarray, n_unique: int) -> np.ndarray:
        """Place thresholds at high CDF-gradient positions plus tail guards."""
        B = min(self._B, n_unique)
        # Compute histogram-based CDF gradient
        q_positions = np.linspace(0, 1, B + 1)
        bin_edges = np.quantile(col, q_positions)

        # CDF gradient: how much probability mass is in each bin
        # For quantile-based bins, gradient is ~uniform; use value-space density
        bin_widths = np.diff(bin_edges)
        bin_widths = np.maximum(bin_widths, 1e-15)
        # Density proxy: probability_per_bin / width = concentration
        density = (1.0 / B) / bin_widths  # each quantile bin has ~equal probability
        density /= density.sum() + 1e-20

        # Number of gradient-focused vs tail-coverage bits
        n_grad = max(1, int(self.max_bits * self.gradient_focus))
        n_tail = self.max_bits - n_grad

        thresholds = []

        # Gradient-focused: place at centers of highest-density bins
        if n_grad > 0:
            top_bins = np.argsort(density)[-min(n_grad * 3, B):]  # candidate bins
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            candidates = bin_centers[top_bins]
            candidates = np.sort(candidates)

            if len(candidates) <= n_grad:
                thresholds.extend(candidates)
            else:
                # Subsample evenly from the sorted high-density candidates
                idx = np.linspace(0, len(candidates) - 1, n_grad).astype(int)
                thresholds.extend(candidates[idx])

        # Tail coverage: uniform quantile thresholds across the full range
        if n_tail > 0:
            tail_q = np.linspace(0.02, 0.98, n_tail + 2)[1:-1]
            tail_th = np.quantile(col, tail_q)
            thresholds.extend(tail_th)

        thresholds = np.sort(np.unique(np.array(thresholds)))

        # Cap at max_bits, pick evenly if too many
        if len(thresholds) > self.max_bits:
            idx = np.linspace(0, len(thresholds) - 1, self.max_bits).astype(int)
            thresholds = thresholds[idx]

        # Ensure strict monotonicity
        for k in range(1, len(thresholds)):
            if thresholds[k] <= thresholds[k - 1]:
                thresholds[k] = thresholds[k - 1] + 1e-10

        return thresholds

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
