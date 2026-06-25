"""
Dual Dynamics Binarizer (DDB)

==========================================================================
CORE PRINCIPLE: TWO-SPEED DYNAMICS + ADAPTIVE BIT ALLOCATION
==========================================================================

Combines insights from signal processing's fast/slow dynamics with
information-theoretic adaptive bit allocation.

DYNAMICS MODEL:
Each feature is modeled as having two characteristic scales:
  - "Fast" dynamics: local variation within dense regions (small Δx)
  - "Slow" dynamics: global range and distribution shape (large Δx)

Thresholds are placed to capture BOTH scales:
  - "Core" thresholds: capture fast dynamics (placed at density peaks
    via percentile clustering)
  - "Edge" thresholds: capture slow dynamics (placed at distribution
    transitions, covering the full range)

ADAPTIVE BIT ALLOCATION:
  - Constant features: 1 bit
  - Binary features: 1 bit
  - Low-cardinality (≤ K): 1 bit per unique value boundary
  - High-cardinality: K bits split between core/edge per dynamics ratio

The dual-dynamics view ensures that both common patterns (core) and
rare events (edge/tail) are captured, which is critical for anomaly
detection in security datasets.

==========================================================================
HYPERPARAMETERS (2): max_bits_per_feature, core_ratio
==========================================================================

    max_bits_per_feature : int (default 10)
        Maximum bits allocated per feature.

    core_ratio : float (default 0.6)
        Fraction of bits for core (fast-dynamic) thresholds.
        Rest goes to edge (slow-dynamic) thresholds.
        Valid range: [0.3, 0.9].

==========================================================================
COMPLEXITY
==========================================================================

    fit:       O(n * d) time, O(d * K) memory
    transform: O(n * d * K) time
    Streaming: O(d * K) per sample
"""

import numpy as np
from .base import ThermometerEncoder


class DualDynamicsBinarizer(ThermometerEncoder):
    """Dual Dynamics Binarizer (DDB).

    Two-speed threshold placement: core (density-focused) and edge
    (range-focused) thresholds, with adaptive bit allocation.
    """

    def __init__(self, max_bits_per_feature: int = 10, core_ratio: float = 0.6):
        super().__init__(K=max_bits_per_feature, name="DDB")
        self.max_bits = max_bits_per_feature
        self.core_ratio = np.clip(core_ratio, 0.3, 0.9)

    def fit(self, X: np.ndarray) -> "DualDynamicsBinarizer":
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
                th = self._dual_thresholds(col)

            self.feature_thresholds_.append(th)
            total_bits += len(th)

        self.total_bits_ = total_bits
        self.fitted = True
        return self

    def _dual_thresholds(self, col: np.ndarray) -> np.ndarray:
        """Place thresholds using dual fast/slow dynamics."""
        n_core = max(1, int(self.max_bits * self.core_ratio))
        n_edge = max(1, self.max_bits - n_core)

        # === CORE thresholds: fast dynamics (density-focused) ===
        # Use quantiles in the interquartile range (where most data lives)
        q1, q3 = np.percentile(col, [15, 85])
        core_mask = (col >= q1) & (col <= q3)
        core_data = col[core_mask] if core_mask.sum() > 10 else col

        core_q = np.linspace(0.05, 0.95, n_core + 2)[1:-1]
        core_th = np.quantile(core_data, core_q)

        # === EDGE thresholds: slow dynamics (range-focused) ===
        # Cover full range including tails, with emphasis on extremes
        # Use log-spaced positions near the edges for better tail resolution
        edge_positions = np.concatenate([
            np.linspace(0.01, 0.15, max(1, n_edge // 2)),
            np.linspace(0.85, 0.99, max(1, n_edge - n_edge // 2)),
        ])
        edge_th = np.quantile(col, edge_positions[:n_edge])

        # === Merge and deduplicate ===
        all_th = np.sort(np.unique(np.concatenate([core_th, edge_th])))

        # Trim to max_bits
        if len(all_th) > self.max_bits:
            idx = np.linspace(0, len(all_th) - 1, self.max_bits).astype(int)
            all_th = all_th[idx]

        # If we got fewer due to dedup, fill with uniform quantiles
        if len(all_th) < self.max_bits:
            n_fill = self.max_bits - len(all_th)
            fill_q = np.linspace(0.05, 0.95, n_fill + 2)[1:-1]
            fill_th = np.quantile(col, fill_q)
            all_th = np.sort(np.unique(np.concatenate([all_th, fill_th])))
            if len(all_th) > self.max_bits:
                idx = np.linspace(0, len(all_th) - 1, self.max_bits).astype(int)
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
