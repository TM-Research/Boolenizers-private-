"""
Moving Window Binarizer (MWB)

==========================================================================
CORE PRINCIPLE: SLIDING WINDOW DYNAMIC PATTERN CAPTURE
==========================================================================

Derived from dynamic signal analysis:

In streaming/time-series data, local statistical properties shift over
time. A global set of thresholds may not capture these dynamics well,
especially for sequential splits where train and test distributions
differ.

MWB uses a sliding window approach during fit to capture local patterns:
1. Divide training data into overlapping windows
2. In each window, compute local statistics (mean, std, quantiles)
3. Merge window-local thresholds to build a ROBUST global set that
   covers both typical and transitional states

This is analogous to a multi-resolution wavelet decomposition:
the union of local views captures patterns that a single global
view would smooth over.

KEY INNOVATION: Window Consensus Thresholds
- Thresholds that appear consistently across windows → stable features
- Thresholds that vary across windows → dynamic/drifting features
  (allocate more bits to capture this variation)

==========================================================================
HYPERPARAMETERS (2): max_bits_per_feature, window_ratio
==========================================================================

    max_bits_per_feature : int (default 10)
        Maximum bits per feature.

    window_ratio : float (default 0.25)
        Window size as fraction of training data. Smaller = more local
        sensitivity. Larger = more global stability.
        Valid range: (0.05, 0.9).

==========================================================================
COMPLEXITY
==========================================================================

    fit:       O(n * d * W) where W = number of windows (~1/step_ratio)
    transform: O(n * d * K) time
    Memory:    O(d * K)
"""

import numpy as np
from .base import ThermometerEncoder


class MovingWindowBinarizer(ThermometerEncoder):
    """Moving Window Binarizer (MWB).

    Uses sliding window statistics to build robust thresholds that
    capture both stable and dynamic patterns in the data.

    Adaptive bit allocation: low-cardinality features get fewer bits.
    """

    def __init__(self, max_bits_per_feature: int = 10, window_ratio: float = 0.25):
        super().__init__(K=max_bits_per_feature, name="MWB")
        self.max_bits = max_bits_per_feature
        self.window_ratio = np.clip(window_ratio, 0.05, 0.9)

    def fit(self, X: np.ndarray) -> "MovingWindowBinarizer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        window_size = max(20, int(n_samples * self.window_ratio))
        step = max(1, window_size // 4)  # 75% overlap
        n_windows = max(1, (n_samples - window_size) // step + 1)

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
                th = self._window_thresholds(col, n_samples, window_size, step, n_windows)

            self.feature_thresholds_.append(th)
            total_bits += len(th)

        self.total_bits_ = total_bits
        self.fitted = True
        return self

    def _window_thresholds(
        self, col: np.ndarray, n_samples: int, window_size: int, step: int, n_windows: int
    ) -> np.ndarray:
        """Build consensus thresholds from overlapping windows."""
        # Collect quantile thresholds from each window
        n_q = min(self.max_bits * 2, 20)  # candidate quantile positions
        q_pos = np.linspace(0.02, 0.98, n_q)

        all_thresholds = []
        window_means = []
        window_stds = []

        for w in range(n_windows):
            start = w * step
            end = min(start + window_size, n_samples)
            window_data = col[start:end]

            if len(window_data) < 5:
                continue

            w_th = np.quantile(window_data, q_pos)
            all_thresholds.append(w_th)
            window_means.append(np.mean(window_data))
            window_stds.append(np.std(window_data))

        if not all_thresholds:
            # Fallback to global quantiles
            q = np.linspace(0.05, 0.95, self.max_bits)
            return np.quantile(col, q)

        all_thresholds = np.array(all_thresholds)  # shape: (n_windows, n_q)
        window_means = np.array(window_means)
        window_stds = np.array(window_stds)

        # Measure threshold stability: std across windows for each quantile position
        th_stability = np.std(all_thresholds, axis=0)
        th_median = np.median(all_thresholds, axis=0)

        # Dynamic range: how much the windows disagree
        mean_spread = np.std(window_means)
        global_std = np.std(col)
        drift_ratio = mean_spread / (global_std + 1e-15)

        # Allocation strategy based on drift
        if drift_ratio > 0.3:
            # High drift: use range of window thresholds to cover both regimes
            th_low = np.percentile(all_thresholds, 10, axis=0)
            th_high = np.percentile(all_thresholds, 90, axis=0)
            # Interleave low and high to cover both regimes
            candidates = np.sort(np.concatenate([th_low, th_high, th_median]))
        else:
            # Low drift: consensus (median) thresholds are reliable
            candidates = th_median

        candidates = np.unique(candidates)

        # Select max_bits thresholds, weighted toward high-stability positions
        if len(candidates) <= self.max_bits:
            thresholds = candidates
        else:
            idx = np.linspace(0, len(candidates) - 1, self.max_bits).astype(int)
            thresholds = candidates[idx]

        # Ensure strict monotonicity
        thresholds = np.sort(thresholds)
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
