"""
TWINE Streaming Values: Copy StandardBinarizer's winning strategy but make it streaming

KEY INSIGHT: StandardBinarizer (94.05%) doesn't use quantile ESTIMATES - it uses
ACTUAL DATA VALUES as thresholds!

Strategy:
1. Maintain reservoir of actual seen values (reservoir sampling)
2. Periodically sample from reservoir to create thresholds
3. Use actual values, not quantile approximations
4. NO hysteresis (data values are already stable)

This is fundamentally different from P²-based methods!
"""

import numpy as np
from .base import ThermometerEncoder


class TWINEStreamingValues(ThermometerEncoder):
    """
    TWINE Streaming Values: Reservoir-based threshold selection.

    Copies StandardBinarizer's approach but online:
    - Keep reservoir of K*M actual values seen
    - Sample K values from reservoir as thresholds
    - Update reservoir with reservoir sampling algorithm

    Parameters:
        K: Number of thresholds per feature
        reservoir_size: How many values to keep per feature (default: K*10)
        update_freq: How often to resample thresholds (default: 100 samples)
    """

    def __init__(
        self,
        K: int = 8,
        reservoir_size: int = 80,  # K * 10
        update_freq: int = 100,
    ):
        super().__init__(K=K, name="TWINE-StreamingValues")
        self.reservoir_size = reservoir_size
        self.update_freq = update_freq

        # State
        self.reservoirs = None  # One per feature
        self.reservoir_counts = None  # How many values seen
        self.thresholds = None
        self.n_features = None
        self.sample_count = 0

    def fit(self, X: np.ndarray) -> 'TWINEStreamingValues':
        """Initialize from batch."""
        self.n_features = X.shape[1]
        self._init_structures()

        # Feed data
        for i in range(X.shape[0]):
            self._encode_single(X[i])

        self.fitted = True
        return self

    def _init_structures(self):
        """Initialize reservoirs and thresholds."""
        # Reservoir for each feature
        self.reservoirs = [[] for _ in range(self.n_features)]
        self.reservoir_counts = np.zeros(self.n_features, dtype=int)

        # Thresholds (initially uniform in [0,1])
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]

        self.fitted = True

    def _cold_start_init(self, x: np.ndarray):
        """Initialize from first sample."""
        self.n_features = len(x)
        self._init_structures()

    def _update_reservoir(self, feature_idx: int, value: float):
        """
        Update reservoir using reservoir sampling algorithm.

        Guarantees uniform random sample of all seen values.
        """
        self.reservoir_counts[feature_idx] += 1
        n = self.reservoir_counts[feature_idx]

        if len(self.reservoirs[feature_idx]) < self.reservoir_size:
            # Reservoir not full, just add
            self.reservoirs[feature_idx].append(value)
        else:
            # Reservoir full, randomly replace with decreasing probability
            j = np.random.randint(0, n)
            if j < self.reservoir_size:
                self.reservoirs[feature_idx][j] = value

    def _sample_thresholds(self, feature_idx: int):
        """
        Sample K thresholds from reservoir.

        Strategy: Sort reservoir values, take evenly spaced samples
        (same as StandardBinarizer's approach)
        """
        if len(self.reservoirs[feature_idx]) < self.K:
            # Not enough values yet, keep initial
            return

        # Sort reservoir
        sorted_values = np.sort(self.reservoirs[feature_idx])

        # Sample K values evenly
        indices = np.linspace(0, len(sorted_values) - 1, self.K, dtype=int)
        self.thresholds[feature_idx] = sorted_values[indices]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform (frozen, no reservoir updates)."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)

        for i in range(n_samples):
            for feat_idx in range(self.n_features):
                for k in range(self.K):
                    bit_idx = feat_idx * self.K + k
                    # Simple threshold (like StandardBinarizer, no hysteresis!)
                    encoded[i, bit_idx] = 1 if X[i, feat_idx] >= self.thresholds[feat_idx, k] else 0

        return encoded

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode single sample and update reservoirs.

        Key: Use actual data values, not quantile approximations!
        """
        if not self.fitted:
            self._cold_start_init(x)

        self.sample_count += 1
        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        for i in range(self.n_features):
            # Update reservoir with this value
            self._update_reservoir(i, x[i])

            # Periodically resample thresholds from reservoir
            if self.sample_count % self.update_freq == 0:
                self._sample_thresholds(i)

            # Encode (simple, no hysteresis - just like StandardBinarizer!)
            for k in range(self.K):
                bit_idx = i * self.K + k
                bits[bit_idx] = 1 if x[i] >= self.thresholds[i, k] else 0

        return bits

    def get_n_output_bits(self) -> int:
        """Get output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K

    def get_metrics(self) -> dict:
        """Get internal metrics."""
        if not self.fitted:
            return {}

        return {
            'reservoir_sizes': [len(r) for r in self.reservoirs],
            'reservoir_counts': self.reservoir_counts.copy(),
            'sample_count': self.sample_count,
            'thresholds_sample': self.thresholds[0].tolist(),  # First feature as example
        }



