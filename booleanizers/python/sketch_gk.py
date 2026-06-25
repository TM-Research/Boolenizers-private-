"""Greenwald-Khanna sketch encoder."""

import numpy as np
from .base import ThermometerEncoder


class SketchGK(ThermometerEncoder):
    """
    Greenwald-Khanna quantile sketch encoder.

    Simplified implementation focusing on quantile estimation.
    For MCU deployment, we use a fixed-size buffer approach
    that approximates the full GK algorithm.
    """

    def __init__(self, K: int = 8, buffer_size: int = 500, epsilon: float = 0.01):
        """
        Initialize GK sketch encoder.

        Args:
            K: Number of thresholds per feature
            buffer_size: Maximum buffer size for sketch
            epsilon: Approximation error bound
        """
        super().__init__(K=K, name="Sketch-GK")
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.buffers = None  # One buffer per feature

    def fit(self, X: np.ndarray) -> 'SketchGK':
        """
        Fit encoder by initializing sketches.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Initialize buffers (simplified: just keep recent samples)
        self.buffers = [[] for _ in range(self.n_features)]

        # Populate buffers with initial data
        for i in range(X.shape[0]):
            for j in range(self.n_features):
                self._update_buffer(j, X[i, j])

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to thermometer codes.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            Encoded data of shape (n_samples, n_features * K)
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)

        for i in range(n_samples):
            encoded[i] = self._encode_single(X[i])

        return encoded

    def _cold_start_init(self, x: np.ndarray):
        """Initialize encoder from a single sample (for streaming)."""
        self.n_features = len(x)
        
        # Initialize buffers (simplified: just keep recent samples)
        self.buffers = [[] for _ in range(self.n_features)]
        
        self.fitted = True
    
    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample.

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Encoded sample of shape (n_features * K,)
        """
        # Lazy initialization for streaming: initialize on first sample
        if not self.fitted:
            self._cold_start_init(x)

        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        for i in range(self.n_features):
            # Compute quantiles from buffer
            thresholds = self._get_quantiles(i)

            # Encode
            for k in range(self.K):
                bit_idx = i * self.K + k
                bits[bit_idx] = 1 if x[i] >= thresholds[k] else 0

            # Update buffer
            self._update_buffer(i, x[i])

        return bits

    def _update_buffer(self, feature_idx: int, value: float):
        """Update buffer for a feature."""
        self.buffers[feature_idx].append(value)

        # Keep only recent samples
        if len(self.buffers[feature_idx]) > self.buffer_size:
            self.buffers[feature_idx].pop(0)

    def _get_quantiles(self, feature_idx: int) -> np.ndarray:
        """Get K quantiles from buffer."""
        if len(self.buffers[feature_idx]) == 0:
            return np.zeros(self.K)

        buffer = np.array(self.buffers[feature_idx])
        probs = np.linspace(1 / (self.K + 1), self.K / (self.K + 1), self.K)

        return np.quantile(buffer, probs)

