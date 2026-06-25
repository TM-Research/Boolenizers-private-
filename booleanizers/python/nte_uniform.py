"""Normal Thermometer Encoder with Uniform bins."""

import numpy as np
from .base import ThermometerEncoder


class NTEUniform(ThermometerEncoder):
    """
    Normal Thermometer Encoder with uniform bins.

    Each feature is divided into K uniformly spaced bins between
    known min and max values (estimated on calibration set).
    The k-th thermometer bit is set if x ≥ k-th bin boundary.

    No drift adaptation or hysteresis.
    """

    def __init__(self, K: int = 8, epsilon: float = 1e-6):
        """
        Initialize NTE-Uniform encoder.

        Args:
            K: Number of thresholds per feature
            epsilon: Small value to avoid edge cases
        """
        super().__init__(K=K, name="NTE-Uniform")
        self.epsilon = epsilon
        self.min_vals = None
        self.max_vals = None
        self.thresholds = None

    def fit(self, X: np.ndarray) -> 'NTEUniform':
        """
        Fit encoder by computing min/max and uniform thresholds.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Compute min and max per feature
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)

        # Handle constant features
        range_vals = self.max_vals - self.min_vals
        range_vals = np.maximum(range_vals, self.epsilon)

        # Create uniform thresholds for each feature
        # thresholds[i, k] = threshold k for feature i
        self.thresholds = np.zeros((self.n_features, self.K))

        for i in range(self.n_features):
            # K thresholds uniformly spaced between min and max
            self.thresholds[i] = np.linspace(
                self.min_vals[i] + range_vals[i] / (self.K + 1),
                self.max_vals[i] - range_vals[i] / (self.K + 1),
                self.K
            )

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

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample.

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Encoded sample of shape (n_features * K,)
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")

        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        for i in range(self.n_features):
            for k in range(self.K):
                # Thermometer coding: bit is 1 if x >= threshold
                bit_idx = i * self.K + k
                bits[bit_idx] = 1 if x[i] >= self.thresholds[i, k] else 0

        return bits

