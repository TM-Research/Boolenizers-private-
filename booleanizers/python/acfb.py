"""Adaptive Continuous Feature Binarization (ACFB) encoder.

Based on: Abeyrathna et al., adaptive continuous feature binarization
that standardizes each feature and samples a small number of thresholds.
"""

import numpy as np
from .base import ThermometerEncoder


class ACFB(ThermometerEncoder):
    """
    Adaptive Continuous Feature Binarization.

    Standardizes each feature (z-score) and samples K thresholds
    based on standard deviations. Thresholds remain fixed after fitting.
    """

    def __init__(self, K: int = 8, std_range: float = 3.0):
        """
        Initialize ACFB encoder.

        Args:
            K: Number of thresholds per feature
            std_range: Range in standard deviations (thresholds placed in [-std_range, std_range])
        """
        super().__init__(K=K, name="ACFB")
        self.std_range = std_range
        self.mean = None
        self.std = None
        self.thresholds = None

    def fit(self, X: np.ndarray) -> 'ACFB':
        """
        Fit by computing mean/std and sampling thresholds.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Compute statistics
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8  # Avoid division by zero

        # Sample thresholds uniformly in standardized space
        # Place K thresholds in [-std_range, std_range]
        std_thresholds = np.linspace(-self.std_range, self.std_range, self.K + 2)[1:-1]

        # Convert back to original space for each feature
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = self.mean[i] + std_thresholds * self.std[i]

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
                bit_idx = i * self.K + k
                bits[bit_idx] = 1 if x[i] >= self.thresholds[i, k] else 0

        return bits

    def get_n_output_bits(self) -> int:
        """Get total number of output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K



