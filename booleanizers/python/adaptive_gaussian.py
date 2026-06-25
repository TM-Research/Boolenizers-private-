"""Adaptive Gaussian Thresholding encoder."""

import numpy as np
from .base import ThermometerEncoder


class AdaptiveGaussian(ThermometerEncoder):
    """
    Adaptive Gaussian thresholding encoder.

    Convolutional TMs binarize images by subtracting a local mean
    and thresholding at a multiple of the standard deviation.

    For streaming data, we use an exponentially weighted moving average (EWMA)
    and standard deviation, then create K thresholds around the adaptive mean.
    """

    def __init__(self, K: int = 8, alpha: float = 0.1, std_multiplier: float = 2.0):
        """
        Initialize Adaptive Gaussian encoder.

        Args:
            K: Number of thresholds per feature
            alpha: Smoothing factor for EWMA (0 < alpha <= 1)
            std_multiplier: Number of standard deviations for threshold spacing
        """
        super().__init__(K=K, name="Adaptive-Gaussian")
        self.alpha = alpha
        self.std_multiplier = std_multiplier

        # Running statistics per feature
        self.mean = None
        self.var = None
        self.count = 0

    def fit(self, X: np.ndarray) -> 'AdaptiveGaussian':
        """
        Fit encoder by initializing running statistics.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Initialize with batch statistics
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        self.count = X.shape[0]

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
        Encode a single sample with adaptive thresholds.

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Encoded sample of shape (n_features * K,)
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")

        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        # Update running statistics with EWMA
        if self.count > 0:
            delta = x - self.mean
            self.mean = self.mean + self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta ** 2)

        self.count += 1

        # Compute adaptive thresholds for each feature
        std = np.sqrt(np.maximum(self.var, 1e-6))

        for i in range(self.n_features):
            # Create K thresholds around mean ± std_multiplier * std
            # Spread thresholds from (mean - std_mult*std) to (mean + std_mult*std)
            thresholds = np.linspace(
                self.mean[i] - self.std_multiplier * std[i],
                self.mean[i] + self.std_multiplier * std[i],
                self.K
            )

            for k in range(self.K):
                bit_idx = i * self.K + k
                bits[bit_idx] = 1 if x[i] >= thresholds[k] else 0

        return bits

