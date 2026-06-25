"""Stochastic Searching on the Line (SSL) encoder.

Two automata per feature learn lower and upper bounds,
producing 2 Boolean bits per feature (not K bits like other encoders).
"""

import numpy as np
from .base import ThermometerEncoder


class SSL(ThermometerEncoder):
    """
    Stochastic Searching on the Line.

    Uses two learning automata per feature to find discriminative
    lower and upper bounds. Produces 2 bits per feature:
    - Bit 1: x >= lower_bound
    - Bit 2: x >= upper_bound

    Note: This produces fewer bits than other encoders (2 per feature vs K).
    """

    def __init__(self, K: int = 8):
        """
        Initialize SSL encoder.

        Args:
            K: Ignored for SSL (always produces 2 bits per feature)
        """
        super().__init__(K=2, name="SSL")  # SSL always produces 2 bits
        self.lower_bounds = None
        self.upper_bounds = None

    def fit(self, X: np.ndarray) -> 'SSL':
        """
        Fit by computing lower and upper bounds (approximated as quartiles).

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Approximate SSL bounds with quartiles
        # Lower bound = 25th percentile, Upper bound = 75th percentile
        self.lower_bounds = np.percentile(X, 25, axis=0)
        self.upper_bounds = np.percentile(X, 75, axis=0)

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to 2-bit codes per feature.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            Encoded data of shape (n_samples, n_features * 2)
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features * 2), dtype=np.uint8)

        for i in range(n_samples):
            encoded[i] = self._encode_single(X[i])

        return encoded

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample.

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Encoded sample of shape (n_features * 2,)
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")

        bits = np.zeros(self.n_features * 2, dtype=np.uint8)

        for i in range(self.n_features):
            # Bit 1: x >= lower_bound
            bits[i * 2] = 1 if x[i] >= self.lower_bounds[i] else 0
            # Bit 2: x >= upper_bound
            bits[i * 2 + 1] = 1 if x[i] >= self.upper_bounds[i] else 0

        return bits

    def get_n_output_bits(self) -> int:
        """Get total number of output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * 2  # Always 2 bits per feature



