"""Wrapper for tmu's StandardBinarizer."""

import numpy as np
from .base import ThermometerEncoder


class StandardBinarizerWrapper(ThermometerEncoder):
    """
    Wrapper around tmu's StandardBinarizer for fair comparison.

    The StandardBinarizer uses unique values from the training set
    as thresholds, sampling up to max_bits_per_feature thresholds
    if there are too many unique values.
    """

    def __init__(self, K: int = 8):
        """
        Initialize StandardBinarizer wrapper.

        Args:
            K: Number of bits per feature (max_bits_per_feature)
        """
        super().__init__(K=K, name="StandardBinarizer")

        # Import here to avoid issues if tmu is not available
        try:
            from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
            self.binarizer = StandardBinarizer(max_bits_per_feature=K)
        except ImportError:
            raise ImportError("tmu package not available for StandardBinarizer")

    def fit(self, X: np.ndarray) -> 'StandardBinarizerWrapper':
        """
        Fit the StandardBinarizer.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]
        self.binarizer.fit(X)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using StandardBinarizer.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            Encoded data of shape (n_samples, n_output_bits)
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        encoded = self.binarizer.transform(X)
        return encoded.astype(np.uint8)

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample.

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Encoded sample
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")

        # StandardBinarizer expects 2D input
        encoded = self.binarizer.transform(x.reshape(1, -1))
        return encoded.flatten().astype(np.uint8)

    def get_n_output_bits(self) -> int:
        """Get total number of output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.binarizer.number_of_features

