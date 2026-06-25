"""Native StandardBinarizer implementation (no tmu dependency).

Implements the algorithm from https://arxiv.org/pdf/1905.04199.pdf, Section 3.3.
"""

import typing
from typing import List

import numpy as np

from .base import ThermometerEncoder


class StandardBinarizerNative(ThermometerEncoder):
    """
    Native implementation of the standard TM binarizer.

    From https://arxiv.org/pdf/1905.04199.pdf, Section 3.3.
    Uses unique values from training data as thresholds, sampling up to K
    thresholds per feature if there are too many unique values.

    Hyperparameters:
        K: max_bits_per_feature - how many threshold values each feature uses.
    """

    unique_values: List[np.ndarray]
    number_of_features: int

    def __init__(self, K: int = 25):
        """
        Initialize StandardBinarizer.

        Args:
            K: max_bits_per_feature - max thresholds per feature
        """
        super().__init__(K=K, name="StandardBinarizerNative")
        self.unique_values = []
        self.number_of_features = 0

    def fit(self, X: np.ndarray) -> "StandardBinarizerNative":
        """
        Fit the binarizer to training data.

        For each feature, computes unique values (excluding min) and samples
        up to K of them as thresholds using uniform step sampling.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]
        self.unique_values = []
        self.number_of_features = 0

        for i in range(X.shape[1]):
            uv = np.unique(X[:, i])[1:]  # exclude min (all bits would be 1)

            if uv.size > self.K:
                unique_vals = np.empty(0)
                step_size = 1.0 * uv.size / self.K
                pos = 0.0
                while int(pos) < uv.size and unique_vals.size < self.K:
                    unique_vals = np.append(unique_vals, np.array(uv[int(pos)]))
                    pos += step_size
            else:
                unique_vals = uv.copy()

            self.unique_values.append(unique_vals)
            self.number_of_features += self.unique_values[-1].size

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to thermometer codes.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            Encoded data of shape (n_samples, number_of_features)
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        X_transformed = np.zeros((X.shape[0], self.number_of_features), dtype=np.uint8)
        pos = 0
        for i in range(X.shape[1]):
            for j in range(self.unique_values[i].size):
                X_transformed[:, pos] = (X[:, i] >= self.unique_values[i][j]).astype(np.uint8)
                pos += 1

        return X_transformed

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

        encoded = np.zeros(self.number_of_features, dtype=np.uint8)
        pos = 0
        for i in range(x.shape[0]):
            for j in range(self.unique_values[i].size):
                encoded[pos] = 1 if x[i] >= self.unique_values[i][j] else 0
                pos += 1
        return encoded

    def get_n_output_bits(self) -> int:
        """Get total number of output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.number_of_features
