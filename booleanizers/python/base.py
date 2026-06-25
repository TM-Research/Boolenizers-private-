"""Base class for thermometer encoders."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class ThermometerEncoder(ABC):
    """
    Base class for thermometer encoders.

    Thermometer encoding converts continuous features into K Boolean bits
    using thresholds τ₁ < τ₂ < ... < τₖ, where bit k is 1 if x ≥ τₖ.

    All encoders track flip statistics for stability analysis.
    """

    def __init__(self, K: int = 8, name: str = "BaseEncoder"):
        """
        Initialize thermometer encoder.

        Args:
            K: Number of thresholds (bits) per feature
            name: Encoder name for identification
        """
        self.K = K
        self.name = name
        self.n_features = None
        self.fitted = False

        # Flip tracking
        self.previous_bits = None
        self.flip_counts = None
        self.total_updates = 0

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'ThermometerEncoder':
        """
        Fit the encoder to training data.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to thermometer codes.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            Encoded data of shape (n_samples, n_features * K)
        """
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Subclasses overriding this for **true streaming** semantics should make
        a single pass over ``X``: each emitted row of bits is computed with
        the state from previous rows only, then state updates to include the
        current row. The default implementation falls back to ``fit`` + ``transform``
        (a two-pass workflow that is not strictly online).
        """
        return self.fit(X).transform(X)

    def encode_online(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample online and track flips.

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Encoded sample of shape (n_features * K,)
        """
        # Get encoded bits
        bits = self._encode_single(x)

        # Track flips
        if self.previous_bits is not None:
            flips = (bits != self.previous_bits).astype(int)
            self.flip_counts += flips
        else:
            # Initialize tracking
            self.flip_counts = np.zeros(len(bits), dtype=np.int64)

        self.previous_bits = bits.copy()
        self.total_updates += 1

        return bits

    @abstractmethod
    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Internal method to encode a single sample.
        Subclasses implement actual encoding logic here.

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Encoded sample of shape (n_features * K,)
        """
        pass

    def get_flip_stats(self) -> Dict[str, Any]:
        """
        Get flip statistics.

        Returns:
            Dictionary with min, mean, max, std of flip rates across bits
        """
        if self.flip_counts is None or self.total_updates == 0:
            return {
                'min': 0.0,
                'mean': 0.0,
                'max': 0.0,
                'std': 0.0,
                'total_updates': 0,
            }

        flip_rates = self.flip_counts / max(self.total_updates, 1)

        return {
            'min': float(np.min(flip_rates)),
            'mean': float(np.mean(flip_rates)),
            'max': float(np.max(flip_rates)),
            'std': float(np.std(flip_rates)),
            'total_updates': self.total_updates,
            'flip_rates': flip_rates.copy(),
        }

    def reset_stats(self):
        """Reset flip tracking statistics."""
        self.previous_bits = None
        self.flip_counts = None
        self.total_updates = 0

    def get_n_output_bits(self) -> int:
        """Get total number of output bits."""
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K

    def __repr__(self) -> str:
        return f"{self.name}(K={self.K}, fitted={self.fitted})"

