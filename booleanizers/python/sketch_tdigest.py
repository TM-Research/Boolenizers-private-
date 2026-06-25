"""t-digest sketch encoder."""

import numpy as np
from .base import ThermometerEncoder


class SketchTDigest(ThermometerEncoder):
    """
    t-digest quantile sketch encoder.

    Simplified implementation using a fixed-size centroid buffer.
    This approximates the full t-digest algorithm for MCU deployment.
    """

    def __init__(self, K: int = 8, compression: int = 100):
        """
        Initialize t-digest sketch encoder.

        Args:
            K: Number of thresholds per feature
            compression: Compression parameter (max centroids)
        """
        super().__init__(K=K, name="Sketch-TDigest")
        self.compression = compression
        self.centroids = None  # List of centroid lists, one per feature
        self.counts = None     # List of count lists, one per feature

    def fit(self, X: np.ndarray) -> 'SketchTDigest':
        """
        Fit encoder by initializing t-digest sketches.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Initialize centroids and counts
        self.centroids = [[] for _ in range(self.n_features)]
        self.counts = [[] for _ in range(self.n_features)]

        # Feed initial data
        for i in range(X.shape[0]):
            for j in range(self.n_features):
                self._update_digest(j, X[i, j])

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
        
        # Initialize centroids and counts
        self.centroids = [[] for _ in range(self.n_features)]
        self.counts = [[] for _ in range(self.n_features)]
        
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
            # Compute quantiles from digest
            thresholds = self._get_quantiles(i)

            # Encode
            for k in range(self.K):
                bit_idx = i * self.K + k
                bits[bit_idx] = 1 if x[i] >= thresholds[k] else 0

            # Update digest
            self._update_digest(i, x[i])

        return bits

    def _update_digest(self, feature_idx: int, value: float):
        """Update digest for a feature."""
        # Simplified: add as new centroid
        self.centroids[feature_idx].append(value)
        self.counts[feature_idx].append(1)

        # Compress if needed
        if len(self.centroids[feature_idx]) > self.compression:
            self._compress(feature_idx)

    def _compress(self, feature_idx: int):
        """Compress centroids by merging nearby ones."""
        if len(self.centroids[feature_idx]) <= 1:
            return

        # Sort centroids
        indices = np.argsort(self.centroids[feature_idx])
        sorted_centroids = [self.centroids[feature_idx][i] for i in indices]
        sorted_counts = [self.counts[feature_idx][i] for i in indices]

        # Merge adjacent pairs to reduce size
        new_centroids = []
        new_counts = []

        i = 0
        while i < len(sorted_centroids):
            if i + 1 < len(sorted_centroids) and len(new_centroids) >= self.compression // 2:
                # Merge pairs
                c1, c2 = sorted_centroids[i], sorted_centroids[i + 1]
                n1, n2 = sorted_counts[i], sorted_counts[i + 1]
                merged_c = (c1 * n1 + c2 * n2) / (n1 + n2)
                merged_n = n1 + n2
                new_centroids.append(merged_c)
                new_counts.append(merged_n)
                i += 2
            else:
                new_centroids.append(sorted_centroids[i])
                new_counts.append(sorted_counts[i])
                i += 1

        self.centroids[feature_idx] = new_centroids
        self.counts[feature_idx] = new_counts

    def _get_quantiles(self, feature_idx: int) -> np.ndarray:
        """Get K quantiles from digest."""
        if len(self.centroids[feature_idx]) == 0:
            return np.zeros(self.K)

        # Sort centroids
        indices = np.argsort(self.centroids[feature_idx])
        sorted_centroids = np.array([self.centroids[feature_idx][i] for i in indices])
        sorted_counts = np.array([self.counts[feature_idx][i] for i in indices])

        # Compute cumulative distribution
        total_count = np.sum(sorted_counts)
        cumulative = np.cumsum(sorted_counts) / total_count

        # Find quantiles
        probs = np.linspace(1 / (self.K + 1), self.K / (self.K + 1), self.K)
        quantiles = np.interp(probs, cumulative, sorted_centroids)

        return quantiles

