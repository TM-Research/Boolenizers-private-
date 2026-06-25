"""Normal Thermometer Encoder with Batch Quantiles."""

import numpy as np
from collections import deque
from .base import ThermometerEncoder


class NTEBatchQuantile(ThermometerEncoder):
    """
    Normal Thermometer Encoder with batch quantile bins.

    Thresholds are computed as batch quantiles of a calibration sample
    or sliding window. Quantile bins better align with signal distribution
    but require storing W samples and cause jumps at refit times.

    No hysteresis.
    """

    def __init__(self, K: int = 8, window_size: int = 1000, refit_interval: int = None):
        """
        Initialize NTE-BatchQ encoder.

        Args:
            K: Number of thresholds per feature
            window_size: Window size for computing quantiles
            refit_interval: How often to recompute quantiles (None = fit once)
        """
        super().__init__(K=K, name="NTE-BatchQ")
        self.window_size = window_size
        self.refit_interval = refit_interval
        self.thresholds = None

        # For online refitting
        self.windows = None  # List of deques, one per feature
        self.update_count = 0
        self.online_mode = False

    def fit(self, X: np.ndarray) -> 'NTEBatchQuantile':
        """
        Fit encoder by computing quantile thresholds.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Compute quantile thresholds
        self.thresholds = np.zeros((self.n_features, self.K))

        # Quantile probabilities for K thresholds
        probs = np.linspace(1 / (self.K + 1), self.K / (self.K + 1), self.K)

        for i in range(self.n_features):
            # Compute K quantiles for feature i
            self.thresholds[i] = np.quantile(X[:, i], probs)

        # Initialize windows for online refitting if needed
        if self.refit_interval is not None:
            self.online_mode = True
            self.windows = [deque(maxlen=self.window_size) for _ in range(self.n_features)]

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

            # Update windows if in online mode
            if self.online_mode:
                for j in range(self.n_features):
                    self.windows[j].append(X[i, j])

                self.update_count += 1

                # Refit if interval reached
                if self.update_count % self.refit_interval == 0:
                    self._refit_from_windows()

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

        # Update windows if in online mode
        if self.online_mode and self.windows is not None:
            for j in range(self.n_features):
                self.windows[j].append(x[j])

            self.update_count += 1

            # Refit if interval reached
            if self.refit_interval is not None and self.update_count % self.refit_interval == 0:
                self._refit_from_windows()

        return bits

    def _refit_from_windows(self):
        """Recompute quantiles from current windows."""
        probs = np.linspace(1 / (self.K + 1), self.K / (self.K + 1), self.K)

        for i in range(self.n_features):
            if len(self.windows[i]) > 0:
                window_data = np.array(self.windows[i])
                self.thresholds[i] = np.quantile(window_data, probs)

