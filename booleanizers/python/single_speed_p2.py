"""Single-Speed P² encoder."""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import P2Quantile


class SingleSpeedP2(ThermometerEncoder):
    """
    Single-Speed P² quantile encoder.

    Uses one P² tracker per feature to directly track quantiles
    as thresholds. No drift gating or hysteresis.

    This isolates the effect of dual-speed tracking and stability knobs.
    """

    def __init__(self, K: int = 8, speed: float = 1.0):
        """
        Initialize SS-P² encoder.

        Args:
            K: Number of thresholds per feature
            speed: Update speed for P² algorithm
        """
        super().__init__(K=K, name="SS-P2")
        self.speed = speed
        self.trackers = None

    def fit(self, X: np.ndarray) -> 'SingleSpeedP2':
        """
        Fit encoder by initializing P² trackers.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Initialize one P² tracker per feature
        self.trackers = [P2Quantile(K=self.K, speed=self.speed) for _ in range(self.n_features)]

        # Feed initial data to trackers
        for i in range(X.shape[0]):
            for j in range(self.n_features):
                self.trackers[j].update(X[i, j])

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
        
        # Initialize one P² tracker per feature (each tracks K quantiles)
        self.trackers = [P2Quantile(K=self.K, speed=self.speed) for _ in range(self.n_features)]
        
        # Initialize thresholds to uniform spacing in [0, 1] range
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]
        
        self.fitted = True
    
    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample and update trackers.

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
            # Get current thresholds
            thresholds = self.trackers[i].get_thresholds()

            # Encode with thermometer coding
            for k in range(self.K):
                bit_idx = i * self.K + k
                bits[bit_idx] = 1 if x[i] >= thresholds[k] else 0

            # Update tracker with new observation
            self.trackers[i].update(x[i])

        return bits

