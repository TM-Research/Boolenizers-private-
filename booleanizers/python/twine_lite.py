"""
TWINE-Lite: Minimalist High-Performance Encoder

PHILOSOPHY: Start with SingleSpeed-P2 (93.15%), add ONLY what helps.

Changes from SingleSpeed-P2:
1. Tiny hysteresis (h=0.03) - just enough to prevent noise
2. That's it!

Changes from TWINE v2:
- Remove: Dual-speed tracking (complexity, no benefit)
- Remove: Drift gate (slows adaptation)
- Remove: Adaptive mechanisms (unnecessary)
- Keep: Minimal hysteresis for stability

This is basically: "SingleSpeed-P2 + minimal hysteresis"
"""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import P2Quantile


class TWINELite(ThermometerEncoder):
    """
    TWINE-Lite: Efficient high-performance streaming encoder.

    Core insight: SingleSpeed-P2 works great. Add MINIMAL modifications.

    Parameters:
        K: Bits per feature
        speed: P² update speed (default: 1.0, same as SingleSpeed-P2)
        h: Minimal hysteresis (default: 0.03, very small)
    """

    def __init__(self, K: int = 8, speed: float = 1.0, h: float = 0.03):
        super().__init__(K=K, name="TWINE-Lite")
        self.speed = speed
        self.h = h  # Tiny hysteresis

        self.trackers = None
        self.bit_state = None  # For hysteresis
        self.n_features = None

    def fit(self, X: np.ndarray) -> 'TWINELite':
        """Initialize from batch."""
        self.n_features = X.shape[1]

        # Initialize P² trackers (same as SingleSpeed-P2)
        self.trackers = [P2Quantile(K=self.K, speed=self.speed) for _ in range(self.n_features)]

        # Initialize bit states
        self.bit_state = np.zeros((self.n_features, self.K), dtype=np.uint8)

        # Feed data
        for i in range(X.shape[0]):
            self._encode_single(X[i])

        self.fitted = True
        return self

    def _cold_start_init(self, x: np.ndarray):
        """Initialize from first sample (same as SingleSpeed-P2)."""
        self.n_features = len(x)

        self.trackers = [P2Quantile(K=self.K, speed=self.speed) for _ in range(self.n_features)]
        self.bit_state = np.zeros((self.n_features, self.K), dtype=np.uint8)

        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform (frozen, with hysteresis state)."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)

        for i in range(n_samples):
            for feat_idx in range(self.n_features):
                thresholds = self.trackers[feat_idx].get_thresholds()

                for k in range(self.K):
                    bit_idx = feat_idx * self.K + k

                    # Minimal hysteresis
                    if self.bit_state[feat_idx, k] == 1:
                        encoded[i, bit_idx] = 1 if X[i, feat_idx] >= (thresholds[k] - self.h) else 0
                    else:
                        encoded[i, bit_idx] = 1 if X[i, feat_idx] >= (thresholds[k] + self.h) else 0

                    self.bit_state[feat_idx, k] = encoded[i, bit_idx]

        return encoded

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode single sample.

        Same as SingleSpeed-P2 but with tiny hysteresis.
        """
        if not self.fitted:
            self._cold_start_init(x)

        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        for i in range(self.n_features):
            # Get thresholds (same as SingleSpeed-P2)
            thresholds = self.trackers[i].get_thresholds()

            # Encode with tiny hysteresis
            for k in range(self.K):
                bit_idx = i * self.K + k

                if self.bit_state[i, k] == 1:
                    # Currently ON: turn OFF if below threshold - h
                    bits[bit_idx] = 1 if x[i] >= (thresholds[k] - self.h) else 0
                else:
                    # Currently OFF: turn ON if above threshold + h
                    bits[bit_idx] = 1 if x[i] >= (thresholds[k] + self.h) else 0

                self.bit_state[i, k] = bits[bit_idx]

            # Update tracker (same as SingleSpeed-P2)
            self.trackers[i].update(x[i])

        return bits

    def get_n_output_bits(self) -> int:
        """Get output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K



