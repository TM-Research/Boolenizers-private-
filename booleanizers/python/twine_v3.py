"""
TWINE v3: Minimalist Adaptive Thermometer Encoding

Philosophy: "Less is More"
- Keep dual-speed P² for drift detection
- Minimal hysteresis (just enough for stability, not too much)
- No drift gate (let thresholds adapt freely)
- No complex rate limiting (use simple exponential smoothing)
- No bit freezing detection (adds complexity without benefit)

Goal: Match SingleSpeed-P2's simplicity while keeping adaptive benefits.
"""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import DualSpeedP2


class TWINEv3(ThermometerEncoder):
    """
    TWINE v3: Minimalist adaptive thermometer encoding.

    Simplifications from v2:
    - Fixed minimal hysteresis (no adaptation)
    - No drift gate
    - No dynamic rate limiting
    - No bit activation boost
    - Pure dual-speed P² with simple threshold updates

    Parameters:
        K: Number of thresholds per feature
        fast_speed: Fast tracker speed (default: 1.5)
        slow_speed: Slow tracker speed (default: 0.5)
        h: Fixed hysteresis width (default: 0.05, much smaller than v2's 0.12)
        alpha: Threshold smoothing factor (default: 0.3)
    """

    def __init__(
        self,
        K: int = 8,
        fast_speed: float = 1.5,
        slow_speed: float = 0.5,
        h: float = 0.05,  # Minimal hysteresis
        alpha: float = 0.3,  # Simple exponential smoothing
    ):
        super().__init__(K=K, name="TWINE-v3")

        self.fast_speed = fast_speed
        self.slow_speed = slow_speed
        self.h = h
        self.alpha = alpha

        # State (initialized on first sample)
        self.trackers = None
        self.thresholds = None
        self.bit_state = None
        self.n_features = None

    def fit(self, X: np.ndarray) -> 'TWINEv3':
        """
        Not used for streaming - encoder adapts online.
        Kept for compatibility with offline comparison.
        """
        self.n_features = X.shape[1]

        # Initialize trackers
        self.trackers = [
            DualSpeedP2(K=self.K, fast_speed=self.fast_speed, slow_speed=self.slow_speed)
            for _ in range(self.n_features)
        ]

        # Initialize thresholds to uniform spacing
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]

        # Initialize bit states to 0
        self.bit_state = np.zeros((self.n_features, self.K), dtype=np.uint8)

        # Feed initial data
        for i in range(X.shape[0]):
            self._encode_single(X[i])

        self.fitted = True
        return self

    def _cold_start_init(self, x: np.ndarray):
        """Initialize encoder from first sample (for streaming)."""
        self.n_features = len(x)

        # Initialize trackers
        self.trackers = [
            DualSpeedP2(K=self.K, fast_speed=self.fast_speed, slow_speed=self.slow_speed)
            for _ in range(self.n_features)
        ]

        # Initialize thresholds to uniform spacing
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]

        # Initialize bit states
        self.bit_state = np.zeros((self.n_features, self.K), dtype=np.uint8)

        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform without updating trackers (frozen encoder)."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)

        for i in range(n_samples):
            for feat_idx in range(self.n_features):
                for k in range(self.K):
                    bit_idx = feat_idx * self.K + k

                    # Simple thermometer encoding with minimal hysteresis
                    if self.bit_state[feat_idx, k] == 1:
                        # Currently ON: turn OFF if x < threshold - h
                        encoded[i, bit_idx] = 1 if X[i, feat_idx] >= (self.thresholds[feat_idx, k] - self.h) else 0
                    else:
                        # Currently OFF: turn ON if x >= threshold + h
                        encoded[i, bit_idx] = 1 if X[i, feat_idx] >= (self.thresholds[feat_idx, k] + self.h) else 0

                    # Update state for next sample
                    self.bit_state[feat_idx, k] = encoded[i, bit_idx]

        return encoded

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode single sample and update trackers.

        Simplified logic:
        1. Update dual-speed P² trackers
        2. Get fast quantile estimates
        3. Smooth threshold updates with exponential moving average
        4. Apply minimal hysteresis
        """
        # Cold-start initialization
        if not self.fitted:
            self._cold_start_init(x)

        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        for i in range(self.n_features):
            # Update trackers with new observation
            self.trackers[i].update(x[i])

            # Get fast quantile estimates (more responsive)
            fast_quantiles = self.trackers[i].get_fast_quantiles()

            # Update thresholds with exponential smoothing
            for k in range(self.K):
                # Simple EMA: new_threshold = alpha * fast_q + (1-alpha) * old_threshold
                new_threshold = self.alpha * fast_quantiles[k] + (1 - self.alpha) * self.thresholds[i, k]
                self.thresholds[i, k] = new_threshold

            # Encode with minimal hysteresis
            for k in range(self.K):
                bit_idx = i * self.K + k

                if self.bit_state[i, k] == 1:
                    # Currently ON: turn OFF if x < threshold - h
                    bits[bit_idx] = 1 if x[i] >= (self.thresholds[i, k] - self.h) else 0
                else:
                    # Currently OFF: turn ON if x >= threshold + h
                    bits[bit_idx] = 1 if x[i] >= (self.thresholds[i, k] + self.h) else 0

                # Update state
                self.bit_state[i, k] = bits[bit_idx]

        return bits

    def get_n_output_bits(self) -> int:
        """Get total number of output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K

    def get_metrics(self) -> dict:
        """Get internal metrics for analysis."""
        if not self.fitted:
            return {}

        return {
            'thresholds': self.thresholds.copy(),
            'bit_state': self.bit_state.copy(),
            'fast_speed': self.fast_speed,
            'slow_speed': self.slow_speed,
            'hysteresis': self.h,
            'alpha': self.alpha,
        }



