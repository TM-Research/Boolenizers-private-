"""
TWINE v5: Meta-Learning Approach

BREAKTHROUGH IDEA: What if we LEARN which parameters work best DURING training?

Instead of fixed hyperparameters, adapt them based on recent performance!

Meta-parameters to learn:
- Hysteresis h (per bit or per feature)
- Learning rate eta
- Fast/slow speeds

Mechanism: Track local accuracy/stability → adjust parameters

This is like AutoML but happening ONLINE during training!
"""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import DualSpeedP2


class TWINEv5MetaLearning(ThermometerEncoder):
    """
    TWINE v5: Meta-learning adaptation of hyperparameters.

    Continuously adjusts h, eta, speeds based on observed performance.

    Performance signals:
    - Bit flip rate (too high → increase h)
    - Drift magnitude (too high → increase eta)
    - Threshold variance (too high → decrease speeds)

    Parameters:
        K: Bits per feature
        h_range: (min, max) hysteresis
        eta_range: (min, max) learning rate
        meta_lr: Rate of meta-parameter updates
    """

    def __init__(
        self,
        K: int = 8,
        h_range: tuple = (0.05, 0.30),
        eta_range: tuple = (0.05, 0.20),
        speed_range: tuple = (0.5, 2.5),
        meta_lr: float = 0.01,
        window_size: int = 100,
    ):
        super().__init__(K=K, name="TWINE-v5-MetaLearning")

        self.h_range = h_range
        self.eta_range = eta_range
        self.speed_range = speed_range
        self.meta_lr = meta_lr
        self.window_size = window_size

        # Adaptive parameters (start at middle of ranges)
        self.h = np.mean(h_range)
        self.eta = np.mean(eta_range)
        self.fast_speed = np.mean(speed_range) * 1.5
        self.slow_speed = np.mean(speed_range) * 0.5

        # State
        self.trackers = None
        self.thresholds = None
        self.bit_state = None
        self.n_features = None

        # Meta-learning tracking
        self.recent_flips = []
        self.recent_drifts = []
        self.sample_count = 0

    def fit(self, X: np.ndarray) -> 'TWINEv5MetaLearning':
        """Initialize."""
        self.n_features = X.shape[1]
        self._init_structures()

        for i in range(X.shape[0]):
            self._encode_single(X[i])

        self.fitted = True
        return self

    def _init_structures(self):
        """Initialize trackers and thresholds."""
        self.trackers = [
            DualSpeedP2(K=self.K, fast_speed=self.fast_speed, slow_speed=self.slow_speed)
            for _ in range(self.n_features)
        ]

        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]

        self.bit_state = np.zeros((self.n_features, self.K), dtype=np.uint8)
        self.fitted = True

    def _cold_start_init(self, x: np.ndarray):
        """Initialize from first sample."""
        self.n_features = len(x)
        self._init_structures()

    def _meta_update(self):
        """Update meta-parameters based on recent performance."""
        if len(self.recent_flips) < 10:
            return  # Not enough data yet

        # Compute statistics over recent window
        mean_flip_rate = np.mean(self.recent_flips[-self.window_size:])
        mean_drift = np.mean(self.recent_drifts[-self.window_size:])

        # Adapt hysteresis based on flip rate
        # High flip rate → increase h (more stability)
        # Low flip rate → decrease h (more responsiveness)
        target_flip_rate = 0.15
        h_gradient = (mean_flip_rate - target_flip_rate) * self.meta_lr
        self.h = np.clip(self.h + h_gradient, self.h_range[0], self.h_range[1])

        # Adapt learning rate based on drift
        # High drift → increase eta (faster adaptation)
        # Low drift → decrease eta (more stability)
        target_drift = 0.10
        eta_gradient = (mean_drift - target_drift) * self.meta_lr
        self.eta = np.clip(self.eta + eta_gradient, self.eta_range[0], self.eta_range[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform (frozen)."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)

        for i in range(n_samples):
            for feat_idx in range(self.n_features):
                for k in range(self.K):
                    bit_idx = feat_idx * self.K + k

                    if self.bit_state[feat_idx, k] == 1:
                        encoded[i, bit_idx] = 1 if X[i, feat_idx] >= (self.thresholds[feat_idx, k] - self.h) else 0
                    else:
                        encoded[i, bit_idx] = 1 if X[i, feat_idx] >= (self.thresholds[feat_idx, k] + self.h) else 0

                    self.bit_state[feat_idx, k] = encoded[i, bit_idx]

        return encoded

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """Encode with meta-learning."""
        if not self.fitted:
            self._cold_start_init(x)

        self.sample_count += 1
        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        # Track flips and drift this sample
        sample_flips = 0
        sample_drift = 0

        for i in range(self.n_features):
            self.trackers[i].update(x[i])

            fast_q = self.trackers[i].get_fast_thresholds()
            slow_q = self.trackers[i].get_slow_thresholds()

            # Update thresholds with current eta
            self.thresholds[i] += self.eta * (fast_q - self.thresholds[i])

            # Track drift (K interior thresholds each)
            sample_drift += np.mean(np.abs(fast_q - slow_q))

            # Encode with current h
            for k in range(self.K):
                bit_idx = i * self.K + k
                old_bit = self.bit_state[i, k]

                if old_bit == 1:
                    new_bit = 1 if x[i] >= (self.thresholds[i, k] - self.h) else 0
                else:
                    new_bit = 1 if x[i] >= (self.thresholds[i, k] + self.h) else 0

                bits[bit_idx] = new_bit
                self.bit_state[i, k] = new_bit

                # Track flip
                if new_bit != old_bit:
                    sample_flips += 1

        # Record metrics
        flip_rate = sample_flips / (self.n_features * self.K)
        self.recent_flips.append(flip_rate)
        self.recent_drifts.append(sample_drift / self.n_features)

        # Meta-update every N samples
        if self.sample_count % 50 == 0:
            self._meta_update()

        return bits

    def get_n_output_bits(self) -> int:
        """Get output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K

    def get_metrics(self) -> dict:
        """Get current meta-parameters."""
        return {
            'h': self.h,
            'eta': self.eta,
            'fast_speed': self.fast_speed,
            'slow_speed': self.slow_speed,
            'mean_flip_rate': np.mean(self.recent_flips[-100:]) if self.recent_flips else 0,
            'mean_drift': np.mean(self.recent_drifts[-100:]) if self.recent_drifts else 0,
        }



