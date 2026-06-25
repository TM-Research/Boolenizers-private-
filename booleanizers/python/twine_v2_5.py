"""
TWINE v2.5: Hybrid Simplicity

Lessons from experiments:
- v2 complexity is justified (v3 failed at 85%)
- But SingleSpeed-P2's simplicity has merit (93.15%)
- Quick test: v2 BEATS SingleSpeed-P2 (+0.20%)
- Full test: v2 LOSES to SingleSpeed-P2 (-0.75%)

Hypothesis: v2 has good mechanisms but poor initialization/warmup

v2.5 Design:
- Keep: Dual-speed P², drift detection
- Simplify: Fixed hysteresis (no adaptation), simpler rate limiting
- Improve: Better cold-start initialization, warmup period
- Add: Drift-magnitude responsive updates
"""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import DualSpeedP2


class TWINEv2_5(ThermometerEncoder):
    """
    TWINE v2.5: Refined balance of complexity and simplicity.

    Changes from v2:
    - ✅ Keep dual-speed P² (core innovation)
    - ✅ Keep drift detection
    - ❌ Remove adaptive hysteresis (use fixed)
    - ❌ Remove bit freeze detection
    - ✅ Add better warm-start initialization
    - ✅ Add drift-magnitude responsive learning

    Parameters:
        K: Number of thresholds per feature
        fast_speed: Fast P² speed (default: 2.0, more aggressive)
        slow_speed: Slow P² speed (default: 0.3, more stable)
        h: Fixed hysteresis (default: 0.15)
        base_lr: Base learning rate for threshold updates (default: 0.1)
        drift_boost: Multiplier when drift detected (default: 2.0)
        warmup_samples: Number of samples for initialization (default: 100)
    """

    def __init__(
        self,
        K: int = 8,
        fast_speed: float = 2.0,
        slow_speed: float = 0.3,
        h: float = 0.15,
        base_lr: float = 0.1,
        drift_boost: float = 2.0,
        warmup_samples: int = 100,
    ):
        super().__init__(K=K, name="TWINE-v2.5")

        self.fast_speed = fast_speed
        self.slow_speed = slow_speed
        self.h = h
        self.base_lr = base_lr
        self.drift_boost = drift_boost
        self.warmup_samples = warmup_samples

        # State
        self.trackers = None
        self.thresholds = None
        self.bit_state = None
        self.n_features = None
        self.sample_count = 0
        self.warmup_buffer = []

    def fit(self, X: np.ndarray) -> 'TWINEv2_5':
        """Initialize with batch data (for offline comparison)."""
        self.n_features = X.shape[1]
        self._init_structures()

        # Feed data
        for i in range(X.shape[0]):
            self._encode_single(X[i])

        self.fitted = True
        return self

    def _init_structures(self):
        """Initialize data structures."""
        # Dual-speed trackers
        self.trackers = [
            DualSpeedP2(K=self.K, fast_speed=self.fast_speed, slow_speed=self.slow_speed)
            for _ in range(self.n_features)
        ]

        # Thresholds - initialize to uniform
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]

        # Bit states
        self.bit_state = np.zeros((self.n_features, self.K), dtype=np.uint8)

        self.fitted = True

    def _cold_start_init(self, x: np.ndarray):
        """
        Smart cold-start initialization.

        Strategy: Collect first N samples, use to initialize thresholds
        based on actual data distribution instead of uniform [0,1].
        """
        self.n_features = len(x)

        # Add to warmup buffer
        self.warmup_buffer.append(x.copy())

        if len(self.warmup_buffer) >= self.warmup_samples:
            # Initialize from warmup data
            warmup_data = np.array(self.warmup_buffer)

            # Initialize structures
            self._init_structures()

            # Set initial thresholds from warmup quantiles
            for i in range(self.n_features):
                quantiles = [(k+1)/(self.K+1) for k in range(self.K)]
                self.thresholds[i] = np.quantile(warmup_data[:, i], quantiles)

            # Feed warmup data to trackers
            for sample in warmup_data:
                for i in range(self.n_features):
                    self.trackers[i].update(sample[i])

            self.sample_count = len(self.warmup_buffer)
            self.warmup_buffer = []  # Clear buffer
            self.fitted = True
        else:
            # Still warming up
            self.fitted = False

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform without updating (frozen encoder)."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)

        for i in range(n_samples):
            for feat_idx in range(self.n_features):
                for k in range(self.K):
                    bit_idx = feat_idx * self.K + k

                    # Hysteresis
                    if self.bit_state[feat_idx, k] == 1:
                        encoded[i, bit_idx] = 1 if X[i, feat_idx] >= (self.thresholds[feat_idx, k] - self.h) else 0
                    else:
                        encoded[i, bit_idx] = 1 if X[i, feat_idx] >= (self.thresholds[feat_idx, k] + self.h) else 0

                    self.bit_state[feat_idx, k] = encoded[i, bit_idx]

        return encoded

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode single sample with drift-responsive updates.

        Key insight: When drift is detected (fast-slow divergence),
        increase learning rate to adapt faster.
        """
        # Cold-start handling
        if not self.fitted:
            self._cold_start_init(x)
            if not self.fitted:  # Still in warmup
                return np.zeros(self.n_features * self.K, dtype=np.uint8)

        self.sample_count += 1
        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        for i in range(self.n_features):
            # Update trackers
            self.trackers[i].update(x[i])

            # Get quantile estimates
            fast_q = self.trackers[i].get_fast_quantiles()
            slow_q = self.trackers[i].get_slow_quantiles()

            # Compute drift magnitude (fast-slow divergence)
            drift_mag = np.mean(np.abs(fast_q - slow_q))

            # Adaptive learning rate based on drift
            if drift_mag > 0.1:  # Significant drift detected
                lr = self.base_lr * self.drift_boost
            else:
                lr = self.base_lr

            # Update thresholds with adaptive LR
            for k in range(self.K):
                # Move threshold towards fast estimate
                self.thresholds[i, k] += lr * (fast_q[k] - self.thresholds[i, k])

            # Encode with fixed hysteresis
            for k in range(self.K):
                bit_idx = i * self.K + k

                if self.bit_state[i, k] == 1:
                    bits[bit_idx] = 1 if x[i] >= (self.thresholds[i, k] - self.h) else 0
                else:
                    bits[bit_idx] = 1 if x[i] >= (self.thresholds[i, k] + self.h) else 0

                self.bit_state[i, k] = bits[bit_idx]

        return bits

    def get_n_output_bits(self) -> int:
        """Get total number of output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K

    def get_metrics(self) -> dict:
        """Get internal metrics."""
        if not self.fitted:
            return {}

        # Compute average drift across features
        drifts = []
        for i in range(self.n_features):
            fast_q = self.trackers[i].get_fast_quantiles()
            slow_q = self.trackers[i].get_slow_quantiles()
            drift = np.mean(np.abs(fast_q - slow_q))
            drifts.append(drift)

        return {
            'thresholds': self.thresholds.copy(),
            'bit_state': self.bit_state.copy(),
            'sample_count': self.sample_count,
            'mean_drift': np.mean(drifts),
            'max_drift': np.max(drifts),
            'fast_speed': self.fast_speed,
            'slow_speed': self.slow_speed,
            'hysteresis': self.h,
        }



