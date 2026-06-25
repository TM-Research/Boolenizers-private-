"""
TWINE: Twin-Tracker Online Quantile Encoding with Hysteresis.

The main encoder from the paper implementing:
- Dual-speed P² quantile tracking (slow/fast)
- Drift gate based on normalized disagreement
- Rate-limited threshold motion
- Per-bit hysteresis with optional cooldown
- Pre-update encoding
"""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import DualSpeedP2


class TWINE(ThermometerEncoder):
    """
    TWINE encoder for stable, drift-aware thermometer coding.

    Parameters from paper (Table 1, Section 3):
    - K: Number of thresholds per feature (typically 4 or 8)
    - τ (tau): Drift gate threshold (0.3-0.6, default 0.4)
    - η (eta): Rate limit on threshold motion (0.04-0.12, default 0.08)
    - h: Hysteresis fraction (0.08-0.20, default 0.12)
    - C: Cooldown samples after flip (0-3, default 1)
    """

    def __init__(
        self,
        K: int = 8,
        tau: float = 0.4,
        eta: float = 0.08,
        h: float = 0.12,
        C: int = 1,
        slow_speed: float = 0.5,
        fast_speed: float = 2.0,
        epsilon_s: float = 1e-6
    ):
        """
        Initialize TWINE encoder.

        Args:
            K: Number of thresholds per feature
            tau: Drift gate threshold for switching to fast tracker
            eta: Rate limit for threshold motion (0 < eta < 1)
            h: Hysteresis fraction for margin width
            C: Cooldown samples after a flip
            slow_speed: Speed multiplier for slow P² tracker
            fast_speed: Speed multiplier for fast P² tracker
            epsilon_s: Small constant to prevent degeneracy
        """
        super().__init__(K=K, name="TWINE")
        self.tau = tau
        self.eta = eta
        self.h = h
        self.C = C
        self.slow_speed = slow_speed
        self.fast_speed = fast_speed
        self.epsilon_s = epsilon_s

        # Per-feature dual trackers
        self.trackers = None

        # Displayed thresholds T_{i,j}
        self.thresholds = None

        # Hysteresis state
        self.bit_state = None     # Current bit states
        self.cooldown_counters = None  # Cooldown counters per bit

    def _cold_start_init(self, x: np.ndarray):
        """Initialize encoder from a single sample (for streaming)."""
        self.n_features = len(x)

        # Initialize dual-speed P² trackers for each feature
        self.trackers = [
            DualSpeedP2(K=self.K, slow_speed=self.slow_speed, fast_speed=self.fast_speed)
            for _ in range(self.n_features)
        ]

        # Initialize thresholds to uniform spacing in [0, 1] range
        # This is a reasonable default until trackers warm up
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]

        # Initialize hysteresis state
        n_bits = self.n_features * self.K
        self.bit_state = np.zeros(n_bits, dtype=np.uint8)
        self.cooldown_counters = np.zeros(n_bits, dtype=np.int32)  # Fixed: was self.cooldown

        self.fitted = True

    def fit(self, X: np.ndarray) -> 'TWINE':
        """
        Fit encoder by initializing trackers and thresholds.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Initialize dual-speed P² trackers for each feature
        self.trackers = [
            DualSpeedP2(K=self.K, slow_speed=self.slow_speed, fast_speed=self.fast_speed)
            for _ in range(self.n_features)
        ]

        # Initialize thresholds
        self.thresholds = np.zeros((self.n_features, self.K))

        # Feed initial data to trackers
        for i in range(X.shape[0]):
            for j in range(self.n_features):
                self.trackers[j].update(X[i, j])

        # Set initial thresholds from slow tracker
        for j in range(self.n_features):
            if self.trackers[j].is_initialized():
                self.thresholds[j] = self.trackers[j].get_slow_thresholds()

        # Initialize hysteresis state
        self.bit_state = np.zeros(self.n_features * self.K, dtype=np.uint8)
        self.cooldown_counters = np.zeros(self.n_features * self.K, dtype=np.int32)

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

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample with TWINE algorithm.

        This implements Algorithm 1 from the paper:
        1. Pre-update encoding: emit bits before updating trackers
        2. Drift gate: select slow or fast tracker based on disagreement
        3. Rate-limited threshold motion
        4. Per-bit hysteresis with cooldown

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Encoded sample of shape (n_features * K,)
        """
        # Lazy initialization for streaming: initialize on first sample
        if not self.fitted:
            self._cold_start_init(x)

        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        # Process each feature
        for i in range(self.n_features):
            if not self.trackers[i].is_initialized():
                # Still initializing, use zeros
                continue

            # 1. Drift gate: compute normalized disagreement
            disagreement = self.trackers[i].compute_disagreement()

            # Select reference quantiles based on drift gate
            if disagreement > self.tau:
                # Use fast tracker (drift detected)
                ref_quantiles = self.trackers[i].get_fast_thresholds()
            else:
                # Use slow tracker (stable regime)
                ref_quantiles = self.trackers[i].get_slow_thresholds()

            # Get full quantile markers for spacing calculation
            if disagreement > self.tau:
                full_q = self.trackers[i].get_fast_quantiles()
            else:
                full_q = self.trackers[i].get_slow_quantiles()

            # 2. Rate-limited threshold motion
            for k in range(self.K):
                # Compute local spacing S_{i,k}
                # S = (q_{k+1} - q_{k-1}) / 2, using full markers (K+2 total)
                q_idx = k + 1  # Index in full markers (0, 1, ..., K, K+1)
                S = 0.5 * (full_q[q_idx + 1] - full_q[q_idx - 1])
                S = max(S, self.epsilon_s)

                # Update displayed threshold T_{i,k} toward reference
                error = ref_quantiles[k] - self.thresholds[i, k]
                step = np.clip(error, -self.eta * S, self.eta * S)
                self.thresholds[i, k] += step

            # Ensure monotonicity (isotonic projection with epsilon margin)
            for k in range(1, self.K):
                if self.thresholds[i, k] <= self.thresholds[i, k - 1]:
                    self.thresholds[i, k] = self.thresholds[i, k - 1] + self.epsilon_s

            # 3. Per-bit hysteresis encoding
            for k in range(self.K):
                bit_idx = i * self.K + k

                # Compute margin δ = h * S
                q_idx = k + 1
                S = 0.5 * (full_q[q_idx + 1] - full_q[q_idx - 1])
                S = max(S, self.epsilon_s)
                margin = self.h * S

                # Get previous bit state
                prev_state = self.bit_state[bit_idx]

                # Check cooldown
                if self.cooldown_counters[bit_idx] > 0:
                    # Still in cooldown, maintain previous state
                    bits[bit_idx] = prev_state
                    self.cooldown_counters[bit_idx] -= 1
                else:
                    # Schmitt trigger with hysteresis
                    threshold = self.thresholds[i, k]

                    if prev_state == 0:
                        # Currently 0, flip to 1 if x >= threshold + margin
                        if x[i] >= threshold + margin:
                            bits[bit_idx] = 1
                            self.cooldown_counters[bit_idx] = self.C
                        else:
                            bits[bit_idx] = 0
                    else:
                        # Currently 1, flip to 0 if x <= threshold - margin
                        if x[i] <= threshold - margin:
                            bits[bit_idx] = 0
                            self.cooldown_counters[bit_idx] = self.C
                        else:
                            bits[bit_idx] = 1

                # Update bit state
                self.bit_state[bit_idx] = bits[bit_idx]

        # 4. Update trackers (post-encoding, to avoid self-conditioning)
        for i in range(self.n_features):
            self.trackers[i].update(x[i])

        return bits

    def get_config(self) -> dict:
        """Get encoder configuration."""
        return {
            'K': self.K,
            'tau': self.tau,
            'eta': self.eta,
            'h': self.h,
            'C': self.C,
            'slow_speed': self.slow_speed,
            'fast_speed': self.fast_speed,
        }

