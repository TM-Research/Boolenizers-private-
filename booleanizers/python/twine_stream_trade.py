"""
TWINE-StreamTrade: Streaming thermometer encoder using online-trading techniques.

Techniques from online trading / streaming finance:
1. EWMA volatility (VolSched-style): short vs long-term "volatility" to adapt step size
2. Regime-aware updates: run-length since last drift to boost responsiveness after stability
3. Kalman-style quantile blending: continuous blend of slow/fast estimates (VWAP-like)
4. Volatility-adjusted eta: high local vol -> larger steps; low vol -> smaller steps for stability
"""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import DualSpeedP2


class TWINEStreamTrade(ThermometerEncoder):
    """
    Stream-based thermometer encoder with online-trading-inspired adaptations.

    - Dual-speed P² quantile tracking (slow/fast) for drift detection
    - EWMA of squared input changes as "volatility" proxy
    - Volatility-adjusted learning rate (VolSched-style)
    - Continuous slow/fast quantile blending (Kalman/VWAP-style)
    - Run-length since drift for regime-aware step boost
    - Adaptive hysteresis and Schmitt trigger for stability
    """

    def __init__(
        self,
        K: int = 8,
        tau: float = 0.6,
        eta: float = 0.08,
        h: float = 0.12,
        slow_speed: float = 0.5,
        fast_speed: float = 2.0,
        # Volatility (EWMA) params
        vol_alpha_short: float = 0.1,   # Short-term vol half-life ~7 samples
        vol_alpha_long: float = 0.01,   # Long-term vol half-life ~70 samples
        eta_vol_min: float = 0.5,        # Min multiplier from vol
        eta_vol_max: float = 2.0,        # Max multiplier from vol
        # Regime / run-length
        run_boost_after: int = 500,     # After this many samples without drift, allow boost on next drift
        run_boost_factor: float = 1.3,  # One-time eta boost when re-entering drift after long stable run
        # Blending: w_fast = blend_slope * disagreement (capped at 1)
        blend_slope: float = 1.5,       # How quickly we shift from slow to fast as disagreement grows
        epsilon_s: float = 1e-6,
    ):
        super().__init__(K=K, name="TWINE-StreamTrade")
        self.tau = tau
        self.eta = eta
        self.h = h
        self.slow_speed = slow_speed
        self.fast_speed = fast_speed
        self.vol_alpha_short = vol_alpha_short
        self.vol_alpha_long = vol_alpha_long
        self.eta_vol_min = eta_vol_min
        self.eta_vol_max = eta_vol_max
        self.run_boost_after = run_boost_after
        self.run_boost_factor = run_boost_factor
        self.blend_slope = blend_slope
        self.epsilon_s = epsilon_s

        self.trackers = None
        self.thresholds = None
        self.bit_state = None
        self.n_features = None

        # Per-feature volatility (EWMA of squared delta)
        self.x_prev = None
        self.vol_short = None
        self.vol_long = None
        # Run-length since last drift event (per feature)
        self.samples_since_drift = None
        self.in_drift = None  # True when disagreement > tau last time

    def _cold_start_init(self, x: np.ndarray):
        self.n_features = len(x)
        self.trackers = [
            DualSpeedP2(K=self.K, slow_speed=self.slow_speed, fast_speed=self.fast_speed)
            for _ in range(self.n_features)
        ]
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]
        self.bit_state = np.zeros(self.n_features * self.K, dtype=np.uint8)
        self.x_prev = np.zeros(self.n_features, dtype=np.float64)
        self.vol_short = np.ones(self.n_features, dtype=np.float64) * 1e-6
        self.vol_long = np.ones(self.n_features, dtype=np.float64) * 1e-6
        self.samples_since_drift = np.zeros(self.n_features, dtype=np.int32)
        self.in_drift = np.zeros(self.n_features, dtype=bool)
        self.fitted = True

    def fit(self, X: np.ndarray) -> "TWINEStreamTrade":
        self.n_features = X.shape[1]
        self.trackers = [
            DualSpeedP2(K=self.K, slow_speed=self.slow_speed, fast_speed=self.fast_speed)
            for _ in range(self.n_features)
        ]
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(X.shape[0]):
            for j in range(self.n_features):
                self.trackers[j].update(X[i, j])
        for j in range(self.n_features):
            if self.trackers[j].is_initialized():
                self.thresholds[j] = self.trackers[j].get_slow_thresholds()
        self.bit_state = np.zeros(self.n_features * self.K, dtype=np.uint8)
        self.x_prev = np.zeros(self.n_features, dtype=np.float64)
        self.vol_short = np.ones(self.n_features, dtype=np.float64) * 1e-6
        self.vol_long = np.ones(self.n_features, dtype=np.float64) * 1e-6
        self.samples_since_drift = np.zeros(self.n_features, dtype=np.int32)
        self.in_drift = np.zeros(self.n_features, dtype=bool)
        if X.shape[0] > 0:
            self.x_prev = X[-1].copy()
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)
        for i in range(n_samples):
            encoded[i] = self._encode_single(X[i])
        return encoded

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            self._cold_start_init(x)

        bits = np.zeros(self.n_features * self.K, dtype=np.uint8)

        for i in range(self.n_features):
            if not self.trackers[i].is_initialized():
                continue

            # 1) Update volatility (EWMA of squared change) — trading-style
            delta_sq = (x[i] - self.x_prev[i]) ** 2
            self.vol_short[i] = (1 - self.vol_alpha_short) * self.vol_short[i] + self.vol_alpha_short * delta_sq
            self.vol_long[i] = (1 - self.vol_alpha_long) * self.vol_long[i] + self.vol_alpha_long * delta_sq
            vol_ratio = self.vol_short[i] / (self.vol_long[i] + self.epsilon_s)
            # Volatility-adjusted eta multiplier (VolSched-style)
            eta_vol_mult = np.clip(np.sqrt(vol_ratio), self.eta_vol_min, self.eta_vol_max)

            disagreement = self.trackers[i].compute_disagreement()

            # 2) Regime: run-length since drift
            if disagreement > self.tau:
                if not self.in_drift[i] and self.samples_since_drift[i] >= self.run_boost_after:
                    run_boost = self.run_boost_factor  # One-time boost after long stable run
                else:
                    run_boost = 1.0
                self.in_drift[i] = True
                self.samples_since_drift[i] = 0
            else:
                run_boost = 1.0
                self.in_drift[i] = False
                self.samples_since_drift[i] += 1

            # 3) Kalman-style blend: ref = (1-w)*slow + w*fast (VWAP-like)
            w_fast = np.tanh(self.blend_slope * disagreement)
            slow_q = self.trackers[i].get_slow_thresholds()
            fast_q = self.trackers[i].get_fast_thresholds()
            ref_quantiles = (1 - w_fast) * slow_q + w_fast * fast_q
            full_q_slow = self.trackers[i].get_slow_quantiles()
            full_q_fast = self.trackers[i].get_fast_quantiles()
            full_q = (1 - w_fast) * full_q_slow + w_fast * full_q_fast

            # 4) Effective eta: base * vol_mult * run_boost (capped)
            effective_eta = self.eta * eta_vol_mult * run_boost
            effective_eta = min(effective_eta, self.eta * 2.5)

            for k in range(self.K):
                q_idx = k + 1
                S = 0.5 * (full_q[q_idx + 1] - full_q[q_idx - 1])
                S = max(S, self.epsilon_s)
                error = ref_quantiles[k] - self.thresholds[i, k]
                step = np.clip(error, -effective_eta * S, effective_eta * S)
                self.thresholds[i, k] += step

            for k in range(1, self.K):
                if self.thresholds[i, k] <= self.thresholds[i, k - 1]:
                    self.thresholds[i, k] = self.thresholds[i, k - 1] + self.epsilon_s

            # 5) Encode with hysteresis (Schmitt trigger)
            for k in range(self.K):
                bit_idx = i * self.K + k
                q_idx = k + 1
                S = 0.5 * (full_q[q_idx + 1] - full_q[q_idx - 1])
                S = max(S, self.epsilon_s)
                margin = self.h * S
                threshold = self.thresholds[i, k]
                prev_state = self.bit_state[bit_idx]

                if prev_state == 0:
                    bits[bit_idx] = 1 if x[i] >= threshold + margin else 0
                else:
                    bits[bit_idx] = 0 if x[i] <= threshold - margin else 1
                self.bit_state[bit_idx] = bits[bit_idx]

        for i in range(self.n_features):
            self.trackers[i].update(x[i])
        self.x_prev = x.copy()

        return bits

    def get_config(self) -> dict:
        return {
            "K": self.K,
            "tau": self.tau,
            "eta": self.eta,
            "h": self.h,
            "slow_speed": self.slow_speed,
            "fast_speed": self.fast_speed,
            "vol_alpha_short": self.vol_alpha_short,
            "vol_alpha_long": self.vol_alpha_long,
            "eta_vol_min": self.eta_vol_min,
            "eta_vol_max": self.eta_vol_max,
            "run_boost_after": self.run_boost_after,
            "run_boost_factor": self.run_boost_factor,
            "blend_slope": self.blend_slope,
        }
