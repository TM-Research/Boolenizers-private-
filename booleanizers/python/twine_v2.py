"""
TWINE v2: Enhanced Twin-Tracker Online Quantile Encoding with Adaptive Mechanisms

Improvements over TWINE v1:
1. Removed cooldown (C=0 always) - data shows it hurts performance
2. Adaptive hysteresis based on per-bit flip frequency
3. Bit activation boost to prevent frozen bits
4. Dynamic rate limiting based on drift magnitude
"""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import DualSpeedP2


class TWINEv2(ThermometerEncoder):
    """
    TWINE v2 encoder with adaptive stability mechanisms.

    Key improvements:
    - Adaptive hysteresis: adjusts per-bit based on observed flip frequency
    - Bit activation boost: prevents bits from staying frozen too long
    - Dynamic rate limiting: faster updates during strong drift
    - No cooldown: data shows C>0 significantly hurts accuracy
    """

    def __init__(
        self,
        K: int = 8,
        tau: float = 0.6,           # From tuning: best was 0.6
        eta: float = 0.08,          # From tuning: best was 0.08
        h: float = 0.12,            # From tuning: best was 0.12
        h_range: tuple = (0.06, 0.20),  # Adaptive hysteresis range
        slow_speed: float = 0.5,
        fast_speed: float = 2.0,
        epsilon_s: float = 1e-6,
        frozen_threshold: int = 500,  # Samples before bit considered frozen
        boost_factor: float = 0.6,    # Margin reduction for frozen bits
    ):
        """
        Initialize TWINE v2 encoder.

        Args:
            K: Number of thresholds per feature
            tau: Drift gate threshold for switching to fast tracker
            eta: Base rate limit for threshold motion
            h: Base hysteresis fraction
            h_range: (min, max) range for adaptive hysteresis
            slow_speed: Speed multiplier for slow P² tracker
            fast_speed: Speed multiplier for fast P² tracker
            epsilon_s: Small constant to prevent degeneracy
            frozen_threshold: Samples without flip before applying boost
            boost_factor: Margin multiplier for frozen bits (< 1.0)
        """
        super().__init__(K=K, name="TWINE-v2")
        self.tau = tau
        self.eta = eta
        self.h = h
        self.h_min, self.h_max = h_range
        self.slow_speed = slow_speed
        self.fast_speed = fast_speed
        self.epsilon_s = epsilon_s
        self.frozen_threshold = frozen_threshold
        self.boost_factor = boost_factor

        # Per-feature dual trackers
        self.trackers = None

        # Metrics storage
        self.metrics = {}

        # Displayed thresholds T_{i,j}
        self.thresholds = None

        # Hysteresis and adaptive state (initialized during fit or cold-start)
        self.bit_state = None
        self.recent_flip_ema = None
        self.samples_since_flip = None
        self.adaptive_h = None

    def _cold_start_init(self, x: np.ndarray):
        """Initialize encoder from a single sample (for streaming)."""
        self.n_features = len(x)

        # Initialize dual-speed P² trackers for each feature
        self.trackers = [
            DualSpeedP2(K=self.K, slow_speed=self.slow_speed, fast_speed=self.fast_speed)
            for _ in range(self.n_features)
        ]

        # Initialize thresholds to uniform spacing in [0, 1] range
        self.thresholds = np.zeros((self.n_features, self.K))
        for i in range(self.n_features):
            self.thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]

        # Initialize hysteresis state
        n_bits = self.n_features * self.K
        self.bit_state = np.zeros(n_bits, dtype=np.uint8)

        # Adaptive mechanisms
        self.recent_flip_ema = np.zeros(n_bits, dtype=np.float32)
        self.samples_since_flip = np.zeros(n_bits, dtype=np.int32)
        self.adaptive_h = np.ones(n_bits, dtype=np.float32) * self.h

        # Comprehensive metrics tracking
        self._init_metrics()

        self.fitted = True

    def fit(self, X: np.ndarray) -> 'TWINEv2':
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
        n_bits = self.n_features * self.K
        self.bit_state = np.zeros(n_bits, dtype=np.uint8)

        # NEW: Adaptive mechanisms
        self.recent_flip_ema = np.zeros(n_bits, dtype=np.float32)
        self.samples_since_flip = np.zeros(n_bits, dtype=np.int32)
        self.adaptive_h = np.ones(n_bits, dtype=np.float32) * self.h

        # Comprehensive metrics tracking
        self._init_metrics()

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
        Encode a single sample with TWINE v2 algorithm.

        Enhancements:
        - Adaptive hysteresis per bit
        - Bit activation boost for frozen bits
        - Dynamic rate limiting based on drift magnitude

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
                continue

            # 1. Drift gate: compute normalized disagreement
            disagreement = self.trackers[i].compute_disagreement()

            # Select reference quantiles based on drift gate
            if disagreement > self.tau:
                ref_quantiles = self.trackers[i].get_fast_thresholds()
                full_q = self.trackers[i].get_fast_quantiles()
            else:
                ref_quantiles = self.trackers[i].get_slow_thresholds()
                full_q = self.trackers[i].get_slow_quantiles()

            # 2. DYNAMIC RATE LIMITING: adjust eta based on drift magnitude
            if disagreement > self.tau * 1.5:
                # Strong drift: move thresholds faster
                effective_eta = self.eta * 2.0
                eta_factor = 2.0
            elif disagreement < self.tau * 0.5:
                # Very stable: move slowly
                effective_eta = self.eta * 0.5
                eta_factor = 0.5
            else:
                effective_eta = self.eta
                eta_factor = 1.0

            # 3. Update thresholds with dynamic rate limit
            for k in range(self.K):
                q_idx = k + 1
                S = 0.5 * (full_q[q_idx + 1] - full_q[q_idx - 1])
                S = max(S, self.epsilon_s)

                error = ref_quantiles[k] - self.thresholds[i, k]
                step = np.clip(error, -effective_eta * S, effective_eta * S)
                self.thresholds[i, k] += step

            # Ensure monotonicity
            for k in range(1, self.K):
                if self.thresholds[i, k] <= self.thresholds[i, k - 1]:
                    self.thresholds[i, k] = self.thresholds[i, k - 1] + self.epsilon_s

            # 4. ADAPTIVE HYSTERESIS encoding
            for k in range(self.K):
                bit_idx = i * self.K + k

                # Compute local spacing
                q_idx = k + 1
                S = 0.5 * (full_q[q_idx + 1] - full_q[q_idx - 1])
                S = max(S, self.epsilon_s)

                # Get previous state
                prev_state = self.bit_state[bit_idx]

                # Update samples since last flip
                self.samples_since_flip[bit_idx] += 1

                # ADAPTIVE HYSTERESIS: adjust based on flip frequency
                # EMA decay: α=0.05 means ~20 samples half-life
                flip_ema = self.recent_flip_ema[bit_idx]

                if flip_ema > 0.3:
                    # Noisy bit: increase hysteresis for stability
                    self.adaptive_h[bit_idx] = min(self.h_max, self.h * 1.5)
                elif flip_ema < 0.05:
                    # Stable bit: decrease hysteresis for responsiveness
                    self.adaptive_h[bit_idx] = max(self.h_min, self.h * 0.7)
                else:
                    # Normal: use base hysteresis
                    self.adaptive_h[bit_idx] = self.h

                # BIT ACTIVATION BOOST: reduce margin for frozen bits
                margin = self.adaptive_h[bit_idx] * S

                if self.samples_since_flip[bit_idx] > self.frozen_threshold:
                    # Bit frozen too long: make it easier to flip
                    margin = margin * self.boost_factor

                # Schmitt trigger with adaptive hysteresis
                threshold = self.thresholds[i, k]
                flipped = False

                if prev_state == 0:
                    if x[i] >= threshold + margin:
                        bits[bit_idx] = 1
                        flipped = True
                    else:
                        bits[bit_idx] = 0
                else:
                    if x[i] <= threshold - margin:
                        bits[bit_idx] = 0
                        flipped = True
                    else:
                        bits[bit_idx] = 1

                # Update flip tracking
                if flipped:
                    self.samples_since_flip[bit_idx] = 0
                    # Update EMA: immediate flip increases it
                    self.recent_flip_ema[bit_idx] = 0.95 * flip_ema + 0.05 * 1.0
                else:
                    # No flip: gradually decay EMA
                    self.recent_flip_ema[bit_idx] = 0.95 * flip_ema + 0.05 * 0.0

                # Update state
                self.bit_state[bit_idx] = bits[bit_idx]

                # Update metrics for this bit
                self._update_metrics(i, bit_idx, flipped, disagreement, eta_factor)

        # 5. Update trackers (post-encoding)
        for i in range(self.n_features):
            self.trackers[i].update(x[i])

        # 6. Update global metrics and snapshots
        self.metrics['total_samples_processed'] += 1
        self._snapshot_metrics()

        return bits

    def get_config(self) -> dict:
        """Get encoder configuration."""
        return {
            'K': self.K,
            'tau': self.tau,
            'eta': self.eta,
            'h': self.h,
            'h_range': (self.h_min, self.h_max),
            'slow_speed': self.slow_speed,
            'fast_speed': self.fast_speed,
            'frozen_threshold': self.frozen_threshold,
            'boost_factor': self.boost_factor,
            'version': 'v2',
        }

    def _init_metrics(self):
        """Initialize comprehensive metrics tracking."""
        n_bits = self.n_features * self.K

        self.metrics = {
            # Per-bit counters
            'bit_flip_count': np.zeros(n_bits, dtype=np.int32),
            'bit_total_samples': np.zeros(n_bits, dtype=np.int32),
            'bit_max_freeze_duration': np.zeros(n_bits, dtype=np.int32),

            # Per-feature trackers
            'feature_disagreement_history': [[] for _ in range(self.n_features)],
            'feature_drift_events': np.zeros(self.n_features, dtype=np.int32),
            'feature_stable_periods': np.zeros(self.n_features, dtype=np.int32),

            # Threshold evolution (sampled periodically)
            'threshold_snapshots': [],  # List of (sample_idx, thresholds_copy)
            'snapshot_interval': 1000,   # Sample every N samples

            # Global counters
            'total_samples_processed': 0,
            'total_flips': 0,
            'total_freeze_boost_activations': 0,
            'total_drift_events': 0,
            'total_stable_periods': 0,

            # Adaptive mechanism stats
            'h_history': [],  # (sample_idx, h_min, h_mean, h_max)
            'h_sample_interval': 1000,

            # Rate limiting stats
            'eta_adjustments': {  # Count of each adjustment type
                'fast': 0,    # 2x eta
                'slow': 0,    # 0.5x eta
                'normal': 0,  # 1x eta
            },

            # Per-bit detailed history (last N samples for diagnosis)
            'recent_bit_states': [],  # Rolling window
            'recent_window_size': 100,
        }

    def _update_metrics(self, feature_idx: int, bit_idx: int, flipped: bool,
                       disagreement: float, eta_factor: float):
        """Update metrics after encoding a bit."""
        m = self.metrics

        # Per-bit updates
        m['bit_total_samples'][bit_idx] += 1
        if flipped:
            m['bit_flip_count'][bit_idx] += 1
            m['total_flips'] += 1

            # Update max freeze duration
            freeze_duration = self.samples_since_flip[bit_idx]
            if freeze_duration > m['bit_max_freeze_duration'][bit_idx]:
                m['bit_max_freeze_duration'][bit_idx] = freeze_duration

        # Track freeze boost activations
        if self.samples_since_flip[bit_idx] > self.frozen_threshold:
            m['total_freeze_boost_activations'] += 1

        # Per-feature updates (only once per feature, not per bit)
        if bit_idx % self.K == 0:  # First bit of this feature
            m['feature_disagreement_history'][feature_idx].append(disagreement)

            # Track drift events
            if disagreement > self.tau:
                m['feature_drift_events'][feature_idx] += 1
                m['total_drift_events'] += 1
            else:
                m['feature_stable_periods'][feature_idx] += 1
                m['total_stable_periods'] += 1

            # Rate limiting stats
            if eta_factor > 1.5:
                m['eta_adjustments']['fast'] += 1
            elif eta_factor < 0.75:
                m['eta_adjustments']['slow'] += 1
            else:
                m['eta_adjustments']['normal'] += 1

    def _snapshot_metrics(self):
        """Periodically snapshot state for time-series analysis."""
        m = self.metrics
        idx = m['total_samples_processed']

        # Threshold evolution
        if idx % m['snapshot_interval'] == 0:
            m['threshold_snapshots'].append((idx, self.thresholds.copy()))

        # Hysteresis evolution
        if idx % m['h_sample_interval'] == 0:
            m['h_history'].append((
                idx,
                float(np.min(self.adaptive_h)),
                float(np.mean(self.adaptive_h)),
                float(np.max(self.adaptive_h))
            ))

    def get_metrics_summary(self) -> dict:
        """Get comprehensive metrics summary for analysis."""
        if not self.fitted or self.metrics['total_samples_processed'] == 0:
            return {}

        m = self.metrics
        n_bits = self.n_features * self.K

        # Compute flip rates
        flip_rates = np.where(
            m['bit_total_samples'] > 0,
            m['bit_flip_count'] / m['bit_total_samples'],
            0.0
        )

        # Identify problematic bits
        frozen_bits = np.where(m['bit_flip_count'] == 0)[0]
        overactive_bits = np.where(flip_rates > 0.5)[0]

        # Per-feature disagreement stats
        avg_disagreement = []
        std_disagreement = []
        for hist in m['feature_disagreement_history']:
            if len(hist) > 0:
                avg_disagreement.append(np.mean(hist))
                std_disagreement.append(np.std(hist))
            else:
                avg_disagreement.append(0.0)
                std_disagreement.append(0.0)

        summary = {
            # Overall statistics
            'total_samples': m['total_samples_processed'],
            'total_flips': m['total_flips'],
            'global_flip_rate': m['total_flips'] / max(m['total_samples_processed'] * n_bits, 1),

            # Bit-level statistics
            'flip_rate_stats': {
                'min': float(np.min(flip_rates)),
                'mean': float(np.mean(flip_rates)),
                'max': float(np.max(flip_rates)),
                'std': float(np.std(flip_rates)),
                'median': float(np.median(flip_rates)),
                'q25': float(np.percentile(flip_rates, 25)),
                'q75': float(np.percentile(flip_rates, 75)),
            },

            'freeze_duration_stats': {
                'min': int(np.min(m['bit_max_freeze_duration'])),
                'mean': float(np.mean(m['bit_max_freeze_duration'])),
                'max': int(np.max(m['bit_max_freeze_duration'])),
                'std': float(np.std(m['bit_max_freeze_duration'])),
            },

            # Problematic bits
            'frozen_bits': {
                'count': len(frozen_bits),
                'percentage': 100.0 * len(frozen_bits) / n_bits,
                'indices': frozen_bits.tolist()[:20],  # First 20 for brevity
            },

            'overactive_bits': {
                'count': len(overactive_bits),
                'percentage': 100.0 * len(overactive_bits) / n_bits,
                'indices': overactive_bits.tolist()[:20],
            },

            # Feature-level statistics
            'disagreement_stats': {
                'per_feature_mean': [float(x) for x in avg_disagreement],
                'per_feature_std': [float(x) for x in std_disagreement],
                'global_mean': float(np.mean(avg_disagreement)),
                'global_std': float(np.std(avg_disagreement)),
            },

            'drift_events': {
                'total': m['total_drift_events'],
                'per_feature': m['feature_drift_events'].tolist(),
                'percentage': 100.0 * m['total_drift_events'] / max(m['total_samples_processed'], 1),
            },

            'stable_periods': {
                'total': m['total_stable_periods'],
                'percentage': 100.0 * m['total_stable_periods'] / max(m['total_samples_processed'], 1),
            },

            # Adaptive mechanism statistics
            'adaptive_hysteresis': {
                'current_min': float(np.min(self.adaptive_h)),
                'current_mean': float(np.mean(self.adaptive_h)),
                'current_max': float(np.max(self.adaptive_h)),
                'current_std': float(np.std(self.adaptive_h)),
                'history': m['h_history'],  # Time series
            },

            'freeze_boost': {
                'total_activations': m['total_freeze_boost_activations'],
                'current_frozen_count': int(np.sum(self.samples_since_flip > self.frozen_threshold)),
                'percentage': 100.0 * m['total_freeze_boost_activations'] / max(m['total_samples_processed'] * n_bits, 1),
            },

            'rate_limiting': {
                'adjustments': m['eta_adjustments'].copy(),
                'fast_percentage': 100.0 * m['eta_adjustments']['fast'] / max(m['total_samples_processed'], 1),
                'slow_percentage': 100.0 * m['eta_adjustments']['slow'] / max(m['total_samples_processed'], 1),
                'normal_percentage': 100.0 * m['eta_adjustments']['normal'] / max(m['total_samples_processed'], 1),
            },

            # Threshold evolution (for plotting)
            'threshold_evolution': m['threshold_snapshots'],

            # Configuration
            'config': self.get_config(),
        }

        return summary

    def get_diagnostic_report(self) -> str:
        """Generate human-readable diagnostic report."""
        summary = self.get_metrics_summary()
        if not summary:
            return "No metrics available (encoder not fitted or no samples processed)"

        report_lines = [
            "="*70,
            "TWINE v2 DIAGNOSTIC REPORT",
            "="*70,
            "",
            f"Total Samples Processed: {summary['total_samples']},",
            f"Total Flips: {summary['total_flips']} (global rate: {summary['global_flip_rate']:.4f})",
            "",
            "--- BIT-LEVEL STATISTICS ---",
            f"Flip Rate: min={summary['flip_rate_stats']['min']:.4f}, "
            f"mean={summary['flip_rate_stats']['mean']:.4f}, "
            f"max={summary['flip_rate_stats']['max']:.4f}, "
            f"std={summary['flip_rate_stats']['std']:.4f}",
            f"Flip Rate Distribution: Q25={summary['flip_rate_stats']['q25']:.4f}, "
            f"Median={summary['flip_rate_stats']['median']:.4f}, "
            f"Q75={summary['flip_rate_stats']['q75']:.4f}",
            "",
            f"Frozen Bits: {summary['frozen_bits']['count']} "
            f"({summary['frozen_bits']['percentage']:.2f}%)",
            f"Overactive Bits (>50% flip): {summary['overactive_bits']['count']} "
            f"({summary['overactive_bits']['percentage']:.2f}%)",
            "",
            f"Max Freeze Duration: min={summary['freeze_duration_stats']['min']}, "
            f"mean={summary['freeze_duration_stats']['mean']:.1f}, "
            f"max={summary['freeze_duration_stats']['max']}",
            "",
            "--- FEATURE-LEVEL STATISTICS ---",
            f"Disagreement: mean={summary['disagreement_stats']['global_mean']:.4f}, "
            f"std={summary['disagreement_stats']['global_std']:.4f}",
            f"Drift Events: {summary['drift_events']['total']} "
            f"({summary['drift_events']['percentage']:.2f}%)",
            f"Stable Periods: {summary['stable_periods']['total']} "
            f"({summary['stable_periods']['percentage']:.2f}%)",
            "",
            "--- ADAPTIVE MECHANISMS ---",
            f"Adaptive Hysteresis: min={summary['adaptive_hysteresis']['current_min']:.3f}, "
            f"mean={summary['adaptive_hysteresis']['current_mean']:.3f}, "
            f"max={summary['adaptive_hysteresis']['current_max']:.3f}",
            f"Freeze Boost Activations: {summary['freeze_boost']['total_activations']} "
            f"({summary['freeze_boost']['percentage']:.4f}%)",
            f"Currently Frozen (>{self.frozen_threshold} samples): "
            f"{summary['freeze_boost']['current_frozen_count']} bits",
            "",
            f"Rate Limiting: Fast={summary['rate_limiting']['fast_percentage']:.2f}%, "
            f"Normal={summary['rate_limiting']['normal_percentage']:.2f}%, "
            f"Slow={summary['rate_limiting']['slow_percentage']:.2f}%",
            "",
            "="*70,
        ]

        return "\n".join(report_lines)

    def get_adaptive_stats(self) -> dict:
        """Get statistics about adaptive mechanisms."""
        if not self.fitted:
            return {}

        return {
            'adaptive_h': {
                'min': float(np.min(self.adaptive_h)),
                'mean': float(np.mean(self.adaptive_h)),
                'max': float(np.max(self.adaptive_h)),
            },
            'frozen_bits': int(np.sum(self.samples_since_flip > self.frozen_threshold)),
            'active_bits': int(np.sum(self.samples_since_flip <= self.frozen_threshold)),
            'flip_ema': {
                'min': float(np.min(self.recent_flip_ema)),
                'mean': float(np.mean(self.recent_flip_ema)),
                'max': float(np.max(self.recent_flip_ema)),
            },
        }

