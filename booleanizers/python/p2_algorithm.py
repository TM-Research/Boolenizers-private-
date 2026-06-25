"""
P² Algorithm for dynamic calculation of quantiles without storing observations.

Based on: Jain, R., & Chlamtac, I. (1985).
"The P² algorithm for dynamic calculation of quantiles and histograms
without storing observations."
Communications of the ACM, 28(10), 1076-1085.
"""

import numpy as np


class P2Quantile:
    """
    P² algorithm for online quantile estimation.

    Maintains K+2 markers with positions and heights to estimate K quantiles.
    Uses parabolic interpolation for smooth marker updates.
    """

    def __init__(self, K: int = 8, speed: float = 1.0):
        """
        Initialize P² quantile tracker.

        Args:
            K: Number of quantiles to track (will use K+2 markers)
            speed: Update speed multiplier (1.0 = standard, <1 = slower, >1 = faster)
        """
        self.K = K
        self.speed = speed
        self.n_markers = K + 2

        # Desired quantile probabilities: 0, 1/(K+1), 2/(K+1), ..., K/(K+1), 1
        self.probabilities = np.linspace(0, 1, self.n_markers)

        # Marker heights (quantile values)
        self.q = np.zeros(self.n_markers)

        # Marker positions (counts)
        self.n = np.arange(1, self.n_markers + 1, dtype=float)

        # Desired positions
        self.n_desired = np.arange(1, self.n_markers + 1, dtype=float)

        # Number of observations seen
        self.count = 0

        # Initialization phase (need K+2 observations to start)
        self.initialized = False
        self.init_buffer = []

    def update(self, x: float):
        """
        Update quantile estimates with a new observation.

        Args:
            x: New observation
        """
        # Initialization phase: collect first K+2 observations
        if not self.initialized:
            self.init_buffer.append(x)
            if len(self.init_buffer) == self.n_markers:
                # Sort and initialize markers
                self.init_buffer.sort()
                self.q = np.array(self.init_buffer, dtype=float)
                self.n = np.arange(1, self.n_markers + 1, dtype=float)
                self.initialized = True
                self.count = self.n_markers
            return

        self.count += 1

        # Find cell k such that q[k] <= x < q[k+1]
        if x < self.q[0]:
            self.q[0] = x
            k = 0
        elif x >= self.q[-1]:
            self.q[-1] = x
            k = self.n_markers - 2
        else:
            k = np.searchsorted(self.q[1:], x, side='right')

        # Increment positions for markers above k
        for i in range(k + 1, self.n_markers):
            self.n[i] += 1

        # Update desired positions
        for i in range(self.n_markers):
            self.n_desired[i] = 1 + (self.count - 1) * self.probabilities[i]

        # Adjust marker heights
        for i in range(1, self.n_markers - 1):
            # Calculate desired change in position
            d = self.n_desired[i] - self.n[i]

            # Apply speed factor
            d = d * self.speed

            # Adjust if necessary
            if (d >= 1 and self.n[i + 1] - self.n[i] > 1) or \
               (d <= -1 and self.n[i - 1] - self.n[i] < -1):

                d_sign = 1 if d >= 0 else -1

                # Try parabolic formula
                q_new = self._parabolic(i, d_sign)

                # Check if parabolic is valid
                if self.q[i - 1] < q_new < self.q[i + 1]:
                    self.q[i] = q_new
                else:
                    # Use linear formula
                    self.q[i] = self._linear(i, d_sign)

                self.n[i] += d_sign

    def _parabolic(self, i: int, d: float) -> float:
        """
        Parabolic formula for marker height adjustment.

        Args:
            i: Marker index
            d: Direction (+1 or -1)

        Returns:
            New marker height
        """
        q_i = self.q[i]
        q_i1 = self.q[i + 1]
        q_i_1 = self.q[i - 1]

        n_i = self.n[i]
        n_i1 = self.n[i + 1]
        n_i_1 = self.n[i - 1]

        # Parabolic interpolation formula
        q_new = q_i + d / (n_i1 - n_i_1) * (
            (n_i - n_i_1 + d) * (q_i1 - q_i) / (n_i1 - n_i) +
            (n_i1 - n_i - d) * (q_i - q_i_1) / (n_i - n_i_1)
        )

        return q_new

    def _linear(self, i: int, d: float) -> float:
        """
        Linear formula for marker height adjustment (fallback).

        Args:
            i: Marker index
            d: Direction (+1 or -1)

        Returns:
            New marker height
        """
        if d == 1:
            return self.q[i] + (self.q[i + 1] - self.q[i]) / (self.n[i + 1] - self.n[i])
        else:
            return self.q[i] - (self.q[i] - self.q[i - 1]) / (self.n[i] - self.n[i - 1])

    def get_quantiles(self) -> np.ndarray:
        """
        Get current quantile estimates.

        Returns:
            Array of K+2 quantile values
        """
        if not self.initialized:
            # Return dummy values if not initialized
            return np.zeros(self.n_markers)
        return self.q.copy()

    def get_thresholds(self) -> np.ndarray:
        """
        Get K interior thresholds (excluding min and max).

        Returns:
            Array of K threshold values
        """
        quantiles = self.get_quantiles()
        return quantiles[1:-1]  # Exclude first and last (0th and 100th percentiles)

    def get_iqr(self) -> float:
        """
        Get interquartile range (IQR) for normalization.

        Returns:
            IQR = Q3 - Q1 (approximate based on K thresholds)
        """
        if not self.initialized:
            return 1.0

        # Use first and last interior quantiles as proxies for Q1 and Q3
        q_low = self.q[1]
        q_high = self.q[-2]
        iqr = max(q_high - q_low, 1e-6)  # Avoid division by zero

        return iqr


class DualSpeedP2:
    """
    Dual-speed P² quantile tracker for TWINE.

    Maintains both slow and fast trackers to detect drift.
    """

    def __init__(self, K: int = 8, slow_speed: float = 0.5, fast_speed: float = 2.0):
        """
        Initialize dual-speed tracker.

        Args:
            K: Number of quantiles
            slow_speed: Speed factor for slow tracker (<1.0 for stability)
            fast_speed: Speed factor for fast tracker (>1.0 for responsiveness)
        """
        self.K = K
        self.slow_tracker = P2Quantile(K=K, speed=slow_speed)
        self.fast_tracker = P2Quantile(K=K, speed=fast_speed)

    def update(self, x: float):
        """Update both trackers with new observation."""
        self.slow_tracker.update(x)
        self.fast_tracker.update(x)

    def get_slow_quantiles(self) -> np.ndarray:
        """Get quantiles from slow tracker."""
        return self.slow_tracker.get_quantiles()

    def get_fast_quantiles(self) -> np.ndarray:
        """Get quantiles from fast tracker."""
        return self.fast_tracker.get_quantiles()

    def get_slow_thresholds(self) -> np.ndarray:
        """Get thresholds from slow tracker."""
        return self.slow_tracker.get_thresholds()

    def get_fast_thresholds(self) -> np.ndarray:
        """Get thresholds from fast tracker."""
        return self.fast_tracker.get_thresholds()

    def compute_disagreement(self) -> float:
        """
        Compute normalized disagreement between slow and fast trackers.

        Returns:
            Normalized disagreement Δ = max|q_fast - q_slow| / IQR_slow
        """
        slow_q = self.get_slow_thresholds()
        fast_q = self.get_fast_thresholds()
        iqr_slow = self.slow_tracker.get_iqr()

        disagreement = np.max(np.abs(fast_q - slow_q)) / iqr_slow
        return disagreement

    def is_initialized(self) -> bool:
        """Check if both trackers are initialized."""
        return self.slow_tracker.initialized and self.fast_tracker.initialized

