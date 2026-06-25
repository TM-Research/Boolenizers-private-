"""
TWINE v4: Ensemble Approach

RADICAL IDEA: What if the problem is that a SINGLE set of thresholds can't
capture the full data distribution?

Solution: Multiple threshold sets voting together!

Inspiration: Random Forest uses multiple trees - we use multiple threshold sets

Architecture:
- K "experts", each with different P² speeds
- Each expert votes on bit values
- Final bit = majority vote or confidence-weighted
- Diversity through different adaptation speeds

Expected benefit: Robustness + accuracy boost
"""

import numpy as np
from .base import ThermometerEncoder
from .p2_algorithm import DualSpeedP2


class TWINEv4Ensemble(ThermometerEncoder):
    """
    TWINE v4: Ensemble of multiple adaptive threshold sets.

    Each expert has different adaptation speeds, creating diversity.
    Final encoding is ensemble decision.

    Parameters:
        K: Bits per feature per expert
        n_experts: Number of parallel experts (default: 3)
        speed_range: (min, max) speeds for experts
        h: Hysteresis per expert
        voting: 'majority' or 'weighted'
    """

    def __init__(
        self,
        K: int = 8,
        n_experts: int = 3,
        speed_range: tuple = (0.5, 2.5),
        h: float = 0.15,
        voting: str = 'majority'
    ):
        # Each expert produces K bits, final output is K bits (ensembled)
        super().__init__(K=K, name="TWINE-v4-Ensemble")

        self.n_experts = n_experts
        self.speed_range = speed_range
        self.h = h
        self.voting = voting

        # Create experts with different speeds
        speeds = np.linspace(speed_range[0], speed_range[1], n_experts)
        self.expert_speeds = [(s * 1.5, s * 0.5) for s in speeds]  # (fast, slow)

        # State per expert
        self.expert_trackers = None
        self.expert_thresholds = None
        self.expert_bit_states = None
        self.expert_confidences = None  # Track each expert's performance

        self.n_features = None

    def fit(self, X: np.ndarray) -> 'TWINEv4Ensemble':
        """Initialize all experts."""
        self.n_features = X.shape[1]
        self._init_experts()

        for i in range(X.shape[0]):
            self._encode_single(X[i])

        self.fitted = True
        return self

    def _init_experts(self):
        """Initialize all expert trackers and states."""
        self.expert_trackers = []
        self.expert_thresholds = []
        self.expert_bit_states = []
        self.expert_confidences = np.ones(self.n_experts)  # Start equal

        for expert_id in range(self.n_experts):
            fast_speed, slow_speed = self.expert_speeds[expert_id]

            # Trackers for this expert
            trackers = [
                DualSpeedP2(K=self.K, fast_speed=fast_speed, slow_speed=slow_speed)
                for _ in range(self.n_features)
            ]
            self.expert_trackers.append(trackers)

            # Thresholds for this expert
            thresholds = np.zeros((self.n_features, self.K))
            for i in range(self.n_features):
                thresholds[i] = np.linspace(0.0, 1.0, self.K + 2)[1:-1]
            self.expert_thresholds.append(thresholds)

            # Bit states for this expert
            bit_states = np.zeros((self.n_features, self.K), dtype=np.uint8)
            self.expert_bit_states.append(bit_states)

        self.fitted = True

    def _cold_start_init(self, x: np.ndarray):
        """Initialize from first sample."""
        self.n_features = len(x)
        self._init_experts()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using ensemble voting."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)

        for i in range(n_samples):
            # Get votes from all experts
            expert_votes = np.zeros((self.n_experts, self.n_features * self.K), dtype=np.uint8)

            for expert_id in range(self.n_experts):
                for feat_idx in range(self.n_features):
                    for k in range(self.K):
                        bit_idx = feat_idx * self.K + k

                        # Expert's vote with hysteresis
                        if self.expert_bit_states[expert_id][feat_idx, k] == 1:
                            vote = 1 if X[i, feat_idx] >= (self.expert_thresholds[expert_id][feat_idx, k] - self.h) else 0
                        else:
                            vote = 1 if X[i, feat_idx] >= (self.expert_thresholds[expert_id][feat_idx, k] + self.h) else 0

                        expert_votes[expert_id, bit_idx] = vote
                        self.expert_bit_states[expert_id][feat_idx, k] = vote

            # Ensemble voting
            if self.voting == 'majority':
                encoded[i] = (np.sum(expert_votes, axis=0) > (self.n_experts / 2)).astype(np.uint8)
            else:  # weighted
                weighted_votes = expert_votes.T @ self.expert_confidences
                encoded[i] = (weighted_votes > (np.sum(self.expert_confidences) / 2)).astype(np.uint8)

        return encoded

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """Encode with ensemble and update all experts."""
        if not self.fitted:
            self._cold_start_init(x)

        # Collect votes from all experts
        expert_votes = np.zeros((self.n_experts, self.n_features * self.K), dtype=np.uint8)

        for expert_id in range(self.n_experts):
            for i in range(self.n_features):
                # Update this expert's tracker
                self.expert_trackers[expert_id][i].update(x[i])

                # Get fast quantiles and update thresholds
                # Use K interior thresholds, not full P² marker array (K+2 values)
                fast_q = self.expert_trackers[expert_id][i].get_fast_thresholds()
                self.expert_thresholds[expert_id][i] = 0.2 * fast_q + 0.8 * self.expert_thresholds[expert_id][i]

                # Encode with this expert
                for k in range(self.K):
                    bit_idx = i * self.K + k

                    if self.expert_bit_states[expert_id][i, k] == 1:
                        vote = 1 if x[i] >= (self.expert_thresholds[expert_id][i, k] - self.h) else 0
                    else:
                        vote = 1 if x[i] >= (self.expert_thresholds[expert_id][i, k] + self.h) else 0

                    expert_votes[expert_id, bit_idx] = vote
                    self.expert_bit_states[expert_id][i, k] = vote

        # Ensemble decision
        if self.voting == 'majority':
            final_bits = (np.sum(expert_votes, axis=0) > (self.n_experts / 2)).astype(np.uint8)
        else:
            weighted_votes = expert_votes.T @ self.expert_confidences
            final_bits = (weighted_votes > (np.sum(self.expert_confidences) / 2)).astype(np.uint8)

        return final_bits

    def get_n_output_bits(self) -> int:
        """Get total output bits."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K



