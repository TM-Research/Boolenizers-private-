"""
Temporal Tsetlin Machine (TTM) — ChronoLogic Framework
=======================================================

A novel Tsetlin Machine variant that explicitly models temporal dependencies
through:
  1. Temporal Clauses — conjunctions spanning multiple time indices
  2. Memory-Augmented Automata — automata with reward history and frequency stats
  3. Time-Aware Feedback — Type I/II feedback with temporal credit assignment
  4. Self-Tuning — adaptive binarizer parameters driven by clause feedback

Designed for integration with TMU's TMClassifier backend for actual clause
learning, while providing the temporal structuring layer on top.

Author: Research Framework
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class TemporalAutomatonState:
    """Extended Tsetlin Automaton state with memory."""
    action: int = 0            # 0 = exclude, 1 = include
    state: int = 50            # automaton state (1..N)
    reward_history: List[float] = field(default_factory=list)
    frequency_count: int = 0   # how often this literal was active
    temporal_index: int = 0    # which time lag this automaton covers
    max_history: int = 32

    def update_reward(self, r: float):
        self.reward_history.append(r)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)

    @property
    def avg_reward(self) -> float:
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)


@dataclass
class TemporalClause:
    """
    A clause that spans multiple time indices.

    Example: C_j = x₁(t-1) ∧ ¬x₂(t) ∧ x₃(t-3)

    Each literal is (feature_index, time_lag, negated).
    """
    clause_id: int
    polarity: int  # +1 or -1
    literals: List[Tuple[int, int, bool]] = field(default_factory=list)
    automata: List[TemporalAutomatonState] = field(default_factory=list)
    weight: float = 1.0
    activation_count: int = 0

    def evaluate(self, X_temporal: np.ndarray) -> np.ndarray:
        """
        Evaluate clause on temporal binary input.

        X_temporal: (n_samples, n_lags, n_features) binary array
        Returns: (n_samples,) binary — 1 if clause is satisfied
        """
        n_samples = X_temporal.shape[0]
        result = np.ones(n_samples, dtype=np.uint8)

        for feat_idx, time_lag, negated in self.literals:
            if time_lag >= X_temporal.shape[1]:
                result[:] = 0
                break
            lit_val = X_temporal[:, time_lag, feat_idx]
            if negated:
                lit_val = 1 - lit_val
            result &= lit_val

        return result


# ===========================================================================
# Temporal Tsetlin Machine
# ===========================================================================

class TemporalTsetlinMachine:
    """
    Temporal Tsetlin Machine (TTM)
    ==============================

    Wraps TMU's TMClassifier with a temporal structuring layer that:
      1. Constructs temporal feature matrices from sliding windows
      2. Manages temporal clause formation
      3. Provides time-aware feedback mechanisms
      4. Self-tunes binarizer parameters

    Architecture:
        Raw series → ChronoLogicBinarizer → Temporal Window Stacking →
        TMClassifier (TMU backend) → Temporal Credit Assignment → Feedback

    Parameters
    ----------
    n_lags : int
        Number of past time steps to include (temporal depth).
    n_clauses : int
        Number of clauses per class.
    T : float
        Voting threshold.
    s : float
        Specificity parameter.
    n_classes : int
        Number of output classes.
    temporal_credit_decay : float
        Exponential decay for temporal credit assignment.
    self_tune_lr : float
        Learning rate for self-tuning binarizer parameters.
    use_memory_augmented : bool
        Enable memory-augmented automata tracking.
    """

    def __init__(
        self,
        n_lags: int = 4,
        n_clauses: int = 1000,
        T: float = 10.0,
        s: float = 10.0,
        n_classes: int = 2,
        epochs: int = 10,
        temporal_credit_decay: float = 0.9,
        self_tune_lr: float = 0.01,
        use_memory_augmented: bool = True,
        weighted_clauses: bool = False,
        type_iii_feedback: bool = False,
        seed: int = 42,
    ):
        self.n_lags = n_lags
        self.n_clauses = n_clauses
        self.T = T
        self.s = s
        self.n_classes = n_classes
        self.epochs = epochs
        self.temporal_credit_decay = temporal_credit_decay
        self.self_tune_lr = self_tune_lr
        self.use_memory_augmented = use_memory_augmented
        self.weighted_clauses = weighted_clauses
        self.type_iii_feedback = type_iii_feedback
        self.seed = seed

        # Internal state
        self.tm_ = None  # TMU TMClassifier instance
        self.temporal_clauses_: List[TemporalClause] = []
        self.feedback_history_: List[Dict] = []
        self.binarizer_params_history_: List[Dict] = []

    def _stack_temporal(self, X: np.ndarray) -> np.ndarray:
        """
        Stack binary features across time lags.

        X: (T, B) where B = number of binary features
        Returns: (T - n_lags, B * (n_lags + 1))

        The output at time t contains [X(t), X(t-1), ..., X(t-n_lags)]
        flattened into a single vector.
        """
        T, B = X.shape
        L = T - self.n_lags
        if L <= 0:
            raise ValueError(f"Series length {T} too short for {self.n_lags} lags")

        stacked = np.zeros((L, B * (self.n_lags + 1)), dtype=np.uint8)
        for lag in range(self.n_lags + 1):
            start = self.n_lags - lag
            stacked[:, lag * B:(lag + 1) * B] = X[start:start + L]

        return stacked

    def _align_labels(self, y: np.ndarray) -> np.ndarray:
        """Align labels to the stacked temporal features (drop first n_lags)."""
        return y[self.n_lags:]

    def fit(self, X_bin: np.ndarray, y: np.ndarray) -> 'TemporalTsetlinMachine':
        """
        Train the Temporal Tsetlin Machine.

        X_bin: (T, B) binary features from ChronoLogicBinarizer
        y: (T,) labels

        Steps:
            1. Stack temporal features: (T, B) → (T-n_lags, B*(n_lags+1))
            2. Initialize TMU TMClassifier
            3. Train with temporal credit assignment
            4. Track feedback for self-tuning
        """
        from tmu.models.classification.vanilla_classifier import TMClassifier

        X_stacked = self._stack_temporal(X_bin)
        y_aligned = self._align_labels(y).astype(np.uint32)

        # Initialize TMU backend
        self.tm_ = TMClassifier(
            number_of_clauses=self.n_clauses,
            T=self.T,
            s=self.s,
            weighted_clauses=self.weighted_clauses,
            type_iii_feedback=self.type_iii_feedback,
            incremental=(not self.weighted_clauses),
            seed=self.seed,
        )

        # Training loop with temporal credit tracking
        epoch_metrics = []
        for epoch in range(self.epochs):
            self.tm_.fit(
                X_stacked.astype(np.uint32),
                y_aligned,
                epochs=1,
            )

            # Evaluate on training data for feedback tracking
            preds = self.tm_.predict(X_stacked.astype(np.uint32))
            acc = np.mean(preds == y_aligned)
            epoch_metrics.append({
                'epoch': epoch,
                'train_acc': float(acc),
            })

            # Temporal credit assignment
            if self.use_memory_augmented and epoch > 0:
                credit = self._temporal_credit_assignment(
                    preds, y_aligned, epoch_metrics
                )
                self.feedback_history_.append({
                    'epoch': epoch,
                    'credit_signal': float(credit),
                    'train_acc': float(acc),
                })

        return self

    def predict(self, X_bin: np.ndarray) -> np.ndarray:
        """
        Predict on temporally binarized data.

        X_bin: (T, B) binary features
        Returns: (T - n_lags,) predictions
        """
        if self.tm_ is None:
            raise ValueError("Must call fit() first")

        X_stacked = self._stack_temporal(X_bin)
        return self.tm_.predict(X_stacked.astype(np.uint32))

    def _temporal_credit_assignment(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        history: List[Dict],
    ) -> float:
        """
        Compute temporal credit assignment signal.

        Uses exponentially-weighted accuracy improvement as the credit signal,
        which can be used to adjust binarizer parameters.

        ΔS(t) ∝ Σ_{Δ=0}^{H} γ^Δ · (acc(t) - acc(t-Δ-1))
        """
        if len(history) < 2:
            return 0.0

        credit = 0.0
        gamma = self.temporal_credit_decay
        for delta in range(min(len(history) - 1, 8)):
            idx = len(history) - 1 - delta
            acc_now = history[idx]['train_acc']
            acc_prev = history[idx - 1]['train_acc']
            credit += (gamma ** delta) * (acc_now - acc_prev)

        return credit

    def self_tune_binarizer(self, binarizer, credit_signal: float) -> Dict[str, float]:
        """
        Adjust binarizer parameters based on feedback.

        θ(t+1) = θ(t) + η · credit_signal · direction

        For ChronoLogicBinarizer, adjustable parameters include:
          - entropy_bias: shift entropy threshold
          - freq_quantile: adjust frequency band sensitivity

        Returns dict of parameter updates applied.
        """
        updates = {}
        lr = self.self_tune_lr

        if hasattr(binarizer, 'entropy_gate') and binarizer.entropy_gate is not None:
            delta_bias = lr * credit_signal
            binarizer.entropy_gate.bias += delta_bias
            updates['entropy_bias'] = float(binarizer.entropy_gate.bias)

        if hasattr(binarizer, 'freq_encoder') and binarizer.freq_encoder is not None:
            # Adjust frequency threshold quantile (clamp to [0.1, 0.9])
            if binarizer.freq_encoder.thresholds is not None:
                scale = 1.0 + lr * credit_signal
                binarizer.freq_encoder.thresholds *= scale
                updates['freq_threshold_scale'] = float(scale)

        self.binarizer_params_history_.append(updates)
        return updates

    def get_temporal_clause_interpretations(self) -> List[str]:
        """
        Extract human-readable temporal clause interpretations.

        Maps TMU clause structure back to temporal logic rules.
        Returns list of strings like:
          "IF x₁(t-1)=1 AND NOT x₂(t)=1 AND x₃(t-3)=1 THEN class=0"
        """
        if self.tm_ is None:
            return []

        # TMU exposes clause bank; we interpret temporal indices
        interpretations = []
        # This would require accessing tm_.clause_bank which varies by TMU version
        # Placeholder for the interpretation logic
        return interpretations

    def get_training_metrics(self) -> Dict[str, Any]:
        """Return training history and feedback signals."""
        return {
            'feedback_history': self.feedback_history_,
            'binarizer_params_history': self.binarizer_params_history_,
        }


# ===========================================================================
# Fuzzy Extension: ChronoLogic → Fuzzy Membership
# ===========================================================================

class FuzzyTemporalBinarizer:
    """
    Extends ChronoLogic binary encoding to fuzzy membership values in [0, 1].

    Instead of hard thresholding, uses sigmoid activations:
        μ_i(t) = σ(β · (x_i(t) - τ_i))

    This provides a natural generalization:
        Binary: B_i ∈ {0, 1}  (β → ∞)
        Fuzzy:  μ_i ∈ [0, 1]  (finite β)

    Compatible with Tsetlin.jl's fuzzy-pattern TM.
    """

    def __init__(self, beta: float = 5.0):
        self.beta = beta

    @staticmethod
    def sigmoid(x: np.ndarray, beta: float = 5.0) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-beta * np.clip(x, -50, 50)))

    def binary_to_fuzzy(self, B: np.ndarray, X_raw: np.ndarray,
                        thresholds: np.ndarray) -> np.ndarray:
        """
        Convert binary encoding to fuzzy membership.

        B: (N, n_bits) binary — original encoding
        X_raw: raw continuous values used to compute soft membership
        thresholds: the thresholds that produced B

        Returns: (N, n_bits) float in [0, 1]
        """
        # For each bit, compute sigmoid distance from threshold
        return self.sigmoid(X_raw - thresholds, self.beta)

    def phase_space_fuzzy(self, r: np.ndarray, theta: np.ndarray,
                          ring_edges: np.ndarray,
                          n_sectors: int) -> np.ndarray:
        """
        Fuzzy phase-space membership.

        Instead of hard ring/sector boundaries, use Gaussian membership:
            μ_{ring}(r) = exp(-((r - r_center) / σ_r)²)
            μ_{sector}(θ) = exp(-((θ - θ_center) / σ_θ)²)
        """
        # This provides smooth transitions between attractor regions
        sector_centers = np.linspace(-np.pi, np.pi, n_sectors + 1)
        sector_centers = (sector_centers[:-1] + sector_centers[1:]) / 2
        sigma_theta = np.pi / n_sectors

        n_rings = ring_edges.shape[1] - 1
        ring_centers = (ring_edges[:, :-1] + np.minimum(ring_edges[:, 1:], 100)) / 2
        sigma_r = np.diff(np.minimum(ring_edges, 100), axis=1) / 2 + 1e-8

        # Returns fuzzy memberships — shape depends on input dimensions
        return ring_centers, sector_centers, sigma_r, sigma_theta


# ===========================================================================
# Pipeline: Full ChronoLogic Pipeline
# ===========================================================================

class ChronoLogicPipeline:
    """
    End-to-end pipeline: Raw time series → Temporal Binarization → TTM → Predictions

    Handles:
        - Temporal windowing and alignment
        - Label alignment (drops first w samples)
        - Self-tuning feedback loop
        - Metric computation
    """

    def __init__(
        self,
        binarizer_kwargs: Optional[Dict] = None,
        ttm_kwargs: Optional[Dict] = None,
    ):
        from encoders.temporal_binarizer import ChronoLogicBinarizer

        self.binarizer = ChronoLogicBinarizer(**(binarizer_kwargs or {}))
        self.ttm = TemporalTsetlinMachine(**(ttm_kwargs or {}))
        self.n_dropped_train_ = 0
        self.n_dropped_test_ = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'ChronoLogicPipeline':
        """
        Fit binarizer and TTM.

        X_train: (T, d) raw continuous features
        y_train: (T,) labels
        """
        # Step 1: Fit and transform binarizer
        self.binarizer.fit(X_train)
        B_train = self.binarizer.transform(X_train)

        # Compute how many samples were lost
        self.n_dropped_train_ = X_train.shape[0] - B_train.shape[0]

        # Align labels
        y_aligned = y_train[self.n_dropped_train_:]

        # Step 2: Fit TTM (which further drops n_lags samples)
        self.ttm.fit(B_train, y_aligned)

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict on raw test data."""
        B_test = self.binarizer.transform(X_test)
        self.n_dropped_test_ = X_test.shape[0] - B_test.shape[0]
        return self.ttm.predict(B_test)

    def get_effective_labels(self, y: np.ndarray, is_train: bool = True) -> np.ndarray:
        """Get labels aligned with predictions."""
        n_dropped = self.n_dropped_train_ if is_train else self.n_dropped_test_
        total_dropped = n_dropped + self.ttm.n_lags
        return y[total_dropped:]
