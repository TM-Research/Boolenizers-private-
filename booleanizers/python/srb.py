"""
Streaming Resonance Binarizer (SRB) v3

A novel binarization algorithm derived from first principles of
information-theoretic signal quantization and adaptive scaling.

==========================================================================
CORE PRINCIPLE: ADAPTIVE PROBIT SCALING
==========================================================================

From information theory, the optimal quantization of a signal places
thresholds at positions that maximize the total entropy of the binary
outputs. For a Gaussian signal, these optimal positions are at the
probit (inverse normal CDF) of equally-spaced probabilities:

    z_k = Phi^{-1}((k+1)/(K+1))  for k = 0, ..., K-1

These probit positions maximize per-bit entropy but only cover ~±1.2σ
for K=8, missing tail structure. Real features often have heavy tails
(attacks, anomalies) that carry crucial discriminative information.

OUR INNOVATION: Adaptive Probit Scaling

We learn each feature's "tail weight" (how far values extend relative
to its standard deviation) and use this to SCALE the probit positions:

    threshold[j,k] = μ_j + σ_j * z_k * scale_j

where scale_j = f(tail_weight_j) adapts per feature:
    - Compact features (low tail weight): scale ≈ 1 (pure probit, max entropy)
    - Heavy-tailed features (high tail weight): scale > 1 (wider coverage)

This gives the optimal balance: maximum information content near the
center (where most data lives) while still covering discriminative
tail structure (where anomalies/attacks live).

The ENTIRE algorithm runs in a single streaming pass with O(d) memory.

==========================================================================
HYPERPARAMETERS (2 only)
==========================================================================

    K : int (default 8)
        Binary bits per feature.

    scale_mode : str (default 'auto')
        - 'auto': per-feature adaptive scaling from tail weight
        - float value: fixed scale for all features

==========================================================================
COMPLEXITY
==========================================================================

    fit:       O(n * d) time, O(d * K) memory
    transform: O(n * d * K) time, O(1) extra memory
    ESP32:     O(d * K) per sample, no division needed in transform
"""

import numpy as np
from .base import ThermometerEncoder


# ---------- Core math: probit function ----------

def _probit(p: float) -> float:
    """Rational approximation of the inverse normal CDF (probit).

    Abramowitz & Stegun 26.2.23. Accuracy ~4.5e-4 for p in (0, 1).
    """
    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    if p > 0.5:
        return -_probit(1.0 - p)
    t = np.sqrt(-2.0 * np.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))


# ---------- Version tracking ----------

SRB_VERSION = "3.0"


class StreamingResonanceBinarizer(ThermometerEncoder):
    """
    Streaming Resonance Binarizer (SRB) v3.

    Single-pass streaming binarizer with adaptive probit scaling.

    Parameters
    ----------
    K : int, default=8
        Bits per feature.
    scale_mode : str or float, default='auto'
        'auto' for per-feature adaptive scaling, or a fixed float.
    """

    def __init__(self, K: int = 8, scale_mode="auto"):
        super().__init__(K=K, name="SRB")
        self.scale_mode = scale_mode

        # Precompute probit positions: z_k = Phi^{-1}((k+1)/(K+1))
        self._z_probit = np.array([_probit((k + 1) / (K + 1)) for k in range(K)])

    def fit(self, X: np.ndarray) -> "StreamingResonanceBinarizer":
        """
        Fit encoder in a single streaming pass.

        Uses Welford's online algorithm for mean/variance plus
        streaming min/max for tail weight estimation.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features = n_features

        # ---- Welford's online algorithm + streaming min/max ----
        mean = np.zeros(n_features, dtype=np.float64)
        m2 = np.zeros(n_features, dtype=np.float64)
        fmin = X[0].copy()
        fmax = X[0].copy()

        for i in range(n_samples):
            x = X[i]
            fmin = np.minimum(fmin, x)
            fmax = np.maximum(fmax, x)
            delta = x - mean
            mean += delta / (i + 1)
            delta2 = x - mean
            m2 += delta * delta2

        var = m2 / max(n_samples - 1, 1)
        std = np.sqrt(np.maximum(var, 1e-20))

        # ---- Per-feature tail weight and adaptive scale ----
        upper_ext = np.abs(fmax - mean)
        lower_ext = np.abs(mean - fmin)
        tail_weight = np.maximum(upper_ext, lower_ext) / np.maximum(std, 1e-10)

        if self.scale_mode == "auto":
            # K-aware adaptive scale: ensures all bits carry useful information.
            #
            # Key insight: for thermometer coding, a bit at threshold z*σ has
            # probability P(1) = Φ(-z). For the bit to be useful (entropy > 5%),
            # we need |z| < ~2.5σ. The outermost probit position is z_probit[-1].
            # So: max_useful_scale = 2.5 / |z_probit[-1]|
            #
            # For K=4: max_scale = 2.5/0.84 = 2.98
            # For K=6: max_scale = 2.5/1.07 = 2.34
            # For K=8: max_scale = 2.5/1.22 = 2.05
            #
            # Base scale from tail weight, clamped by K-dependent maximum:
            outer_probit = abs(self._z_probit[-1])
            max_scale = 2.5 / max(outer_probit, 0.1)

            scale = 0.5 * np.clip(tail_weight, 2.4, 6.0)
            scale = np.clip(scale, 1.2, max_scale)
        else:
            scale = np.full(n_features, float(self.scale_mode))

        # ---- Place thresholds at scaled probit positions ----
        # threshold[j,k] = mean[j] + std[j] * z_k * scale[j]
        self.thresholds_ = (
            mean[:, np.newaxis]
            + std[:, np.newaxis] * self._z_probit[np.newaxis, :] * scale[:, np.newaxis]
        )

        # ---- Ensure strict monotonicity ----
        for j in range(n_features):
            self.thresholds_[j] = np.sort(self.thresholds_[j])
            for k in range(1, self.K):
                if self.thresholds_[j, k] <= self.thresholds_[j, k - 1]:
                    self.thresholds_[j, k] = self.thresholds_[j, k - 1] + 1e-10

        # Store diagnostics
        self.mu_ = mean
        self.std_ = std
        self.scale_ = scale
        self.tail_weight_ = tail_weight

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to binary thermometer codes. Fully vectorized."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        bits = (
            X[:, :, np.newaxis] >= self.thresholds_[np.newaxis, :, :]
        ).astype(np.uint8)
        return bits.reshape(X.shape[0], -1)

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """Encode one sample. O(d * K) — ESP32 compatible."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")
        x = np.asarray(x, dtype=np.float64)
        bits = (x[:, np.newaxis] >= self.thresholds_).astype(np.uint8)
        return bits.ravel()

    def get_n_output_bits(self) -> int:
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K

    def get_threshold_info(self) -> dict:
        """Return threshold details for interpretability analysis."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return {
            "version": SRB_VERSION,
            "thresholds": self.thresholds_.copy(),
            "feature_means": self.mu_.copy(),
            "feature_stds": self.std_.copy(),
            "scale_per_feature": self.scale_.copy(),
            "tail_weight_per_feature": self.tail_weight_.copy(),
            "probit_positions": self._z_probit.copy(),
            "K": self.K,
            "n_features": self.n_features,
        }
