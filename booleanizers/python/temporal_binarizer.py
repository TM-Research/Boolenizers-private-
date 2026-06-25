"""
ChronoLogic Temporal Binarizer (Φ_CL)
======================================
A novel temporal binarization framework that transforms multivariate time series
into temporally-enriched binary representations for Tsetlin Machine learning.

Core Novel Ideas:
  1. Phase-Space Attractor Binarization — maps (X(t), dX/dt) into binary attractor regions
  2. Temporal Entropy Gating — activates bits only when local window entropy exceeds
     an adaptive threshold, filtering temporally uninformative segments
  3. Frequency-Band Binary Encoding — FFT-based decomposition into binary frequency
     band activations

Reference: Designed for integration with TMU (TMClassifier) and Tsetlin.jl.

Author: Research Framework
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from encoders.base import ThermometerEncoder


# ---------------------------------------------------------------------------
# Utility: rolling windows
# ---------------------------------------------------------------------------

def _rolling_windows(X: np.ndarray, w: int) -> np.ndarray:
    """Create rolling windows of size w along axis 0.

    Args:
        X: shape (T, d)
        w: window size

    Returns:
        shape (T - w + 1, w, d)
    """
    T, d = X.shape
    if T < w:
        raise ValueError(f"Series length {T} < window {w}")
    shape = (T - w + 1, w, d)
    strides = (X.strides[0], X.strides[0], X.strides[1])
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


# ---------------------------------------------------------------------------
# Module 1: Phase-Space Attractor Binarizer
# ---------------------------------------------------------------------------

class PhaseSpaceAttractorEncoder:
    r"""
    Maps each feature into a 2D phase space (x, dx/dt) and quantizes into
    binary attractor region indicators.

    Given x_i(t) and its finite difference \dot{x}_i(t) = x_i(t) - x_i(t-1),
    the phase plane is partitioned into R radial sectors × A annular rings,
    yielding R*A binary bits per feature.

    The partition adapts: centroids and radii are learned from training data.

    Mathematical formulation:
        Let z_i(t) = (x_i(t) - μ_i, ẋ_i(t) - μ̇_i) ∈ ℝ²
        Polar: (r, θ) = (‖z_i‖, atan2(z_i[1], z_i[0]))
        B_{i,a,s}(t) = 1  iff  r ∈ [r_a, r_{a+1})  and  θ ∈ [θ_s, θ_{s+1})
    """

    def __init__(self, n_rings: int = 2, n_sectors: int = 4, eps: float = 1e-8):
        self.n_rings = n_rings
        self.n_sectors = n_sectors
        self.bits_per_feature = n_rings * n_sectors
        self.eps = eps
        # Learned parameters
        self.mu_x: Optional[np.ndarray] = None       # (d,)
        self.mu_dx: Optional[np.ndarray] = None      # (d,)
        self.ring_edges: Optional[np.ndarray] = None  # (d, n_rings+1)

    @property
    def K(self) -> int:
        return self.bits_per_feature

    def fit(self, X: np.ndarray) -> 'PhaseSpaceAttractorEncoder':
        """Fit on (T, d) time series."""
        dx = np.diff(X, axis=0)               # (T-1, d)
        x = X[1:]                              # align with dx
        self.mu_x = np.mean(x, axis=0)        # (d,)
        self.mu_dx = np.mean(dx, axis=0)       # (d,)

        # Compute radii for ring edge quantiles
        zx = x - self.mu_x
        zdx = dx - self.mu_dx
        r = np.sqrt(zx**2 + zdx**2 + self.eps)  # (T-1, d)

        quantiles = np.linspace(0, 1, self.n_rings + 1)[1:-1]  # interior edges
        self.ring_edges = np.zeros((X.shape[1], self.n_rings + 1))
        self.ring_edges[:, 0] = 0.0
        for i, q in enumerate(quantiles):
            self.ring_edges[:, i + 1] = np.quantile(r, q, axis=0)
        self.ring_edges[:, -1] = np.inf
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform (T, d) → (T-1, d * n_rings * n_sectors) binary."""
        dx = np.diff(X, axis=0)
        x = X[1:]
        T, d = x.shape

        zx = x - self.mu_x
        zdx = dx - self.mu_dx
        r = np.sqrt(zx**2 + zdx**2 + self.eps)
        theta = np.arctan2(zdx, zx)  # in [-π, π]

        sector_edges = np.linspace(-np.pi, np.pi, self.n_sectors + 1)

        out = np.zeros((T, d * self.bits_per_feature), dtype=np.uint8)

        for feat in range(d):
            for a in range(self.n_rings):
                ring_mask = (r[:, feat] >= self.ring_edges[feat, a]) & \
                            (r[:, feat] < self.ring_edges[feat, a + 1])
                for s in range(self.n_sectors):
                    sec_mask = (theta[:, feat] >= sector_edges[s]) & \
                               (theta[:, feat] < sector_edges[s + 1])
                    bit_idx = feat * self.bits_per_feature + a * self.n_sectors + s
                    out[:, bit_idx] = (ring_mask & sec_mask).astype(np.uint8)

        return out


# ---------------------------------------------------------------------------
# Module 2: Temporal Entropy Gating
# ---------------------------------------------------------------------------

class TemporalEntropyGate:
    r"""
    Computes local Shannon entropy over a sliding window and gates binary
    activations: bits are active only when entropy exceeds an adaptive threshold.

    Mathematical formulation:
        H_i(t) = -∑_k p_k(t) log₂(p_k(t) + ε)
        where p_k(t) is the histogram probability of x_i in window [t-w, t]

        τ_i(t) = α · τ_i(t-1) + (1-α) · H̄_i(t)      (EMA of mean entropy)
        G_i(t) = 1  iff  H_i(t) > τ_i(t)

    This filters out temporally flat (uninformative) regions, ensuring TM
    clauses focus on dynamically interesting segments.
    """

    def __init__(self, window: int = 8, n_bins: int = 8, alpha: float = 0.9,
                 bias: float = 0.0):
        self.window = window
        self.n_bins = n_bins
        self.alpha = alpha
        self.bias = bias
        # Learned
        self.bin_edges: Optional[np.ndarray] = None  # (d, n_bins+1)
        self.tau: Optional[np.ndarray] = None         # (d,) adaptive threshold

    def fit(self, X: np.ndarray) -> 'TemporalEntropyGate':
        """Fit histogram bin edges and initial thresholds from training series."""
        d = X.shape[1]
        self.bin_edges = np.zeros((d, self.n_bins + 1))
        for j in range(d):
            self.bin_edges[j] = np.linspace(
                np.min(X[:, j]) - 1e-8, np.max(X[:, j]) + 1e-8, self.n_bins + 1
            )
        # Compute entropies on training data to set initial tau
        H = self._compute_entropy(X)  # (T', d)
        self.tau = np.mean(H, axis=0) + self.bias  # (d,)
        return self

    def _compute_entropy(self, X: np.ndarray) -> np.ndarray:
        """Compute windowed entropy for each feature. Returns (T-w+1, d)."""
        T, d = X.shape
        w = self.window
        L = T - w + 1
        H = np.zeros((L, d))
        eps = 1e-12
        for t in range(L):
            seg = X[t:t + w, :]  # (w, d)
            for j in range(d):
                counts, _ = np.histogram(seg[:, j], bins=self.bin_edges[j])
                p = counts / w
                p = p[p > 0]
                H[t, j] = -np.sum(p * np.log2(p + eps))
        return H

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return binary gate mask (T-w+1, d) where 1 = high entropy."""
        H = self._compute_entropy(X)
        return (H > self.tau[np.newaxis, :]).astype(np.uint8)

    def transform_adaptive(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (gate_mask, updated_tau) with EMA threshold adaptation."""
        H = self._compute_entropy(X)
        T = H.shape[0]
        tau = self.tau.copy()
        gates = np.zeros_like(H, dtype=np.uint8)
        for t in range(T):
            gates[t] = (H[t] > tau).astype(np.uint8)
            tau = self.alpha * tau + (1 - self.alpha) * H[t]
        self.tau = tau  # persist updated threshold
        return gates, H


# ---------------------------------------------------------------------------
# Module 3: Frequency-Band Binary Encoding
# ---------------------------------------------------------------------------

class FrequencyBandEncoder:
    r"""
    Applies FFT over a sliding window and binarizes the presence/absence
    of energy in K frequency bands.

    Mathematical formulation:
        For window x_i[t-w:t], compute DFT: F_i(ω) = FFT(x_i[t-w:t])
        Partition positive frequencies into K bands: [ω_{k-1}, ω_k)
        Energy in band k: E_k = Σ_{ω ∈ band_k} |F_i(ω)|²
        B_{i,k}(t) = 1  iff  E_k > τ_k

    Thresholds τ_k are learned from training data (quantile-based).
    """

    def __init__(self, window: int = 16, n_bands: int = 4, quantile: float = 0.5):
        self.window = window
        self.n_bands = n_bands
        self.quantile = quantile
        # Learned
        self.thresholds: Optional[np.ndarray] = None  # (d, n_bands)

    @property
    def K(self) -> int:
        return self.n_bands

    def _band_energies(self, X: np.ndarray) -> np.ndarray:
        """Compute per-band FFT energies. X: (T, d) → (T-w+1, d, n_bands)."""
        T, d = X.shape
        w = self.window
        L = T - w + 1
        n_freqs = w // 2  # positive frequencies
        band_size = max(1, n_freqs // self.n_bands)

        energies = np.zeros((L, d, self.n_bands))
        for t in range(L):
            seg = X[t:t + w, :]  # (w, d)
            F = np.fft.rfft(seg, axis=0)  # (w//2+1, d)
            power = np.abs(F[1:])**2       # exclude DC, (w//2, d)
            for k in range(self.n_bands):
                lo = k * band_size
                hi = min((k + 1) * band_size, power.shape[0])
                if lo < power.shape[0]:
                    energies[t, :, k] = np.sum(power[lo:hi], axis=0)
        return energies

    def fit(self, X: np.ndarray) -> 'FrequencyBandEncoder':
        """Learn band energy thresholds from training data."""
        energies = self._band_energies(X)  # (L, d, n_bands)
        d = X.shape[1]
        self.thresholds = np.zeros((d, self.n_bands))
        for j in range(d):
            for k in range(self.n_bands):
                self.thresholds[j, k] = np.quantile(energies[:, j, k], self.quantile)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform (T, d) → (L, d * n_bands) binary."""
        energies = self._band_energies(X)  # (L, d, n_bands)
        L, d, _ = energies.shape
        out = np.zeros((L, d * self.n_bands), dtype=np.uint8)
        for j in range(d):
            for k in range(self.n_bands):
                out[:, j * self.n_bands + k] = \
                    (energies[:, j, k] > self.thresholds[j, k]).astype(np.uint8)
        return out


# ---------------------------------------------------------------------------
# Module 4: Temporal Logic Propositions
# ---------------------------------------------------------------------------

class TemporalLogicEncoder:
    r"""
    Converts raw time-series segments into temporal logic propositions:
      - "increasing" / "decreasing" (monotonicity)
      - "oscillating" (sign changes in derivative)
      - "bursting" (local variance spike)
      - "stable" (low variance)

    Each proposition becomes a binary bit per feature, per lag window.

    Mathematical formulation:
        Let Δ_i(t) = x_i(t) - x_i(t-1)
        increasing_i(t): 1 iff  Σ_{s=t-w}^{t} 1[Δ_i(s)>0] > 0.7w
        decreasing_i(t): 1 iff  Σ_{s=t-w}^{t} 1[Δ_i(s)<0] > 0.7w
        oscillating_i(t): 1 iff  #sign_changes(Δ_i[t-w:t]) > 0.4w
        bursting_i(t): 1 iff  Var(x_i[t-w:t]) > γ · Var_global_i
        stable_i(t): 1 iff  Var(x_i[t-w:t]) < δ · Var_global_i

    5 propositions per feature → 5d bits.
    """

    PROPOSITIONS = ['increasing', 'decreasing', 'oscillating', 'bursting', 'stable']

    def __init__(self, window: int = 8, mono_threshold: float = 0.7,
                 osc_threshold: float = 0.4, burst_gamma: float = 2.0,
                 stable_delta: float = 0.3):
        self.window = window
        self.mono_threshold = mono_threshold
        self.osc_threshold = osc_threshold
        self.burst_gamma = burst_gamma
        self.stable_delta = stable_delta
        self.n_props = len(self.PROPOSITIONS)
        # Learned
        self.global_var: Optional[np.ndarray] = None  # (d,)

    @property
    def K(self) -> int:
        return self.n_props

    def fit(self, X: np.ndarray) -> 'TemporalLogicEncoder':
        self.global_var = np.var(X, axis=0) + 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (T, d) → (T-w, d * 5) binary."""
        T, d = X.shape
        w = self.window
        L = T - w
        dx = np.diff(X, axis=0)  # (T-1, d)

        out = np.zeros((L, d * self.n_props), dtype=np.uint8)
        for t in range(L):
            seg_dx = dx[t:t + w, :]       # (w, d) — derivative segment
            seg_x = X[t:t + w + 1, :]     # (w+1, d) — value segment

            # Increasing: majority positive derivatives
            frac_pos = np.mean(seg_dx > 0, axis=0)
            out[t, 0::self.n_props] = (frac_pos > self.mono_threshold).astype(np.uint8)

            # Decreasing: majority negative derivatives
            frac_neg = np.mean(seg_dx < 0, axis=0)
            out[t, 1::self.n_props] = (frac_neg > self.mono_threshold).astype(np.uint8)

            # Oscillating: high sign-change rate
            signs = np.sign(seg_dx)
            sign_changes = np.sum(np.abs(np.diff(signs, axis=0)) > 0, axis=0)
            osc_frac = sign_changes / max(w - 1, 1)
            out[t, 2::self.n_props] = (osc_frac > self.osc_threshold).astype(np.uint8)

            # Bursting: local variance spike
            local_var = np.var(seg_x, axis=0)
            out[t, 3::self.n_props] = (local_var > self.burst_gamma * self.global_var).astype(np.uint8)

            # Stable: very low local variance
            out[t, 4::self.n_props] = (local_var < self.stable_delta * self.global_var).astype(np.uint8)

        return out


# ===========================================================================
# UNIFIED: ChronoLogic Temporal Binarizer (Φ_CL)
# ===========================================================================

class ChronoLogicBinarizer(ThermometerEncoder):
    r"""
    ChronoLogic Temporal Binarizer — Φ_CL
    =======================================

    Unified temporal binarization that fuses:
      1. Phase-Space Attractor regions (structural dynamics)
      2. Temporal Entropy Gating (information-theoretic filtering)
      3. Frequency-Band activations (spectral structure)
      4. Temporal Logic Propositions (symbolic dynamics)

    The gating mechanism ensures bits are active only during dynamically
    interesting time windows, improving clause separability.

    Total bits per sample:
        B = d * (n_rings * n_sectors + n_bands + n_props)
    where the entropy gate acts as a multiplicative mask on the phase-space
    and frequency bits.

    Parameters
    ----------
    window : int
        Temporal context window size (k in the formulation).
    n_rings, n_sectors : int
        Phase-space partition granularity.
    n_bands : int
        Number of FFT frequency bands.
    entropy_alpha : float
        EMA smoothing for adaptive entropy threshold.
    use_entropy_gate : bool
        Whether to apply entropy gating (multiplicative mask).
    use_phase_space : bool
        Include phase-space attractor bits.
    use_frequency : bool
        Include frequency-band bits.
    use_temporal_logic : bool
        Include temporal logic proposition bits.
    """

    def __init__(
        self,
        K: int = 8,
        window: int = 8,
        n_rings: int = 2,
        n_sectors: int = 4,
        n_bands: int = 4,
        entropy_alpha: float = 0.9,
        entropy_bias: float = 0.0,
        freq_quantile: float = 0.5,
        use_entropy_gate: bool = True,
        use_phase_space: bool = True,
        use_frequency: bool = True,
        use_temporal_logic: bool = True,
        name: str = "ChronoLogic",
    ):
        # K is informational; actual bits depend on sub-modules
        super().__init__(K=K, name=name)
        self.window = window
        self.use_entropy_gate = use_entropy_gate
        self.use_phase_space = use_phase_space
        self.use_frequency = use_frequency
        self.use_temporal_logic = use_temporal_logic

        # Sub-modules
        self.phase_encoder = PhaseSpaceAttractorEncoder(
            n_rings=n_rings, n_sectors=n_sectors
        ) if use_phase_space else None

        self.entropy_gate = TemporalEntropyGate(
            window=window, alpha=entropy_alpha, bias=entropy_bias
        ) if use_entropy_gate else None

        self.freq_encoder = FrequencyBandEncoder(
            window=window, n_bands=n_bands, quantile=freq_quantile
        ) if use_frequency else None

        self.logic_encoder = TemporalLogicEncoder(
            window=window
        ) if use_temporal_logic else None

    def fit(self, X: np.ndarray) -> 'ChronoLogicBinarizer':
        """
        Fit all sub-modules.

        For tabular data (n_samples, n_features), we treat each sample
        as independent. For true temporal data, X should be (T, d).

        If X has no temporal structure (standard tabular), we add a synthetic
        temporal dimension by treating samples as sequential.
        """
        self.n_features = X.shape[1]

        if self.phase_encoder is not None:
            self.phase_encoder.fit(X)
        if self.entropy_gate is not None:
            self.entropy_gate.fit(X)
        if self.freq_encoder is not None:
            self.freq_encoder.fit(X)
        if self.logic_encoder is not None:
            self.logic_encoder.fit(X)

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series X: (T, d) → (T', B) binary array.

        T' < T due to windowing; the effective length depends on the
        maximum window requirement across sub-modules.
        """
        if not self.fitted:
            raise ValueError("Must call fit() first")

        T, d = X.shape
        parts = []
        lengths = []

        # Phase-space: needs T-1 samples (due to diff)
        if self.phase_encoder is not None:
            ps_bits = self.phase_encoder.transform(X)  # (T-1, d*rings*sectors)
            parts.append(('phase', ps_bits))
            lengths.append(ps_bits.shape[0])

        # Frequency bands: needs T-w+1 samples
        if self.freq_encoder is not None:
            fb_bits = self.freq_encoder.transform(X)  # (T-w+1, d*n_bands)
            parts.append(('freq', fb_bits))
            lengths.append(fb_bits.shape[0])

        # Temporal logic: needs T-w samples
        if self.logic_encoder is not None:
            tl_bits = self.logic_encoder.transform(X)  # (T-w, d*5)
            parts.append(('logic', tl_bits))
            lengths.append(tl_bits.shape[0])

        # Entropy gate: needs T-w+1 samples
        gate_mask = None
        if self.entropy_gate is not None:
            gate_mask, _ = self.entropy_gate.transform_adaptive(X)  # (T-w+1, d)
            lengths.append(gate_mask.shape[0])

        if not lengths:
            return np.zeros((T, 0), dtype=np.uint8)

        # Align all to minimum length (trim from front = causal alignment)
        L = min(lengths)
        aligned = []

        for name, bits in parts:
            aligned.append(bits[-L:])  # take last L rows (causal: most recent)

        # Apply entropy gate as multiplicative mask
        if gate_mask is not None and len(aligned) > 0:
            g = gate_mask[-L:]  # (L, d)
            gated = []
            for bits in aligned:
                n_bits_per_feat = bits.shape[1] // d
                # Expand gate to match bit dimensions
                g_expanded = np.repeat(g, n_bits_per_feat, axis=1)
                gated.append(bits * g_expanded)
            aligned = gated

        if not aligned:
            return np.zeros((L, 0), dtype=np.uint8)

        result = np.concatenate(aligned, axis=1).astype(np.uint8)
        return result

    def transform_tabular(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience for tabular (non-temporal) data.

        Treats the sample dimension as a pseudo-temporal dimension,
        applies temporal binarization, then pads the first few samples
        that are lost to windowing with zeros.

        This allows the binarizer to integrate into the standard
        encoder pipeline (fit/transform on tabular matrices).
        """
        B = self.transform(X)
        T = X.shape[0]
        L = B.shape[0]
        lost = T - L
        if lost > 0:
            pad = np.zeros((lost, B.shape[1]), dtype=np.uint8)
            B = np.concatenate([pad, B], axis=0)
        return B

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """Single-sample encoding (for flip tracking). Uses last window."""
        # For single-sample online use, this is a stub
        # Real temporal encoding requires a buffer
        return np.zeros(self.get_n_output_bits(), dtype=np.uint8)

    def get_n_output_bits(self) -> int:
        """Total binary features produced."""
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        d = self.n_features
        total = 0
        if self.phase_encoder is not None:
            total += d * self.phase_encoder.bits_per_feature
        if self.freq_encoder is not None:
            total += d * self.freq_encoder.n_bands
        if self.logic_encoder is not None:
            total += d * self.logic_encoder.n_props
        return total

    def describe_bits(self) -> Dict[str, Any]:
        """Return a description of what each bit group encodes."""
        d = self.n_features or 0
        desc = {}
        offset = 0
        if self.phase_encoder is not None:
            n = d * self.phase_encoder.bits_per_feature
            desc['phase_space'] = {
                'offset': offset, 'n_bits': n,
                'semantics': f'{self.phase_encoder.n_rings} rings × '
                             f'{self.phase_encoder.n_sectors} sectors per feature'
            }
            offset += n
        if self.freq_encoder is not None:
            n = d * self.freq_encoder.n_bands
            desc['frequency_bands'] = {
                'offset': offset, 'n_bits': n,
                'semantics': f'{self.freq_encoder.n_bands} FFT bands per feature'
            }
            offset += n
        if self.logic_encoder is not None:
            n = d * self.logic_encoder.n_props
            desc['temporal_logic'] = {
                'offset': offset, 'n_bits': n,
                'semantics': 'increasing/decreasing/oscillating/bursting/stable per feature'
            }
            offset += n
        desc['total_bits'] = offset
        desc['entropy_gated'] = self.use_entropy_gate
        return desc
