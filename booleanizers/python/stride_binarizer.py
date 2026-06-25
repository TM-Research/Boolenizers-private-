"""
STRIDE Binarizer — Novel Time-Series Binarization
Signal-TRaded Incremental Delta Encoder

Inspired by:
  • MACD (trading)      — fast/slow EMA difference = momentum signal
  • ATR  (trading)      — Average True Range = local volatility normalization
  • Schmitt Trigger     — hysteresis deadband: output holds state in noise band
  • Delta Modulation    — event-driven: only store STATE CHANGES

WHY FEWER BITS THAN ANY CLASSIC BINARIZER:
  Traditional binarizer : N samples → N bits   (every sample forced to a bit)
  STRIDE                : hysteresis locks state until a REAL change occurs
                          → only transitions are "meaningful bits"
                          → 1000-sample series → typically 10–40 transitions
                          → store as (position, value) pairs: ≈ 40–80 bytes
                            vs 125 bytes packed-bits, vs 4000 bytes float32

TRADING INSIGHT — MACD crossover normalized by ATR:
  In trading you only act on a move when it is large relative to RECENT NOISE.
  STRIDE applies the same principle: only flip the output bit when the EMA
  momentum exceeds alpha × local ATR.  ATR is computed via a slow EMA of
  |Δx|, so it is immune to global inflation from past spikes.

2 HYPERPARAMETERS ONLY:
  alpha — sensitivity  (like RSI 70/30 threshold)
  fast  — EMA speed    (like MACD fast period ≈ 1/fast samples)

O(n) time, O(1) space per sample — suitable for ESP32 / MicroPython
"""

import numpy as np
from scipy.signal import lfilter
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  STRIDE Binarizer
# ─────────────────────────────────────────────────────────────────────────────

class STRIDEBinarizer:
    """
    STRIDE: Signal-TRaded Incremental Delta Encoder

    Per-sample algorithm (O(1)):
      1. EMA_fast ← fast * x + (1-fast) * EMA_fast    [fast momentum tracker]
      2. EMA_slow ← slow * x + (1-slow) * EMA_slow    [trend baseline]
      3. MACD     = EMA_fast − EMA_slow                [net momentum]
      4. ATR      ← slow * |Δx| + (1-slow) * ATR      [local noise floor]
      5. norm     = MACD / ATR                         [normalised momentum]
      6. Schmitt trigger on norm with threshold ±alpha  [hysteresis output]

    Parameters
    ----------
    alpha : float  [0.5 – 3.0], default 1.5
        Detection threshold in ATR units.
        Higher → fewer output bits, only strong spikes detected.
    fast  : float  [0.1 – 0.5], default 0.25
        Fast EMA factor (≈ 1/fast samples response time).
        slow = fast/3 is derived automatically (MACD 4:12 ratio analogue).
    """

    def __init__(self, alpha: float = 1.5, fast: float = 0.25):
        self.alpha = alpha
        self.fast  = fast
        self.slow  = fast / 3.0          # slow EMA ≈ 3× slower (MACD ratio)

    # ── Fast vectorised EMA ────────────────────────────────────────────────────

    @staticmethod
    def _ema(x: np.ndarray, a: float) -> np.ndarray:
        """IIR EMA initialised to x[0] — matches streaming step() exactly."""
        zi = np.array([(1.0 - a) * x[0]])
        y, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=zi)
        return y

    # ── Core encoder ──────────────────────────────────────────────────────────

    def _encode_1d(self, x: np.ndarray) -> np.ndarray:
        n   = len(x)
        x   = x.astype(np.float64)

        # 1–2. Fast & slow EMAs (vectorised via IIR)
        ef  = self._ema(x, self.fast)
        es  = self._ema(x, self.slow)

        # 3. MACD = momentum signal
        macd = ef - es

        # 4. ATR = EMA of |first-difference|  ← LOCAL noise floor, spike-immune
        #    Using slow EMA so ATR is stable and not thrown off by single spikes.
        absdiff = np.abs(np.diff(x, prepend=x[0]))
        atr     = self._ema(absdiff, self.slow) + 1e-8

        # 5. Normalised momentum: MACD in units of local ATR (like breakout z-score)
        norm = macd / atr

        # 6. Schmitt trigger (sequential — each output depends on previous state)
        #    Asymmetric band: spikes are upward (IoT attacks) → alpha upper
        #                     recovery is downward → alpha*0.5 lower (easier exit)
        bits  = np.empty(n, dtype=np.uint8)
        state = 0
        hi    =  self.alpha
        lo    = -self.alpha * 0.5
        for i in range(n):
            v = norm[i]
            if   v > hi:  state = 1   # attack / spike detected
            elif v < lo:  state = 0   # recovery to baseline
            # else: hold (hysteresis zone) → NO BIT CHANGE = fewer bits
            bits[i] = state

        return bits

    # ── Public API — sklearn-compatible ───────────────────────────────────────

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        """
        Returns uint8 bit array, same shape as X.
        X : (length,) or (n_series, length)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return self._encode_1d(X)
        return np.vstack([self._encode_1d(row) for row in X])

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.transform(X)

    # ── Compact / low-bit outputs ─────────────────────────────────────────────

    def transform_packed(self, X) -> np.ndarray:
        """
        Pack 8 bits → 1 byte.  8× memory vs uint8 full array.
        Output shape: (..., ceil(length/8))
        Perfect for ESP32 BLE / UART payload.
        """
        bits = self.transform(X)
        return np.packbits(bits, axis=-1)

    def transform_events(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ultra-compact: only state-transition positions and their new values.

        For a 1000-sample normal signal with 2 attacks:
          → ~8–16 events instead of 1000 bits  (60–125× fewer bytes)

        Returns
        -------
        positions : int array  — sample indices where bit changes
        values    : uint8 array — new bit value at each change
        """
        bits = self._encode_1d(np.asarray(x, dtype=np.float64))
        mask = np.concatenate([[True], np.diff(bits) != 0])
        idx  = np.where(mask)[0]
        return idx, bits[idx]

    # ── Streaming API — O(1) per sample ───────────────────────────────────────

    def reset_stream(self):
        """Call before feeding a new time-series stream."""
        self._prev  = None
        self._ef    = None     # fast EMA
        self._es    = None     # slow EMA
        self._atr   = None     # ATR (slow EMA of |Δx|)
        self._state = 0
        return self

    def step(self, x: float) -> int:
        """
        Feed one sample, return one bit.  O(1).
        6 multiplies + 6 adds — runs on ESP32 / MicroPython.
        Output is identical to batch transform().
        """
        if self._prev is None:
            self._ef    = x
            self._es    = x
            self._atr   = 1e-8    # non-zero to avoid /0 on first step
            self._prev  = x
            return 0

        # EMA updates
        self._ef += self.fast * (x - self._ef)
        self._es += self.slow * (x - self._es)
        macd = self._ef - self._es

        # ATR update: slow EMA of |delta|
        absdiff = abs(x - self._prev)
        self._atr += self.slow * (absdiff - self._atr)
        self._prev = x

        norm = macd / (self._atr + 1e-8)

        if   norm >  self.alpha:           self._state = 1
        elif norm < -(self.alpha * 0.5):   self._state = 0

        return self._state

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def sparsity(self, X) -> float:
        """Fraction of zero bits (higher = sparser = fewer effective bits)."""
        bits = self.transform(np.asarray(X, dtype=np.float64))
        return float(1.0 - bits.mean())

    def compression_ratio(self, x) -> float:
        """Ratio: (N samples) / (number of state transitions)."""
        pos, _ = self.transform_events(np.asarray(x))
        return len(x) / max(len(pos), 1)

    def __repr__(self):
        return f"STRIDEBinarizer(alpha={self.alpha}, fast={self.fast})"


# ─────────────────────────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    n = 300

    # ── Synthetic IoT attack signal ───────────────────────────────────────────
    signal = np.random.normal(0, 0.3, n)     # baseline Gaussian noise
    signal[80:92]   += 6.0                   # attack 1: sudden spike
    signal[150:155] += 4.0                   # attack 2: short burst
    signal[220:240] += 2.5                   # attack 3: sustained anomaly

    bz = STRIDEBinarizer(alpha=1.5, fast=0.25)

    bits         = bz.transform(signal)
    packed       = bz.transform_packed(signal)
    pos, vals    = bz.transform_events(signal)

    print("=" * 58)
    print("  STRIDE Binarizer — Bit Budget")
    print("=" * 58)
    print(f"  Raw float32 signal : {n} × 4 bytes  = {n*4:>5} bytes")
    print(f"  Full uint8 bit arr : {n} × 1 bit    = {n//8:>5} bytes")
    print(f"  Packed bytes       :                  {len(packed):>5} bytes")
    print(f"  Events (pos+val)   : {len(pos)} events × 2     = {len(pos)*2:>5} bytes")
    print(f"\n  Sparsity  : {bz.sparsity(signal)*100:.1f}%  zeros")
    print(f"  Compress  : {bz.compression_ratio(signal):.1f}×  vs 1-bit-per-sample")

    # ── Verify streaming == batch ─────────────────────────────────────────────
    bz.reset_stream()
    stream_bits = np.array([bz.step(float(v)) for v in signal], dtype=np.uint8)
    match = np.array_equal(bits, stream_bits)
    print(f"\n  Streaming == Batch : {'OK' if match else 'FAIL'}")

    # ── Attack detection ──────────────────────────────────────────────────────
    print(f"\n  Attack 1 (t=80–91)  detected : {any(75  <= p <= 95  for p in pos)}")
    print(f"  Attack 2 (t=150–54) detected : {any(145 <= p <= 160 for p in pos)}")
    print(f"  Attack 3 (t=220–39) detected : {any(215 <= p <= 245 for p in pos)}")
    print(f"\n  Transition positions : {pos}")
    print(f"  Transition values    : {vals}")
    print("=" * 58)
