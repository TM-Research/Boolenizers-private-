"""
KBinsThermometer
================
Thin wrapper around ``sklearn.preprocessing.KBinsDiscretizer`` that converts
its ordinal bin labels into a thermometer code (K bits per feature, with
bit ``k`` set when the bin index ``≥ k+1``).

This is an **offline batch** encoder — ``fit`` calls ``KBinsDiscretizer.fit``
which examines all training rows at once. We include it here purely as a
**baseline** for the online encoder family.

Two strategies are exposed:
  * ``KBinsThermometer-Quantile`` (strategy="quantile")
  * ``KBinsThermometer-Uniform``  (strategy="uniform")

Both use ``n_bins = K + 1`` so K thermometer bits are emitted per feature
(matching the per-feature bit count of OQTB and OQSB).
"""

from __future__ import annotations

import numpy as np

from .base import ThermometerEncoder


class KBinsThermometer(ThermometerEncoder):
    def __init__(self, K: int = 8, strategy: str = "quantile",
                 encode: str = "ordinal", subsample: int = 200_000):
        super().__init__(K=K, name=f"KBins-{strategy.capitalize()}")
        if strategy not in ("quantile", "uniform", "kmeans"):
            raise ValueError("strategy must be 'quantile', 'uniform' or 'kmeans'")
        self.strategy = strategy
        self.subsample = int(subsample)
        self._n_bins = K + 1  # K thresholds → K+1 bins, K thermometer bits
        self._kbd = None

    def fit(self, X: np.ndarray) -> "KBinsThermometer":
        from sklearn.preprocessing import KBinsDiscretizer

        X = np.asarray(X, dtype=np.float64)
        self.n_features = X.shape[1]
        # KBinsDiscretizer in recent sklearn accepts ``subsample`` to keep fit
        # tractable on multi-million-row inputs (it raises a warning otherwise
        # under strategy='quantile'). Old versions don't have the kwarg.
        try:
            self._kbd = KBinsDiscretizer(
                n_bins=self._n_bins,
                encode="ordinal",
                strategy=self.strategy,
                subsample=self.subsample if self.strategy == "quantile" else None,
            )
        except TypeError:
            self._kbd = KBinsDiscretizer(
                n_bins=self._n_bins,
                encode="ordinal",
                strategy=self.strategy,
            )
        self._kbd.fit(X)
        # Record actual n_bins per feature in case sklearn coalesced bins.
        self._actual_n_bins = np.asarray(self._kbd.n_bins_, dtype=np.int64)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        # Ordinal bins ∈ {0, ..., n_bins-1}
        ordinal = np.asarray(self._kbd.transform(X), dtype=np.int64)
        n_samples, n_features = ordinal.shape
        out = np.zeros((n_samples, n_features * self.K), dtype=np.uint8)
        # Thermometer: bit k = 1 iff ordinal >= k+1. Each feature emits exactly
        # K bits (we clip to K even when sklearn returns fewer bins).
        for j in range(n_features):
            col = ordinal[:, j]
            for k in range(self.K):
                out[:, j * self.K + k] = (col >= (k + 1)).astype(np.uint8)
        return out

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        return self.transform(np.asarray(x, dtype=np.float64).reshape(1, -1))[0]

    def get_n_output_bits(self) -> int:
        if self.n_features is None:
            raise ValueError("Encoder must be fitted first")
        return self.n_features * self.K
