"""ThermometerEncoder adapter for TMU's official StandardBinarizer.

This wraps ``tmu.preprocessing.standard_binarizer.binarizer.StandardBinarizer``
so it can be benchmarked side-by-side with our online encoders via the same
registry / harness. TMU's StandardBinarizer is the canonical (offline)
preprocessor distributed with the Tsetlin Machine Unified library.
"""

from __future__ import annotations

import numpy as np

from .base import ThermometerEncoder


class TMUStandardBinarizer(ThermometerEncoder):
    def __init__(self, max_bits_per_feature: int = 8):
        super().__init__(K=max_bits_per_feature, name=f"TMU-Std-K{max_bits_per_feature}")
        self.max_bits_per_feature = int(max_bits_per_feature)
        self._enc = None

    def fit(self, X: np.ndarray) -> "TMUStandardBinarizer":
        from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer

        X = np.asarray(X, dtype=np.float64)
        self.n_features = X.shape[1]
        self._enc = StandardBinarizer(max_bits_per_feature=self.max_bits_per_feature)
        self._enc.fit(X)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self._enc.transform(X), dtype=np.uint8)

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return self.transform(x)[0]

    def get_n_output_bits(self) -> int:
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        b = self.transform(np.zeros((1, self.n_features))).shape[1]
        return int(b)
