"""Wrapper for GLADE (Gap-aware Lightweight Adaptive Discretisation Engine).

GLADE is an OFFLINE batch encoder — its ``fit`` examines the entire training
set (or a 30k-sample subsample) and learns adaptive per-feature thresholds:

    * Categorical features (n_unique <= n_bins): one threshold per unique value
    * Numeric features (n_unique > n_bins): hybrid budget allocator
      - zero-inflated columns (>30% zeros) get a log2-scaled smaller budget
      - thresholds chosen at evenly-spaced percentiles
      - "structural gap" snap when max gap > gap_ratio × median gap
      - local perturbation around each edge picks the position with highest
        Bernoulli variance p(1-p)

This wrapper makes GLADE conform to the ThermometerEncoder protocol used by
our benchmark harness so it can be compared directly with OGB / OQTB / OQSB
/ KBinsDiscretizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import ThermometerEncoder


class GLADEBooleanizer:
    """Gap-aware Lightweight Adaptive Discretisation Engine.

    The transformer learns threshold literals from a numeric matrix and turns
    each row into Boolean features:

    ``bit_j = X[feature_index_j] >= threshold_j``.

    ``transform(..., pack_bits=True)`` returns the same Boolean matrix packed
    along the feature axis with ``numpy.packbits``.
    """

    def __init__(self, n_bins: int = 15, gap_ratio: float = 5.0):
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        if not (gap_ratio > 0):
            raise ValueError("gap_ratio must be > 0 (use math.inf to disable snap)")
        self.n_bins = int(n_bins)
        self.gap_ratio = float(gap_ratio)
        self._cat_cols: list[int] = []
        self._num_cols: list[int] = []
        self._cat_edges: list[np.ndarray] = []
        self._num_edges: list[np.ndarray] = []
        self._feat_idx: np.ndarray | None = None
        self._thresh: np.ndarray | None = None
        self._n_features = 0
        self._fitted = False

    def fit(self, X: np.ndarray) -> "GLADEBooleanizer":
        X = self._as_2d_float(X)
        n_rows, n_features = X.shape
        self._n_features = int(n_features)

        sample_limit = 30_000
        if n_rows > sample_limit:
            rng = np.random.RandomState(42)
            X_sample = X[rng.choice(n_rows, sample_limit, replace=False)]
        else:
            X_sample = X

        self._cat_cols = []
        self._num_cols = []
        for feature in range(n_features):
            n_unique = np.unique(X_sample[:, feature]).size
            if n_unique <= 1:
                continue
            if n_unique <= self.n_bins:
                self._cat_cols.append(feature)
            else:
                self._num_cols.append(feature)

        self._cat_edges = [np.sort(np.unique(X[:, j])) for j in self._cat_cols]
        self._num_edges = [
            self._find_edges(X_sample[:, j], self._hybrid_budget(X_sample[:, j]))
            for j in self._num_cols
        ]

        feature_indices: list[int] = []
        thresholds: list[float] = []

        for edges, feature in zip(self._cat_edges, self._cat_cols, strict=True):
            for value in edges[1:]:
                feature_indices.append(feature)
                thresholds.append(float(value))

        for edges, feature in zip(self._num_edges, self._num_cols, strict=True):
            for value in edges:
                feature_indices.append(feature)
                thresholds.append(float(value))

        self._feat_idx = np.asarray(feature_indices, dtype=np.int32)
        self._thresh = np.asarray(thresholds, dtype=np.float32)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, pack_bits: bool = False) -> np.ndarray:
        self._require_fitted()
        X = self._as_2d_float(X)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"expected {self._n_features} features, got {X.shape[1]}"
            )

        feat_idx = self._feature_indices
        thresholds = self.thresholds
        n_rows = X.shape[0]
        n_bits = thresholds.size
        out = np.empty((n_rows, n_bits), dtype=np.uint8)

        chunk_size = 64
        for start in range(0, n_bits, chunk_size):
            end = min(start + chunk_size, n_bits)
            np.greater_equal(
                X[:, feat_idx[start:end]],
                thresholds[start:end][np.newaxis, :],
                out=out[:, start:end],
            )

        if not pack_bits:
            return out

        pad = (-n_bits) % 8
        if pad:
            out = np.pad(out, ((0, 0), (0, pad)), mode="constant")
        return np.packbits(out, axis=1)

    def fit_transform(self, X: np.ndarray, pack_bits: bool = False) -> np.ndarray:
        return self.fit(X).transform(X, pack_bits=pack_bits)

    @property
    def n_bits(self) -> int:
        return int(self.thresholds.size) if self._fitted else 0

    @property
    def n_features_in(self) -> int:
        return int(self._n_features)

    @property
    def feature_indices(self) -> np.ndarray:
        return self._feature_indices.copy()

    @property
    def thresholds(self) -> np.ndarray:
        self._require_fitted()
        assert self._thresh is not None
        return self._thresh

    @property
    def _feature_indices(self) -> np.ndarray:
        self._require_fitted()
        assert self._feat_idx is not None
        return self._feat_idx

    def _hybrid_budget(self, col: np.ndarray) -> int:
        n_unique = np.unique(col).size
        if n_unique <= 1:
            return 0
        if n_unique <= self.n_bins:
            return max(1, n_unique - 1)

        zero_fraction = float(np.mean(col == 0))
        if zero_fraction > 0.3:
            nonzero = col[col != 0]
            nonzero_unique = np.unique(nonzero).size if nonzero.size else n_unique
            density = max(1.0 - zero_fraction, 0.01)
            effective = max(nonzero_unique * density ** 2, 2.0)
            budget = int(np.ceil(np.log2(effective)) + 2)
            return max(1, min(budget, self.n_bins))

        return self.n_bins

    def _find_edges(self, col: np.ndarray, n_edges: int) -> np.ndarray:
        unique = np.sort(np.unique(col))
        if unique.size <= 1 or n_edges <= 0:
            return np.empty(0, dtype=np.float64)
        if unique.size <= n_edges:
            return (unique[:-1] + unique[1:]) / 2.0

        prefix: list[float] = []
        work = col
        if float(np.mean(col == 0)) > 0.3:
            nonzero = col[col > 0]
            if nonzero.size > 10 and np.unique(nonzero).size > 1:
                prefix = [float(np.min(nonzero)) * 0.5]
                work = nonzero

        remaining = n_edges - len(prefix)
        if remaining <= 0:
            return np.asarray(prefix[:n_edges], dtype=np.float64)

        work_unique = np.sort(np.unique(work))
        if work_unique.size <= remaining:
            edges = (work_unique[:-1] + work_unique[1:]) / 2.0
            return np.sort(np.unique(prefix + edges.tolist()))[:n_edges]

        percentiles = np.linspace(
            100 / (remaining + 1),
            100 * remaining / (remaining + 1),
            remaining,
        )
        raw_edges = np.percentile(work, percentiles)
        edges = self._snap_structural_gaps(work_unique, raw_edges)
        edges = self._local_perturb(work, edges)
        return np.sort(np.unique(prefix + edges.tolist()))[:n_edges]

    def _snap_structural_gaps(
        self, unique_values: np.ndarray, raw_edges: np.ndarray
    ) -> np.ndarray:
        gaps = np.diff(unique_values)
        median_gap = np.median(gaps)
        max_gap = np.max(gaps)
        has_structural_gap = (median_gap > 0) and (max_gap > self.gap_ratio * median_gap)
        if not has_structural_gap:
            return np.unique(raw_edges)

        midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
        idx = np.searchsorted(midpoints, raw_edges).clip(0, midpoints.size - 1)
        left = (idx - 1).clip(0)
        right_distance = np.abs(midpoints[idx] - raw_edges)
        left_distance = np.abs(midpoints[left] - raw_edges)
        best = np.where(left_distance < right_distance, left, idx)
        return np.unique(midpoints[best])

    @staticmethod
    def _local_perturb(col: np.ndarray, edges: np.ndarray) -> np.ndarray:
        if edges.size == 0:
            return edges

        sorted_edges = np.sort(np.unique(edges))
        lo = float(col.min())
        hi = float(col.max())
        out: list[float] = []

        for i, edge in enumerate(sorted_edges):
            left = sorted_edges[i - 1] if i > 0 else lo
            right = sorted_edges[i + 1] if i < sorted_edges.size - 1 else hi
            candidates = [
                edge - (edge - left) * 0.25,
                edge,
                edge + (right - edge) * 0.25,
            ]

            best_edge = float(edge)
            best_variance = -1.0
            for candidate in candidates:
                p = float(np.mean(col >= candidate))
                variance = p * (1.0 - p)
                if variance > best_variance:
                    best_variance = variance
                    best_edge = float(candidate)
            out.append(best_edge)

        return np.asarray(out, dtype=np.float64)

    @staticmethod
    def _as_2d_float(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
        return arr

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("GLADEBooleanizer is not fitted")


GLADEv2 = GLADEBooleanizer


# -------- ThermometerEncoder adapter -----------------------------------------
class GLADEEncoder(ThermometerEncoder):
    """ThermometerEncoder adapter so GLADE can run inside the existing
    benchmark harness next to OGB / OQTB / OQSB / KBinsDiscretizer."""

    def __init__(self, n_bins: int = 15, gap_ratio: float = 5.0):
        super().__init__(K=n_bins, name=f"GLADE-K{n_bins}")
        self._glade = GLADEBooleanizer(n_bins=n_bins, gap_ratio=gap_ratio)

    def fit(self, X: np.ndarray) -> "GLADEEncoder":
        X = np.asarray(X, dtype=np.float64)
        self.n_features = X.shape[1]
        self._glade.fit(X)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        return self._glade.transform(X).astype(np.uint8)

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return self._glade.transform(x)[0].astype(np.uint8)

    def get_n_output_bits(self) -> int:
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return int(self._glade.n_bits)
