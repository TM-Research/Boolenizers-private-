"""Quantile Binning + Espresso Logic Minimization (QB-Esp) encoder.

Discretizes features into quantile bins then applies logic minimization
to reduce redundant columns. This is a simplified version without actual
Espresso (uses correlation-based column removal instead).
"""

import numpy as np
from .base import ThermometerEncoder


class QBEspresso(ThermometerEncoder):
    """
    Quantile Binning + Espresso Logic Minimization.

    First discretizes into quantile bins, then removes redundant columns
    using correlation-based logic minimization (simplified Espresso).

    Note: This is a simplified implementation that approximates Espresso
    by removing highly correlated columns instead of full logic minimization.
    """

    def __init__(self, K: int = 8, correlation_threshold: float = 0.95):
        """
        Initialize QB-Espresso encoder.

        Args:
            K: Initial number of quantile bins per feature
            correlation_threshold: Threshold for removing redundant columns
        """
        super().__init__(K=K, name="QB-Espresso")
        self.correlation_threshold = correlation_threshold
        self.thresholds = None
        self.selected_columns = None

    def fit(self, X: np.ndarray) -> 'QBEspresso':
        """
        Fit by computing quantiles and selecting non-redundant columns.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = X.shape[1]

        # Step 1: Compute quantile thresholds
        self.thresholds = np.zeros((self.n_features, self.K))

        for i in range(self.n_features):
            quantiles = [(k + 1) / (self.K + 1) for k in range(self.K)]
            self.thresholds[i] = np.quantile(X[:, i], quantiles)

        # Step 2: Encode training data
        n_samples = X.shape[0]
        encoded_full = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)

        for i in range(n_samples):
            for feat_idx in range(self.n_features):
                for k in range(self.K):
                    bit_idx = feat_idx * self.K + k
                    encoded_full[i, bit_idx] = 1 if X[i, feat_idx] >= self.thresholds[feat_idx, k] else 0

        # Step 3: Logic minimization (simplified - remove highly correlated columns)
        self.selected_columns = self._minimize_columns(encoded_full)

        self.fitted = True
        return self

    def _minimize_columns(self, encoded: np.ndarray) -> np.ndarray:
        """
        Simplified logic minimization using variance and sampling.

        Instead of expensive full correlation matrix, use:
        1. Remove low-variance columns (constant or near-constant)
        2. Sample-based redundancy detection

        Args:
            encoded: Encoded data of shape (n_samples, n_features * K)

        Returns:
            Array of selected column indices
        """
        n_cols = encoded.shape[1]

        # Step 1: Remove low-variance columns
        variances = np.var(encoded, axis=0)
        var_threshold = 0.01  # Keep columns with variance > 0.01
        high_var_cols = np.where(variances > var_threshold)[0]

        # Step 2: Sample-based redundancy detection (much faster)
        # Only check correlation for a random sample of column pairs
        selected = []
        used = set()

        max_comparisons = min(5000, len(high_var_cols) * 10)  # Limit comparisons
        comparison_count = 0

        for i in high_var_cols:
            if i in used or comparison_count >= max_comparisons:
                if i not in used:
                    selected.append(i)
                continue

            selected.append(i)

            # Sample a subset of remaining columns to check
            remaining = [j for j in high_var_cols if j > i and j not in used]
            sample_size = min(20, len(remaining))  # Check at most 20 columns

            if remaining:
                sample_indices = np.random.choice(remaining, size=min(sample_size, len(remaining)), replace=False)

                for j in sample_indices:
                    comparison_count += 1

                    # Quick correlation check
                    correlation = np.corrcoef(encoded[:, i], encoded[:, j])[0, 1]

                    if not np.isnan(correlation) and abs(correlation) > self.correlation_threshold:
                        used.add(j)

        return np.array(sorted(selected))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to minimized thermometer codes.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            Encoded data of shape (n_samples, len(selected_columns))
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        n_samples = X.shape[0]
        # First encode fully
        encoded_full = np.zeros((n_samples, self.n_features * self.K), dtype=np.uint8)

        for i in range(n_samples):
            for feat_idx in range(self.n_features):
                for k in range(self.K):
                    bit_idx = feat_idx * self.K + k
                    encoded_full[i, bit_idx] = 1 if X[i, feat_idx] >= self.thresholds[feat_idx, k] else 0

        # Then select minimized columns
        return encoded_full[:, self.selected_columns]

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample.

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Encoded sample with minimized columns
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")

        # First encode fully
        bits_full = np.zeros(self.n_features * self.K, dtype=np.uint8)

        for i in range(self.n_features):
            for k in range(self.K):
                bit_idx = i * self.K + k
                bits_full[bit_idx] = 1 if x[i] >= self.thresholds[i, k] else 0

        # Then select minimized columns
        return bits_full[self.selected_columns]

    def get_n_output_bits(self) -> int:
        """Get total number of output bits after minimization."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        return len(self.selected_columns)

