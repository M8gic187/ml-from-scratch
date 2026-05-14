"""Principal Component Analysis (PCA) via eigendecomposition of the covariance matrix."""

from __future__ import annotations

import numpy as np


class PCA:
    """Linear dimensionality reduction using Singular Value Decomposition.

    Centers the data, computes the covariance matrix, and projects it onto the
    top ``n_components`` eigenvectors (principal components).  The components
    are ordered by descending explained variance.

    Parameters
    ----------
    n_components : int | None
        Number of principal components to keep.  ``None`` keeps all components.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space (eigenvectors), ordered by explained
        variance descending.
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each selected component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each component.
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, used for centering.
    n_components_ : int
        Resolved number of components after fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_scratch.decomposition import PCA
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 5))
    >>> pca = PCA(n_components=2).fit(X)
    >>> X_reduced = pca.transform(X)   # shape (100, 2)
    >>> X_back = pca.inverse_transform(X_reduced)   # approximate reconstruction
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components

        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.n_components_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "PCA":
        """Compute principal components from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix (unbiased): (n_features, n_features)
        cov = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending order — reverse to descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]  # columns are eigenvectors

        n_keep = self.n_components if self.n_components is not None else n_features
        n_keep = min(n_keep, n_features)
        self.n_components_ = n_keep

        self.components_ = eigenvectors[:, :n_keep].T  # (n_components, n_features)
        self.explained_variance_ = eigenvalues[:n_keep]
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_var if total_var > 0 else np.zeros(n_keep)
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the principal components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and project in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Approximately reconstruct original-space data from reduced representation.

        Parameters
        ----------
        X_reduced : ndarray of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features)
        """
        self._check_fitted()
        X_reduced = np.asarray(X_reduced, dtype=float)
        return X_reduced @ self.components_ + self.mean_

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")
