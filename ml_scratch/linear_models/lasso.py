"""Lasso Regression (L1-regularised least squares) via coordinate descent."""

import numpy as np


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Element-wise soft-thresholding operator used in Lasso coordinate descent."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


class Lasso:
    """Lasso regression trained with coordinate descent.

    The objective minimised is::

        (1 / (2*n)) * ||y - Xw - b||^2  +  alpha * ||w||_1

    Coordinate descent cycles over each weight and applies the
    soft-thresholding closed-form update, which naturally drives
    irrelevant coefficients to exactly zero.

    Parameters
    ----------
    alpha : float
        Regularisation strength. Larger values produce sparser solutions.
    n_iterations : int
        Number of full coordinate-descent passes over all features.
    fit_intercept : bool
        Whether to include a bias term (not regularised).
    tol : float
        Convergence tolerance; training stops when the maximum weight
        change across a full pass is smaller than ``tol``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.tol = tol

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Lasso":
        """Fit the model using coordinate descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        # Precompute column norms squared for efficiency
        col_norms_sq = np.sum(X ** 2, axis=0)  # shape (n_features,)

        for iteration in range(self.n_iterations):
            weights_old = self.weights_.copy()

            # Update intercept (unregularised)
            if self.fit_intercept:
                residual = y - X @ self.weights_
                self.bias_ = float(np.mean(residual))

            # Coordinate descent update for each weight
            for j in range(n_features):
                if col_norms_sq[j] == 0.0:
                    continue
                # Partial residual excluding contribution of feature j
                r_j = y - self.bias_ - X @ self.weights_ + X[:, j] * self.weights_[j]
                rho_j = X[:, j] @ r_j / n_samples
                self.weights_[j] = float(
                    _soft_threshold(np.array([rho_j]), self.alpha)[0]
                    / (col_norms_sq[j] / n_samples)
                )

            self.n_iter_ = iteration + 1
            if np.max(np.abs(self.weights_ - weights_old)) < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values for X."""
        self._check_fitted()
        return np.asarray(X, dtype=float) @ self.weights_ + self.bias_

    def _check_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
