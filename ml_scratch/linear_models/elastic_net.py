"""ElasticNet Regression (combined L1 + L2 penalty) via coordinate descent."""

import numpy as np


class ElasticNet:
    """ElasticNet regression: a convex combination of Lasso and Ridge penalties.

    The objective minimised is::

        (1 / (2*n)) * ||y - Xw - b||^2
        +  alpha * l1_ratio * ||w||_1
        +  0.5 * alpha * (1 - l1_ratio) * ||w||^2

    When ``l1_ratio=1`` this reduces to Lasso; when ``l1_ratio=0`` it
    reduces to Ridge (but using the coordinate-descent solver).
    Setting ``0 < l1_ratio < 1`` blends both penalties, which often
    improves on pure Lasso when features are correlated.

    Parameters
    ----------
    alpha : float
        Overall regularisation strength.
    l1_ratio : float
        Mix between L1 (1.0) and L2 (0.0) penalty. Must be in [0, 1].
    n_iterations : int
        Maximum number of coordinate-descent passes.
    fit_intercept : bool
        Whether to include a bias term (not regularised).
    tol : float
        Convergence tolerance on the maximum weight change per pass.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        tol: float = 1e-4,
    ) -> None:
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.tol = tol

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNet":
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

        col_norms_sq = np.sum(X ** 2, axis=0)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)

        for iteration in range(self.n_iterations):
            weights_old = self.weights_.copy()

            if self.fit_intercept:
                residual = y - X @ self.weights_
                self.bias_ = float(np.mean(residual))

            for j in range(n_features):
                if col_norms_sq[j] == 0.0:
                    continue
                r_j = y - self.bias_ - X @ self.weights_ + X[:, j] * self.weights_[j]
                rho_j = X[:, j] @ r_j / n_samples

                # Soft-threshold for L1, then scale down by L2 term
                numerator = np.sign(rho_j) * max(abs(rho_j) - l1, 0.0)
                denominator = col_norms_sq[j] / n_samples + l2
                self.weights_[j] = numerator / denominator

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
