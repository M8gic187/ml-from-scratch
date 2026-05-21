"""Ridge Regression (L2-regularised least squares)."""

import numpy as np


class Ridge:
    """Ordinary least squares with L2 penalty, trained via gradient descent.

    The objective minimised is::

        (1/n) * ||y - Xw - b||^2  +  alpha * ||w||^2

    Parameters
    ----------
    alpha : float
        Regularisation strength. Larger values force weights closer to zero.
    learning_rate : float
        Step size for gradient descent updates.
    n_iterations : int
        Number of gradient descent steps.
    fit_intercept : bool
        Whether to include a bias term (not regularised).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
    ) -> None:
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        """Fit the model.

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
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            y_pred = self._forward(X)
            error = y_pred - y

            grad_w = (2 / n_samples) * (X.T @ error) + 2 * self.alpha * self.weights_
            grad_b = (2 / n_samples) * np.sum(error) if self.fit_intercept else 0.0

            self.weights_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * grad_b

            mse = float(np.mean(error ** 2))
            reg = float(self.alpha * np.dot(self.weights_, self.weights_))
            self.loss_history_.append(mse + reg)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values for X."""
        self._check_fitted()
        return self._forward(np.asarray(X, dtype=float))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights_ + self.bias_

    def _check_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
