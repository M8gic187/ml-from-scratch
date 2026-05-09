"""Linear Regression via batch gradient descent."""

import numpy as np


class LinearRegression:
    """Ordinary least squares linear regression trained with gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates.
    n_iterations : int
        Number of gradient descent steps.
    fit_intercept : bool
        Whether to include a bias term.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Train the model on data X with targets y.

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

            grad_w = (2 / n_samples) * X.T @ error
            grad_b = (2 / n_samples) * np.sum(error) if self.fit_intercept else 0.0

            self.weights_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * grad_b

            self.loss_history_.append(float(np.mean(error ** 2)))

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
