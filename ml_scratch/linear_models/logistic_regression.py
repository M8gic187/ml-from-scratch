"""Logistic Regression for binary classification via gradient descent."""

import numpy as np


class LogisticRegression:
    """Binary logistic regression trained with gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient updates.
    n_iterations : int
        Number of gradient descent steps.
    threshold : float
        Decision threshold for converting probabilities to class labels.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        threshold: float = 0.5,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit model to binary-labelled data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,), values in {0, 1}

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
            y_pred = self._sigmoid(X @ self.weights_ + self.bias_)

            # Clamp predictions to avoid log(0)
            y_pred_clamped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(
                y * np.log(y_pred_clamped) + (1 - y) * np.log(1 - y_pred_clamped)
            )
            self.loss_history_.append(float(loss))

            error = y_pred - y
            self.weights_ -= self.learning_rate * (X.T @ error) / n_samples
            self.bias_ -= self.learning_rate * np.mean(error)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-1 probabilities for each sample."""
        self._check_fitted()
        return self._sigmoid(
            np.asarray(X, dtype=float) @ self.weights_ + self.bias_
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (0 or 1)."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid: clip avoids exp overflow without branching.
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _check_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
