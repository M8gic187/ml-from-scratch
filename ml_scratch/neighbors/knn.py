"""K-Nearest Neighbors for classification and regression."""

from collections import Counter

import numpy as np


class KNNClassifier:
    """K-Nearest Neighbors classifier.

    Supports uniform and distance-weighted voting, and both Euclidean and
    Manhattan distance metrics.

    Parameters
    ----------
    k : int
        Number of neighbors to consider.
    metric : {'euclidean', 'manhattan'}
        Distance metric used to find neighbors.
    weights : {'uniform', 'distance'}
        'uniform' gives equal vote to all k neighbors; 'distance' weights
        each neighbor's vote inversely by its distance.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        self.k = k
        self.metric = metric
        self.weights = weights

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for each sample in X."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> object:
        dists = self._distances(x)
        nn_idx = np.argsort(dists)[: self.k]
        nn_labels = self.y_train_[nn_idx]

        if self.weights == "uniform":
            return Counter(nn_labels.tolist()).most_common(1)[0][0]

        nn_dists = dists[nn_idx]
        with np.errstate(divide="ignore"):
            w = np.where(nn_dists == 0.0, np.inf, 1.0 / nn_dists)
        classes = np.unique(nn_labels)
        scores = {c: w[nn_labels == c].sum() for c in classes}
        return max(scores, key=scores.get)

    def _distances(self, x: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
        if self.metric == "manhattan":
            return np.sum(np.abs(self.X_train_ - x), axis=1)
        raise ValueError(f"Unknown metric '{self.metric}'. Use 'euclidean' or 'manhattan'.")

    def _check_fitted(self) -> None:
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict().")


class KNNRegressor:
    """K-Nearest Neighbors regressor.

    Predicts the (weighted) mean of the k nearest neighbors' target values.

    Parameters
    ----------
    k : int
        Number of neighbors to consider.
    metric : {'euclidean', 'manhattan'}
        Distance metric used to find neighbors.
    weights : {'uniform', 'distance'}
        'uniform' averages targets equally; 'distance' weights by inverse
        distance so closer neighbors contribute more.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        self.k = k
        self.metric = metric
        self.weights = weights

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        """Store training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values for each sample in X."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> float:
        dists = self._distances(x)
        nn_idx = np.argsort(dists)[: self.k]
        nn_targets = self.y_train_[nn_idx]

        if self.weights == "uniform":
            return float(np.mean(nn_targets))

        nn_dists = dists[nn_idx]
        with np.errstate(divide="ignore"):
            w = np.where(nn_dists == 0.0, np.inf, 1.0 / nn_dists)
        # If any neighbor is an exact match, only those votes count.
        if np.any(np.isinf(w)):
            w = np.isinf(w).astype(float)
        return float(np.average(nn_targets, weights=w))

    def _distances(self, x: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
        if self.metric == "manhattan":
            return np.sum(np.abs(self.X_train_ - x), axis=1)
        raise ValueError(f"Unknown metric '{self.metric}'. Use 'euclidean' or 'manhattan'.")

    def _check_fitted(self) -> None:
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict().")
