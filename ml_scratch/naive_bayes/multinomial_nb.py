"""Multinomial Naive Bayes classifier for discrete count features."""

from __future__ import annotations

import numpy as np


class MultinomialNB:
    """Multinomial Naive Bayes for multi-class classification.

    Models each feature as a discrete multinomial random variable.
    Designed for non-negative integer (or frequency) features such as
    word counts in a bag-of-words representation.  Classification is
    performed entirely in log-space for numerical stability.

    Parameters
    ----------
    alpha : float
        Additive (Laplace / Lidstone) smoothing parameter.  ``alpha=1``
        gives classic Laplace smoothing; ``alpha=0`` disables smoothing
        (use only when every feature is guaranteed to appear in every
        class during training).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

        self.classes_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        # Log-probability of each feature given a class, shape (n_classes, n_features)
        self.feature_log_prob_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        """Estimate class priors and per-class feature log-probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Non-negative feature matrix (e.g. word counts).
        y : ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative feature values.")

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / len(y))

        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            # Sum of each feature across all class samples + smoothing
            feature_counts = X_cls.sum(axis=0) + self.alpha
            # Total count across all features for this class + smoothing
            total_count = feature_counts.sum()
            self.feature_log_prob_[i] = np.log(feature_counts) - np.log(total_count)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return un-normalised log-posterior for each class.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        # (n_samples, n_classes) = X @ feature_log_prob_.T + class_log_prior_
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalised posterior probabilities for each class.

        Applies a numerically stable softmax over the log-posteriors.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        log_p = self.predict_log_proba(X)
        log_p -= log_p.max(axis=1, keepdims=True)
        proba = np.exp(log_p)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.classes_ is None:
            raise RuntimeError("Call fit() before predict().")
