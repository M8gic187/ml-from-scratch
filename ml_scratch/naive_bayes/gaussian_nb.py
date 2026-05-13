"""Gaussian Naive Bayes classifier for multi-class classification."""

from __future__ import annotations

import numpy as np


class GaussianNB:
    """Gaussian Naive Bayes for multi-class classification.

    Assumes that each feature is conditionally independent given the class
    label, and that the likelihood of each feature follows a Gaussian
    distribution. Classification is performed in log-space for numerical
    stability.

    Parameters
    ----------
    var_smoothing : float
        A fraction of the largest per-feature variance added to all
        variances to prevent division-by-zero on constant features.
        Mirrors scikit-learn's default behaviour.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

        self.classes_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        # Per-class feature means and variances, shape (n_classes, n_features)
        self.theta_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        """Estimate class priors and per-class Gaussian parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.log(counts / len(y))
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.theta_[i] = X_cls.mean(axis=0)
            self.sigma_[i] = X_cls.var(axis=0)

        # Smoothing: add a fraction of the global maximum variance
        max_var = self.sigma_.max() if self.sigma_.size > 0 else 1.0
        self.sigma_ += self.var_smoothing * max_var

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return un-normalised log-posterior for each class.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        log_proba = np.empty((len(X), n_classes))

        for i in range(n_classes):
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.sigma_[i])
                + (X - self.theta_[i]) ** 2 / self.sigma_[i],
                axis=1,
            )
            log_proba[:, i] = self.class_log_prior_[i] + log_likelihood

        return log_proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalised posterior probabilities for each class.

        Applies a numerically stable softmax over the log-posteriors.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        log_p = self.predict_log_proba(X)
        log_p -= log_p.max(axis=1, keepdims=True)   # shift for stability
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
