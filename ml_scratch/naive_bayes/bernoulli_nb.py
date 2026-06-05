"""Bernoulli Naive Bayes classifier for binary feature vectors."""

from __future__ import annotations

import numpy as np


class BernoulliNB:
    """Bernoulli Naive Bayes for multi-class classification.

    Assumes each feature is a binary indicator (0 or 1).  Unlike
    MultinomialNB, this model explicitly accounts for features that are
    *absent* in a sample by including the complementary log-probability
    ``log(1 - P(x_i=1 | y=c))`` for every zero feature.  This makes it
    particularly effective for short or sparse binary documents.

    Parameters
    ----------
    alpha : float
        Additive (Laplace / Lidstone) smoothing applied to the per-class
        feature presence counts.
    binarize : float | None
        Threshold below which feature values are set to 0 and above which
        they are set to 1.  Pass ``None`` to skip binarization (inputs are
        assumed to already be binary).
    """

    def __init__(self, alpha: float = 1.0, binarize: float | None = 0.0) -> None:
        self.alpha = alpha
        self.binarize = binarize

        self.classes_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        # Log P(x_i=1 | y=c), shape (n_classes, n_features)
        self.feature_log_prob_: np.ndarray | None = None
        # Log P(x_i=0 | y=c) = log(1 - P(x_i=1 | y=c))
        self._neg_feature_log_prob: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNB":
        """Estimate class priors and per-class binary feature probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Binary feature matrix (binarized internally if ``binarize`` is set).
        y : ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        X = self._binarize(np.asarray(X, dtype=float))
        y = np.asarray(y)

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / len(y))

        n_classes = len(self.classes_)
        self.feature_log_prob_ = np.zeros((n_classes, X.shape[1]))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            n_cls = len(X_cls)
            # Count of samples where each feature is 1, with Laplace smoothing
            presence = X_cls.sum(axis=0) + self.alpha
            total = n_cls + 2.0 * self.alpha   # binary: 2 possible values
            self.feature_log_prob_[i] = np.log(presence) - np.log(total)

        self._neg_feature_log_prob = np.log1p(-np.exp(self.feature_log_prob_))
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return un-normalised log-posterior for each class.

        Sums log-likelihoods for both present (x=1) and absent (x=0)
        features, then adds the class log-prior.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = self._binarize(np.asarray(X, dtype=float))
        # Contribution from present features
        log_p_present = X @ self.feature_log_prob_.T
        # Contribution from absent features
        log_p_absent = (1.0 - X) @ self._neg_feature_log_prob.T
        return log_p_present + log_p_absent + self.class_log_prior_

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

    def _binarize(self, X: np.ndarray) -> np.ndarray:
        if self.binarize is not None:
            return (X > self.binarize).astype(float)
        return X

    def _check_fitted(self) -> None:
        if self.classes_ is None:
            raise RuntimeError("Call fit() before predict().")
