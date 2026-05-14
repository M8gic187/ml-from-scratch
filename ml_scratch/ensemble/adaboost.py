"""AdaBoost: Adaptive Boosting for binary classification."""

from __future__ import annotations

import numpy as np


class _DecisionStump:
    """Single-split binary decision tree (depth-1) used as weak learner."""

    def __init__(self) -> None:
        self.feature_index: int = 0
        self.threshold: float = 0.0
        self.polarity: int = 1  # +1 or -1, flips the stump direction

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
    ) -> float:
        """Fit stump by minimising weighted misclassification error.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) with values in {-1, +1}
        sample_weight : ndarray of shape (n_samples,) — must sum to 1

        Returns
        -------
        float
            Weighted error of this stump on the training set.
        """
        n_samples, n_features = X.shape
        best_error = float("inf")

        for j in range(n_features):
            thresholds = np.unique(X[:, j])
            for thresh in thresholds:
                for polarity in (1, -1):
                    preds = np.where(polarity * X[:, j] < polarity * thresh, 1, -1)
                    err = np.dot(sample_weight, preds != y)
                    if err < best_error:
                        best_error = err
                        self.feature_index = j
                        self.threshold = thresh
                        self.polarity = polarity

        return best_error

    def predict(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.feature_index]
        return np.where(self.polarity * col < self.polarity * self.threshold, 1, -1)


class AdaBoostClassifier:
    """AdaBoost (Adaptive Boosting) for binary classification.

    Trains an ensemble of ``n_estimators`` decision stumps sequentially.
    Each stump focuses on the samples that its predecessors misclassified
    by re-weighting the training distribution.  Final predictions are a
    weighted majority vote of all stumps.

    Labels are internally mapped to {-1, +1} and back to the original
    classes on output, so any two-class input works.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (weak learners).
    learning_rate : float
        Scales each learner's contribution.  Smaller values require more
        estimators but typically generalise better.
    random_state : int | None
        Unused; kept for API consistency with other estimators.

    Attributes
    ----------
    estimators_ : list of _DecisionStump
        Fitted weak learners.
    estimator_weights_ : ndarray of shape (n_estimators,)
        Alpha coefficient for each learner (higher = better learner).
    estimator_errors_ : ndarray of shape (n_estimators,)
        Weighted training error for each learner.
    classes_ : ndarray of shape (2,)
        Original class labels encountered during fit.

    Examples
    --------
    >>> from ml_scratch.ensemble import AdaBoostClassifier
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 4))
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> clf = AdaBoostClassifier(n_estimators=50).fit(X, y)
    >>> acc = (clf.predict(X) == y).mean()
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.estimators_: list[_DecisionStump] = []
        self.estimator_weights_: np.ndarray | None = None
        self.estimator_errors_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """Train the boosted ensemble.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) — binary labels (any two values)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples = len(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"AdaBoost supports binary classification only; got {len(self.classes_)} classes."
            )

        # Map labels to {-1, +1}
        y_encoded = np.where(y == self.classes_[1], 1, -1).astype(float)

        weights = np.full(n_samples, 1.0 / n_samples)
        alphas = np.zeros(self.n_estimators)
        errors = np.zeros(self.n_estimators)
        self.estimators_ = []

        for t in range(self.n_estimators):
            stump = _DecisionStump()
            err = stump.fit(X, y_encoded, weights)

            # Clip to avoid division by zero and log(0)
            err = np.clip(err, 1e-10, 1 - 1e-10)
            alpha = self.learning_rate * 0.5 * np.log((1.0 - err) / err)

            preds = stump.predict(X)
            # Update and renormalise sample weights
            weights *= np.exp(-alpha * y_encoded * preds)
            weights /= weights.sum()

            self.estimators_.append(stump)
            alphas[t] = alpha
            errors[t] = err

        self.estimator_weights_ = alphas
        self.estimator_errors_ = errors
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the raw boosted score (positive → class 1, negative → class 0).

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        score = sum(
            alpha * stump.predict(X)
            for stump, alpha in zip(self.estimators_, self.estimator_weights_)
        )
        return score

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Returns
        -------
        ndarray of shape (n_samples,) with values from ``classes_``
        """
        score = self.decision_function(X)
        return np.where(score >= 0, self.classes_[1], self.classes_[0])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")
