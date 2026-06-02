"""VotingClassifier: ensemble of heterogeneous classifiers via majority or probability voting."""

from __future__ import annotations

import numpy as np


class VotingClassifier:
    """Soft- and hard-voting ensemble of heterogeneous classifiers.

    Combines the predictions of fitted base estimators, either by majority
    vote (hard) or by averaging predicted class probabilities (soft).

    Each estimator must expose ``fit(X, y)`` and ``predict(X)``.  Soft voting
    additionally requires ``predict_proba(X)`` on every estimator.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Name–estimator pairs.  Names must be unique and are used only for
        identification; they do not affect training.
    voting : {'hard', 'soft'}
        Aggregation strategy.

        - ``'hard'``: the class label predicted by the strict (optionally
          weighted) majority of estimators wins.
        - ``'soft'``: the class with the highest (optionally weighted) average
          posterior probability wins.  Requires every estimator to implement
          ``predict_proba``.
    weights : array-like of shape (n_estimators,) or None
        Relative contribution of each estimator.  ``None`` gives equal weight
        to every estimator.

    Attributes
    ----------
    estimators_ : list
        Fitted estimator objects in the order they were passed.
    classes_ : ndarray of shape (n_classes,)
        Sorted unique class labels seen during ``fit``.

    Examples
    --------
    >>> from ml_scratch.ensemble import VotingClassifier
    >>> from ml_scratch.tree import DecisionTreeClassifier
    >>> from ml_scratch.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((120, 4))
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> clf = VotingClassifier(
    ...     [('dt', DecisionTreeClassifier(max_depth=3, random_state=0)),
    ...      ('rf', RandomForestClassifier(n_estimators=20, random_state=0))],
    ...     voting='soft',
    ... ).fit(X, y)
    >>> acc = (clf.predict(X) == y).mean()
    """

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        voting: str = "hard",
        weights: list[float] | np.ndarray | None = None,
    ) -> None:
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

        self.estimators_: list = []
        self.classes_: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingClassifier":
        """Fit all base estimators on the full training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        if self.voting not in ("hard", "soft"):
            raise ValueError(
                f"voting must be 'hard' or 'soft'; got '{self.voting}'."
            )
        if len(self.estimators) == 0:
            raise ValueError("estimators must not be empty.")

        names = [n for n, _ in self.estimators]
        if len(names) != len(set(names)):
            raise ValueError("Estimator names must be unique.")

        if self.weights is not None:
            w = np.asarray(self.weights, dtype=float)
            if w.shape[0] != len(self.estimators):
                raise ValueError(
                    f"len(weights) = {w.shape[0]} but len(estimators) = {len(self.estimators)}."
                )
            self._weights = w
        else:
            self._weights = None

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self.estimators_ = []
        for _, est in self.estimators:
            est.fit(X, y)
            self.estimators_.append(est)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)

        if self.voting == "soft":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

        return self._hard_predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged class probabilities (soft voting only).

        Probabilities from each estimator are averaged — or weighted-averaged
        when *weights* are supplied.  Columns follow the global ``classes_``
        ordering; estimators that were trained on a class-aligned dataset are
        handled correctly via column remapping.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)

        Raises
        ------
        AttributeError
            If ``voting != 'soft'``.
        """
        self._check_fitted()
        if self.voting != "soft":
            raise AttributeError(
                "predict_proba is only available when voting='soft'."
            )

        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        avg = np.zeros((n_samples, n_classes))
        total_weight = 0.0

        for i, est in enumerate(self.estimators_):
            proba = np.asarray(est.predict_proba(X))
            w = 1.0 if self._weights is None else float(self._weights[i])

            est_classes = (
                np.asarray(est.classes_)
                if hasattr(est, "classes_")
                else self.classes_
            )

            if np.array_equal(est_classes, self.classes_):
                avg += w * proba
            else:
                # Remap columns to match global class order.
                for local_k, cls in enumerate(est_classes):
                    global_k = int(np.searchsorted(self.classes_, cls))
                    avg[:, global_k] += w * proba[:, local_k]

            total_weight += w

        return avg / total_weight

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hard_predict(self, X: np.ndarray) -> np.ndarray:
        """Majority-vote prediction (uniform or weighted)."""
        if self._weights is None:
            predictions = np.stack(
                [est.predict(X) for est in self.estimators_], axis=1
            )
            voted = []
            for row in predictions:
                values, counts = np.unique(row, return_counts=True)
                voted.append(values[np.argmax(counts)])
            return np.array(voted)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        tally = np.zeros((n_samples, n_classes))
        for i, est in enumerate(self.estimators_):
            preds = est.predict(X)
            for k, cls in enumerate(self.classes_):
                tally[:, k] += self._weights[i] * (preds == cls)
        return self.classes_[np.argmax(tally, axis=1)]

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")
