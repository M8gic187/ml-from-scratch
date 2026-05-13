"""Random Forest classifier: bootstrap-aggregated decision trees."""

from __future__ import annotations

import numpy as np

from ..tree.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """Random Forest for multi-class classification.

    Fits ``n_estimators`` CART decision trees on bootstrap samples of the
    training data, each considering a random feature subspace at every split.
    Prediction uses soft majority voting (averaged probabilities) across
    all trees, which generally outperforms hard majority vote.

    Parameters
    ----------
    n_estimators : int
        Number of trees to grow in the forest.
    max_depth : int | None
        Maximum depth of each individual tree.
    min_samples_split : int
        Minimum number of samples required to split a node.
    min_samples_leaf : int
        Minimum number of samples allowed in a leaf node.
    n_features : int | None
        Number of features considered at each split.  ``None`` defaults to
        ``floor(sqrt(n_features))``, the standard random-forest heuristic.
    bootstrap : bool
        If ``True``, each tree is trained on a bootstrap sample (sampling
        with replacement). If ``False``, the full dataset is used for each
        tree (pasting).
    random_state : int | None
        Seed for the random number generator controlling bootstrap sampling
        and per-tree feature subspaces.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_features: int | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Grow the forest by fitting trees on (bootstrap) training data.

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
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_features_in_ = n_features
        self.estimators_ = []

        n_feat_split = self.n_features or max(1, int(np.sqrt(n_features)))
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            tree_seed = int(rng.integers(0, 2**31))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_features=n_feat_split,
                random_state=tree_seed,
            )
            if self.bootstrap:
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                tree.fit(X[idx], y[idx])
            else:
                tree.fit(X, y)
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged class probability estimates across all trees.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        votes = np.zeros((len(X), n_classes))

        for tree in self.estimators_:
            preds = tree.predict(X)
            for j, cls in enumerate(self.classes_):
                votes[:, j] += preds == cls

        return votes / self.n_estimators

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels via soft majority vote across all trees."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")
