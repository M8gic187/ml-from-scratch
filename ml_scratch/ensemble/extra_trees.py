"""Extra Trees Classifier: extremely randomised decision tree ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class _ETNode:
    """Internal node or leaf of an Extra Tree."""

    feature: int = -1
    threshold: float = 0.0
    left: Optional["_ETNode"] = field(default=None, repr=False)
    right: Optional["_ETNode"] = field(default=None, repr=False)
    value: object = None


class _ExtraTreeClassifier:
    """Single Extra-randomised tree for classification.

    At each split candidate, the threshold is drawn *uniformly at random*
    from the feature's observed range [min, max] rather than searched
    exhaustively for the best Gini gain.  The best gain among the drawn
    random splits selects the actual split, so impurity still decreases —
    just with less precision.  The resulting diversity across trees replaces
    the need for bootstrap sampling.

    Parameters
    ----------
    max_depth : int | None
        Maximum depth. ``None`` grows until pure leaves or min_samples_split.
    min_samples_split : int
        Minimum samples required to attempt a split.
    min_samples_leaf : int
        Minimum samples in each child after a split.
    n_features : int | None
        Number of random candidate features per split.  ``None`` uses all.
    random_state : int | None
        Seed for the RNG controlling both feature subspace and thresholds.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_features: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.random_state = random_state

        self.root_: _ETNode | None = None
        self.n_features_in_: int = 0
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ExtraTreeClassifier":
        """Build the Extra Tree on training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow(X, y, depth=0, rng=rng)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root_) for x in X])

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _grow(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        rng: np.random.Generator,
    ) -> _ETNode:
        n_samples = len(y)
        majority = _majority_class(y)

        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        )
        if stop:
            return _ETNode(value=majority)

        n_feat = min(self.n_features or self.n_features_in_, self.n_features_in_)
        candidate_features = rng.choice(self.n_features_in_, size=n_feat, replace=False)

        best_feat, best_thresh, best_gain = -1, 0.0, -1.0
        parent_gini = _gini(y)

        for feat in candidate_features:
            col = X[:, feat]
            lo, hi = col.min(), col.max()
            if lo >= hi:
                continue
            # Extremely Randomised Trees: uniformly random threshold
            thresh = rng.uniform(lo, hi)
            mask = col <= thresh
            n_left, n_right = int(mask.sum()), int((~mask).sum())
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            child_gini = (n_left * _gini(y[mask]) + n_right * _gini(y[~mask])) / n_samples
            gain = parent_gini - child_gini
            if gain > best_gain:
                best_gain, best_thresh, best_feat = gain, thresh, feat

        if best_feat == -1:
            return _ETNode(value=majority)

        mask = X[:, best_feat] <= best_thresh
        left = self._grow(X[mask], y[mask], depth + 1, rng)
        right = self._grow(X[~mask], y[~mask], depth + 1, rng)
        return _ETNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _traverse(self, x: np.ndarray, node: _ETNode) -> object:
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


# ---------------------------------------------------------------------------
# Public ensemble class
# ---------------------------------------------------------------------------


class ExtraTreesClassifier:
    """Extremely Randomised Trees classifier.

    Grows an ensemble of ``_ExtraTreeClassifier`` trees.  Unlike
    :class:`~ml_scratch.ensemble.RandomForestClassifier`, each split
    threshold is drawn *uniformly at random* from the feature's observed
    range instead of being optimised — trading split precision for
    construction speed and additional diversity across trees.

    Key differences from Random Forest
    ------------------------------------
    - **Random thresholds** (not optimal Gini splits) → faster, more diverse.
    - **bootstrap=False** by default → every tree sees the full training set.
    - **n_features=None** by default → all features are candidates; diversity
      comes from random thresholds rather than feature subsampling.

    Parameters
    ----------
    n_estimators : int
        Number of trees to grow.
    max_depth : int | None
        Maximum depth of each individual tree.
    min_samples_split : int
        Minimum samples required to attempt a split.
    min_samples_leaf : int
        Minimum samples required in each leaf after a split.
    n_features : int | None
        Number of features considered at each split.  ``None`` uses all
        features (the Geurts et al. default for Extra Trees).
    bootstrap : bool
        If ``True``, each tree is trained on a bootstrap sample.
        Default ``False`` — full dataset per tree, as in the original paper.
    random_state : int | None
        Seed for reproducibility.

    Attributes
    ----------
    estimators_ : list of _ExtraTreeClassifier
        The fitted ensemble members.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during ``fit``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    References
    ----------
    Geurts, Ernst & Wehenkel (2006) — *Extremely Randomized Trees*,
    Machine Learning, 63(1), 3–42.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_features: int | None = None,
        bootstrap: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: list[_ExtraTreeClassifier] = []
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesClassifier":
        """Grow the ensemble on training data.

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

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            tree_seed = int(rng.integers(0, 2**31))
            tree = _ExtraTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_features=self.n_features,
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
        """Return averaged class-probability estimates across all trees.

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


# ---------------------------------------------------------------------------
# Module-level helpers (shared by _ExtraTreeClassifier)
# ---------------------------------------------------------------------------


def _gini(y: np.ndarray) -> float:
    """Gini impurity of label array *y*."""
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return float(1.0 - np.sum(probs**2))


def _majority_class(y: np.ndarray) -> object:
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]
