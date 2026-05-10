"""CART Decision Tree for multi-class classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class _Node:
    """Internal node or leaf of the decision tree."""

    feature: int = -1
    threshold: float = 0.0
    left: Optional["_Node"] = field(default=None, repr=False)
    right: Optional["_Node"] = field(default=None, repr=False)
    # Leaf-only: majority class
    value: object = None


class DecisionTreeClassifier:
    """Binary CART decision tree for multi-class classification.

    Grows a full tree using Gini impurity as the splitting criterion, then
    prunes depth and minimum-sample constraints to control overfitting.

    Parameters
    ----------
    max_depth : int | None
        Maximum tree depth. ``None`` grows until leaves are pure or too small.
    min_samples_split : int
        Minimum samples required to attempt a split.
    min_samples_leaf : int
        Minimum samples required in each child after a split.
    n_features : int | None
        Number of features to consider at each split (random subspace).
        ``None`` uses all features.
    random_state : int | None
        Seed for feature-subspace randomness.
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

        self.root_: _Node | None = None
        self.n_features_in_: int = 0
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Build the decision tree on training data.

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
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow(X, y, depth=0, rng=rng)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root_) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates (leaf majority ratio).

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            One row per sample; columns follow ``self.classes_`` order.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        preds = self.predict(X)
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for i, cls in enumerate(self.classes_):
            proba[:, i] = (preds == cls).astype(float)
        return proba

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _grow(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        rng: np.random.Generator,
    ) -> _Node:
        n_samples = len(y)
        majority = self._majority_class(y)

        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        )
        if stop:
            return _Node(value=majority)

        # Random feature subspace
        n_feat = self.n_features or self.n_features_in_
        n_feat = min(n_feat, self.n_features_in_)
        feature_indices = rng.choice(self.n_features_in_, size=n_feat, replace=False)

        best_feat, best_thresh, best_gain = -1, 0.0, -1.0
        for feat in feature_indices:
            gain, thresh = self._best_split(X[:, feat], y)
            if gain > best_gain:
                best_gain, best_thresh, best_feat = gain, thresh, feat

        if best_feat == -1 or best_gain <= 0.0:
            return _Node(value=majority)

        mask = X[:, best_feat] <= best_thresh
        left_mask, right_mask = mask, ~mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return _Node(value=majority)

        left = self._grow(X[left_mask], y[left_mask], depth + 1, rng)
        right = self._grow(X[right_mask], y[right_mask], depth + 1, rng)
        return _Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, column: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Return (best_gini_gain, threshold) for a single feature column."""
        n = len(y)
        parent_gini = self._gini(y)
        sorted_idx = np.argsort(column)
        col_sorted = column[sorted_idx]
        y_sorted = y[sorted_idx]

        best_gain, best_thresh = 0.0, col_sorted[0]

        for i in range(1, n):
            if col_sorted[i] == col_sorted[i - 1]:
                continue

            y_left = y_sorted[:i]
            y_right = y_sorted[i:]

            if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                continue

            child_gini = (len(y_left) * self._gini(y_left) + len(y_right) * self._gini(y_right)) / n
            gain = parent_gini - child_gini

            if gain > best_gain:
                best_gain = gain
                best_thresh = (col_sorted[i - 1] + col_sorted[i]) / 2.0

        return best_gain, best_thresh

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _traverse(self, x: np.ndarray, node: _Node) -> object:
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """Gini impurity of label array y."""
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return float(1.0 - np.sum(probs ** 2))

    @staticmethod
    def _majority_class(y: np.ndarray) -> object:
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _check_fitted(self) -> None:
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict().")
