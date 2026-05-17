"""Gradient Boosting: stage-wise additive models for classification and regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal regression tree (CART, MSE criterion)
# ---------------------------------------------------------------------------

@dataclass
class _RNode:
    """Node for the internal regression tree."""

    feature: int = -1
    threshold: float = 0.0
    left: Optional["_RNode"] = field(default=None, repr=False)
    right: Optional["_RNode"] = field(default=None, repr=False)
    value: float = 0.0  # leaf mean; ignored for internal nodes


class _RegressionTree:
    """Shallow CART regression tree that minimises MSE.

    Designed as a fast internal component for gradient boosting — not part of
    the public API.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.  Gradient boosting typically uses depth 1–5.
    min_samples_leaf : int
        Minimum number of samples in every leaf.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 1) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root_: _RNode | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RegressionTree":
        self.root_ = self._grow(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(x, self.root_) for x in X])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _RNode:
        leaf_val = float(np.mean(y))

        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return _RNode(value=leaf_val)

        best_feat, best_thresh, best_mse = -1, 0.0, float("inf")
        n = len(y)

        for j in range(X.shape[1]):
            col = X[:, j]
            order = np.argsort(col)
            col_s, y_s = col[order], y[order]

            for i in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                if col_s[i] == col_s[i - 1]:
                    continue
                y_l, y_r = y_s[:i], y_s[i:]
                mse = (
                    np.var(y_l) * len(y_l) + np.var(y_r) * len(y_r)
                ) / n
                if mse < best_mse:
                    best_mse = mse
                    best_feat = j
                    best_thresh = (col_s[i - 1] + col_s[i]) / 2.0

        if best_feat == -1:
            return _RNode(value=leaf_val)

        mask = X[:, best_feat] <= best_thresh
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return _RNode(value=leaf_val)

        left = self._grow(X[mask], y[mask], depth + 1)
        right = self._grow(X[~mask], y[~mask], depth + 1)
        return _RNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _traverse(self, x: np.ndarray, node: _RNode) -> float:
        if node.left is None:  # leaf
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ---------------------------------------------------------------------------
# GradientBoostingClassifier
# ---------------------------------------------------------------------------

class GradientBoostingClassifier:
    """Gradient Boosting for binary classification (log-loss).

    Fits an additive model ``F(x) = F₀ + Σ ηₜ · hₜ(x)`` where each weak
    learner ``hₜ`` is a shallow regression tree trained on the negative
    gradient of the log-loss (i.e. the residuals ``y - σ(F)``).

    Labels are mapped to {0, 1} internally; any two-class input is accepted.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage applied to each tree's contribution.  Smaller values
        require more estimators but often improve generalisation.
    max_depth : int
        Maximum depth of each regression tree.
    min_samples_leaf : int
        Minimum samples per leaf in the regression trees.
    subsample : float
        Fraction of training samples drawn (without replacement) for each
        tree.  Values < 1.0 introduce stochastic gradient boosting.
    random_state : int | None
        Seed for the subsample RNG.

    Attributes
    ----------
    estimators_ : list of _RegressionTree
        Fitted weak learners in boosting order.
    init_score_ : float
        Log-odds of the positive class (constant initial prediction).
    classes_ : ndarray of shape (2,)
        Original class labels.
    train_loss_ : list of float
        Mean log-loss after each boosting round.

    Examples
    --------
    >>> from ml_scratch.ensemble import GradientBoostingClassifier
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((300, 4))
    >>> y = (X[:, 0] - X[:, 1] > 0).astype(int)
    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1).fit(X, y)
    >>> acc = (clf.predict(X) == y).mean()
    >>> acc > 0.90
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

        self.estimators_: list[_RegressionTree] = []
        self.init_score_: float = 0.0
        self.classes_: np.ndarray | None = None
        self.train_loss_: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        """Train the gradient boosted ensemble.

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
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"GradientBoostingClassifier supports binary classification; "
                f"got {len(self.classes_)} classes."
            )

        y_enc = (y == self.classes_[1]).astype(float)
        n_samples = len(y_enc)
        rng = np.random.default_rng(self.random_state)

        # Constant initial prediction: log-odds of positive class
        p0 = np.clip(y_enc.mean(), 1e-10, 1 - 1e-10)
        self.init_score_ = float(np.log(p0 / (1.0 - p0)))

        F = np.full(n_samples, self.init_score_)
        self.estimators_ = []
        self.train_loss_ = []

        for _ in range(self.n_estimators):
            proba = _sigmoid(F)
            # Negative gradient of log-loss = y - σ(F)
            residuals = y_enc - proba

            # Stochastic subsampling
            if self.subsample < 1.0:
                idx = rng.choice(
                    n_samples, size=int(n_samples * self.subsample), replace=False
                )
            else:
                idx = np.arange(n_samples)

            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X[idx], residuals[idx])
            update = tree.predict(X)
            F += self.learning_rate * update
            self.estimators_.append(tree)

            # Track mean log-loss
            p = np.clip(_sigmoid(F), 1e-15, 1 - 1e-15)
            loss = -np.mean(y_enc * np.log(p) + (1 - y_enc) * np.log(1 - p))
            self.train_loss_.append(float(loss))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw log-odds scores (positive → class 1).

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_score_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Columns correspond to ``classes_[0]`` and ``classes_[1]``.
        """
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        threshold : float
            Probability cutoff for class 1.

        Returns
        -------
        ndarray of shape (n_samples,) with values from ``classes_``
        """
        p1 = self.predict_proba(X)[:, 1]
        return np.where(p1 >= threshold, self.classes_[1], self.classes_[0])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")


# ---------------------------------------------------------------------------
# GradientBoostingRegressor
# ---------------------------------------------------------------------------

class GradientBoostingRegressor:
    """Gradient Boosting for regression.

    Supports two loss functions:

    * ``'mse'`` (Mean Squared Error) — pseudo-residuals are ``y - F``;
      initialised with ``mean(y)``.
    * ``'mae'`` (Mean Absolute Error) — pseudo-residuals are ``sign(y - F)``;
      initialised with ``median(y)``.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage factor applied to each tree's contribution.
    max_depth : int
        Maximum depth of each regression tree.
    min_samples_leaf : int
        Minimum samples per leaf.
    subsample : float
        Fraction of samples used to fit each tree (stochastic GB if < 1.0).
    loss : {'mse', 'mae'}
        Loss function to optimise.
    random_state : int | None
        Seed for the subsample RNG.

    Attributes
    ----------
    estimators_ : list of _RegressionTree
        Fitted weak learners in boosting order.
    init_value_ : float
        Constant initial prediction (mean or median of training targets).
    train_loss_ : list of float
        Training loss after each boosting round.

    Examples
    --------
    >>> from ml_scratch.ensemble import GradientBoostingRegressor
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 3))
    >>> y = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.1, 200)
    >>> reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1).fit(X, y)
    >>> mse = np.mean((reg.predict(X) - y) ** 2)
    >>> mse < 0.1
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        loss: Literal["mse", "mae"] = "mse",
        random_state: int | None = None,
    ) -> None:
        if loss not in ("mse", "mae"):
            raise ValueError(f"loss must be 'mse' or 'mae'; got {loss!r}")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = loss
        self.random_state = random_state

        self.estimators_: list[_RegressionTree] = []
        self.init_value_: float = 0.0
        self.train_loss_: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        """Train the gradient boosted regressor.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) — continuous targets

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = len(y)
        rng = np.random.default_rng(self.random_state)

        self.init_value_ = float(np.median(y) if self.loss == "mae" else np.mean(y))
        F = np.full(n_samples, self.init_value_)
        self.estimators_ = []
        self.train_loss_ = []

        for _ in range(self.n_estimators):
            residuals = (
                np.sign(y - F) if self.loss == "mae" else y - F
            )

            if self.subsample < 1.0:
                idx = rng.choice(
                    n_samples, size=int(n_samples * self.subsample), replace=False
                )
            else:
                idx = np.arange(n_samples)

            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X[idx], residuals[idx])
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)

            loss_val = (
                float(np.mean(np.abs(y - F)))
                if self.loss == "mae"
                else float(np.mean((y - F) ** 2))
            )
            self.train_loss_.append(loss_val)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_value_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")
