"""Isolation Forest: unsupervised anomaly detection via random isolation trees."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal node / leaf dataclass
# ---------------------------------------------------------------------------

class _IsolationNode:
    """A node in an isolation tree (binary split or leaf)."""

    __slots__ = ("feature", "threshold", "left", "right", "size")

    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None = None,
        left: "_IsolationNode | None" = None,
        right: "_IsolationNode | None" = None,
        size: int = 0,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.size = size  # number of training samples reaching this node


# ---------------------------------------------------------------------------
# Single isolation tree
# ---------------------------------------------------------------------------

class _IsolationTree:
    """A single randomised isolation tree.

    Recursively partitions data by choosing a feature and split threshold
    uniformly at random.  Anomalous points (sparse regions) are isolated
    near the root; normal points (dense regions) require many splits.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.  Typically set to ``ceil(log2(subsample_size))``.
    random_state : int | None
        Seed for the per-tree RNG.
    """

    def __init__(self, max_depth: int, random_state: int | None = None) -> None:
        self.max_depth = max_depth
        self.rng = np.random.default_rng(random_state)
        self.root_: _IsolationNode | None = None

    def fit(self, X: np.ndarray) -> "_IsolationTree":
        self.root_ = self._grow(X, depth=0)
        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """Return the path length of each sample through the tree."""
        return np.array([self._traverse(x, self.root_, depth=0) for x in X])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _grow(self, X: np.ndarray, depth: int) -> _IsolationNode:
        n, n_features = X.shape
        if depth >= self.max_depth or n <= 1:
            return _IsolationNode(size=n)

        feat = int(self.rng.integers(0, n_features))
        col = X[:, feat]
        lo, hi = col.min(), col.max()
        if lo == hi:
            return _IsolationNode(size=n)

        threshold = float(self.rng.uniform(lo, hi))
        mask = col < threshold
        left = self._grow(X[mask], depth + 1)
        right = self._grow(X[~mask], depth + 1)
        return _IsolationNode(feature=feat, threshold=threshold, left=left, right=right, size=n)

    def _traverse(self, x: np.ndarray, node: _IsolationNode, depth: int) -> float:
        """Walk x down the tree and return adjusted path length at the leaf."""
        if node.left is None:  # leaf
            return depth + _c(node.size)
        if x[node.feature] < node.threshold:
            return self._traverse(x, node.left, depth + 1)
        return self._traverse(x, node.right, depth + 1)


# ---------------------------------------------------------------------------
# Expected path-length correction factor  c(n)
# ---------------------------------------------------------------------------

def _c(n: int) -> float:
    """Average path length of an unsuccessful BST search in a tree of n nodes.

    Used to normalise raw path lengths so that scores are comparable across
    different subsample sizes.  Derived from the harmonic number approximation.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    H = np.log(n - 1) + 0.5772156649  # Euler–Mascheroni constant
    return 2.0 * H - 2.0 * (n - 1) / n


# ---------------------------------------------------------------------------
# IsolationForest
# ---------------------------------------------------------------------------

class IsolationForest:
    """Isolation Forest for unsupervised anomaly detection.

    Fits an ensemble of randomised isolation trees on subsamples of the
    training data.  Points that are isolated near the root of many trees
    (short average path length) receive a high anomaly score.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees to build.
    max_samples : int | "auto"
        Number of samples drawn (without replacement) to train each tree.
        ``"auto"`` uses ``min(256, n_samples)``, following the original paper.
    contamination : float
        Expected fraction of outliers in the dataset, used to set the
        decision threshold in ``predict``.  Must be in ``(0, 0.5]``.
    random_state : int | None
        Seed for reproducible results.

    Attributes
    ----------
    estimators_ : list[_IsolationTree]
        Fitted isolation trees.
    offset_ : float
        Decision threshold derived from ``contamination`` after fitting.
    n_features_in_ : int
        Number of features seen during ``fit``.

    References
    ----------
    Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008).
    *Isolation Forest*.  In ICDM 2008.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        contamination: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        if not (0 < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5].")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.estimators_: list[_IsolationTree] = []
        self.offset_: float = 0.0
        self.n_features_in_: int = 0
        self._max_samples_actual: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "IsolationForest":
        """Build the isolation forest.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (unlabelled).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        psi = (
            min(256, n_samples)
            if self.max_samples == "auto"
            else int(self.max_samples)
        )
        psi = min(psi, n_samples)
        self._max_samples_actual = psi
        max_depth = int(np.ceil(np.log2(max(psi, 2))))

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree_seed = int(rng.integers(0, 2**31))
            idx = rng.choice(n_samples, size=psi, replace=False)
            tree = _IsolationTree(max_depth=max_depth, random_state=tree_seed)
            tree.fit(X[idx])
            self.estimators_.append(tree)

        scores = self.score_samples(X)
        self.offset_ = float(np.percentile(scores, 100 * self.contamination))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute the anomaly score for each sample.

        Scores are in ``(-inf, 0]``.  More negative means more anomalous.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        avg_path = np.mean(
            [tree.path_length(X) for tree in self.estimators_], axis=0
        )
        c_psi = _c(self._max_samples_actual)
        if c_psi == 0:
            return np.zeros(len(X))
        return -np.power(2.0, -avg_path / c_psi)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to the decision boundary.

        Positive values indicate inliers; negative values indicate outliers.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict whether each sample is an inlier (+1) or outlier (-1).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,) with values in {-1, +1}
        """
        decisions = self.decision_function(X)
        return np.where(decisions >= 0, 1, -1)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict() / score_samples().")
