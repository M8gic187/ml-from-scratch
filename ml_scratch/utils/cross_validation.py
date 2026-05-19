"""Cross-validation utilities for model evaluation."""

from __future__ import annotations

import numpy as np
from typing import Iterator


class KFold:
    """K-Fold cross-validator.

    Splits data into *k* consecutive folds. On each round, one fold is
    used as the validation set and the remaining k-1 folds form the
    training set.

    Parameters
    ----------
    n_splits : int
        Number of folds. Must be at least 2.
    shuffle : bool
        Whether to shuffle indices before splitting.
    random_state : int or None
        Seed for the random number generator (used only when shuffle=True).
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation index pairs.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)

        Yields
        ------
        train_idx : ndarray of int
        val_idx   : ndarray of int
        """
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1  # distribute remainder
        starts = np.concatenate([[0], np.cumsum(fold_sizes)])

        for k in range(self.n_splits):
            val_idx = indices[starts[k] : starts[k + 1]]
            train_idx = np.concatenate([indices[: starts[k]], indices[starts[k + 1] :]])
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class StratifiedKFold:
    """Stratified K-Fold cross-validator.

    Like :class:`KFold` but ensures that each fold preserves the
    class distribution of the full dataset (to within ±1 sample per
    class per fold).

    Parameters
    ----------
    n_splits : int
        Number of folds. Must be at least 2.
    shuffle : bool
        Whether to shuffle indices within each class before splitting.
    random_state : int or None
        Seed for the random number generator (used only when shuffle=True).
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate stratified train/validation index pairs.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
        y : ndarray of shape (n_samples,)
            Class labels used for stratification.

        Yields
        ------
        train_idx : ndarray of int
        val_idx   : ndarray of int
        """
        y = np.asarray(y)
        classes, y_encoded = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.random_state) if self.shuffle else None

        # Build per-class index lists (optionally shuffled)
        class_indices: list[np.ndarray] = []
        for c in range(len(classes)):
            idx = np.where(y_encoded == c)[0]
            if rng is not None:
                rng.shuffle(idx)
            class_indices.append(idx)

        # Distribute each class into k folds, then merge
        fold_indices: list[list[np.ndarray]] = [[] for _ in range(self.n_splits)]
        for idx in class_indices:
            n_c = len(idx)
            fold_sizes = np.full(self.n_splits, n_c // self.n_splits)
            fold_sizes[: n_c % self.n_splits] += 1
            starts = np.concatenate([[0], np.cumsum(fold_sizes)])
            for k in range(self.n_splits):
                fold_indices[k].append(idx[starts[k] : starts[k + 1]])

        all_indices = np.arange(len(y))
        for k in range(self.n_splits):
            val_idx = np.sort(np.concatenate(fold_indices[k]))
            train_idx = np.setdiff1d(all_indices, val_idx)
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


def cross_val_score(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: int | KFold | StratifiedKFold = 5,
    scoring=None,
) -> np.ndarray:
    """Evaluate an estimator using cross-validation.

    Parameters
    ----------
    estimator
        Any object with ``fit(X, y)`` and ``predict(X)`` methods.
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    cv : int or cross-validator instance
        If an int is given, a :class:`StratifiedKFold` with that many
        splits is used.
    scoring : callable or None
        ``scoring(y_true, y_pred) -> float``.  Defaults to accuracy
        (fraction of correctly classified samples).

    Returns
    -------
    scores : ndarray of shape (n_splits,)
        Score on each validation fold.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if scoring is None:
        scoring = lambda y_true, y_pred: float(np.mean(y_true == y_pred))

    if isinstance(cv, int):
        splitter: KFold | StratifiedKFold = StratifiedKFold(n_splits=cv, shuffle=False)
        splits = splitter.split(X, y)
    elif isinstance(cv, StratifiedKFold):
        splits = cv.split(X, y)
    else:
        splits = cv.split(X)

    scores = []
    for train_idx, val_idx in splits:
        estimator.fit(X[train_idx], y[train_idx])
        preds = estimator.predict(X[val_idx])
        scores.append(scoring(y[val_idx], preds))

    return np.array(scores)
