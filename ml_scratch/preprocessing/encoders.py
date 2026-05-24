"""Categorical encoders: LabelEncoder and OneHotEncoder."""

import numpy as np


class LabelEncoder:
    """Encode a 1-D array of categorical labels as integers in [0, n_classes-1].

    The mapping is determined by the sorted order of unique values seen
    during ``fit``, matching scikit-learn's convention.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique label values in sorted order.
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None

    @property
    def n_classes_(self) -> int:
        self._check_fitted()
        return len(self.classes_)

    def fit(self, y: np.ndarray) -> "LabelEncoder":
        """Learn the set of unique labels from y.

        Parameters
        ----------
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"Expected 1-D array, got shape {y.shape}.")
        self.classes_ = np.unique(y)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Encode labels in y as integers.

        Parameters
        ----------
        y : array-like of shape (n_samples,)

        Returns
        -------
        y_encoded : ndarray of shape (n_samples,), dtype int
        """
        self._check_fitted()
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"Expected 1-D array, got shape {y.shape}.")
        lookup = {label: idx for idx, label in enumerate(self.classes_)}
        try:
            return np.array([lookup[label] for label in y], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Unknown label: {exc}. Call fit() with all labels first.")

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and encode in one step."""
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Decode integer-encoded labels back to original values.

        Parameters
        ----------
        y : array-like of shape (n_samples,), dtype int

        Returns
        -------
        y_decoded : ndarray of shape (n_samples,)
        """
        self._check_fitted()
        y = np.asarray(y, dtype=int)
        if y.ndim != 1:
            raise ValueError(f"Expected 1-D array, got shape {y.shape}.")
        if y.min() < 0 or y.max() >= self.n_classes_:
            raise ValueError(
                f"Values must be in [0, {self.n_classes_ - 1}], "
                f"got range [{y.min()}, {y.max()}]."
            )
        return self.classes_[y]

    def _check_fitted(self) -> None:
        if self.classes_ is None:
            raise RuntimeError("Call fit() before transform().")


class OneHotEncoder:
    """Encode categorical features as a binary (0/1) sparse-style matrix.

    Each feature column is expanded into ``n_categories`` binary columns.
    By default the first category is dropped per feature to avoid
    multicollinearity (``drop='first'``); pass ``drop=None`` to keep all.

    Parameters
    ----------
    drop : {'first', None}
        Strategy to reduce redundant columns:

        - ``'first'`` : drop the first category per feature (default).
        - ``None``    : keep all category columns.

    Attributes
    ----------
    categories_ : list of ndarray
        The unique categories discovered per feature, in sorted order.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    def __init__(self, drop: str | None = "first") -> None:
        if drop not in ("first", None):
            raise ValueError("drop must be 'first' or None.")
        self.drop = drop
        self.categories_: list[np.ndarray] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y=None) -> "OneHotEncoder":
        """Learn unique categories for each feature column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        X = np.asarray(X)
        self._validate_input(X)
        self.n_features_in_ = X.shape[1]
        self.categories_ = [np.unique(X[:, j]) for j in range(self.n_features_in_)]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """One-hot-encode X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_encoded : ndarray of shape (n_samples, n_out_features), dtype float
            Binary indicator matrix.
        """
        self._check_fitted()
        X = np.asarray(X)
        self._validate_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        parts: list[np.ndarray] = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            lookup = {cat: idx for idx, cat in enumerate(cats)}
            try:
                indices = np.array([lookup[v] for v in col], dtype=int)
            except KeyError as exc:
                raise ValueError(
                    f"Feature {j}: unknown category {exc}. "
                    "Fit with all categories first."
                )
            block = np.eye(len(cats), dtype=float)[indices]
            if self.drop == "first" and len(cats) > 1:
                block = block[:, 1:]
            parts.append(block)

        return np.hstack(parts)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> list[str]:
        """Return output feature names for the encoded columns.

        Returns
        -------
        names : list of str
            One string per output column in the form ``"x<j>_<category>"``.
        """
        self._check_fitted()
        names: list[str] = []
        for j, cats in enumerate(self.categories_):
            start = 1 if (self.drop == "first" and len(cats) > 1) else 0
            for cat in cats[start:]:
                names.append(f"x{j}_{cat}")
        return names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")

    def _check_fitted(self) -> None:
        if self.categories_ is None:
            raise RuntimeError("Call fit() before transform().")
