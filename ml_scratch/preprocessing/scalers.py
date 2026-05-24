"""Feature scaling transformers: StandardScaler and MinMaxScaler."""

import numpy as np


class StandardScaler:
    """Standardise features to zero mean and unit variance.

    Computes per-feature mean and standard deviation on ``fit`` and applies
    the transform ``(X - mean) / std`` on subsequent calls.  Features with
    zero variance are left unchanged (std replaced by 1 to avoid division by
    zero), matching scikit-learn's behaviour.

    Parameters
    ----------
    copy : bool
        If True, always operate on a copy of the input array.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Per-feature mean computed during ``fit``.
    scale_ : ndarray of shape (n_features,)
        Per-feature standard deviation (std) computed during ``fit``.
        Zero-variance features get a scale of 1.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    def __init__(self, copy: bool = True) -> None:
        self.copy = copy
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y=None) -> "StandardScaler":
        """Compute mean and std for later scaling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        self._validate_input(X)
        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features of X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        self._check_fitted()
        X = np.array(X, dtype=float, copy=self.copy)
        self._validate_n_features(X)
        X -= self.mean_
        X /= self.scale_
        return X

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling: ``X * std + mean``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_orig : ndarray of shape (n_samples, n_features)
        """
        self._check_fitted()
        X = np.array(X, dtype=float, copy=self.copy)
        self._validate_n_features(X)
        X *= self.scale_
        X += self.mean_
        return X

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")

    def _validate_n_features(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")


class MinMaxScaler:
    """Scale features to a fixed range [feature_range[0], feature_range[1]].

    Computes per-feature minimum and maximum on ``fit`` and applies the
    transform ``(X - min) / (max - min) * (high - low) + low``.  Features
    with identical min and max (constant features) are mapped to 0.

    Parameters
    ----------
    feature_range : tuple of (float, float)
        Desired range of the transformed data.  Defaults to (0, 1).
    copy : bool
        If True, always operate on a copy of the input array.

    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per-feature minimum seen during ``fit``.
    max_ : ndarray of shape (n_features,)
        Per-feature maximum seen during ``fit``.
    scale_ : ndarray of shape (n_features,)
        Scaling factor ``(high - low) / (max - min)``.  Zero for constant
        features.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    def __init__(
        self, feature_range: tuple[float, float] = (0, 1), copy: bool = True
    ) -> None:
        low, high = feature_range
        if low >= high:
            raise ValueError(
                f"feature_range must satisfy low < high, got {feature_range}."
            )
        self.feature_range = feature_range
        self.copy = copy
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y=None) -> "MinMaxScaler":
        """Compute per-feature min/max for later scaling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        self._validate_input(X)
        self.n_features_in_ = X.shape[1]
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        data_range = self.max_ - self.min_
        low, high = self.feature_range
        # Use safe divisor to avoid divide-by-zero RuntimeWarning from NumPy;
        # np.where still evaluates both branches eagerly.
        safe_range = np.where(data_range == 0, 1.0, data_range)
        self.scale_ = np.where(data_range == 0, 0.0, (high - low) / safe_range)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features of X to ``feature_range``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        self._check_fitted()
        X = np.array(X, dtype=float, copy=self.copy)
        self._validate_n_features(X)
        low, _ = self.feature_range
        X -= self.min_
        X *= self.scale_
        X += low
        return X

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the min-max scaling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_orig : ndarray of shape (n_samples, n_features)
        """
        self._check_fitted()
        X = np.array(X, dtype=float, copy=self.copy)
        self._validate_n_features(X)
        low, _ = self.feature_range
        # Avoid divide-by-zero for constant features (scale_ == 0)
        safe_scale = np.where(self.scale_ == 0, 1.0, self.scale_)
        X -= low
        X /= safe_scale
        X += self.min_
        # Constant features stay at min_ after inverse (scale_==0 means no info)
        X[:, self.scale_ == 0] = self.min_[self.scale_ == 0]
        return X

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")

    def _validate_n_features(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

    def _check_fitted(self) -> None:
        if self.min_ is None:
            raise RuntimeError("Call fit() before transform().")
