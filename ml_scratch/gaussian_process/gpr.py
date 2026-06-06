"""Gaussian Process Regression (GPR) implemented with NumPy."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .kernels import matern52_kernel, periodic_kernel, rbf_kernel, linear_kernel

_KERNELS: dict[str, Callable[..., np.ndarray]] = {
    "rbf": rbf_kernel,
    "matern52": matern52_kernel,
    "periodic": periodic_kernel,
    "linear": linear_kernel,
}


class GaussianProcessRegressor:
    """Non-parametric Bayesian regression using a Gaussian Process prior.

    Fits a GP to observed data and produces predictions with calibrated
    uncertainty estimates.  Inference is exact (Cholesky-based) and runs in
    O(n³) time, which is appropriate for datasets up to a few thousand points.

    The posterior predictive distribution at test points X* is Gaussian:

        f* | X, y, X*  ~  N(μ*, Σ*)

    where:
        μ* = K(X*, X) [K(X,X) + σ²_n I]⁻¹ y
        Σ* = K(X*, X*) - K(X*, X) [K(X,X) + σ²_n I]⁻¹ K(X, X*)

    Parameters
    ----------
    kernel : {"rbf", "matern52", "periodic", "linear"} or callable
        Kernel function.  A string selects a built-in kernel; a callable must
        have the signature ``k(X1, X2, **kernel_params) -> ndarray``.
    length_scale : float
        Characteristic length-scale of the kernel (ignored for "linear").
    signal_var : float
        Output variance (amplitude squared).
    noise_var : float
        Observation noise variance σ²_n (added to the diagonal of K(X, X)).
    period : float
        Periodicity parameter — only used when ``kernel="periodic"``.
    bias : float
        Constant offset — only used when ``kernel="linear"``.

    Attributes
    ----------
    X_train_ : ndarray of shape (n_samples, n_features)
        Training inputs stored after fit.
    alpha_ : ndarray of shape (n_samples,)
        Dual coefficients: [K + σ²_n I]⁻¹ y, used for fast prediction.
    L_ : ndarray of shape (n_samples, n_samples)
        Lower-triangular Cholesky factor of K(X, X) + σ²_n I.
    log_marginal_likelihood_ : float
        Log marginal likelihood of the training data under the fitted GP.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_scratch.gaussian_process import GaussianProcessRegressor
    >>> rng = np.random.default_rng(42)
    >>> X_train = rng.uniform(-3, 3, (20, 1))
    >>> y_train = np.sin(X_train.ravel()) + rng.normal(0, 0.1, 20)
    >>> gpr = GaussianProcessRegressor(kernel="rbf", length_scale=1.0, noise_var=0.01)
    >>> gpr.fit(X_train, y_train)
    GaussianProcessRegressor(kernel='rbf', length_scale=1.0, noise_var=0.01)
    >>> X_test = np.linspace(-3, 3, 5).reshape(-1, 1)
    >>> mean, std = gpr.predict(X_test, return_std=True)
    """

    def __init__(
        self,
        kernel: Literal["rbf", "matern52", "periodic", "linear"] | Callable = "rbf",
        *,
        length_scale: float = 1.0,
        signal_var: float = 1.0,
        noise_var: float = 1e-3,
        period: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        self.kernel = kernel
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.period = period
        self.bias = bias

        self.X_train_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.L_: np.ndarray | None = None
        self.log_marginal_likelihood_: float = float("nan")

    # ------------------------------------------------------------------
    # Kernel helper
    # ------------------------------------------------------------------

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if callable(self.kernel) and not isinstance(self.kernel, str):
            return self.kernel(X1, X2)

        name = self.kernel
        if name not in _KERNELS:
            raise ValueError(
                f"Unknown kernel '{name}'. Choose from {list(_KERNELS)} or pass a callable."
            )
        fn = _KERNELS[name]
        if name == "periodic":
            return fn(X1, X2, length_scale=self.length_scale,
                      period=self.period, signal_var=self.signal_var)
        if name == "linear":
            return fn(X1, X2, signal_var=self.signal_var, bias=self.bias)
        return fn(X1, X2, length_scale=self.length_scale, signal_var=self.signal_var)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        """Fit the GP to training data.

        Computes the Cholesky decomposition of the noisy kernel matrix and the
        dual coefficients α = [K + σ²_n I]⁻¹ y.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim == 1:
            X = X[:, np.newaxis]
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.X_train_ = X
        n = X.shape[0]

        K = self._compute_kernel(X, X)
        K_noisy = K + self.noise_var * np.eye(n)

        # Cholesky: K_noisy = L L^T
        self.L_ = np.linalg.cholesky(K_noisy)

        # α = L^{-T} L^{-1} y  (solve two triangular systems)
        v = np.linalg.solve(self.L_, y)              # L v = y
        self.alpha_ = np.linalg.solve(self.L_.T, v)  # L^T α = v

        # Log marginal likelihood: -½ y^T α - Σ log(L_ii) - n/2 log(2π)
        self.log_marginal_likelihood_ = (
            -0.5 * y @ self.alpha_
            - np.sum(np.log(np.diag(self.L_)))
            - 0.5 * n * np.log(2.0 * np.pi)
        )
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        return_cov: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict posterior mean (and optionally uncertainty) at test points.

        Parameters
        ----------
        X : ndarray of shape (n_test, n_features)
        return_std : bool
            If True, also return the posterior standard deviation per point.
        return_cov : bool
            If True, also return the full posterior covariance matrix.
            Mutually exclusive with ``return_std``.

        Returns
        -------
        mean : ndarray of shape (n_test,)
        std : ndarray of shape (n_test,), only when ``return_std=True``
        cov : ndarray of shape (n_test, n_test), only when ``return_cov=True``
        """
        self._check_fitted()
        if return_std and return_cov:
            raise ValueError("return_std and return_cov are mutually exclusive.")

        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim == 1:
            X = X[:, np.newaxis]

        K_trans = self._compute_kernel(X, self.X_train_)  # (n_test, n_train)
        mean = K_trans @ self.alpha_

        if not return_std and not return_cov:
            return mean

        # V = L^{-1} K(X*, X)^T  →  shape (n_train, n_test)
        V = np.linalg.solve(self.L_, K_trans.T)

        if return_cov:
            K_ss = self._compute_kernel(X, X)
            cov = K_ss - V.T @ V
            return mean, cov

        # return_std
        K_diag = self._compute_kernel(X, X)  # full matrix, take diagonal
        var = np.diag(K_diag) - np.sum(V ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std

    def sample_y(
        self,
        X: np.ndarray,
        n_samples: int = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw samples from the posterior predictive distribution.

        Parameters
        ----------
        X : ndarray of shape (n_test, n_features)
        n_samples : int
            Number of function samples to draw.
        random_state : int, Generator, or None

        Returns
        -------
        y_samples : ndarray of shape (n_test, n_samples)
        """
        mean, cov = self.predict(X, return_cov=True)
        rng = np.random.default_rng(random_state)
        # Jitter for numerical stability
        cov += 1e-9 * np.eye(len(mean))
        return rng.multivariate_normal(mean, cov, size=n_samples).T

    def log_marginal_likelihood(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> float:
        """Return the log marginal likelihood of the training data.

        If X and y are provided, recomputes for those data (without updating
        the fitted model).  Otherwise returns the value stored from ``fit()``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or None
        y : ndarray of shape (n_samples,) or None

        Returns
        -------
        lml : float
        """
        if X is None and y is None:
            self._check_fitted()
            return self.log_marginal_likelihood_
        if X is None or y is None:
            raise ValueError("Provide both X and y, or neither.")
        # Temporary fit to get LML without storing state
        tmp = GaussianProcessRegressor(
            kernel=self.kernel,
            length_scale=self.length_scale,
            signal_var=self.signal_var,
            noise_var=self.noise_var,
            period=self.period,
            bias=self.bias,
        )
        tmp.fit(X, y)
        return tmp.log_marginal_likelihood_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R² on test data.

        Parameters
        ----------
        X : ndarray of shape (n_test, n_features)
        y : ndarray of shape (n_test,)

        Returns
        -------
        r2 : float
        """
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    def __repr__(self) -> str:
        kernel_name = self.kernel if isinstance(self.kernel, str) else self.kernel.__name__
        return (
            f"GaussianProcessRegressor("
            f"kernel={kernel_name!r}, "
            f"length_scale={self.length_scale}, "
            f"noise_var={self.noise_var})"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict().")
