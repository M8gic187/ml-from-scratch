"""Gaussian Mixture Model (GMM) with full EM algorithm."""

import numpy as np


class GaussianMixture:
    """Gaussian Mixture Model fitted with the Expectation-Maximisation algorithm.

    Each cluster is modelled as a multivariate Gaussian.  The EM algorithm
    alternates between:

    * **E-step** – compute soft cluster assignments (responsibilities) for
      every sample given the current parameters.
    * **M-step** – update mixture weights, means, and covariance matrices to
      maximise the expected complete-data log-likelihood.

    Parameters
    ----------
    n_components : int
        Number of mixture components (clusters).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence threshold on the change in log-likelihood between
        successive iterations.
    covariance_type : {'full', 'diag', 'spherical'}
        Shape of the covariance matrices:

        * ``'full'``      – each component has its own arbitrary covariance.
        * ``'diag'``      – diagonal covariance (features are independent).
        * ``'spherical'`` – single variance shared across all features.
    reg_covar : float
        Small value added to the diagonal of covariance matrices to ensure
        numerical stability (regularisation).
    random_state : int | None
        Seed for the random number generator used during initialisation.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        Mixture weights (sum to 1).
    means_ : ndarray of shape (n_components, n_features)
        Component means.
    covariances_ : ndarray
        Component covariance matrices.  Shape depends on *covariance_type*:
        ``(n_components, n_features, n_features)`` for ``'full'``,
        ``(n_components, n_features)`` for ``'diag'``,
        ``(n_components,)`` for ``'spherical'``.
    converged_ : bool
        ``True`` if the EM algorithm converged within *max_iter* iterations.
    n_iter_ : int
        Number of EM iterations actually performed.
    lower_bound_ : float
        Log-likelihood of the training data under the fitted model.
    """

    _VALID_COV_TYPES = ("full", "diag", "spherical")

    def __init__(
        self,
        n_components: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if covariance_type not in self._VALID_COV_TYPES:
            raise ValueError(
                f"covariance_type must be one of {self._VALID_COV_TYPES}, "
                f"got '{covariance_type}'"
            )
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = -np.inf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GaussianMixture":
        """Fit the Gaussian mixture model to *X* via EM.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if n_samples < self.n_components:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_components ({self.n_components})."
            )

        rng = np.random.default_rng(self.random_state)
        self._initialise(X, rng)

        prev_ll = -np.inf
        for i in range(1, self.max_iter + 1):
            log_resp, log_ll = self._e_step(X)
            self._m_step(X, log_resp)

            change = log_ll - prev_ll
            self.n_iter_ = i
            self.lower_bound_ = log_ll

            if abs(change) < self.tol:
                self.converged_ = True
                break
            prev_ll = log_ll

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely component index for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        self._check_fitted()
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probabilities of component membership.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        resp : ndarray of shape (n_samples, n_components)
            Each row sums to 1.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        log_resp, _ = self._e_step(X)
        return np.exp(log_resp)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return component labels for *X*."""
        return self.fit(X).predict(X)

    def score(self, X: np.ndarray) -> float:
        """Return mean per-sample log-likelihood on *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        float
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        _, log_ll = self._e_step(X)
        return float(log_ll)

    def aic(self, X: np.ndarray) -> float:
        """Akaike Information Criterion (lower is better).

        AIC = 2k − 2 * n * log-likelihood, where *k* is the number of free
        parameters in the model.
        """
        return 2.0 * self._n_parameters() - 2.0 * X.shape[0] * self.score(X)

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion (lower is better).

        BIC = k * log(n) − 2 * n * log-likelihood.
        """
        n = X.shape[0]
        return self._n_parameters() * np.log(n) - 2.0 * n * self.score(X)

    # ------------------------------------------------------------------
    # EM internals
    # ------------------------------------------------------------------

    def _initialise(self, X: np.ndarray, rng: np.random.Generator) -> None:
        """Initialise parameters using random data-point means."""
        n_samples, n_features = X.shape
        k = self.n_components

        self.weights_ = np.full(k, 1.0 / k)

        indices = rng.choice(n_samples, size=k, replace=False)
        self.means_ = X[indices].copy()

        var = np.var(X, axis=0).mean() + self.reg_covar
        if self.covariance_type == "full":
            self.covariances_ = np.stack([np.eye(n_features) * var for _ in range(k)])
        elif self.covariance_type == "diag":
            self.covariances_ = np.full((k, n_features), var)
        else:  # spherical
            self.covariances_ = np.full(k, var)

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute log-responsibilities and mean log-likelihood."""
        log_weights = np.log(self.weights_ + 1e-300)
        log_gauss = self._log_gaussian(X)  # (n_samples, n_components)

        weighted = log_gauss + log_weights[np.newaxis, :]
        log_norm = self._logsumexp(weighted, axis=1)  # (n_samples,)

        log_resp = weighted - log_norm[:, np.newaxis]
        mean_log_ll = float(np.mean(log_norm))
        return log_resp, mean_log_ll

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray) -> None:
        """Update mixture weights, means, and covariances."""
        n_samples = X.shape[0]
        resp = np.exp(log_resp)  # (n_samples, n_components)

        nk = resp.sum(axis=0) + 1e-300  # (n_components,)
        self.weights_ = nk / n_samples
        self.means_ = (resp.T @ X) / nk[:, np.newaxis]

        if self.covariance_type == "full":
            self.covariances_ = self._update_full(X, resp, nk)
        elif self.covariance_type == "diag":
            self.covariances_ = self._update_diag(X, resp, nk)
        else:
            self.covariances_ = self._update_spherical(X, resp, nk)

    def _update_full(
        self, X: np.ndarray, resp: np.ndarray, nk: np.ndarray
    ) -> np.ndarray:
        n_features = X.shape[1]
        k = self.n_components
        covs = np.zeros((k, n_features, n_features))
        for j in range(k):
            diff = X - self.means_[j]
            covs[j] = (resp[:, j, np.newaxis] * diff).T @ diff / nk[j]
            covs[j] += np.eye(n_features) * self.reg_covar
        return covs

    def _update_diag(
        self, X: np.ndarray, resp: np.ndarray, nk: np.ndarray
    ) -> np.ndarray:
        k = self.n_components
        n_features = X.shape[1]
        covs = np.zeros((k, n_features))
        for j in range(k):
            diff = X - self.means_[j]
            covs[j] = (resp[:, j, np.newaxis] * diff ** 2).sum(axis=0) / nk[j]
            covs[j] += self.reg_covar
        return covs

    def _update_spherical(
        self, X: np.ndarray, resp: np.ndarray, nk: np.ndarray
    ) -> np.ndarray:
        n_features = X.shape[1]
        covs = np.zeros(self.n_components)
        for j in range(self.n_components):
            diff = X - self.means_[j]
            covs[j] = (resp[:, j] * np.sum(diff ** 2, axis=1)).sum() / (nk[j] * n_features)
            covs[j] += self.reg_covar
        return covs

    # ------------------------------------------------------------------
    # Log-density helpers
    # ------------------------------------------------------------------

    def _log_gaussian(self, X: np.ndarray) -> np.ndarray:
        """Return log N(x | mu_k, Sigma_k) for all samples and components."""
        n_samples, n_features = X.shape
        k = self.n_components
        log_p = np.zeros((n_samples, k))

        for j in range(k):
            mu = self.means_[j]
            if self.covariance_type == "full":
                log_p[:, j] = self._log_mvn_full(X, mu, self.covariances_[j], n_features)
            elif self.covariance_type == "diag":
                log_p[:, j] = self._log_mvn_diag(X, mu, self.covariances_[j], n_features)
            else:
                log_p[:, j] = self._log_mvn_spherical(X, mu, self.covariances_[j], n_features)
        return log_p

    @staticmethod
    def _log_mvn_full(
        X: np.ndarray, mu: np.ndarray, cov: np.ndarray, d: int
    ) -> np.ndarray:
        diff = X - mu
        try:
            L = np.linalg.cholesky(cov)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            sol = np.linalg.solve(L, diff.T)
            maha = np.sum(sol ** 2, axis=0)
        except np.linalg.LinAlgError:
            sign, log_det = np.linalg.slogdet(cov)
            log_det = log_det if sign > 0 else 0.0
            cov_inv = np.linalg.pinv(cov)
            maha = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)

    @staticmethod
    def _log_mvn_diag(
        X: np.ndarray, mu: np.ndarray, var: np.ndarray, d: int
    ) -> np.ndarray:
        diff = X - mu
        log_det = np.sum(np.log(var))
        maha = np.sum(diff ** 2 / var, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)

    @staticmethod
    def _log_mvn_spherical(
        X: np.ndarray, mu: np.ndarray, sigma2: float, d: int
    ) -> np.ndarray:
        diff = X - mu
        log_det = d * np.log(sigma2)
        maha = np.sum(diff ** 2, axis=1) / sigma2
        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
        """Numerically stable logsumexp along *axis*."""
        a_max = np.max(a, axis=axis, keepdims=True)
        return np.log(np.sum(np.exp(a - a_max), axis=axis)) + np.squeeze(a_max, axis=axis)

    def _n_parameters(self) -> int:
        """Count the free parameters of the model."""
        d = self.means_.shape[1]
        k = self.n_components
        mean_params = k * d
        weight_params = k - 1
        if self.covariance_type == "full":
            cov_params = k * d * (d + 1) // 2
        elif self.covariance_type == "diag":
            cov_params = k * d
        else:
            cov_params = k
        return mean_params + weight_params + cov_params

    def _check_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
