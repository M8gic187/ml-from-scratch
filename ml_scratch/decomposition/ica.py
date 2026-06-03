"""Independent Component Analysis (FastICA) via fixed-point iteration."""

from __future__ import annotations

import numpy as np


class FastICA:
    """Extract statistically independent components using the FastICA algorithm.

    FastICA maximises non-Gaussianity of the projected signals as a proxy for
    statistical independence.  It operates on pre-whitened data and iterates a
    fixed-point update rule until the mixing vectors converge.

    Two extraction strategies are supported:

    * ``'parallel'``  — all components are updated simultaneously each
      iteration (default, faster on large datasets).
    * ``'deflation'`` — components are extracted one at a time; each new
      component is orthogonalised against all previously found ones via
      Gram–Schmidt.

    Parameters
    ----------
    n_components : int | None
        Number of independent components to extract.  ``None`` uses all
        input features after whitening.
    algorithm : {'parallel', 'deflation'}
        Extraction strategy.
    fun : {'logcosh', 'exp', 'cube'}
        Non-linearity (contrast function) used to approximate negentropy:

        * ``'logcosh'`` — ``g(u) = tanh(alpha * u)``, stable default.
        * ``'exp'``     — ``g(u) = u * exp(-u² / 2)``, good for super-Gaussian.
        * ``'cube'``    — ``g(u) = u³``, fast but sensitive to outliers.
    fun_alpha : float
        Shape parameter for the ``'logcosh'`` non-linearity (typically 1).
    max_iter : int
        Maximum number of fixed-point iterations.
    tol : float
        Convergence tolerance: ``1 - |w_new · w_old|``.
    whiten : bool
        If ``True`` (default), centre and whiten ``X`` before extraction.
        Set to ``False`` only if ``X`` is already whitened.
    random_state : int | None
        Seed for reproducible weight initialisation.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Estimated unmixing directions in the *whitened* feature space,
        mapped back to the original feature space.
    mixing_ : ndarray of shape (n_features, n_components)
        Pseudo-inverse of ``components_``; maps component signals back to
        the original feature space.
    mean_ : ndarray of shape (n_features,) or None
        Per-feature mean subtracted during whitening.  ``None`` when
        ``whiten=False``.
    whitening_ : ndarray of shape (n_components, n_features) or None
        Whitening matrix (PCA-based).  ``None`` when ``whiten=False``.
    n_iter_ : list[int]
        Number of iterations taken per component (deflation) or a single
        value (parallel).

    Examples
    --------
    >>> import numpy as np
    >>> from ml_scratch.decomposition import FastICA
    >>> rng = np.random.default_rng(0)
    >>> S = rng.laplace(size=(500, 3))          # independent sources
    >>> A = rng.standard_normal((3, 3))         # unknown mixing matrix
    >>> X = S @ A.T                             # observed mixed signals
    >>> ica = FastICA(n_components=3, random_state=0).fit(X)
    >>> S_hat = ica.transform(X)               # recovered sources
    """

    def __init__(
        self,
        n_components: int | None = None,
        algorithm: str = "parallel",
        fun: str = "logcosh",
        fun_alpha: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        whiten: bool = True,
        random_state: int | None = None,
    ) -> None:
        if algorithm not in ("parallel", "deflation"):
            raise ValueError(f"algorithm must be 'parallel' or 'deflation', got '{algorithm}'")
        if fun not in ("logcosh", "exp", "cube"):
            raise ValueError(f"fun must be 'logcosh', 'exp', or 'cube', got '{fun}'")

        self.n_components = n_components
        self.algorithm = algorithm
        self.fun = fun
        self.fun_alpha = fun_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.whiten = whiten
        self.random_state = random_state

        self.components_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.whitening_: np.ndarray | None = None
        self.n_iter_: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FastICA":
        """Estimate the unmixing matrix from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self._fit(np.asarray(X, dtype=float))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the independent components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        # components_ already encodes the whitening transform when whiten=True,
        # so we only need to subtract the mean and project directly.
        if self.whiten:
            X = X - self.mean_
        return X @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and project in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, S: np.ndarray) -> np.ndarray:
        """Reconstruct approximate original-space signals from components.

        Parameters
        ----------
        S : ndarray of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features)
        """
        self._check_fitted()
        S = np.asarray(S, dtype=float)
        # mixing_ = pinv(components_) maps components → original feature space.
        X_centered = S @ self.mixing_.T
        if self.whiten:
            return X_centered + self.mean_
        return X_centered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit(self, X: np.ndarray) -> None:
        n_samples, n_features = X.shape
        n_components = self.n_components if self.n_components is not None else n_features

        if n_components > n_features:
            raise ValueError(
                f"n_components ({n_components}) must be <= n_features ({n_features})"
            )

        rng = np.random.default_rng(self.random_state)

        if self.whiten:
            self.mean_ = X.mean(axis=0)
            X_c = X - self.mean_
            # PCA whitening: project to n_components, scale to unit variance
            cov = np.cov(X_c, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # descending order, keep top n_components
            idx = np.argsort(eigvals)[::-1][:n_components]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            # Whitening matrix: (n_components, n_features)
            self.whitening_ = (eigvecs / np.sqrt(np.maximum(eigvals, 1e-12))).T
            X_white = X_c @ self.whitening_.T  # (n_samples, n_components)
        else:
            self.mean_ = None
            self.whitening_ = np.eye(n_components, n_features)
            X_white = X[:, :n_components]
            n_components = X_white.shape[1]

        W = rng.standard_normal((n_components, n_components))
        W = self._sym_decorr(W)

        if self.algorithm == "parallel":
            W, n_iters = self._parallel(X_white, W)
            self.n_iter_ = [n_iters]
        else:
            W, iter_list = self._deflation(X_white, W)
            self.n_iter_ = iter_list

        # components_ maps whitened space → ICs; shape (n_components, n_components)
        # For inverse operations we also need the back-projection to original space.
        self.components_ = W  # (n_components, n_components) in whitened space
        self.mixing_ = np.linalg.pinv(W)

        # Map components_ to original feature space for transform() convenience
        if self.whiten:
            self.components_ = W @ self.whitening_  # (n_components, n_features)
            self.mixing_ = np.linalg.pinv(self.components_)

    def _parallel(
        self, X: np.ndarray, W: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Simultaneous update of all weight vectors."""
        n_samples = X.shape[0]
        for i in range(1, self.max_iter + 1):
            GX, G_prime_mean = self._g_and_dgmean(X @ W.T)  # (n, k), scalar
            W_new = (GX.T @ X) / n_samples - G_prime_mean * W
            W_new = self._sym_decorr(W_new)
            # Convergence: max off-diag deviation from identity in W_new @ W.T
            delta = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1))
            W = W_new
            if delta < self.tol:
                return W, i
        return W, self.max_iter

    def _deflation(
        self, X: np.ndarray, W_init: np.ndarray
    ) -> tuple[np.ndarray, list[int]]:
        """Sequential extraction with Gram–Schmidt decorrelation."""
        n_components = W_init.shape[0]
        n_samples = X.shape[0]
        W = np.zeros_like(W_init)
        iter_counts: list[int] = []

        for c in range(n_components):
            w = W_init[c].copy()
            w /= np.linalg.norm(w) + 1e-12

            for i in range(1, self.max_iter + 1):
                wx = X @ w  # (n,)
                g, dg_mean = self._g_scalar_and_dgmean(wx)
                w_new = (X.T @ g) / n_samples - dg_mean * w
                # Gram–Schmidt against already-found components
                w_new -= W[:c].T @ (W[:c] @ w_new)
                norm = np.linalg.norm(w_new)
                if norm < 1e-12:
                    # Degenerate: restart with random vector
                    rng = np.random.default_rng(None)
                    w_new = rng.standard_normal(w.shape)
                    w_new -= W[:c].T @ (W[:c] @ w_new)
                    norm = np.linalg.norm(w_new)
                w_new /= norm
                delta = abs(abs(np.dot(w_new, w)) - 1)
                w = w_new
                if delta < self.tol:
                    break

            W[c] = w
            iter_counts.append(i)

        return W, iter_counts

    # ------------------------------------------------------------------
    # Non-linearities
    # ------------------------------------------------------------------

    def _g_and_dgmean(
        self, U: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Vectorised g(U) and mean(g'(U)) for the parallel algorithm."""
        if self.fun == "logcosh":
            a = self.fun_alpha
            tanh_au = np.tanh(a * U)
            return tanh_au, float(a * (1 - tanh_au ** 2).mean())
        if self.fun == "exp":
            exp_term = np.exp(-0.5 * U ** 2)
            return U * exp_term, float(((1 - U ** 2) * exp_term).mean())
        # cube
        return U ** 3, float(3 * (U ** 2).mean())

    def _g_scalar_and_dgmean(
        self, u: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """g(u) as a 1-D vector and scalar mean(g'(u)) for deflation."""
        U = u[:, np.newaxis]
        g_matrix, dg_mean = self._g_and_dgmean(U)
        return g_matrix[:, 0], dg_mean

    # ------------------------------------------------------------------
    # Orthogonalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _sym_decorr(W: np.ndarray) -> np.ndarray:
        """Symmetric decorrelation: W ← (W W^T)^{-1/2} W."""
        S, U = np.linalg.eigh(W @ W.T)
        S = np.maximum(S, 1e-12)
        return (U * (S ** -0.5)) @ U.T @ W

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")
