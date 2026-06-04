"""Non-negative Matrix Factorization (NMF) via multiplicative updates and ALS."""

from __future__ import annotations

import numpy as np


class NMF:
    """Decompose a non-negative matrix V into two non-negative factors W and H.

    Finds an approximate factorisation ``V ≈ W @ H`` where both ``W`` and ``H``
    are element-wise non-negative.  Two solvers are provided:

    * ``'mu'``  — Multiplicative Update (Lee & Seung, 2001).  Guarantees
      non-negativity by construction and minimises either the Frobenius norm
      or the generalised Kullback–Leibler divergence.
    * ``'als'`` — Alternating Least Squares.  Solves a non-negative least
      squares sub-problem for each factor in turn; typically converges faster
      than MU for the Frobenius objective.

    Parameters
    ----------
    n_components : int
        Number of latent components (rank of the factorisation).
    solver : {'mu', 'als'}
        Optimisation algorithm.  ``'mu'`` supports both ``'frobenius'`` and
        ``'kl'`` beta-loss; ``'als'`` uses Frobenius only.
    beta_loss : {'frobenius', 'kl'}
        Loss function for the ``'mu'`` solver:

        * ``'frobenius'`` — squared Euclidean distance ``‖V - WH‖_F²``.
        * ``'kl'``        — generalised Kullback–Leibler divergence
          ``∑ V log(V / WH) - V + WH``.

        Ignored when ``solver='als'``.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance: stop when the relative change in reconstruction
        error drops below ``tol``.
    random_state : int | None
        Seed for reproducible initialisation.
    l1_reg_W : float
        L1 regularisation coefficient applied to ``W``.
    l1_reg_H : float
        L1 regularisation coefficient applied to ``H``.
    l2_reg_W : float
        L2 regularisation coefficient applied to ``W``.
    l2_reg_H : float
        L2 regularisation coefficient applied to ``H``.
    eps : float
        Small constant added to denominators to prevent division by zero.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorisation matrix ``H`` — the dictionary / basis vectors in feature
        space.  Each row is one latent topic / component.
    n_components_ : int
        Effective number of components used.
    n_iter_ : int
        Number of iterations executed until convergence (or ``max_iter``).
    reconstruction_err_ : float
        Frobenius reconstruction error ``‖V - W @ H‖_F`` after fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_scratch.decomposition import NMF
    >>> rng = np.random.default_rng(0)
    >>> V = np.abs(rng.standard_normal((80, 20)))   # non-negative matrix
    >>> nmf = NMF(n_components=5, random_state=0).fit(V)
    >>> W = nmf.transform(V)                        # shape (80, 5)
    >>> V_approx = nmf.inverse_transform(W)         # shape (80, 20)
    >>> print(f"Reconstruction error: {nmf.reconstruction_err_:.4f}")
    """

    def __init__(
        self,
        n_components: int = 2,
        solver: str = "mu",
        beta_loss: str = "frobenius",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int | None = None,
        l1_reg_W: float = 0.0,
        l1_reg_H: float = 0.0,
        l2_reg_W: float = 0.0,
        l2_reg_H: float = 0.0,
        eps: float = 1e-10,
    ) -> None:
        if solver not in ("mu", "als"):
            raise ValueError(f"solver must be 'mu' or 'als', got '{solver}'")
        if beta_loss not in ("frobenius", "kl"):
            raise ValueError(f"beta_loss must be 'frobenius' or 'kl', got '{beta_loss}'")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if solver == "als" and beta_loss == "kl":
            raise ValueError("beta_loss='kl' is only supported with solver='mu'")

        self.n_components = n_components
        self.solver = solver
        self.beta_loss = beta_loss
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.l1_reg_W = l1_reg_W
        self.l1_reg_H = l1_reg_H
        self.l2_reg_W = l2_reg_W
        self.l2_reg_H = l2_reg_H
        self.eps = eps

        self.components_: np.ndarray | None = None
        self.n_components_: int = n_components
        self.n_iter_: int = 0
        self.reconstruction_err_: float = float("inf")

        # Internal: W stored for inverse_transform reference is not kept;
        # only H (components_) is retained after fit.
        self._W_fit: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "NMF":
        """Fit the NMF model to the non-negative matrix *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            All entries must be non-negative.

        Returns
        -------
        self
        """
        X = self._validate(X)
        W, H = self._initialise(X)

        if self.solver == "mu":
            W, H, n_iter = self._mu_solve(X, W, H)
        else:
            W, H, n_iter = self._als_solve(X, W, H)

        self.components_ = H
        self.n_components_ = H.shape[0]
        self.n_iter_ = n_iter
        self.reconstruction_err_ = float(np.linalg.norm(X - W @ H, "fro"))
        self._W_fit = W
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode *X* into the latent component space.

        Solves the non-negative least squares problem
        ``min_{W ≥ 0} ‖X - W H‖_F²`` with ``H`` fixed at the fitted value.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            Activation matrix ``W``.
        """
        self._check_fitted()
        X = self._validate(X)
        H = self.components_
        rng = np.random.default_rng(self.random_state)
        W = rng.uniform(0, 1, (X.shape[0], self.n_components_))
        W, _ = self._nnls_update_W(X, W, H, n_iter=100)
        return W

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return the activation matrix in one step.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
        """
        self.fit(X)
        return self._W_fit  # type: ignore[return-value]

    def inverse_transform(self, W: np.ndarray) -> np.ndarray:
        """Reconstruct the data matrix from activations *W*.

        Parameters
        ----------
        W : ndarray of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features)
        """
        self._check_fitted()
        return np.maximum(0.0, np.asarray(W, dtype=float) @ self.components_)

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _mu_solve(
        self,
        X: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Lee & Seung multiplicative update rules."""
        eps = self.eps
        prev_err = float("inf")

        for i in range(1, self.max_iter + 1):
            if self.beta_loss == "frobenius":
                # Update H
                WtX = W.T @ X
                WtWH = W.T @ W @ H + eps
                H *= WtX / WtWH
                if self.l1_reg_H > 0:
                    H = np.maximum(0.0, H - self.l1_reg_H)
                if self.l2_reg_H > 0:
                    H /= 1.0 + self.l2_reg_H
                H = np.maximum(H, eps)

                # Update W
                XHt = X @ H.T
                WHHt = W @ H @ H.T + eps
                W *= XHt / WHHt
                if self.l1_reg_W > 0:
                    W = np.maximum(0.0, W - self.l1_reg_W)
                if self.l2_reg_W > 0:
                    W /= 1.0 + self.l2_reg_W
                W = np.maximum(W, eps)

            else:  # kl
                WH = W @ H + eps
                # Update H: H ← H * (W^T (X / WH)) / sum(W, axis=0)^T
                H *= (W.T @ (X / WH)) / (W.sum(axis=0, keepdims=True).T + eps)
                H = np.maximum(H, eps)

                WH = W @ H + eps
                # Update W: W ← W * ((X / WH) H^T) / sum(H, axis=1)^T
                W *= ((X / WH) @ H.T) / (H.sum(axis=1, keepdims=True).T + eps)
                W = np.maximum(W, eps)

            # Convergence check
            err = float(np.linalg.norm(X - W @ H, "fro"))
            if prev_err < float("inf"):
                rel_change = abs(prev_err - err) / (prev_err + eps)
                if rel_change < self.tol:
                    return W, H, i
            prev_err = err

        return W, H, self.max_iter

    def _als_solve(
        self,
        X: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Alternating Least Squares with non-negativity projection."""
        eps = self.eps
        prev_err = float("inf")

        for i in range(1, self.max_iter + 1):
            # Update H: solve min_H ‖X - WH‖² s.t. H ≥ 0
            WtW = W.T @ W
            if self.l2_reg_H > 0:
                WtW += self.l2_reg_H * np.eye(WtW.shape[0])
            WtX = W.T @ X
            H = np.linalg.solve(WtW + eps * np.eye(WtW.shape[0]), WtX)
            if self.l1_reg_H > 0:
                H -= self.l1_reg_H
            H = np.maximum(H, eps)

            # Update W: solve min_W ‖X - WH‖² s.t. W ≥ 0
            HHt = H @ H.T
            if self.l2_reg_W > 0:
                HHt += self.l2_reg_W * np.eye(HHt.shape[0])
            XHt = X @ H.T
            W = np.linalg.solve(HHt + eps * np.eye(HHt.shape[0]), XHt.T).T
            if self.l1_reg_W > 0:
                W -= self.l1_reg_W
            W = np.maximum(W, eps)

            # Convergence check
            err = float(np.linalg.norm(X - W @ H, "fro"))
            if prev_err < float("inf"):
                rel_change = abs(prev_err - err) / (prev_err + eps)
                if rel_change < self.tol:
                    return W, H, i
            prev_err = err

        return W, H, self.max_iter

    def _nnls_update_W(
        self,
        X: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        n_iter: int = 100,
    ) -> tuple[np.ndarray, int]:
        """Multiplicative update for W only (H fixed), used in transform()."""
        eps = self.eps
        XHt = X @ H.T
        HHt = H @ H.T
        for i in range(n_iter):
            denom = W @ HHt + eps
            W_new = W * XHt / denom
            W_new = np.maximum(W_new, eps)
            if np.max(np.abs(W_new - W)) < self.tol:
                return W_new, i + 1
            W = W_new
        return W, n_iter

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _initialise(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """NNDSVD-like initialisation using random non-negative values."""
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        scale = np.sqrt(X.mean() / self.n_components)
        W = rng.uniform(0, 2 * scale, (n_samples, self.n_components))
        H = rng.uniform(0, 2 * scale, (self.n_components, n_features))
        return np.maximum(W, self.eps), np.maximum(H, self.eps)

    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.min() < 0:
            raise ValueError("NMF requires a non-negative input matrix X.")
        return X

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform() or inverse_transform().")
