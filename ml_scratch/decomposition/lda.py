"""Linear Discriminant Analysis (LDA) for supervised dimensionality reduction and classification."""

from __future__ import annotations

import numpy as np


class LinearDiscriminantAnalysis:
    """Linear Discriminant Analysis (LDA).

    Finds the linear subspace that maximises between-class separation relative
    to within-class variance.  At most ``min(n_features, n_classes - 1)``
    discriminant directions exist; ``n_components`` selects how many to keep.

    Can be used in two ways:

    * **Dimensionality reduction** — call :meth:`fit` then :meth:`transform`.
    * **Classification** — call :meth:`fit` then :meth:`predict` / :meth:`predict_proba`
      (Gaussian assumption, nearest-centroid in the discriminant subspace).

    Parameters
    ----------
    n_components : int | None
        Number of discriminant directions to retain.  ``None`` keeps all
        ``min(n_features, n_classes - 1)`` directions.
    solver : {'eigen', 'svd'}
        ``'eigen'`` solves SW⁻¹ SB directly (fast, may be less stable for
        ill-conditioned SW).  ``'svd'`` whitens the data first for better
        numerical stability.
    tol : float
        Threshold below which singular values are treated as zero when
        inverting SW (used by both solvers).
    priors : array-like of shape (n_classes,) | None
        Class prior probabilities in the order of ``classes_``.  ``None``
        uses empirical class frequencies.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Discriminant axes, ordered by descending discriminability.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total between-class variance captured by each component.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels encountered during fit.
    means_ : ndarray of shape (n_classes, n_features)
        Per-class feature means.
    priors_ : ndarray of shape (n_classes,)
        Class prior probabilities used for classification.
    scalings_ : ndarray of shape (n_features, n_components)
        Scaling vectors (eigenvectors in feature space), one column per
        discriminant direction.
    xbar_ : ndarray of shape (n_features,)
        Overall weighted mean of the training data.
    n_components_ : int
        Resolved number of discriminant directions after fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_scratch.decomposition import LinearDiscriminantAnalysis
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal([0, 0], 1, (50, 2)),
    ...                rng.normal([3, 3], 1, (50, 2)),
    ...                rng.normal([6, 0], 1, (50, 2))])
    >>> y = np.array([0]*50 + [1]*50 + [2]*50)
    >>> lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
    >>> X_t = lda.transform(X)          # shape (150, 2)
    >>> acc = (lda.predict(X) == y).mean()
    >>> acc > 0.95
    True
    """

    def __init__(
        self,
        n_components: int | None = None,
        solver: str = "eigen",
        tol: float = 1e-4,
        priors: np.ndarray | None = None,
    ) -> None:
        if solver not in ("eigen", "svd"):
            raise ValueError(f"solver must be 'eigen' or 'svd'; got {solver!r}")
        self.n_components = n_components
        self.solver = solver
        self.tol = tol
        self.priors = priors

        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None
        self.scalings_: np.ndarray | None = None
        self.xbar_: np.ndarray | None = None
        self.n_components_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearDiscriminantAnalysis":
        """Compute discriminant directions from labelled training data.

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
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Class counts and means
        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)

        if self.priors is not None:
            self.priors_ = np.asarray(self.priors, dtype=float)
            if len(self.priors_) != n_classes:
                raise ValueError(
                    f"priors length {len(self.priors_)} != n_classes {n_classes}"
                )
            self.priors_ = self.priors_ / self.priors_.sum()
        else:
            self.priors_ = counts / n_samples

        self.means_ = np.array(
            [X[y == c].mean(axis=0) for c in self.classes_], dtype=float
        )
        self.xbar_ = (self.priors_[:, None] * self.means_).sum(axis=0)

        max_components = min(n_features, n_classes - 1)
        n_keep = self.n_components if self.n_components is not None else max_components
        n_keep = min(n_keep, max_components)
        self.n_components_ = n_keep

        if self.solver == "eigen":
            scalings, ev_ratio = self._solve_eigen(X, y, counts, n_features, n_keep)
        else:
            scalings, ev_ratio = self._solve_svd(X, y, counts, n_features, n_keep)

        self.scalings_ = scalings                          # (n_features, n_keep)
        self.components_ = scalings.T                     # (n_keep, n_features)
        self.explained_variance_ratio_ = ev_ratio
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the discriminant subspace.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return (X - self.xbar_) @ self.scalings_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and project in one step."""
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using nearest centroid in the discriminant space.

        Each sample is assigned to the class whose projected mean is closest
        (Euclidean distance), weighted by log-prior.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,) with values from ``classes_``
        """
        self._check_fitted()
        log_proba = self._log_posterior(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior class probabilities (softmax of log-posteriors).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        log_p = self._log_posterior(X)
        # Numerically stable softmax
        log_p -= log_p.max(axis=1, keepdims=True)
        exp_p = np.exp(log_p)
        return exp_p / exp_p.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _solve_eigen(
        self,
        X: np.ndarray,
        y: np.ndarray,
        counts: np.ndarray,
        n_features: int,
        n_keep: int,
    ):
        """Solve the generalised eigenvalue problem SW⁻¹ SB."""
        n_samples = len(y)

        # Within-class scatter SW
        SW = np.zeros((n_features, n_features))
        for k, c in enumerate(self.classes_):
            Xc = X[y == c] - self.means_[k]
            SW += Xc.T @ Xc

        # Between-class scatter SB
        SB = np.zeros((n_features, n_features))
        for k in range(len(self.classes_)):
            diff = (self.means_[k] - self.xbar_)[:, None]
            SB += counts[k] * (diff @ diff.T)

        # Regularise SW for stability
        SW += self.tol * np.eye(n_features)

        # Solve SW⁻¹ SB — use eigh on symmetric form for stability
        try:
            SW_inv = np.linalg.inv(SW)
        except np.linalg.LinAlgError:
            SW_inv = np.linalg.pinv(SW)

        M = SW_inv @ SB
        eigenvalues, eigenvectors = np.linalg.eig(M)

        # Keep real parts (numerical noise may introduce tiny imaginary parts)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues = np.maximum(eigenvalues, 0.0)
        total = eigenvalues.sum()
        ev_ratio = (eigenvalues[:n_keep] / total) if total > 0 else np.zeros(n_keep)

        scalings = eigenvectors[:, :n_keep]
        return scalings, ev_ratio

    def _solve_svd(
        self,
        X: np.ndarray,
        y: np.ndarray,
        counts: np.ndarray,
        n_features: int,
        n_keep: int,
    ):
        """SVD-based solver: whiten via within-class SVD, then decompose SB."""
        # Build class-centred data matrix
        Xc_list = []
        for k, c in enumerate(self.classes_):
            Xc_list.append(X[y == c] - self.means_[k])
        Xc_all = np.vstack(Xc_list)

        # Whiten using SVD of within-class scatter
        _, S, Vt = np.linalg.svd(Xc_all, full_matrices=False)
        rank = np.sum(S > self.tol * S[0])
        S_inv = 1.0 / S[:rank]
        W = (Vt[:rank].T * S_inv)  # whitening matrix (n_features, rank)

        # Between-class scatter in whitened space
        means_w = self.means_ @ W          # (n_classes, rank)
        xbar_w = self.xbar_ @ W

        SB_w = np.zeros((rank, rank))
        for k in range(len(self.classes_)):
            diff = (means_w[k] - xbar_w)[:, None]
            SB_w += counts[k] * (diff @ diff.T)

        # SVD of between-class scatter (symmetric → eigh)
        eigenvalues, eigenvectors = np.linalg.eigh(SB_w)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[idx], 0.0)
        eigenvectors = eigenvectors[:, idx]

        total = eigenvalues.sum()
        ev_ratio = (eigenvalues[:n_keep] / total) if total > 0 else np.zeros(n_keep)

        # Map back to original feature space
        scalings = W @ eigenvectors[:, :n_keep]   # (n_features, n_keep)
        return scalings, ev_ratio

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _log_posterior(self, X: np.ndarray) -> np.ndarray:
        """Compute log-posterior scores for each class (n_samples, n_classes)."""
        X_t = self.transform(X)                                   # (n, k)
        means_t = (self.means_ - self.xbar_) @ self.scalings_    # (C, k)

        # Squared Euclidean distances to projected class means
        dists = np.sum((X_t[:, None, :] - means_t[None, :, :]) ** 2, axis=2)
        return -0.5 * dists + np.log(self.priors_)[None, :]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform() / predict().")
