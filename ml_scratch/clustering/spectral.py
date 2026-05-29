"""Spectral Clustering via graph Laplacian embedding."""

import numpy as np
from .kmeans import KMeans


class SpectralClustering:
    """Graph-based clustering via spectral embedding.

    Constructs an affinity graph from the input data, embeds points into a
    low-dimensional space as the leading eigenvectors of the normalised
    symmetric graph Laplacian, then clusters the embedding with K-Means.

    This approach separates non-convex clusters (e.g. concentric rings or
    interlocking crescents) that distance-only methods like K-Means cannot
    handle on the raw feature space.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (must be >= 2).
    affinity : {"rbf", "nearest_neighbors"}
        Strategy for building the affinity matrix:

        * ``"rbf"`` – Gaussian kernel
          ``W[i,j] = exp(-||xᵢ-xⱼ||² / (2·gamma²))``.
        * ``"nearest_neighbors"`` – symmetric binary k-NN graph; two nodes
          are connected if either is among the other's k nearest neighbours.
    gamma : float
        Bandwidth of the RBF kernel.  Larger values make the kernel wider
        (smoother affinity).  Ignored when ``affinity="nearest_neighbors"``.
    n_neighbors : int
        Number of neighbours for the k-NN graph.  Ignored when
        ``affinity="rbf"``.
    random_state : int | None
        Seed for K-Means reproducibility.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label (0 … n_clusters-1) for every input sample.
    embedding_ : ndarray of shape (n_samples, n_clusters)
        Row-normalised spectral embedding passed to K-Means.
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Affinity matrix constructed during ``fit``.
    """

    _VALID_AFFINITY = ("rbf", "nearest_neighbors")

    def __init__(
        self,
        n_clusters: int = 2,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        random_state: int | None = None,
    ) -> None:
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if affinity not in self._VALID_AFFINITY:
            raise ValueError(
                f"affinity must be one of {self._VALID_AFFINITY}, got '{affinity}'"
            )
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")

        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self.labels_: np.ndarray | None = None
        self.embedding_: np.ndarray | None = None
        self.affinity_matrix_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "SpectralClustering":
        """Compute spectral clustering on *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if n < self.n_clusters:
            raise ValueError(
                f"n_samples ({n}) must be >= n_clusters ({self.n_clusters})."
            )

        # 1. Affinity matrix W  (symmetric, non-negative)
        W = self._build_affinity(X)
        self.affinity_matrix_ = W

        # 2. Normalised symmetric Laplacian:  L_sym = I - D^{-1/2} W D^{-1/2}
        d = W.sum(axis=1)                                       # degree vector
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)   # isolated → 0
        # Efficient: scale rows then columns instead of full matrix products
        Wn = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]     # D^{-1/2}WD^{-1/2}
        L_sym = np.eye(n) - Wn

        # 3. Eigendecomposition — eigh returns values in ascending order
        _, eigvecs = np.linalg.eigh(L_sym)
        U = eigvecs[:, : self.n_clusters]                       # (n, k)

        # 4. Row-normalise so K-Means operates on the unit sphere
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        U_norm = U / norms
        self.embedding_ = U_norm

        # 5. K-Means on the spectral embedding
        km = KMeans(k=self.n_clusters, random_state=self.random_state)
        km.fit(U_norm)
        self.labels_ = km.labels_
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels (0 … n_clusters-1)."""
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Affinity builders
    # ------------------------------------------------------------------

    def _build_affinity(self, X: np.ndarray) -> np.ndarray:
        if self.affinity == "rbf":
            return self._rbf_affinity(X)
        return self._knn_affinity(X)

    def _rbf_affinity(self, X: np.ndarray) -> np.ndarray:
        """W[i,j] = exp(-||xᵢ-xⱼ||² / (2·gamma²))."""
        sq = np.sum(X ** 2, axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-dist_sq / (2.0 * self.gamma ** 2))

    def _knn_affinity(self, X: np.ndarray) -> np.ndarray:
        """Symmetric binary k-NN affinity matrix."""
        n = X.shape[0]
        k = min(self.n_neighbors, n - 1)
        sq = np.sum(X ** 2, axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        dist_sq = np.maximum(dist_sq, 0.0)

        W = np.zeros((n, n), dtype=float)
        for i in range(n):
            nn_idx = np.argsort(dist_sq[i])[1 : k + 1]   # exclude self
            W[i, nn_idx] = 1.0

        return np.maximum(W, W.T)   # symmetrise: connect if either direction holds

    def _check_fitted(self) -> None:
        if self.labels_ is None:
            raise RuntimeError("Call fit() before accessing fitted attributes.")
