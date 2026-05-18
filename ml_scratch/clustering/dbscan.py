"""DBSCAN: Density-Based Spatial Clustering of Applications with Noise."""

import numpy as np


class DBSCAN:
    """Density-based clustering that discovers arbitrarily shaped clusters.

    Points are classified as core, border, or noise based on the density
    of their local neighbourhood.  Cluster labels are assigned by
    breadth-first expansion from every core point.

    Parameters
    ----------
    eps : float
        Radius of the neighbourhood (epsilon).  Two points are considered
        neighbours if their distance is <= *eps*.
    min_samples : int
        Minimum number of points (including the point itself) within
        *eps* to classify a point as a *core* point.
    metric : {"euclidean", "manhattan"}
        Distance metric used to compute pairwise distances.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label for each input point.  Noise points receive ``-1``.
    core_sample_indices_ : ndarray
        Indices of all core points in the fitted data.
    n_clusters_ : int
        Number of clusters found (excluding noise).
    """

    NOISE = -1

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(f"Unknown metric '{metric}'. Choose 'euclidean' or 'manhattan'.")

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.labels_: np.ndarray | None = None
        self.core_sample_indices_: np.ndarray | None = None
        self.n_clusters_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """Compute DBSCAN clustering on *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        dist = self._pairwise_distances(X)
        # neighbours[i] = indices of all points within eps of point i (i included)
        neighbours = [np.where(dist[i] <= self.eps)[0] for i in range(n)]
        is_core = np.array([len(nb) >= self.min_samples for nb in neighbours])

        labels = np.full(n, self.NOISE, dtype=int)
        cluster_id = 0

        for i in range(n):
            if not is_core[i] or labels[i] != self.NOISE:
                continue
            # Start a new cluster from core point i
            self._expand_cluster(i, cluster_id, labels, neighbours, is_core)
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels.  Noise points are labelled ``-1``."""
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_cluster(
        self,
        seed: int,
        cluster_id: int,
        labels: np.ndarray,
        neighbours: list,
        is_core: np.ndarray,
    ) -> None:
        """BFS expansion: assign *cluster_id* to all density-reachable points."""
        queue = list(neighbours[seed])
        labels[seed] = cluster_id

        while queue:
            point = queue.pop()
            if labels[point] == self.NOISE:
                # Border point absorbed into this cluster
                labels[point] = cluster_id
            if labels[point] != self.NOISE:
                continue
            labels[point] = cluster_id
            if is_core[point]:
                for nb in neighbours[point]:
                    if labels[nb] == self.NOISE:
                        queue.append(nb)

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Return the full (n, n) pairwise distance matrix."""
        if self.metric == "euclidean":
            # ||a - b||² = ||a||² + ||b||² - 2 a·b  (vectorised)
            sq = np.sum(X ** 2, axis=1)
            dist_sq = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
            # Numerical noise can yield tiny negatives; clip before sqrt
            return np.sqrt(np.maximum(dist_sq, 0.0))
        # manhattan
        return np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=2)

    def _check_fitted(self) -> None:
        if self.labels_ is None:
            raise RuntimeError("Call fit() before accessing fitted attributes.")
