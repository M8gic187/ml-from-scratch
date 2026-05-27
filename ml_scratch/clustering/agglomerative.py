"""Agglomerative (Hierarchical) Clustering with multiple linkage criteria."""

import numpy as np


class AgglomerativeClustering:
    """Bottom-up hierarchical clustering via greedy cluster merging.

    At each step the two closest clusters are merged according to the chosen
    *linkage* criterion.  Merging continues until the desired number of
    clusters is reached.

    Parameters
    ----------
    n_clusters : int
        Target number of clusters.
    linkage : {"single", "complete", "average", "ward"}
        Criterion used to measure distance between two clusters:

        * ``"single"``   – minimum pairwise distance (susceptible to chaining).
        * ``"complete"`` – maximum pairwise distance (produces compact clusters).
        * ``"average"``  – mean pairwise distance (UPGMA; balanced trade-off).
        * ``"ward"``     – minimises total within-cluster variance; tends to
                           produce equally sized, spherical clusters.
    metric : {"euclidean", "manhattan"}
        Distance metric used to compute pairwise point distances.
        Ward linkage requires ``metric="euclidean"``.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label (0 … n_clusters-1) for every input sample.
    n_clusters_ : int
        Number of clusters (equals *n_clusters*).
    """

    _VALID_LINKAGE = ("single", "complete", "average", "ward")
    _VALID_METRIC = ("euclidean", "manhattan")

    def __init__(
        self,
        n_clusters: int = 2,
        linkage: str = "ward",
        metric: str = "euclidean",
    ) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if linkage not in self._VALID_LINKAGE:
            raise ValueError(
                f"linkage must be one of {self._VALID_LINKAGE}, got '{linkage}'"
            )
        if metric not in self._VALID_METRIC:
            raise ValueError(
                f"metric must be one of {self._VALID_METRIC}, got '{metric}'"
            )
        if linkage == "ward" and metric != "euclidean":
            raise ValueError("Ward linkage requires metric='euclidean'.")

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

        self.labels_: np.ndarray | None = None
        self.n_clusters_: int = n_clusters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "AgglomerativeClustering":
        """Compute agglomerative clustering on *X*.

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

        # Bookkeeping
        members: dict[int, list[int]] = {i: [i] for i in range(n)}
        sizes: dict[int, int] = {i: 1 for i in range(n)}
        # Centroids are only needed for Ward distance; cheap to track always.
        centroids: dict[int, np.ndarray] = {i: X[i].copy() for i in range(n)}
        active: set[int] = set(range(n))

        # Build initial pairwise cluster-distance dictionary (upper triangle).
        point_dist = self._pairwise(X)
        cdist: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                cdist[(i, j)] = float(point_dist[i, j])

        next_id = n  # IDs for merged clusters start at n

        while len(active) > self.n_clusters:
            # Locate the pair of active clusters with minimum distance.
            min_d = float("inf")
            best_a = best_b = -1
            for (a, b), d in cdist.items():
                if a in active and b in active and d < min_d:
                    min_d = d
                    best_a, best_b = a, b

            a, b = best_a, best_b
            n_a, n_b = sizes[a], sizes[b]
            n_ab = n_a + n_b

            # Merge a and b into a new cluster.
            members[next_id] = members[a] + members[b]
            sizes[next_id] = n_ab
            centroids[next_id] = (n_a * centroids[a] + n_b * centroids[b]) / n_ab

            active.discard(a)
            active.discard(b)

            # Compute distance from the new cluster to every remaining cluster.
            key_ab = (min(a, b), max(a, b))
            d_ab = cdist.get(key_ab, 0.0)

            for c in active:
                key_ac = (min(a, c), max(a, c))
                key_bc = (min(b, c), max(b, c))
                d_ac = cdist.get(key_ac, float("inf"))
                d_bc = cdist.get(key_bc, float("inf"))
                n_c = sizes[c]

                new_d = self._merge_dist(
                    d_ac, d_bc, d_ab, n_a, n_b, n_c, next_id, c, centroids
                )
                cdist[(min(next_id, c), max(next_id, c))] = new_d

            active.add(next_id)
            next_id += 1

        # Assign contiguous labels 0 … n_clusters-1.
        self.labels_ = np.empty(n, dtype=int)
        for label, cid in enumerate(sorted(active)):
            for idx in members[cid]:
                self.labels_[idx] = label

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels for *X*."""
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pairwise(self, X: np.ndarray) -> np.ndarray:
        """Return the (n, n) pairwise distance matrix for *X*."""
        if self.metric == "euclidean":
            sq = np.sum(X ** 2, axis=1)
            d_sq = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
            return np.sqrt(np.maximum(d_sq, 0.0))
        # Manhattan
        return np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=2)

    def _merge_dist(
        self,
        d_ac: float,
        d_bc: float,
        d_ab: float,
        n_a: int,
        n_b: int,
        n_c: int,
        new_id: int,
        c: int,
        centroids: dict[int, np.ndarray],
    ) -> float:
        """Distance from the merged cluster (a ∪ b) to cluster c.

        Uses the Lance-Williams update formula for single / complete / average
        and the centroid-based Ward formula for ward linkage.
        """
        n_ab = n_a + n_b

        if self.linkage == "single":
            return min(d_ac, d_bc)

        if self.linkage == "complete":
            return max(d_ac, d_bc)

        if self.linkage == "average":
            # UPGMA: weighted average of existing distances
            return (n_a * d_ac + n_b * d_bc) / n_ab

        # ward: sqrt(n_ab * n_c / (n_ab + n_c)) * ||centroid_ab - centroid_c||
        n_abc = n_ab + n_c
        return float(
            np.sqrt(n_ab * n_c / n_abc)
            * np.linalg.norm(centroids[new_id] - centroids[c])
        )

    def _check_fitted(self) -> None:
        if self.labels_ is None:
            raise RuntimeError("Call fit() before accessing fitted attributes.")
