"""K-Means clustering with k-means++ initialization."""

import numpy as np


class KMeans:
    """K-Means clustering algorithm with k-means++ centroid seeding.

    Parameters
    ----------
    k : int
        Number of clusters.
    n_iterations : int
        Maximum number of expectation-maximization iterations.
    tol : float
        Convergence tolerance: stop when centroid shift is below this value.
    random_state : int | None
        Seed for the random number generator.
    """

    def __init__(
        self,
        k: int = 3,
        n_iterations: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.k = k
        self.n_iterations = n_iterations
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray) -> "KMeans":
        """Compute k-means clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        self.centroids_ = self._init_plus_plus(X, rng)

        for i in range(1, self.n_iterations + 1):
            labels = self._assign(X)
            new_centroids = self._update(X, labels)

            shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            self.labels_ = labels
            self.n_iter_ = i

            if shift < self.tol:
                break

        self.labels_ = self._assign(X)
        self.inertia_ = self._compute_inertia(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample in X to the nearest centroid."""
        self._check_fitted()
        return self._assign(np.asarray(X, dtype=float))

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster labels for X."""
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_plus_plus(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """K-means++ seeding: spread initial centroids probabilistically."""
        n_samples = X.shape[0]
        first_idx = rng.integers(0, n_samples)
        centroids = [X[first_idx]]

        for _ in range(1, self.k):
            dists = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dists / dists.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _assign(self, X: np.ndarray) -> np.ndarray:
        """Return index of nearest centroid for each sample (E-step)."""
        dists = np.stack(
            [np.sum((X - c) ** 2, axis=1) for c in self.centroids_], axis=1
        )
        return np.argmin(dists, axis=1)

    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recompute centroids as cluster means (M-step)."""
        new_centroids = np.zeros_like(self.centroids_)
        for j in range(self.k):
            members = X[labels == j]
            # Keep the old centroid if a cluster becomes empty
            new_centroids[j] = members.mean(axis=0) if len(members) > 0 else self.centroids_[j]
        return new_centroids

    def _compute_inertia(self, X: np.ndarray) -> float:
        """Sum of squared distances of each point to its assigned centroid."""
        total = 0.0
        for j in range(self.k):
            members = X[self.labels_ == j]
            if len(members) > 0:
                total += float(np.sum((members - self.centroids_[j]) ** 2))
        return total

    def _check_fitted(self) -> None:
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict().")
