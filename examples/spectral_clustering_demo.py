"""Demo: SpectralClustering on four synthetic datasets."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import SpectralClustering, KMeans
from ml_scratch.utils.metrics import silhouette_score

rng = np.random.default_rng(42)


def _purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Cluster purity: fraction of points that match the majority true label."""
    n = len(labels_true)
    total = 0
    for c in np.unique(labels_pred):
        mask = labels_pred == c
        if mask.sum() == 0:
            continue
        total += np.bincount(labels_true[mask]).max()
    return total / n


# ── Dataset 1: three well-separated blobs (RBF affinity) ────────────────────
centres = [[-8, 0], [8, 0], [0, 8]]
X_blobs = np.vstack([
    rng.multivariate_normal(c, np.eye(2) * 0.4, 80) for c in centres
])
y_blobs = np.repeat([0, 1, 2], 80)

sc_blobs = SpectralClustering(n_clusters=3, affinity="rbf", gamma=2.0, random_state=0)
sc_blobs.fit(X_blobs)

print("── Three blobs  (RBF affinity, gamma=2.0) ─────────────────────────────")
print(f"  Samples         : {X_blobs.shape[0]}")
print(f"  Clusters found  : {len(np.unique(sc_blobs.labels_))}  (expected 3)")
print(f"  Purity          : {_purity(y_blobs, sc_blobs.labels_):.4f}")
print(f"  Silhouette score: {silhouette_score(X_blobs, sc_blobs.labels_):.4f}")

# ── Dataset 2: concentric rings — k-NN affinity ──────────────────────────────
t = np.linspace(0, 2 * np.pi, 150)
inner = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, 0.06, (150, 2))
outer = np.column_stack([3 * np.cos(t), 3 * np.sin(t)]) + rng.normal(0, 0.06, (150, 2))
X_rings = np.vstack([inner, outer])
y_rings = np.array([0] * 150 + [1] * 150)

sc_rings = SpectralClustering(
    n_clusters=2, affinity="nearest_neighbors", n_neighbors=10, random_state=0
)
sc_rings.fit(X_rings)

km_rings = KMeans(k=2, random_state=0).fit(X_rings)

print()
print("── Concentric rings  (k-NN affinity, k=10) ────────────────────────────")
print(f"  Samples         : {X_rings.shape[0]}")
print(f"  SpectralClustering purity   : {_purity(y_rings, sc_rings.labels_):.4f}  "
      "(spectral separates non-convex shapes)")
print(f"  K-Means purity  (baseline)  : {_purity(y_rings, km_rings.labels_):.4f}  "
      "(K-Means fails on rings)")

# ── Dataset 3: interlocking crescents ────────────────────────────────────────
t_half = np.linspace(0, np.pi, 120)
arc_a = np.column_stack([np.cos(t_half), np.sin(t_half)]) * 4.0
arc_b = np.column_stack([np.cos(t_half) + 2.0, -np.sin(t_half)]) * 4.0
arc_a += rng.normal(0, 0.15, arc_a.shape)
arc_b += rng.normal(0, 0.15, arc_b.shape)
X_moons = np.vstack([arc_a, arc_b])
y_moons = np.array([0] * 120 + [1] * 120)

sc_moons = SpectralClustering(
    n_clusters=2, affinity="nearest_neighbors", n_neighbors=8, random_state=0
)
sc_moons.fit(X_moons)

print()
print("── Interlocking crescents  (k-NN affinity, k=8) ───────────────────────")
print(f"  Samples         : {X_moons.shape[0]}")
print(f"  SpectralClustering purity   : {_purity(y_moons, sc_moons.labels_):.4f}")
print(f"  Clusters found  : {len(np.unique(sc_moons.labels_))}  (expected 2)")

# ── Dataset 4: RBF vs k-NN affinity comparison on blobs ─────────────────────
X_cmp = np.vstack([
    rng.multivariate_normal([-5, -5], np.eye(2) * 0.5, 60),
    rng.multivariate_normal([5, -5], np.eye(2) * 0.5, 60),
    rng.multivariate_normal([0, 5], np.eye(2) * 0.5, 60),
])
y_cmp = np.repeat([0, 1, 2], 60)

sc_rbf = SpectralClustering(n_clusters=3, affinity="rbf", gamma=0.5, random_state=0).fit(X_cmp)
sc_knn = SpectralClustering(
    n_clusters=3, affinity="nearest_neighbors", n_neighbors=10, random_state=0
).fit(X_cmp)

print()
print("── RBF vs k-NN affinity (3 balanced blobs) ────────────────────────────")
print(f"  Spectral (RBF, gamma=0.5)  purity: {_purity(y_cmp, sc_rbf.labels_):.4f}  "
      f"silhouette: {silhouette_score(X_cmp, sc_rbf.labels_):.4f}")
print(f"  Spectral (k-NN, k=10)      purity: {_purity(y_cmp, sc_knn.labels_):.4f}  "
      f"silhouette: {silhouette_score(X_cmp, sc_knn.labels_):.4f}")
