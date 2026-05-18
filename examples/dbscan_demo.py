"""Demo: DBSCAN on three synthetic datasets (blobs, moons, noise)."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import DBSCAN
from ml_scratch.utils.metrics import silhouette_score

rng = np.random.default_rng(42)

# ── Dataset 1: well-separated blobs ─────────────────────────────────────────
centers = [[-6, 0], [0, 6], [6, 0]]
X_blobs = np.vstack([
    rng.multivariate_normal(c, np.eye(2) * 0.4, 80)
    for c in centers
])

db_blobs = DBSCAN(eps=1.2, min_samples=5)
db_blobs.fit(X_blobs)

print("── Blobs dataset ──────────────────────────────────────────────")
print(f"  Points          : {X_blobs.shape[0]}")
print(f"  Clusters found  : {db_blobs.n_clusters_}  (expected 3)")
print(f"  Noise points    : {(db_blobs.labels_ == -1).sum()}")
print(f"  Core points     : {len(db_blobs.core_sample_indices_)}")
print(f"  Silhouette score: {silhouette_score(X_blobs, db_blobs.labels_):.4f}")

# ── Dataset 2: crescent / interlocking arcs ──────────────────────────────────
t = np.linspace(0, np.pi, 120)
arc_a = np.column_stack([np.cos(t), np.sin(t)]) * 3.0
arc_b = np.column_stack([np.cos(t) + 1.5, -np.sin(t)]) * 3.0
noise_a = rng.normal(0, 0.15, arc_a.shape)
noise_b = rng.normal(0, 0.15, arc_b.shape)
X_moons = np.vstack([arc_a + noise_a, arc_b + noise_b])

db_moons = DBSCAN(eps=0.5, min_samples=4)
db_moons.fit(X_moons)

print()
print("── Crescent arcs dataset ──────────────────────────────────────")
print(f"  Points          : {X_moons.shape[0]}")
print(f"  Clusters found  : {db_moons.n_clusters_}  (expected 2)")
print(f"  Noise points    : {(db_moons.labels_ == -1).sum()}")
print(f"  Silhouette score: {silhouette_score(X_moons, db_moons.labels_):.4f}")

# ── Dataset 3: cluster + scattered outliers ──────────────────────────────────
cluster  = rng.multivariate_normal([0, 0], np.eye(2) * 0.3, 60)
outliers = rng.uniform(-10, 10, (15, 2))
X_noisy  = np.vstack([cluster, outliers])

db_noisy = DBSCAN(eps=0.7, min_samples=5)
db_noisy.fit(X_noisy)

print()
print("── Cluster + outliers dataset ─────────────────────────────────")
print(f"  Points          : {X_noisy.shape[0]}")
print(f"  Clusters found  : {db_noisy.n_clusters_}  (expected 1)")
print(f"  Noise / outliers: {(db_noisy.labels_ == -1).sum()}  (injected: 15)")
print(f"  Silhouette score: {silhouette_score(X_noisy, db_noisy.labels_):.4f}")

# ── Comparison: DBSCAN vs KMeans on noisy data ───────────────────────────────
from ml_scratch.clustering import KMeans

km = KMeans(k=2, random_state=0).fit(X_noisy)
print()
print("── DBSCAN vs K-Means on noisy data ────────────────────────────")
print(f"  K-Means  silhouette: {silhouette_score(X_noisy, km.labels_):.4f}"
      "  (forced k=2, no noise concept)")
print(f"  DBSCAN   silhouette: {silhouette_score(X_noisy, db_noisy.labels_):.4f}"
      "  (noise excluded from score)")
