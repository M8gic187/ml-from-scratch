"""Demo: AgglomerativeClustering — four linkage strategies on synthetic data."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import AgglomerativeClustering
from ml_scratch.utils.metrics import silhouette_score

rng = np.random.default_rng(42)


def _make_blobs(centers, n_per=60, spread=0.5):
    return np.vstack([
        rng.multivariate_normal(c, np.eye(2) * spread, n_per)
        for c in centers
    ])


# ── Demo 1: 2-cluster separation ─────────────────────────────────────────────
centers_2 = [[-5, 0], [5, 0]]
X2 = _make_blobs(centers_2, n_per=80)

agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
agg.fit(X2)

print("── Demo 1: Ward linkage on 2 well-separated blobs ─────────────")
print(f"  Samples         : {X2.shape[0]}")
print(f"  Clusters found  : {len(np.unique(agg.labels_))}")
print(f"  Cluster sizes   : {[int((agg.labels_ == l).sum()) for l in np.unique(agg.labels_)]}")
print(f"  Silhouette score: {silhouette_score(X2, agg.labels_):.4f}")


# ── Demo 2: 3-cluster, all four linkage methods compared ─────────────────────
centers_3 = [[-6, -4], [6, -4], [0, 5]]
X3 = _make_blobs(centers_3, n_per=60, spread=0.6)

print()
print("── Demo 2: 3 clusters — linkage method comparison ─────────────")
print(f"  {'Linkage':<12} {'Sizes':<24} {'Silhouette':>10}")
print(f"  {'-'*50}")

for linkage in ("single", "complete", "average", "ward"):
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    labels = model.fit_predict(X3)
    sizes = sorted([(labels == l).sum() for l in np.unique(labels)], reverse=True)
    sil = silhouette_score(X3, labels)
    print(f"  {linkage:<12} {str(sizes):<24} {sil:>10.4f}")


# ── Demo 3: 4-cluster data, Ward vs complete ─────────────────────────────────
centers_4 = [[-5, -5], [5, -5], [-5, 5], [5, 5]]
X4 = _make_blobs(centers_4, n_per=50, spread=0.7)

print()
print("── Demo 3: 4 clusters — Ward vs complete linkage ──────────────")
for linkage in ("ward", "complete"):
    model = AgglomerativeClustering(n_clusters=4, linkage=linkage)
    labels = model.fit_predict(X4)
    sil = silhouette_score(X4, labels)
    sizes = sorted([(labels == l).sum() for l in np.unique(labels)], reverse=True)
    print(f"  {linkage:<10}: sizes={sizes}  silhouette={sil:.4f}")


# ── Demo 4: Manhattan metric (single + complete linkage) ─────────────────────
X_m = _make_blobs([[-4, 0], [4, 0]], n_per=50, spread=0.4)

print()
print("── Demo 4: Manhattan metric ────────────────────────────────────")
for linkage in ("single", "complete", "average"):
    model = AgglomerativeClustering(n_clusters=2, linkage=linkage, metric="manhattan")
    labels = model.fit_predict(X_m)
    sil = silhouette_score(X_m, labels)
    sizes = sorted([(labels == l).sum() for l in np.unique(labels)], reverse=True)
    print(f"  {linkage:<10}: sizes={sizes}  silhouette={sil:.4f}")

print()
print("All demos completed successfully.")
