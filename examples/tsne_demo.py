"""t-SNE demonstration: visualise high-dimensional cluster structure in 2-D."""

from __future__ import annotations

import numpy as np

from ml_scratch.decomposition import PCA, TSNE

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Synthetic dataset: 4 Gaussian clusters in 20-D
# ---------------------------------------------------------------------------

n_per_class = 60
centres = rng.uniform(-8, 8, (4, 20))
X = np.vstack([rng.standard_normal((n_per_class, 20)) + c for c in centres])
y = np.repeat(np.arange(4), n_per_class)

print("Dataset: 240 samples × 20 features, 4 classes")
print()

# ---------------------------------------------------------------------------
# Example 1 — Default settings (perplexity=30, 1000 iterations)
# ---------------------------------------------------------------------------

tsne = TSNE(n_components=2, perplexity=30, n_iterations=1000, random_state=42)
Y = tsne.fit_transform(X)

print("=== Example 1: Default t-SNE (perplexity=30) ===")
print(f"Embedding shape : {Y.shape}")
print(f"KL divergence   : {tsne.kl_divergence_:.4f}")
print(f"Iterations      : {tsne.n_iter_}")
print()

# Cluster separation: intra-class vs inter-class centroid distance
centroids = np.array([Y[y == k].mean(axis=0) for k in range(4)])
intra = np.mean([np.mean(np.linalg.norm(Y[y == k] - centroids[k], axis=1)) for k in range(4)])
inter = np.mean(
    [np.linalg.norm(centroids[i] - centroids[j])
     for i in range(4) for j in range(i + 1, 4)]
)
print(f"  Mean intra-cluster distance : {intra:.2f}")
print(f"  Mean inter-cluster distance : {inter:.2f}")
print(f"  Separation ratio            : {inter / intra:.2f}x")
print()

# ---------------------------------------------------------------------------
# Example 2 — Low perplexity (focus on very local structure)
# ---------------------------------------------------------------------------

tsne_low = TSNE(n_components=2, perplexity=5, n_iterations=800, random_state=0)
Y_low = tsne_low.fit_transform(X)

print("=== Example 2: Low perplexity (perplexity=5) ===")
print(f"KL divergence : {tsne_low.kl_divergence_:.4f}")
print()

# ---------------------------------------------------------------------------
# Example 3 — High perplexity (capture more global structure)
# ---------------------------------------------------------------------------

tsne_high = TSNE(n_components=2, perplexity=50, n_iterations=800, random_state=0)
Y_high = tsne_high.fit_transform(X)

print("=== Example 3: High perplexity (perplexity=50) ===")
print(f"KL divergence : {tsne_high.kl_divergence_:.4f}")
print()

# ---------------------------------------------------------------------------
# Example 4 — 3-D embedding
# ---------------------------------------------------------------------------

tsne_3d = TSNE(n_components=3, perplexity=20, n_iterations=500, random_state=7)
Y_3d = tsne_3d.fit_transform(X)

print("=== Example 4: 3-D embedding ===")
print(f"Embedding shape : {Y_3d.shape}")
print(f"KL divergence   : {tsne_3d.kl_divergence_:.4f}")
print()

# ---------------------------------------------------------------------------
# Example 5 — PCA pre-whitening (common practice for high-D data)
# ---------------------------------------------------------------------------

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

tsne_pca = TSNE(n_components=2, perplexity=25, n_iterations=800, random_state=3)
Y_pca = tsne_pca.fit_transform(X_pca)

print("=== Example 5: PCA(10) → t-SNE pipeline ===")
print(f"PCA variance retained : {pca.explained_variance_ratio_.sum():.3f}")
print(f"t-SNE KL divergence   : {tsne_pca.kl_divergence_:.4f}")
print()

# ---------------------------------------------------------------------------
# Example 6 — Auto learning rate
# ---------------------------------------------------------------------------

tsne_auto = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    n_iterations=500,
    random_state=42,
)
Y_auto = tsne_auto.fit_transform(X)

print("=== Example 6: learning_rate='auto' ===")
print(f"KL divergence : {tsne_auto.kl_divergence_:.4f}")
print()

print("All examples completed successfully.")
