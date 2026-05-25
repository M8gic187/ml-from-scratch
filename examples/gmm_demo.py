"""Demo: Gaussian Mixture Model on synthetic datasets.

Four examples are shown:

1. Basic 3-component GMM on well-separated blobs.
2. Comparison of covariance types (full / diag / spherical).
3. Soft assignment: inspect posterior probabilities on boundary points.
4. Model selection via BIC to recover the correct number of components.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import GaussianMixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blobs(centers, n_per=150, cov_scale=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.multivariate_normal(c, np.eye(len(c)) * cov_scale, size=n_per)
        for c in centers
    ])


def _cluster_purity(labels_pred, labels_true, k):
    """Fraction of majority-class samples in each predicted cluster (mean)."""
    purities = []
    for c in range(k):
        mask = labels_pred == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels_true[mask], minlength=k)
        purities.append(counts.max() / mask.sum())
    return float(np.mean(purities))


# ---------------------------------------------------------------------------
# Example 1 – Basic GMM with 3 components
# ---------------------------------------------------------------------------

print("=" * 60)
print("Example 1: Basic 3-component GMM (full covariance)")
print("=" * 60)

centers_3 = [[0.0, 0.0], [6.0, 0.0], [3.0, 5.2]]
X3 = _make_blobs(centers_3, n_per=150, cov_scale=0.6, seed=1)
true_labels = np.repeat([0, 1, 2], 150)

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm.fit(X3)
pred = gmm.predict(X3)

print(f"  Converged     : {gmm.converged_} ({gmm.n_iter_} iterations)")
print(f"  Log-likelihood: {gmm.lower_bound_:.4f}")
print(f"  Cluster purity: {_cluster_purity(pred, true_labels, 3):.3f}")
print(f"  Mixture weights: {np.round(gmm.weights_, 3)}")
print(f"  Learned means vs true centers:")
for j, mu in enumerate(gmm.means_):
    closest = centers_3[int(np.argmin([np.linalg.norm(mu - np.array(c)) for c in centers_3]))]
    print(f"    Component {j}: [{mu[0]:+.3f}, {mu[1]:+.3f}]  "
          f"(nearest true: {closest})")

# ---------------------------------------------------------------------------
# Example 2 – Covariance type comparison
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Example 2: Covariance type comparison")
print("=" * 60)
print(f"  {'Type':12s}  {'Converged':10s}  {'Iters':6s}  {'Score':10s}  {'AIC':10s}  {'BIC':10s}")
print("  " + "-" * 58)

for cov_type in ("full", "diag", "spherical"):
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
    gmm.fit(X3)
    print(f"  {cov_type:12s}  {str(gmm.converged_):10s}  {gmm.n_iter_:6d}  "
          f"{gmm.score(X3):10.4f}  {gmm.aic(X3):10.2f}  {gmm.bic(X3):10.2f}")

# ---------------------------------------------------------------------------
# Example 3 – Soft assignments on boundary samples
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Example 3: Soft posterior probabilities on boundary points")
print("=" * 60)

gmm_soft = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm_soft.fit(X3)

# Construct points that lie between each pair of cluster centers
boundary_points = [
    (np.array(centers_3[0]) + np.array(centers_3[1])) / 2,
    (np.array(centers_3[1]) + np.array(centers_3[2])) / 2,
    (np.array(centers_3[0]) + np.array(centers_3[2])) / 2,
]
descriptions = ["midpoint(C0, C1)", "midpoint(C1, C2)", "midpoint(C0, C2)"]

proba = gmm_soft.predict_proba(np.array(boundary_points))
print(f"  {'Point':20s}  {'P(C0)':8s}  {'P(C1)':8s}  {'P(C2)':8s}  {'Hard label':10s}")
print("  " + "-" * 60)
for desc, p in zip(descriptions, proba):
    print(f"  {desc:20s}  {p[0]:.5f}   {p[1]:.5f}   {p[2]:.5f}   C{np.argmax(p)}")

# ---------------------------------------------------------------------------
# Example 4 – Model selection via BIC
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Example 4: Model selection via BIC (true k = 3)")
print("=" * 60)

X_sel = _make_blobs([[0, 0], [8, 0], [4, 7]], n_per=200, cov_scale=0.4, seed=7)
print(f"  {'k':4s}  {'BIC':12s}  {'AIC':12s}  {'Score':12s}")
print("  " + "-" * 44)
best_k, best_bic = None, np.inf
for k in range(1, 7):
    gmm_k = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm_k.fit(X_sel)
    bic = gmm_k.bic(X_sel)
    marker = " <-- best" if bic < best_bic else ""
    if bic < best_bic:
        best_k, best_bic = k, bic
    print(f"  {k:4d}  {bic:12.2f}  {gmm_k.aic(X_sel):12.2f}  {gmm_k.score(X_sel):12.4f}{marker}")

print(f"\n  BIC-optimal k = {best_k}")
