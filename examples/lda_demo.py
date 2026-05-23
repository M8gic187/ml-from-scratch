"""Linear Discriminant Analysis demo.

Four examples that illustrate the main capabilities:
  1. Dimensionality reduction on a 3-class 2-D toy problem
  2. LDA vs PCA on iris-like 4-D data
  3. Binary classification with custom class priors
  4. Solver comparison (eigen vs svd) on high-dimensional data
"""

from __future__ import annotations

import numpy as np

from ml_scratch.decomposition import LinearDiscriminantAnalysis, PCA


def _accuracy(lda, X, y):
    return (lda.predict(X) == y).mean()


# ---------------------------------------------------------------------------
# Example 1 – Dimensionality reduction on a 3-class toy problem
# ---------------------------------------------------------------------------

def example_1_three_class_reduction():
    print("=" * 60)
    print("Example 1 – 3-class dimensionality reduction (2-D → 1-D)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    centers = [(0, 0), (6, 0), (3, 5)]
    X = np.vstack([rng.normal(c, 1.0, (50, 2)) for c in centers])
    y = np.repeat([0, 1, 2], 50)

    lda = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
    X_t = lda.transform(X)

    print(f"  Original shape  : {X.shape}")
    print(f"  Reduced shape   : {X_t.shape}")
    print(f"  EVR[0]          : {lda.explained_variance_ratio_[0]:.4f}")
    print(f"  Train accuracy  : {_accuracy(lda, X, y):.3f}")
    print()


# ---------------------------------------------------------------------------
# Example 2 – LDA vs PCA on iris-like 4-D data
# ---------------------------------------------------------------------------

def example_2_lda_vs_pca():
    print("=" * 60)
    print("Example 2 – LDA vs PCA on iris-like 4-D data")
    print("=" * 60)

    rng = np.random.default_rng(1)
    # Three linearly separable classes in 4-D feature space
    means = np.array([[0, 0, 0, 0],
                      [2, 2, 0, 0],
                      [0, 0, 2, 2]], dtype=float)
    X = np.vstack([rng.normal(m, 0.8, (50, 4)) for m in means])
    y = np.repeat([0, 1, 2], 50)

    lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
    pca = PCA(n_components=2).fit(X)

    # Nearest-centroid classifier in PCA space for fair comparison
    X_pca = pca.transform(X)
    X_lda = lda.transform(X)

    pca_means = np.array([X_pca[y == c].mean(axis=0) for c in [0, 1, 2]])
    dists_pca = np.linalg.norm(X_pca[:, None] - pca_means[None], axis=2)
    pred_pca = np.argmin(dists_pca, axis=1)
    acc_pca = (pred_pca == y).mean()

    print(f"  LDA accuracy    : {_accuracy(lda, X, y):.3f}")
    print(f"  PCA + NC acc.   : {acc_pca:.3f}  (nearest-centroid in PCA space)")
    print(f"  LDA EVR         : {lda.explained_variance_ratio_.round(3)}")
    print(f"  PCA EVR         : {pca.explained_variance_ratio_[:2].round(3)}")
    print()


# ---------------------------------------------------------------------------
# Example 3 – Binary classification with imbalanced priors
# ---------------------------------------------------------------------------

def example_3_custom_priors():
    print("=" * 60)
    print("Example 3 – Binary classification with custom priors")
    print("=" * 60)

    rng = np.random.default_rng(2)
    # 90 % class 0, 10 % class 1
    X0 = rng.normal([0, 0], 1.0, (180, 2))
    X1 = rng.normal([2, 2], 1.0, (20, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 180 + [1] * 20)

    lda_emp = LinearDiscriminantAnalysis().fit(X, y)                     # empirical priors
    lda_uni = LinearDiscriminantAnalysis(priors=[0.5, 0.5]).fit(X, y)   # uniform priors
    lda_true = LinearDiscriminantAnalysis(priors=[0.9, 0.1]).fit(X, y)  # true priors

    for name, lda in [("Empirical", lda_emp), ("Uniform", lda_uni), ("True (0.9/0.1)", lda_true)]:
        pred = lda.predict(X)
        recall_1 = (pred[y == 1] == 1).mean()
        acc = (pred == y).mean()
        print(f"  {name:20s}  acc={acc:.3f}  recall(class 1)={recall_1:.3f}")
    print()


# ---------------------------------------------------------------------------
# Example 4 – Solver comparison on high-dimensional data
# ---------------------------------------------------------------------------

def example_4_solver_comparison():
    print("=" * 60)
    print("Example 4 – eigen vs svd solver on 20-D, 5-class data")
    print("=" * 60)

    rng = np.random.default_rng(3)
    centers = rng.normal(0, 4, (5, 20))
    X_train = np.vstack([rng.normal(c, 1, (60, 20)) for c in centers])
    X_test = np.vstack([rng.normal(c, 1, (20, 20)) for c in centers])
    y_train = np.repeat(np.arange(5), 60)
    y_test = np.repeat(np.arange(5), 20)

    for solver in ("eigen", "svd"):
        lda = LinearDiscriminantAnalysis(n_components=4, solver=solver).fit(X_train, y_train)
        train_acc = _accuracy(lda, X_train, y_train)
        test_acc = _accuracy(lda, X_test, y_test)
        evr_total = lda.explained_variance_ratio_.sum()
        print(f"  solver={solver:5s}  train={train_acc:.3f}  test={test_acc:.3f}  EVR_total={evr_total:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_1_three_class_reduction()
    example_2_lda_vs_pca()
    example_3_custom_priors()
    example_4_solver_comparison()
    print("All examples complete.")
