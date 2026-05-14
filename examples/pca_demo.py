"""Demo: PCA for dimensionality reduction and reconstruction on synthetic data."""

import numpy as np

from ml_scratch.decomposition import PCA
from ml_scratch.utils.metrics import mse


def make_dataset(rng: np.random.Generator) -> np.ndarray:
    """10-D dataset with 3 dominant directions."""
    t = rng.standard_normal((300, 3))
    W = rng.standard_normal((3, 10))
    noise = 0.2 * rng.standard_normal((300, 10))
    return t @ W + noise


def main() -> None:
    rng = np.random.default_rng(0)
    X = make_dataset(rng)

    pca_full = PCA().fit(X)

    print("Explained variance ratio per component (all 10):")
    for i, r in enumerate(pca_full.explained_variance_ratio_, 1):
        bar = "#" * int(r * 40)
        print(f"  PC{i:2d}: {r:.4f}  {bar}")

    cumulative = pca_full.explained_variance_ratio_.cumsum()
    for k in (1, 2, 3, 5, 7, 10):
        print(f"  Cumulative variance (top {k:2d} PCs): {cumulative[k-1]:.4f}")

    print()
    print(f"{'n_components':<16} {'Reconstruction MSE':>20}")
    print("-" * 38)
    for k in (1, 2, 3, 5, 7, 10):
        pca = PCA(n_components=k).fit(X)
        X_back = pca.inverse_transform(pca.transform(X))
        print(f"{k:<16} {mse(X.ravel(), X_back.ravel()):>20.4f}")


if __name__ == "__main__":
    main()
