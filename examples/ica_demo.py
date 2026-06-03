"""FastICA usage examples.

Six demos covering the main use-cases:

  Demo 1: Blind source separation — recover three independent signals from
          a random linear mixture and measure recovery quality.
  Demo 2: Algorithm comparison — parallel vs deflation on the same data.
  Demo 3: Contrast function sweep — logcosh / exp / cube on Laplace sources.
  Demo 4: Feature extraction — reduce a 20-D dataset to 6 independent
          components and compare variance explained vs PCA.
  Demo 5: PCA + ICA pipeline — pre-whiten with PCA before FastICA on a
          high-dimensional dataset.
  Demo 6: Reconstruction accuracy — measure MSE of inverse_transform across
          different n_components.
"""

import numpy as np

from ml_scratch.decomposition import FastICA, PCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sources(n: int, rng: np.random.Generator) -> np.ndarray:
    """Three independent non-Gaussian sources: Laplace, uniform, sinusoidal."""
    return np.column_stack([
        rng.laplace(size=n),
        rng.uniform(-1.5, 1.5, size=n),
        np.sin(np.linspace(0, 10 * np.pi, n)) + 0.05 * rng.standard_normal(n),
    ])


def _max_abs_corr_per_component(S_true: np.ndarray, S_hat: np.ndarray) -> list[float]:
    """For each recovered component, find its highest absolute correlation
    with any true source."""
    result = []
    for j in range(S_hat.shape[1]):
        best = 0.0
        b = S_hat[:, j] - S_hat[:, j].mean()
        for i in range(S_true.shape[1]):
            a = S_true[:, i] - S_true[:, i].mean()
            r = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            best = max(best, r)
        result.append(best)
    return result


# ---------------------------------------------------------------------------
# Demo 1: Blind source separation
# ---------------------------------------------------------------------------

def demo_blind_source_separation() -> None:
    print("=" * 60)
    print("Demo 1: Blind source separation")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n = 800
    S_true = _make_sources(n, rng)

    # Full-rank random mixing matrix
    A = rng.standard_normal((3, 3))
    while abs(np.linalg.det(A)) < 0.5:
        A = rng.standard_normal((3, 3))
    X = S_true @ A.T

    ica = FastICA(n_components=3, random_state=0, max_iter=500)
    S_hat = ica.fit_transform(X)

    corrs = _max_abs_corr_per_component(S_true, S_hat)
    print(f"  Max |corr| per recovered component: {[f'{c:.3f}' for c in corrs]}")
    print(f"  Mean max |corr|: {np.mean(corrs):.3f}")
    print(f"  Iterations to convergence: {ica.n_iter_}")
    print()


# ---------------------------------------------------------------------------
# Demo 2: Algorithm comparison — parallel vs deflation
# ---------------------------------------------------------------------------

def demo_algorithm_comparison() -> None:
    print("=" * 60)
    print("Demo 2: Parallel vs Deflation algorithm")
    print("=" * 60)

    rng = np.random.default_rng(1)
    n = 600
    S_true = _make_sources(n, rng)
    A = np.array([[2.0, 0.5, 0.1],
                  [0.3, 1.5, 0.4],
                  [0.2, 0.7, 2.0]])
    X = S_true @ A.T

    for algo in ("parallel", "deflation"):
        ica = FastICA(n_components=3, algorithm=algo, random_state=0, max_iter=500)
        S_hat = ica.fit_transform(X)
        corrs = _max_abs_corr_per_component(S_true, S_hat)
        print(f"  {algo:10s}  mean |corr|={np.mean(corrs):.3f}  "
              f"iters={ica.n_iter_}")
    print()


# ---------------------------------------------------------------------------
# Demo 3: Contrast function sweep
# ---------------------------------------------------------------------------

def demo_contrast_functions() -> None:
    print("=" * 60)
    print("Demo 3: Contrast function comparison (logcosh / exp / cube)")
    print("=" * 60)

    rng = np.random.default_rng(2)
    n = 700
    S_true = _make_sources(n, rng)
    A = rng.standard_normal((3, 3))
    X = S_true @ A.T

    for fun in ("logcosh", "exp", "cube"):
        ica = FastICA(n_components=3, fun=fun, random_state=0, max_iter=500)
        S_hat = ica.fit_transform(X)
        corrs = _max_abs_corr_per_component(S_true, S_hat)
        print(f"  fun={fun:7s}  mean |corr|={np.mean(corrs):.3f}  "
              f"iters={ica.n_iter_}")
    print()


# ---------------------------------------------------------------------------
# Demo 4: Feature extraction
# ---------------------------------------------------------------------------

def demo_feature_extraction() -> None:
    print("=" * 60)
    print("Demo 4: Feature extraction (ICA vs PCA)")
    print("=" * 60)

    rng = np.random.default_rng(3)
    n, d = 400, 20
    X = rng.standard_normal((n, d))

    for k in (3, 6, 10):
        ica = FastICA(n_components=k, random_state=0).fit(X)
        pca = PCA(n_components=k).fit(X)

        X_ica = ica.transform(X)
        X_pca = pca.transform(X)

        mse_ica = float(np.mean((X - ica.inverse_transform(X_ica)) ** 2))
        mse_pca = float(np.mean((X - pca.inverse_transform(X_pca)) ** 2))

        print(f"  k={k:2d}  ICA recon MSE={mse_ica:.4f}  "
              f"PCA recon MSE={mse_pca:.4f}  "
              f"(PCA optimal for MSE — ICA targets independence)")
    print()


# ---------------------------------------------------------------------------
# Demo 5: PCA + ICA pipeline on high-dimensional data
# ---------------------------------------------------------------------------

def demo_pca_ica_pipeline() -> None:
    print("=" * 60)
    print("Demo 5: PCA pre-whitening + FastICA pipeline")
    print("=" * 60)

    rng = np.random.default_rng(4)
    n, d = 500, 50
    X = rng.standard_normal((n, d))

    # Step 1: PCA noise reduction
    pca = PCA(n_components=15).fit(X)
    X_pca = pca.transform(X)
    print(f"  PCA: {d}D → 15D  "
          f"(var explained: {pca.explained_variance_ratio_.sum():.3f})")

    # Step 2: FastICA on the reduced representation
    ica = FastICA(n_components=8, random_state=0).fit(X_pca)
    S = ica.transform(X_pca)
    print(f"  ICA: 15D → 8 independent components  "
          f"(iters: {ica.n_iter_})")
    print(f"  Output shape: {S.shape}")
    print(f"  Component kurtoses: "
          f"{[f'{float(np.mean(S[:, j]**4) / max(float(np.mean(S[:, j]**2))**2, 1e-9)):.2f}' for j in range(8)]}")
    print()


# ---------------------------------------------------------------------------
# Demo 6: Reconstruction MSE vs n_components
# ---------------------------------------------------------------------------

def demo_reconstruction_accuracy() -> None:
    print("=" * 60)
    print("Demo 6: Reconstruction MSE vs number of components")
    print("=" * 60)

    rng = np.random.default_rng(5)
    n, d = 300, 10
    S = _make_sources(n, rng)
    pad = rng.standard_normal((n, 7))
    X = np.hstack([S, pad])     # 3 true sources + 7 noise dims

    print(f"  {'k':>3}  {'ICA MSE':>10}  {'PCA MSE':>10}")
    for k in range(1, d + 1):
        ica = FastICA(n_components=k, random_state=0).fit(X)
        pca = PCA(n_components=k).fit(X)
        mse_ica = float(np.mean((X - ica.inverse_transform(ica.transform(X))) ** 2))
        mse_pca = float(np.mean((X - pca.inverse_transform(pca.transform(X))) ** 2))
        print(f"  {k:>3}  {mse_ica:>10.4f}  {mse_pca:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# Run all demos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_blind_source_separation()
    demo_algorithm_comparison()
    demo_contrast_functions()
    demo_feature_extraction()
    demo_pca_ica_pipeline()
    demo_reconstruction_accuracy()
