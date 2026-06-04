"""NMF (Non-negative Matrix Factorization) usage examples.

Six demos covering the main use-cases:

  Demo 1: Synthetic rank-3 recovery — decompose a low-rank non-negative
          matrix and measure reconstruction quality.
  Demo 2: Solver comparison — Multiplicative Updates (mu) vs Alternating
          Least Squares (als) on the same data; convergence speed and error.
  Demo 3: Beta-loss sweep — Frobenius vs KL-divergence for the mu solver;
          impact on reconstruction and sparsity of learned components.
  Demo 4: Topic modelling analogy — decompose a synthetic document-term
          matrix and inspect the top terms per latent topic.
  Demo 5: Regularisation effects — L1 on H drives sparse components; L2
          on W produces smoother activations.
  Demo 6: Reconstruction MSE vs n_components for NMF and PCA side-by-side.
"""

import numpy as np

from ml_scratch.decomposition import NMF, PCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_low_rank(n_samples: int, n_features: int, rank: int, seed: int) -> np.ndarray:
    """Non-negative rank-*rank* matrix with additive noise."""
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((n_samples, rank)))
    H = np.abs(rng.standard_normal((rank, n_features)))
    noise = rng.uniform(0, 0.1, (n_samples, n_features))
    return np.maximum(0.0, W @ H + noise)


def _rel_err(X: np.ndarray, X_hat: np.ndarray) -> float:
    return float(np.linalg.norm(X - X_hat, "fro") / np.linalg.norm(X, "fro"))


# ---------------------------------------------------------------------------
# Demo 1: Synthetic rank-3 recovery
# ---------------------------------------------------------------------------

def demo_low_rank_recovery() -> None:
    print("=" * 60)
    print("Demo 1: Low-rank matrix recovery")
    print("=" * 60)

    X = _make_low_rank(n_samples=120, n_features=25, rank=3, seed=0)

    for solver in ("mu", "als"):
        nmf = NMF(n_components=3, solver=solver, max_iter=400, tol=1e-5, random_state=0)
        W = nmf.fit_transform(X)
        X_hat = nmf.inverse_transform(W)
        rel = _rel_err(X, X_hat)
        print(f"  solver={solver!r}  iters={nmf.n_iter_:>4d}  "
              f"rel_err={rel:.4f}  components_min={nmf.components_.min():.6f}")
    print()


# ---------------------------------------------------------------------------
# Demo 2: Solver comparison — MU vs ALS
# ---------------------------------------------------------------------------

def demo_solver_comparison() -> None:
    print("=" * 60)
    print("Demo 2: Solver comparison (MU vs ALS)")
    print("=" * 60)

    rng = np.random.default_rng(7)
    X = np.abs(rng.standard_normal((200, 30)))

    results = {}
    for solver in ("mu", "als"):
        nmf = NMF(n_components=6, solver=solver, max_iter=300, tol=1e-5, random_state=7)
        nmf.fit(X)
        results[solver] = (nmf.n_iter_, nmf.reconstruction_err_)

    baseline = float(np.linalg.norm(X, "fro"))
    print(f"  Baseline ‖X‖_F: {baseline:.3f}")
    for solver, (n_iter, err) in results.items():
        print(f"  solver={solver!r}  iters={n_iter:>4d}  ‖X-WH‖_F={err:.4f}  "
              f"rel_err={err/baseline:.4f}")
    winner = min(results, key=lambda s: results[s][1])
    print(f"  Lower reconstruction error: {winner!r}")
    print()


# ---------------------------------------------------------------------------
# Demo 3: Beta-loss sweep (Frobenius vs KL)
# ---------------------------------------------------------------------------

def demo_beta_loss() -> None:
    print("=" * 60)
    print("Demo 3: Beta-loss comparison (Frobenius vs KL)")
    print("=" * 60)

    # Count-like data (integers, appropriate for KL divergence)
    rng = np.random.default_rng(1)
    X = rng.poisson(lam=3.0, size=(100, 20)).astype(float)

    for loss in ("frobenius", "kl"):
        nmf = NMF(n_components=5, solver="mu", beta_loss=loss,
                  max_iter=400, tol=1e-5, random_state=1)
        W = nmf.fit_transform(X)
        X_hat = nmf.inverse_transform(W)
        rel = _rel_err(X, X_hat)
        sparsity = float((nmf.components_ < 1e-4).mean())
        print(f"  beta_loss={loss!r:<12s}  iters={nmf.n_iter_:>4d}  "
              f"rel_err={rel:.4f}  H_sparsity={sparsity:.3f}")
    print()


# ---------------------------------------------------------------------------
# Demo 4: Topic modelling analogy
# ---------------------------------------------------------------------------

def demo_topic_modelling() -> None:
    print("=" * 60)
    print("Demo 4: Topic modelling analogy (document-term matrix)")
    print("=" * 60)

    # Synthetic vocabulary
    vocab = [
        "data", "model", "learn", "train", "accuracy",   # ML topic
        "matrix", "vector", "eigen", "norm", "linear",   # Math topic
        "pixel", "image", "colour", "filter", "convolution",  # Vision topic
    ]
    n_docs, n_topics = 80, 3
    rng = np.random.default_rng(42)

    # Generate documents as mixtures of three pure topics
    topic_weights = np.abs(rng.standard_normal((n_docs, n_topics)))
    topic_profiles = np.zeros((n_topics, len(vocab)))
    for t in range(n_topics):
        start = t * 5
        topic_profiles[t, start:start + 5] = rng.uniform(2, 5, 5)
        topic_profiles[t] += rng.uniform(0, 0.2, len(vocab))  # background noise

    X = topic_weights @ topic_profiles  # (n_docs, n_vocab), non-negative

    nmf = NMF(n_components=3, max_iter=400, tol=1e-5, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_  # (n_topics, n_vocab)

    print(f"  Document-term matrix: {X.shape}")
    print(f"  Activation matrix W:  {W.shape}")
    print(f"  Component matrix H:   {H.shape}\n")
    print("  Top 5 terms per recovered component:")
    for k in range(H.shape[0]):
        top_idx = np.argsort(H[k])[::-1][:5]
        top_terms = [vocab[i] for i in top_idx]
        print(f"    Component {k + 1}: {top_terms}")
    print()


# ---------------------------------------------------------------------------
# Demo 5: Regularisation effects
# ---------------------------------------------------------------------------

def demo_regularisation() -> None:
    print("=" * 60)
    print("Demo 5: Regularisation effects on H sparsity")
    print("=" * 60)

    rng = np.random.default_rng(3)
    X = np.abs(rng.standard_normal((100, 20)))

    configs = [
        ("No regularisation",   dict()),
        ("L1 on H (0.02)",      dict(l1_reg_H=0.02)),
        ("L1 on H (0.1)",       dict(l1_reg_H=0.1)),
        ("L2 on W (0.5)",       dict(l2_reg_W=0.5)),
    ]

    for label, kwargs in configs:
        nmf = NMF(n_components=5, max_iter=300, tol=1e-5, random_state=3, **kwargs)
        nmf.fit(X)
        sparsity = float((nmf.components_ < 1e-4).mean())
        err = nmf.reconstruction_err_
        print(f"  {label:<26s}  ‖X-WH‖_F={err:7.3f}  H_sparsity={sparsity:.3f}")
    print()


# ---------------------------------------------------------------------------
# Demo 6: Reconstruction MSE vs n_components (NMF vs PCA)
# ---------------------------------------------------------------------------

def demo_nmf_vs_pca() -> None:
    print("=" * 60)
    print("Demo 6: Reconstruction error vs n_components (NMF vs PCA)")
    print("=" * 60)

    X = _make_low_rank(n_samples=150, n_features=30, rank=5, seed=9)

    print(f"  {'k':>3}  {'NMF rel_err':>12}  {'PCA rel_err':>12}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*12}")

    for k in (1, 2, 3, 5, 8, 12, 20):
        # NMF
        nmf = NMF(n_components=k, max_iter=400, tol=1e-5, random_state=0)
        W_nmf = nmf.fit_transform(X)
        X_nmf = nmf.inverse_transform(W_nmf)
        err_nmf = _rel_err(X, X_nmf)

        # PCA (truncated reconstruction)
        pca = PCA(n_components=k)
        W_pca = pca.fit_transform(X)
        X_pca = pca.inverse_transform(W_pca)
        err_pca = _rel_err(X, np.maximum(0, X_pca))

        print(f"  {k:>3d}  {err_nmf:>12.4f}  {err_pca:>12.4f}")
    print()
    print("  Note: PCA is the optimal linear reconstructor for Frobenius norm;")
    print("  NMF trades reconstruction accuracy for interpretable, non-negative parts.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_low_rank_recovery()
    demo_solver_comparison()
    demo_beta_loss()
    demo_topic_modelling()
    demo_regularisation()
    demo_nmf_vs_pca()
