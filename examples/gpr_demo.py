"""Gaussian Process Regression (GPR) usage examples.

Five demos covering the main use-cases:

  Demo 1: 1-D curve fitting — fit a GPR on noisy sine data and inspect
          the posterior mean ± 2σ confidence band.
  Demo 2: Kernel comparison — RBF, Matern52, and Periodic on the same
          dataset; log marginal likelihood and R² for each kernel.
  Demo 3: Noise sensitivity — vary noise_var across two orders of magnitude
          and observe how uncertainty grows in sparse regions.
  Demo 4: Posterior sampling — draw five function samples from the
          posterior and visualise the distribution of predictions.
  Demo 5: 2-D regression surface — fit a GPR on a 2-D Branin-style
          function; report MSE and R² on a held-out grid.
"""

import numpy as np

from ml_scratch.gaussian_process import GaussianProcessRegressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------------
# Demo 1: 1-D curve fitting
# ---------------------------------------------------------------------------

def demo_1d_curve_fitting() -> None:
    print("=" * 60)
    print("Demo 1: 1-D curve fitting (noisy sine)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    X_train = rng.uniform(-3.0, 3.0, (30, 1))
    y_train = np.sin(X_train.ravel()) + rng.normal(0, 0.1, 30)

    gpr = GaussianProcessRegressor(kernel="rbf", length_scale=1.0, noise_var=0.01)
    gpr.fit(X_train, y_train)

    X_test = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
    mean, std = gpr.predict(X_test, return_std=True)
    y_true = np.sin(X_test.ravel())

    print(f"  Training samples   : {len(X_train)}")
    print(f"  Test points        : {len(X_test)}")
    print(f"  Log marginal LH    : {gpr.log_marginal_likelihood_:.3f}")
    print(f"  R² (test vs sin)   : {gpr.score(X_test, y_true):.4f}")
    print(f"  MSE (test vs sin)  : {_mse(y_true, mean):.5f}")
    print(f"  Mean posterior std : {std.mean():.4f}")

    # Coverage: fraction of true values inside mean ± 2σ band
    lower = mean - 2 * std
    upper = mean + 2 * std
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    print(f"  95 % band coverage : {coverage:.1%}  (expected ≈ 95 %)")
    print()


# ---------------------------------------------------------------------------
# Demo 2: Kernel comparison
# ---------------------------------------------------------------------------

def demo_kernel_comparison() -> None:
    print("=" * 60)
    print("Demo 2: Kernel comparison on noisy sine data")
    print("=" * 60)

    rng = np.random.default_rng(7)
    X = rng.uniform(-3.0, 3.0, (40, 1))
    y = np.sin(X.ravel()) + rng.normal(0, 0.1, 40)

    X_test = np.linspace(-3.0, 3.0, 100).reshape(-1, 1)
    y_true = np.sin(X_test.ravel())

    configs = [
        {"kernel": "rbf",       "length_scale": 1.0,  "noise_var": 0.01},
        {"kernel": "matern52",  "length_scale": 1.0,  "noise_var": 0.01},
        {"kernel": "periodic",  "length_scale": 1.0,  "period": np.pi, "noise_var": 0.01},
        {"kernel": "linear",    "signal_var": 2.0,    "noise_var": 0.05},
    ]

    print(f"  {'Kernel':<12}  {'LML':>9}  {'R²':>7}  {'MSE':>10}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*7}  {'-'*10}")
    for cfg in configs:
        gpr = GaussianProcessRegressor(**cfg)
        gpr.fit(X, y)
        r2 = gpr.score(X_test, y_true)
        mse = _mse(y_true, gpr.predict(X_test))
        lml = gpr.log_marginal_likelihood_
        print(f"  {cfg['kernel']:<12}  {lml:>9.2f}  {r2:>7.4f}  {mse:>10.6f}")
    print()


# ---------------------------------------------------------------------------
# Demo 3: Noise sensitivity
# ---------------------------------------------------------------------------

def demo_noise_sensitivity() -> None:
    print("=" * 60)
    print("Demo 3: Noise sensitivity (varying noise_var)")
    print("=" * 60)

    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 6.0, (25, 1))
    y = np.sin(X.ravel()) + rng.normal(0, 0.15, 25)

    X_test = np.linspace(0.0, 6.0, 80).reshape(-1, 1)
    y_true = np.sin(X_test.ravel())

    noise_levels = [1e-5, 0.01, 0.1, 1.0]
    print(f"  {'noise_var':<12}  {'Avg σ (train)':>14}  {'Avg σ (extrap)':>15}  {'R²':>7}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*15}  {'-'*7}")

    # Extrapolation region: beyond training range
    X_extrap = np.linspace(6.5, 9.0, 20).reshape(-1, 1)

    for nv in noise_levels:
        gpr = GaussianProcessRegressor(kernel="rbf", length_scale=1.0, noise_var=nv)
        gpr.fit(X, y)
        _, std_train = gpr.predict(X_test, return_std=True)
        _, std_ext = gpr.predict(X_extrap, return_std=True)
        r2 = gpr.score(X_test, y_true)
        print(f"  {nv:<12.5f}  {std_train.mean():>14.4f}  {std_ext.mean():>15.4f}  {r2:>7.4f}")
    print()


# ---------------------------------------------------------------------------
# Demo 4: Posterior sampling
# ---------------------------------------------------------------------------

def demo_posterior_sampling() -> None:
    print("=" * 60)
    print("Demo 4: Posterior function sampling")
    print("=" * 60)

    rng = np.random.default_rng(11)
    X_train = rng.uniform(-2.0, 2.0, (15, 1))
    y_train = X_train.ravel() ** 2 + rng.normal(0, 0.2, 15)

    gpr = GaussianProcessRegressor(kernel="rbf", length_scale=0.8, noise_var=0.05)
    gpr.fit(X_train, y_train)

    X_test = np.linspace(-2.5, 2.5, 50).reshape(-1, 1)
    n_samples = 5
    samples = gpr.sample_y(X_test, n_samples=n_samples, random_state=42)
    mean, std = gpr.predict(X_test, return_std=True)

    print(f"  Training samples      : {len(X_train)}")
    print(f"  Posterior samples     : {n_samples}")
    print(f"  Sample array shape    : {samples.shape}")
    print(f"  Posterior mean range  : [{mean.min():.3f}, {mean.max():.3f}]")

    # Samples should be broadly consistent with mean ± 3σ
    sample_within = np.mean(
        (samples >= (mean - 3 * std)[:, None]) & (samples <= (mean + 3 * std)[:, None])
    )
    print(f"  Samples within ±3σ    : {sample_within:.1%}")
    print()


# ---------------------------------------------------------------------------
# Demo 5: 2-D regression
# ---------------------------------------------------------------------------

def demo_2d_regression() -> None:
    print("=" * 60)
    print("Demo 5: 2-D regression (modified Rosenbrock)")
    print("=" * 60)

    def target(X: np.ndarray) -> np.ndarray:
        x1, x2 = X[:, 0], X[:, 1]
        return np.sin(x1) * np.cos(x2) + 0.1 * (x1 ** 2 + x2 ** 2)

    rng = np.random.default_rng(22)
    X_train = rng.uniform(-2.0, 2.0, (80, 2))
    y_train = target(X_train) + rng.normal(0, 0.05, 80)

    gpr = GaussianProcessRegressor(kernel="rbf", length_scale=1.0, noise_var=0.01)
    gpr.fit(X_train, y_train)

    # Regular grid test set
    g = np.linspace(-2.0, 2.0, 12)
    X_test = np.array([[a, b] for a in g for b in g])
    y_true = target(X_test)

    mean = gpr.predict(X_test)
    r2 = gpr.score(X_test, y_true)
    mse = _mse(y_true, mean)

    print(f"  Training samples : {len(X_train)}")
    print(f"  Test grid points : {len(X_test)}  ({len(g)}×{len(g)})")
    print(f"  R² (test)        : {r2:.4f}")
    print(f"  MSE (test)       : {mse:.6f}")
    print(f"  LML              : {gpr.log_marginal_likelihood_:.2f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_1d_curve_fitting()
    demo_kernel_comparison()
    demo_noise_sensitivity()
    demo_posterior_sampling()
    demo_2d_regression()
    print("All demos completed successfully.")
