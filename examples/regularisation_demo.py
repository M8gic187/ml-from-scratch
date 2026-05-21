"""Regularisation demo: Ridge, Lasso, and ElasticNet.

This script illustrates three canonical use-cases for regularised regression:

1. Ridge on a noisy dense problem — shows L2 weight shrinkage.
2. Lasso on a sparse problem — shows that L1 drives irrelevant weights to zero.
3. ElasticNet on correlated features — shows the blend outperforms pure Lasso
   when predictors are highly collinear.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.linear_models import Ridge, Lasso, ElasticNet, LinearRegression
from ml_scratch.utils.metrics import r2_score


# ─── helpers ─────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ─── 1. Ridge: weight shrinkage on a dense noisy problem ─────────────────────

_print_section("1. Ridge Regression — L2 regularisation")

rng = np.random.default_rng(42)
n, p = 300, 6
X_dense = rng.uniform(-2, 2, (n, p))
w_true = rng.uniform(-2, 2, p)
y_dense = X_dense @ w_true + rng.normal(0, 1.5, n)   # high noise

X_tr, y_tr = X_dense[:240], y_dense[:240]
X_te, y_te = X_dense[240:], y_dense[240:]

ols  = LinearRegression(learning_rate=0.02, n_iterations=2000).fit(X_tr, y_tr)
low  = Ridge(alpha=0.01, learning_rate=0.02, n_iterations=2000).fit(X_tr, y_tr)
high = Ridge(alpha=5.0,  learning_rate=0.02, n_iterations=2000).fit(X_tr, y_tr)

print(f"  {'Model':<25}  {'||w||':>8}  {'R² test':>9}")
print(f"  {'-'*46}")
for name, m in [("OLS (no reg)", ols), ("Ridge α=0.01", low), ("Ridge α=5.0", high)]:
    norm = np.linalg.norm(m.weights_)
    r2   = r2_score(y_te, m.predict(X_te))
    print(f"  {name:<25}  {norm:>8.4f}  {r2:>9.4f}")

print("\n  → Larger alpha shrinks weights and regularises overfitting.")


# ─── 2. Lasso: sparse feature selection ──────────────────────────────────────

_print_section("2. Lasso Regression — L1 sparse selection")

n, p = 400, 10
X_sparse = rng.uniform(-2, 2, (n, p))
w_sparse = np.zeros(p)
w_sparse[[0, 3, 7]] = [2.0, -1.5, 1.0]   # only 3 of 10 features matter
y_sparse = X_sparse @ w_sparse + rng.normal(0, 0.3, n)

X_tr, y_tr = X_sparse[:320], y_sparse[:320]
X_te, y_te = X_sparse[320:], y_sparse[320:]

alphas = [0.01, 0.1, 0.5]
print(f"\n  {'alpha':<10}  {'non-zeros':>10}  {'||w||_1':>10}  {'R² test':>9}")
print(f"  {'-' * 46}")
for a in alphas:
    m = Lasso(alpha=a, n_iterations=2000).fit(X_tr, y_tr)
    nz  = int(np.sum(np.abs(m.weights_) > 1e-6))
    l1n = np.linalg.norm(m.weights_, 1)
    r2  = r2_score(y_te, m.predict(X_te))
    print(f"  {a:<10.2f}  {nz:>10}  {l1n:>10.4f}  {r2:>9.4f}")

print("\n  → Moderate alpha recovers the 3 true features and zeros out the rest.")
print("  True non-zero indices: [0, 3, 7]")
best = Lasso(alpha=0.05, n_iterations=2000).fit(X_tr, y_tr)
print(f"  Estimated weights (α=0.05): {best.weights_.round(3)}")


# ─── 3. ElasticNet: correlated features ──────────────────────────────────────

_print_section("3. ElasticNet — blended penalty on correlated features")

n = 500
x1 = rng.uniform(-2, 2, n)
x2 = x1 + rng.normal(0, 0.2, n)    # highly correlated with x1
x3 = x1 + rng.normal(0, 0.4, n)    # moderately correlated
x4 = rng.uniform(-2, 2, n)         # irrelevant
x5 = rng.uniform(-2, 2, n)         # irrelevant
X_corr = np.column_stack([x1, x2, x3, x4, x5])
y_corr = 1.5 * x1 - 1.2 * x2 + rng.normal(0, 0.3, n)

X_tr, y_tr = X_corr[:400], y_corr[:400]
X_te, y_te = X_corr[400:], y_corr[400:]

alpha = 0.05
models = [
    ("Lasso (l1=1.0)",     Lasso(alpha=alpha, n_iterations=2000)),
    ("ElasticNet (l1=0.7)", ElasticNet(alpha=alpha, l1_ratio=0.7, n_iterations=2000)),
    ("ElasticNet (l1=0.5)", ElasticNet(alpha=alpha, l1_ratio=0.5, n_iterations=2000)),
    ("Ridge (via EN l1=0)", ElasticNet(alpha=alpha, l1_ratio=0.0, n_iterations=2000)),
]

print(f"\n  {'Model':<28}  {'non-zeros':>10}  {'R² test':>9}")
print(f"  {'-' * 52}")
for name, m in models:
    m.fit(X_tr, y_tr)
    nz = int(np.sum(np.abs(m.weights_) > 1e-6))
    r2 = r2_score(y_te, m.predict(X_te))
    print(f"  {name:<28}  {nz:>10}  {r2:>9.4f}")

print("\n  → ElasticNet handles correlated predictors more stably than pure Lasso.")

print("\n[done]")
