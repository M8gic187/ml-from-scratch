"""MLP demo: MLPClassifier and MLPRegressor on synthetic datasets.

Run with:
    python examples/mlp_demo.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.neural_network import MLPClassifier, MLPRegressor
from ml_scratch.utils.metrics import (
    accuracy,
    confusion_matrix,
    classification_report,
    r2_score,
    mse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_test_split(X, y, test_ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def make_blobs(centres, n_per_class=150, seed=0):
    rng = np.random.default_rng(seed)
    Xs = [rng.multivariate_normal(c, np.eye(len(c)), n_per_class) for c in centres]
    X = np.vstack(Xs)
    y = np.hstack([np.full(n_per_class, k) for k in range(len(centres))])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def make_xor(n=400, noise=0.15, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += rng.normal(0, noise, X.shape)
    return X, y


def _sep(title=""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Demo 1: Binary classification on XOR (non-linear problem)
# ---------------------------------------------------------------------------

def demo_xor():
    _sep("Demo 1: Binary classification — XOR (non-linear)")

    X, y = make_xor(n=600, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=0.25, seed=1)

    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        learning_rate=5e-3,
        n_iterations=400,
        batch_size=32,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    proba = clf.predict_proba(X_te)

    print(f"Architecture : input(2) → 32 → 16 → output(2), ReLU")
    print(f"Accuracy     : {accuracy(y_te, preds):.4f}")
    print(f"Initial loss : {clf.loss_history_[0]:.4f}")
    print(f"Final loss   : {clf.loss_history_[-1]:.4f}")
    print(f"\nClassification report:")
    print(classification_report(y_te, preds))
    print(f"Confusion matrix:\n{confusion_matrix(y_te, preds)}")
    print(f"\nSample probabilities (first 5 test rows):")
    for i in range(5):
        print(f"  true={y_te[i]}  P(0)={proba[i,0]:.3f}  P(1)={proba[i,1]:.3f}")


# ---------------------------------------------------------------------------
# Demo 2: Multi-class classification (3 Gaussian blobs)
# ---------------------------------------------------------------------------

def demo_multiclass():
    _sep("Demo 2: Multi-class classification — 3-class Gaussian blobs")

    centres = [[-4, 0], [4, 0], [0, 4]]
    X, y = make_blobs(centres, n_per_class=150, seed=7)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=0.25, seed=2)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        learning_rate=1e-2,
        n_iterations=300,
        batch_size=64,
        random_state=0,
    )
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    proba = clf.predict_proba(X_te)

    print(f"Architecture : input(2) → 64 → 32 → output(3), ReLU")
    print(f"Classes      : {clf.classes_.tolist()}")
    print(f"Accuracy     : {accuracy(y_te, preds):.4f}")
    print(f"Initial loss : {clf.loss_history_[0]:.4f}")
    print(f"Final loss   : {clf.loss_history_[-1]:.4f}")
    print(f"\nClassification report:")
    print(classification_report(y_te, preds))
    print(f"Confusion matrix:\n{confusion_matrix(y_te, preds)}")
    print(f"\nProbability matrix shape: {proba.shape}")
    print(f"Row sums (should be 1.0): {proba[:5].sum(axis=1).round(6)}")


# ---------------------------------------------------------------------------
# Demo 3: Regression — noisy sine function
# ---------------------------------------------------------------------------

def demo_regression_sine():
    _sep("Demo 3: Regression — noisy sine function")

    rng = np.random.default_rng(10)
    X = rng.uniform(-np.pi, np.pi, (600, 1))
    y = np.sin(X.ravel()) + rng.normal(0, 0.15, 600)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=0.25, seed=3)

    reg = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="tanh",
        learning_rate=5e-3,
        n_iterations=600,
        batch_size=64,
        random_state=0,
    )
    reg.fit(X_tr, y_tr)

    preds = reg.predict(X_te)

    print(f"Architecture : input(1) → 64 → 32 → output(1), tanh")
    print(f"R²           : {r2_score(y_te, preds):.4f}")
    print(f"RMSE         : {np.sqrt(mse(y_te, preds)):.4f}")
    print(f"Initial loss : {reg.loss_history_[0]:.4f}")
    print(f"Final loss   : {reg.loss_history_[-1]:.6f}")
    print(f"Prediction output shape: {preds.shape}")


# ---------------------------------------------------------------------------
# Demo 4: Multi-output regression
# ---------------------------------------------------------------------------

def demo_regression_multioutput():
    _sep("Demo 4: Multi-output regression — two targets")

    rng = np.random.default_rng(20)
    X = rng.standard_normal((500, 4))
    Y = np.column_stack([
        3 * X[:, 0] - X[:, 1],
        X[:, 2] ** 2 - 2 * X[:, 3],
    ]) + rng.normal(0, 0.1, (500, 2))

    split = 400
    X_tr, X_te = X[:split], X[split:]
    Y_tr, Y_te = Y[:split], Y[split:]

    reg = MLPRegressor(
        hidden_layer_sizes=(64,),
        activation="relu",
        learning_rate=1e-2,
        n_iterations=500,
        batch_size=64,
        random_state=5,
    )
    reg.fit(X_tr, Y_tr)

    preds = reg.predict(X_te)
    print(f"Architecture  : input(4) → 64 → output(2), ReLU")
    print(f"Output shape  : {preds.shape}  (n_samples, n_outputs)")
    for k in range(2):
        r2 = r2_score(Y_te[:, k], preds[:, k])
        print(f"  Target {k+1} R² : {r2:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_xor()
    demo_multiclass()
    demo_regression_sine()
    demo_regression_multioutput()
    print("\n" + "=" * 60)
    print("  All demos completed successfully.")
    print("=" * 60)
