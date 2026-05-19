"""SVM demo: LinearSVC and kernel SVC on linearly-separable and non-linear data.

Run with:
    python examples/svm_demo.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.svm import LinearSVC, SVC
from ml_scratch.utils.metrics import (
    accuracy,
    confusion_matrix,
    classification_report,
)
from ml_scratch.utils.cross_validation import cross_val_score, StratifiedKFold


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

def make_blobs(seed: int = 0, n: int = 300):
    """Two well-separated Gaussian blobs (linearly separable)."""
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal([-3, -3], np.eye(2), n // 2)
    X1 = rng.multivariate_normal([3, 3], np.eye(2), n // 2)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_moons(seed: int = 7, n: int = 300, noise: float = 0.15):
    """Two interleaved half-circles (not linearly separable)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n // 2)
    X0 = np.column_stack([np.cos(t), np.sin(t)])
    X1 = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5])
    X = np.vstack([X0, X1]) + rng.normal(0, noise, (n, 2))
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def train_test_split(X, y, test_ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


# ---------------------------------------------------------------------------
# Demo 1: LinearSVC on linearly-separable data
# ---------------------------------------------------------------------------

print("=" * 60)
print("Demo 1: LinearSVC (Pegasos) on linearly-separable blobs")
print("=" * 60)

X, y = make_blobs()
X_tr, X_te, y_tr, y_te = train_test_split(X, y)

model = LinearSVC(C=1.0, n_iterations=3000)
model.fit(X_tr, y_tr)
preds = model.predict(X_te)

print(f"  Training samples : {len(y_tr)}")
print(f"  Test samples     : {len(y_te)}")
print(f"  Test accuracy    : {accuracy(y_te, preds):.4f}")
print(f"  Weight vector    : {model.weights_.round(4)}")
print(f"  Bias             : {model.bias_:.4f}")

print("\nConfusion matrix:")
cm = confusion_matrix(y_te, preds)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

print("\nClassification report:")
print(classification_report(y_te, preds))


# ---------------------------------------------------------------------------
# Demo 2: Kernel SVC — linear kernel vs RBF on moons
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Demo 2: SVC with linear and RBF kernels on moon dataset")
print("=" * 60)

X, y = make_moons()
X_tr, X_te, y_tr, y_te = train_test_split(X, y)

for kernel, C, gamma in [("linear", 1.0, "scale"), ("rbf", 5.0, 1.5)]:
    clf = SVC(kernel=kernel, C=C, gamma=gamma, n_iterations=100)
    clf.fit(X_tr, y_tr)
    acc = accuracy(y_te, clf.predict(X_te))
    n_sv = len(clf.support_vectors_)
    print(f"  kernel={kernel:6s}  C={C}  gamma={gamma!r:6}  "
          f"acc={acc:.4f}  support_vectors={n_sv}")


# ---------------------------------------------------------------------------
# Demo 3: Cross-validation with StratifiedKFold
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Demo 3: 5-fold stratified cross-validation")
print("=" * 60)

X, y = make_blobs(n=400)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n  LinearSVC (C=1.0):")
scores = cross_val_score(LinearSVC(C=1.0, n_iterations=3000), X, y, cv=skf)
print(f"    fold scores : {scores.round(4)}")
print(f"    mean ± std  : {scores.mean():.4f} ± {scores.std():.4f}")

print("\n  SVC RBF (C=5.0, gamma='scale'):")
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores2 = cross_val_score(SVC(kernel="rbf", C=5.0, n_iterations=80), X, y, cv=skf2)
print(f"    fold scores : {scores2.round(4)}")
print(f"    mean ± std  : {scores2.mean():.4f} ± {scores2.std():.4f}")


# ---------------------------------------------------------------------------
# Demo 4: Polynomial kernel on blobs
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Demo 4: SVC with polynomial kernel (degree=2)")
print("=" * 60)

X, y = make_blobs(n=200)
X_tr, X_te, y_tr, y_te = train_test_split(X, y)

clf = SVC(kernel="poly", degree=2, C=1.0, gamma="scale", n_iterations=60)
clf.fit(X_tr, y_tr)
acc = accuracy(y_te, clf.predict(X_te))
print(f"  Poly(degree=2) test accuracy : {acc:.4f}")
print(f"  Number of support vectors    : {len(clf.support_vectors_)}")

print("\nAll demos complete.")
