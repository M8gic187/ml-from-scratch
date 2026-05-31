"""Demo: ExtraTreesClassifier vs RandomForest on four scenarios."""

import numpy as np

from ml_scratch.ensemble import ExtraTreesClassifier, RandomForestClassifier
from ml_scratch.utils.metrics import accuracy


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def train_test_split(X, y, test_size=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int((1 - test_size) * len(y))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def compare_row(label, train_acc, test_acc):
    print(f"  {label:<40} train={train_acc:.4f}  test={test_acc:.4f}")


# ---------------------------------------------------------------------------
# Demo 1: binary blobs — accuracy parity with Random Forest
# ---------------------------------------------------------------------------


def demo_binary_blobs():
    print("Demo 1: Binary blobs (well-separated)")
    print("-" * 60)
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([3, 3], 0.8, (200, 2)),
                   rng.normal([-3, -3], 0.8, (200, 2))])
    y = np.array([0] * 200 + [1] * 200)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=1)

    for n in [10, 50, 100]:
        et = ExtraTreesClassifier(n_estimators=n, random_state=42).fit(X_tr, y_tr)
        rf = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_tr, y_tr)
        compare_row(f"ExtraTrees(n={n})", accuracy(y_tr, et.predict(X_tr)),
                    accuracy(y_te, et.predict(X_te)))
        compare_row(f"RandomForest(n={n})", accuracy(y_tr, rf.predict(X_tr)),
                    accuracy(y_te, rf.predict(X_te)))
    print()


# ---------------------------------------------------------------------------
# Demo 2: three-class problem
# ---------------------------------------------------------------------------


def demo_multiclass():
    print("Demo 2: Three-class problem")
    print("-" * 60)
    rng = np.random.default_rng(3)
    centres = np.array([[0, 5], [-5, -2], [5, -2]], dtype=float)
    X = np.vstack([rng.normal(c, 1.0, (150, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 150)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=7)

    et = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
    compare_row("ExtraTrees(n=100)", accuracy(y_tr, et.predict(X_tr)),
                accuracy(y_te, et.predict(X_te)))
    compare_row("RandomForest(n=100)", accuracy(y_tr, rf.predict(X_tr)),
                accuracy(y_te, rf.predict(X_te)))
    print()


# ---------------------------------------------------------------------------
# Demo 3: high-dimensional data (20 features, 5 informative)
# ---------------------------------------------------------------------------


def demo_high_dimensional():
    print("Demo 3: High-dimensional data (20 features, 5 informative)")
    print("-" * 60)
    rng = np.random.default_rng(5)
    n = 300
    X_info = np.hstack([rng.normal(0, 1, (n, 5)),
                        rng.normal(0, 1, (n, 15))])
    y = (X_info[:, 0] + X_info[:, 1] - X_info[:, 2] > 0).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X_info, y, seed=2)

    for n_feat in [None, 5, 3]:
        et = ExtraTreesClassifier(n_estimators=50, n_features=n_feat,
                                  random_state=0).fit(X_tr, y_tr)
        compare_row(f"ExtraTrees(n_features={n_feat})",
                    accuracy(y_tr, et.predict(X_tr)),
                    accuracy(y_te, et.predict(X_te)))
    print()


# ---------------------------------------------------------------------------
# Demo 4: bootstrap vs no-bootstrap (Extra Trees default)
# ---------------------------------------------------------------------------


def demo_bootstrap_effect():
    print("Demo 4: Bootstrap=True vs False on noisy data")
    print("-" * 60)
    rng = np.random.default_rng(8)
    X_a = rng.normal([1, 1], 1.5, (200, 2))
    X_b = rng.normal([-1, -1], 1.5, (200, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * 200 + [1] * 200)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=9)

    for boot in [False, True]:
        et = ExtraTreesClassifier(n_estimators=100, bootstrap=boot,
                                  random_state=42).fit(X_tr, y_tr)
        compare_row(f"ExtraTrees(bootstrap={boot})",
                    accuracy(y_tr, et.predict(X_tr)),
                    accuracy(y_te, et.predict(X_te)))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    demo_binary_blobs()
    demo_multiclass()
    demo_high_dimensional()
    demo_bootstrap_effect()


if __name__ == "__main__":
    main()
