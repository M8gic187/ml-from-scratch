"""Demo: KNNClassifier and KNNRegressor on synthetic data."""

import numpy as np

from ml_scratch.neighbors import KNNClassifier, KNNRegressor
from ml_scratch.utils.metrics import accuracy, mae


def classification_demo() -> None:
    rng = np.random.default_rng(0)
    X_train = np.vstack([
        rng.normal([-3, -3], 0.8, (80, 2)),
        rng.normal([+3, +3], 0.8, (80, 2)),
    ])
    y_train = np.array([0] * 80 + [1] * 80)
    X_test = np.vstack([
        rng.normal([-3, -3], 0.8, (20, 2)),
        rng.normal([+3, +3], 0.8, (20, 2)),
    ])
    y_test = np.array([0] * 20 + [1] * 20)

    for k, w in [(1, "uniform"), (5, "uniform"), (5, "distance")]:
        clf = KNNClassifier(k=k, weights=w).fit(X_train, y_train)
        acc = accuracy(y_test, clf.predict(X_test))
        print(f"  k={k:2d}, weights={w:<10s} → accuracy={acc:.4f}")


def regression_demo() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 2 * np.pi, (200, 1))
    y = np.sin(X.ravel()) + rng.normal(0, 0.15, 200)
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    for k, w in [(3, "uniform"), (7, "uniform"), (7, "distance")]:
        reg = KNNRegressor(k=k, weights=w).fit(X_train, y_train)
        err = mae(y_test, reg.predict(X_test))
        print(f"  k={k:2d}, weights={w:<10s} → MAE={err:.4f}")


if __name__ == "__main__":
    print("=== KNN Classification ===")
    classification_demo()
    print("\n=== KNN Regression ===")
    regression_demo()
