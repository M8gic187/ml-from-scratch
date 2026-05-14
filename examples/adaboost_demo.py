"""Demo: AdaBoost vs single Decision Stump on a noisy binary dataset."""

import numpy as np

from ml_scratch.ensemble import AdaBoostClassifier
from ml_scratch.tree import DecisionTreeClassifier
from ml_scratch.utils.metrics import accuracy


def make_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X_a = rng.normal(loc=[2.0, 1.5], scale=1.2, size=(150, 2))
    X_b = rng.normal(loc=[-2.0, -1.5], scale=1.2, size=(150, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * 150 + [1] * 150)
    return X, y


def main() -> None:
    rng = np.random.default_rng(0)
    X, y = make_dataset(rng)

    idx = rng.permutation(len(y))
    split = int(0.75 * len(y))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    print(f"{'Model':<35} {'Train acc':>10} {'Test acc':>10}")
    print("-" * 57)

    # Single stumps as baseline
    stump = DecisionTreeClassifier(max_depth=1, random_state=0)
    stump.fit(X_train, y_train)
    print(
        f"{'DecisionTree(max_depth=1)':<35} "
        f"{accuracy(y_train, stump.predict(X_train)):>10.4f} "
        f"{accuracy(y_test, stump.predict(X_test)):>10.4f}"
    )

    # AdaBoost with varying ensemble sizes
    for n in [10, 25, 50, 100, 200]:
        clf = AdaBoostClassifier(n_estimators=n)
        clf.fit(X_train, y_train)
        label = f"AdaBoost(n_estimators={n})"
        print(
            f"{label:<35} "
            f"{accuracy(y_train, clf.predict(X_train)):>10.4f} "
            f"{accuracy(y_test, clf.predict(X_test)):>10.4f}"
        )

    # Learning curve: test accuracy vs number of estimators
    print("\nLearning curve (test accuracy):")
    clf_full = AdaBoostClassifier(n_estimators=200)
    clf_full.fit(X_train, y_train)
    scores = clf_full.estimator_weights_
    print(f"  Mean alpha: {scores.mean():.4f}  |  Std alpha: {scores.std():.4f}")


if __name__ == "__main__":
    main()
