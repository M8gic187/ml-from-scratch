"""Demo: RandomForestClassifier vs single DecisionTree on a noisy dataset."""

import numpy as np

from ml_scratch.ensemble import RandomForestClassifier
from ml_scratch.tree import DecisionTreeClassifier
from ml_scratch.utils.metrics import accuracy


def make_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    centres = np.array([[0, 4], [-4, -2], [4, -2]], dtype=float)
    X = np.vstack([rng.normal(c, 1.2, (120, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 120)
    return X, y


def main() -> None:
    rng = np.random.default_rng(0)
    X, y = make_dataset(rng)

    idx = rng.permutation(len(y))
    split = int(0.75 * len(y))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    print(f"{'Model':<30} {'Train acc':>10} {'Test acc':>10}")
    print("-" * 52)

    # Single decision trees at varying depth
    for depth in [2, 4, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        label = f"DecisionTree(max_depth={depth})"
        print(
            f"{label:<30} "
            f"{accuracy(y_train, tree.predict(X_train)):>10.4f} "
            f"{accuracy(y_test, tree.predict(X_test)):>10.4f}"
        )

    # Random forests at varying ensemble sizes
    for n in [10, 50, 100]:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        label = f"RandomForest(n={n})"
        print(
            f"{label:<30} "
            f"{accuracy(y_train, rf.predict(X_train)):>10.4f} "
            f"{accuracy(y_test, rf.predict(X_test)):>10.4f}"
        )


if __name__ == "__main__":
    main()
