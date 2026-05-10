"""Demo: DecisionTreeClassifier on a multi-class synthetic dataset."""

import numpy as np

from ml_scratch.tree import DecisionTreeClassifier
from ml_scratch.utils.metrics import accuracy


def main() -> None:
    rng = np.random.default_rng(42)
    centres = np.array([[0.0, 4.0], [-4.0, -2.0], [4.0, -2.0]])
    X = np.vstack([rng.normal(c, 0.7, (100, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 100)

    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    print(f"{'max_depth':<12} {'train_acc':>10} {'test_acc':>10}")
    print("-" * 34)
    for depth in [1, 2, 3, 5, None]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(X_train, y_train)
        train_acc = accuracy(y_train, clf.predict(X_train))
        test_acc = accuracy(y_test, clf.predict(X_test))
        label = str(depth) if depth is not None else "None"
        print(f"{label:<12} {train_acc:>10.4f} {test_acc:>10.4f}")


if __name__ == "__main__":
    main()
