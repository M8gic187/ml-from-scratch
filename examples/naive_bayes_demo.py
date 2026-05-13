"""Demo: GaussianNB on a multi-class synthetic dataset."""

import numpy as np

from ml_scratch.naive_bayes import GaussianNB
from ml_scratch.utils.metrics import accuracy


def main() -> None:
    rng = np.random.default_rng(42)
    centres = np.array([[5, 0], [-5, 0], [0, 5], [0, -5]], dtype=float)
    X = np.vstack([rng.normal(c, 1.0, (100, 2)) for c in centres])
    y = np.repeat([0, 1, 2, 3], 100)

    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    clf = GaussianNB(var_smoothing=1e-9)
    clf.fit(X_train, y_train)

    train_acc = accuracy(y_train, clf.predict(X_train))
    test_acc = accuracy(y_test, clf.predict(X_test))

    print("Gaussian Naive Bayes — 4-class synthetic data")
    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Test  accuracy : {test_acc:.4f}")

    proba = clf.predict_proba(X_test[:5])
    print("\nPosterior probabilities for first 5 test samples:")
    print(f"  {'Sample':>6}  " + "  ".join(f"P(y={c})" for c in clf.classes_))
    for i, row in enumerate(proba):
        print(f"  {i:>6}  " + "  ".join(f"{p:7.4f}" for p in row))


if __name__ == "__main__":
    main()
