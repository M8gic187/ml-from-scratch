"""Tests for LogisticRegression."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.linear_models import LogisticRegression
from ml_scratch.utils.metrics import accuracy


def _make_blobs(seed: int = 7):
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal([0, 0], np.eye(2), size=200)
    X1 = rng.multivariate_normal([4, 4], np.eye(2), size=200)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(200), np.ones(200)])
    idx = rng.permutation(400)
    return X[idx], y[idx]


def test_fit_returns_self():
    X, y = _make_blobs()
    model = LogisticRegression()
    assert model.fit(X, y) is model


def test_loss_decreases():
    X, y = _make_blobs()
    model = LogisticRegression(learning_rate=0.1, n_iterations=300)
    model.fit(X, y)
    assert model.loss_history_[0] > model.loss_history_[-1]


def test_proba_in_unit_interval():
    X, y = _make_blobs()
    model = LogisticRegression(n_iterations=200)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_high_accuracy_on_separable_data():
    X, y = _make_blobs()
    split = 320
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    assert accuracy(y[split:].astype(int), preds) > 0.95


def test_predict_before_fit_raises():
    model = LogisticRegression()
    try:
        model.predict(np.array([[1.0, 2.0]]))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_sigmoid_stable_on_large_inputs():
    X = np.array([[500.0, -500.0], [-500.0, 500.0]])
    y = np.array([1.0, 0.0])
    model = LogisticRegression(n_iterations=1)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(np.isfinite(proba))


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
