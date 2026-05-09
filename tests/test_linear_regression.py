"""Tests for LinearRegression."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.linear_models import LinearRegression
from ml_scratch.utils.metrics import r2_score


def _make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(300, 1))
    y = 2.5 * X.squeeze() + 4.0 + rng.normal(0, 0.3, size=300)
    return X, y


def test_fit_returns_self():
    X, y = _make_data()
    model = LinearRegression()
    assert model.fit(X, y) is model


def test_loss_decreases():
    X, y = _make_data()
    model = LinearRegression(learning_rate=0.05, n_iterations=200)
    model.fit(X, y)
    assert model.loss_history_[0] > model.loss_history_[-1]


def test_recovers_linear_coefficients():
    X, y = _make_data()
    model = LinearRegression(learning_rate=0.05, n_iterations=1000)
    model.fit(X, y)
    assert abs(model.weights_[0] - 2.5) < 0.2
    assert abs(model.bias_ - 4.0) < 0.2


def test_r2_above_threshold():
    X, y = _make_data()
    model = LinearRegression(learning_rate=0.05, n_iterations=1000)
    model.fit(X[:250], y[:250])
    preds = model.predict(X[250:])
    assert r2_score(y[250:], preds) > 0.95


def test_predict_before_fit_raises():
    model = LinearRegression()
    try:
        model.predict(np.array([[1.0]]))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


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
