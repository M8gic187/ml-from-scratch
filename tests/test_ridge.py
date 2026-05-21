"""Tests for Ridge regression."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.linear_models import Ridge
from ml_scratch.utils.metrics import r2_score


def _make_data(n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(300, n_features))
    w = rng.uniform(-2, 2, size=n_features)
    y = X @ w + rng.normal(0, 0.2, size=300)
    return X, y, w


def test_fit_returns_self():
    X, y, _ = _make_data()
    model = Ridge()
    assert model.fit(X, y) is model


def test_predict_before_fit_raises():
    try:
        Ridge().predict(np.ones((5, 3)))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_loss_decreases():
    X, y, _ = _make_data()
    model = Ridge(alpha=0.1, learning_rate=0.02, n_iterations=500)
    model.fit(X, y)
    assert model.loss_history_[0] > model.loss_history_[-1]


def test_r2_above_threshold():
    X, y, _ = _make_data(seed=7)
    model = Ridge(alpha=0.01, learning_rate=0.02, n_iterations=2000)
    model.fit(X[:250], y[:250])
    assert r2_score(y[250:], model.predict(X[250:])) > 0.95


def test_higher_alpha_shrinks_weights():
    """Larger alpha should produce smaller weight norms."""
    X, y, _ = _make_data(seed=1)
    low = Ridge(alpha=0.001, learning_rate=0.02, n_iterations=2000).fit(X, y)
    high = Ridge(alpha=10.0, learning_rate=0.02, n_iterations=2000).fit(X, y)
    assert np.linalg.norm(high.weights_) < np.linalg.norm(low.weights_)


def test_no_intercept():
    rng = np.random.default_rng(99)
    X = rng.uniform(-1, 1, (200, 2))
    y = X @ np.array([1.0, -1.0])  # no bias
    model = Ridge(alpha=0.001, learning_rate=0.05, n_iterations=2000, fit_intercept=False)
    model.fit(X, y)
    assert abs(model.bias_) < 1e-9
    assert r2_score(y, model.predict(X)) > 0.99


def test_weights_shape():
    X, y, _ = _make_data(n_features=5)
    model = Ridge().fit(X, y)
    assert model.weights_.shape == (5,)


def test_predict_shape():
    X, y, _ = _make_data()
    model = Ridge().fit(X[:200], y[:200])
    assert model.predict(X[200:]).shape == (100,)


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
