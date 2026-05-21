"""Tests for Lasso regression."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.linear_models import Lasso
from ml_scratch.utils.metrics import r2_score


def _make_sparse_data(seed: int = 0):
    """Return a dataset where only 3 of 8 features are relevant."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(400, 8))
    w_true = np.array([2.0, -1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    y = X @ w_true + rng.normal(0, 0.2, size=400)
    return X, y, w_true


def test_fit_returns_self():
    X, y, _ = _make_sparse_data()
    model = Lasso()
    assert model.fit(X, y) is model


def test_predict_before_fit_raises():
    try:
        Lasso().predict(np.ones((5, 8)))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_r2_above_threshold():
    X, y, _ = _make_sparse_data(seed=3)
    model = Lasso(alpha=0.05, n_iterations=1000)
    model.fit(X[:320], y[:320])
    assert r2_score(y[320:], model.predict(X[320:])) > 0.90


def test_sparsity_with_high_alpha():
    """High alpha should zero out most weights."""
    X, y, _ = _make_sparse_data()
    model = Lasso(alpha=0.5, n_iterations=1000).fit(X, y)
    n_zeros = int(np.sum(np.abs(model.weights_) < 1e-6))
    assert n_zeros >= 4


def test_sparsity_recovers_support():
    """Moderate alpha should keep the 3 relevant features non-zero."""
    X, y, w_true = _make_sparse_data(seed=5)
    model = Lasso(alpha=0.05, n_iterations=2000).fit(X, y)
    relevant = np.where(w_true != 0.0)[0]
    for idx in relevant:
        assert abs(model.weights_[idx]) > 0.2, (
            f"Feature {idx} should be non-zero but got {model.weights_[idx]:.4f}"
        )


def test_n_iter_set_after_fit():
    X, y, _ = _make_sparse_data()
    model = Lasso(n_iterations=500).fit(X, y)
    assert 1 <= model.n_iter_ <= 500


def test_early_stopping():
    """Well-conditioned data should converge well before max iterations."""
    X, y, _ = _make_sparse_data(seed=10)
    model = Lasso(alpha=0.05, n_iterations=5000, tol=1e-4).fit(X, y)
    assert model.n_iter_ < 5000


def test_weights_shape():
    X, y, _ = _make_sparse_data()
    model = Lasso().fit(X, y)
    assert model.weights_.shape == (8,)


def test_predict_shape():
    X, y, _ = _make_sparse_data()
    model = Lasso(alpha=0.05).fit(X[:350], y[:350])
    assert model.predict(X[350:]).shape == (50,)


def test_l1_ratio_zero_gives_ridge_like_result():
    """alpha=0 should give near-OLS solution (no regularisation)."""
    X, y, w_true = _make_sparse_data(seed=2)
    model = Lasso(alpha=1e-6, n_iterations=2000).fit(X, y)
    assert r2_score(y, model.predict(X)) > 0.95


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
