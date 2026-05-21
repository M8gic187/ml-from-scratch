"""Tests for ElasticNet regression."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.linear_models import ElasticNet
from ml_scratch.utils.metrics import r2_score


def _make_correlated_data(seed: int = 0):
    """Return a dataset with correlated features — ElasticNet's sweet spot."""
    rng = np.random.default_rng(seed)
    n = 400
    x1 = rng.uniform(-2, 2, n)
    # x2 and x3 are correlated with x1
    x2 = x1 + rng.normal(0, 0.3, n)
    x3 = x1 + rng.normal(0, 0.5, n)
    x4 = rng.uniform(-2, 2, n)   # irrelevant
    x5 = rng.uniform(-2, 2, n)   # irrelevant
    X = np.column_stack([x1, x2, x3, x4, x5])
    y = 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.2, n)
    return X, y


def test_fit_returns_self():
    X, y = _make_correlated_data()
    model = ElasticNet()
    assert model.fit(X, y) is model


def test_predict_before_fit_raises():
    try:
        ElasticNet().predict(np.ones((5, 5)))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_invalid_l1_ratio_raises():
    try:
        ElasticNet(l1_ratio=1.5)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_r2_above_threshold():
    X, y = _make_correlated_data(seed=4)
    model = ElasticNet(alpha=0.05, l1_ratio=0.5, n_iterations=1000)
    model.fit(X[:320], y[:320])
    assert r2_score(y[320:], model.predict(X[320:])) > 0.70


def test_l1_ratio_1_gives_sparse_solution():
    """Pure L1 (l1_ratio=1) should yield at least some zero weights."""
    X, y = _make_correlated_data()
    model = ElasticNet(alpha=0.2, l1_ratio=1.0, n_iterations=2000).fit(X, y)
    n_zeros = int(np.sum(np.abs(model.weights_) < 1e-6))
    assert n_zeros >= 1


def test_l1_ratio_0_no_sparsity():
    """Pure L2 (l1_ratio=0) generally does not zero out weights."""
    X, y = _make_correlated_data(seed=8)
    model = ElasticNet(alpha=0.01, l1_ratio=0.0, n_iterations=2000).fit(X, y)
    n_nonzero = int(np.sum(np.abs(model.weights_) > 1e-3))
    assert n_nonzero >= 4


def test_early_stopping():
    X, y = _make_correlated_data(seed=11)
    model = ElasticNet(alpha=0.05, l1_ratio=0.5, n_iterations=5000, tol=1e-4).fit(X, y)
    assert model.n_iter_ < 5000


def test_n_iter_set_after_fit():
    X, y = _make_correlated_data()
    model = ElasticNet(n_iterations=200).fit(X, y)
    assert 1 <= model.n_iter_ <= 200


def test_weights_shape():
    X, y = _make_correlated_data()
    model = ElasticNet().fit(X, y)
    assert model.weights_.shape == (5,)


def test_predict_shape():
    X, y = _make_correlated_data()
    model = ElasticNet(alpha=0.05).fit(X[:350], y[:350])
    assert model.predict(X[350:]).shape == (50,)


def test_blended_penalty_between_ridge_and_lasso():
    """l1_ratio=0.5 weight norm should be between pure-L1 and pure-L2."""
    X, y = _make_correlated_data(seed=6)
    lasso_like = ElasticNet(alpha=0.1, l1_ratio=1.0, n_iterations=2000).fit(X, y)
    ridge_like = ElasticNet(alpha=0.1, l1_ratio=0.0, n_iterations=2000).fit(X, y)
    blended = ElasticNet(alpha=0.1, l1_ratio=0.5, n_iterations=2000).fit(X, y)
    assert (
        np.linalg.norm(lasso_like.weights_, 1)
        <= np.linalg.norm(blended.weights_, 1)
        <= np.linalg.norm(ridge_like.weights_, 1)
    )


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
