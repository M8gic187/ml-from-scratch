"""Tests for LinearSVC and SVC."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.svm import LinearSVC, SVC
from ml_scratch.utils.metrics import accuracy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _linearly_separable(seed: int = 0):
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal([-2, -2], np.eye(2), size=100)
    X1 = rng.multivariate_normal([2, 2], np.eye(2), size=100)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(100), np.ones(100)]).astype(int)
    idx = rng.permutation(200)
    return X[idx], y[idx]


def _nonlinear_moons(seed: int = 7, n: int = 200):
    """Two interleaved half-circles (XOR-like, not linearly separable)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n // 2)
    X0 = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, 0.1, (n // 2, 2))
    X1 = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5]) + rng.normal(0, 0.1, (n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    idx = rng.permutation(n)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# LinearSVC tests
# ---------------------------------------------------------------------------

def test_linear_svc_fit_returns_self():
    X, y = _linearly_separable()
    model = LinearSVC()
    assert model.fit(X, y) is model


def test_linear_svc_attributes_after_fit():
    X, y = _linearly_separable()
    model = LinearSVC()
    model.fit(X, y)
    assert model.weights_ is not None
    assert model.weights_.shape == (2,)
    assert isinstance(model.bias_, float)


def test_linear_svc_high_accuracy_separable():
    X, y = _linearly_separable()
    split = 160
    model = LinearSVC(C=1.0, n_iterations=3000)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    assert accuracy(y[split:], preds) >= 0.90


def test_linear_svc_decision_function_shape():
    X, y = _linearly_separable()
    model = LinearSVC()
    model.fit(X, y)
    scores = model.decision_function(X)
    assert scores.shape == (200,)


def test_linear_svc_decision_function_sign_matches_predict():
    X, y = _linearly_separable()
    model = LinearSVC(n_iterations=3000)
    model.fit(X, y)
    scores = model.decision_function(X)
    preds = model.predict(X)
    # prediction=1 iff score>=0
    assert np.all((scores >= 0) == (preds == 1))


def test_linear_svc_predict_before_fit_raises():
    model = LinearSVC()
    try:
        model.predict(np.array([[1.0, 2.0]]))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_linear_svc_accepts_pm1_labels():
    X, y = _linearly_separable()
    y_pm = np.where(y == 0, -1, 1)
    model = LinearSVC(n_iterations=3000)
    model.fit(X, y_pm)
    preds = model.predict(X)
    assert accuracy(y, preds) >= 0.90


def test_linear_svc_output_binary():
    X, y = _linearly_separable()
    model = LinearSVC()
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------------------------------------------------------------------------
# SVC (kernel) tests
# ---------------------------------------------------------------------------

def test_svc_fit_returns_self():
    X, y = _linearly_separable()
    model = SVC(kernel="linear", n_iterations=20)
    assert model.fit(X, y) is model


def test_svc_linear_kernel_accuracy():
    X, y = _linearly_separable()
    split = 160
    model = SVC(kernel="linear", C=1.0, n_iterations=50)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    assert accuracy(y[split:], preds) >= 0.88


def test_svc_rbf_kernel_accuracy():
    X, y = _linearly_separable()
    split = 160
    model = SVC(kernel="rbf", C=5.0, n_iterations=60)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    assert accuracy(y[split:], preds) >= 0.88


def test_svc_rbf_nonlinear_moons():
    X, y = _nonlinear_moons()
    split = 160
    model = SVC(kernel="rbf", C=5.0, gamma=1.0, n_iterations=80)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    assert accuracy(y[split:], preds) >= 0.80


def test_svc_poly_kernel_runs():
    X, y = _linearly_separable()
    model = SVC(kernel="poly", degree=2, C=1.0, n_iterations=20)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_svc_support_vectors_stored():
    X, y = _linearly_separable()
    model = SVC(kernel="rbf", n_iterations=40)
    model.fit(X, y)
    assert model.support_vectors_ is not None
    assert model.support_vectors_.shape[1] == 2
    assert len(model.support_alpha_) == len(model.support_vectors_)


def test_svc_decision_function_shape():
    X, y = _linearly_separable()
    model = SVC(kernel="linear", n_iterations=30)
    model.fit(X, y)
    scores = model.decision_function(X)
    assert scores.shape == (200,)


def test_svc_decision_function_sign_matches_predict():
    X, y = _linearly_separable()
    model = SVC(kernel="linear", n_iterations=50)
    model.fit(X, y)
    scores = model.decision_function(X)
    preds = model.predict(X)
    assert np.all((scores >= 0) == (preds == 1))


def test_svc_predict_before_fit_raises():
    model = SVC()
    try:
        model.predict(np.array([[1.0, 2.0]]))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_svc_gamma_scale():
    X, y = _linearly_separable()
    model = SVC(kernel="rbf", gamma="scale", n_iterations=30)
    model.fit(X, y)
    assert model._gamma_fit > 0


def test_svc_unknown_kernel_raises():
    X, y = _linearly_separable()
    model = SVC(kernel="sigmoid", n_iterations=5)
    try:
        model.fit(X, y)
        assert False, "Expected ValueError"
    except ValueError:
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
