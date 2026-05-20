"""Tests for MLPClassifier and MLPRegressor."""

import numpy as np
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.neural_network import MLPClassifier, MLPRegressor
from ml_scratch.utils.metrics import accuracy, r2_score


# ---------------------------------------------------------------------------
# Shared data fixtures
# ---------------------------------------------------------------------------

def _binary_data(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal([-2, -2], np.eye(2), size=n // 2)
    X1 = rng.multivariate_normal([2, 2], np.eye(2), size=n // 2)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def _multiclass_data(seed: int = 5, n_per_class: int = 80):
    rng = np.random.default_rng(seed)
    centres = [[-3, 0], [3, 0], [0, 3]]
    Xs = [rng.multivariate_normal(c, np.eye(2), size=n_per_class) for c in centres]
    X = np.vstack(Xs)
    y = np.hstack([np.full(n_per_class, k) for k in range(3)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _regression_data(seed: int = 1, n: int = 300):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.1, n)
    return X, y


# ---------------------------------------------------------------------------
# MLPClassifier — lifecycle
# ---------------------------------------------------------------------------

def test_classifier_fit_returns_self():
    X, y = _binary_data()
    clf = MLPClassifier(hidden_layer_sizes=(16,), n_iterations=5, random_state=0)
    assert clf.fit(X, y) is clf


def test_classifier_attributes_after_fit():
    X, y = _binary_data()
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), n_iterations=5, random_state=0)
    clf.fit(X, y)
    assert clf._fitted
    assert len(clf.weights_) == 3        # input→h1, h1→h2, h2→output
    assert len(clf.biases_) == 3
    assert clf.weights_[0].shape == (2, 16)
    assert clf.weights_[1].shape == (16, 8)
    assert clf.weights_[2].shape == (8, 2)
    assert hasattr(clf, "classes_")
    assert list(clf.classes_) == [0, 1]


def test_classifier_loss_history_length():
    X, y = _binary_data()
    n_iter = 12
    clf = MLPClassifier(n_iterations=n_iter, random_state=0)
    clf.fit(X, y)
    assert len(clf.loss_history_) == n_iter


def test_classifier_loss_decreases():
    X, y = _binary_data()
    clf = MLPClassifier(
        hidden_layer_sizes=(32,), n_iterations=200,
        learning_rate=1e-2, random_state=42,
    )
    clf.fit(X, y)
    assert clf.loss_history_[-1] < clf.loss_history_[0]


def test_classifier_predict_shape():
    X, y = _binary_data()
    clf = MLPClassifier(n_iterations=5, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X[:20])
    assert preds.shape == (20,)


def test_classifier_predict_proba_shape():
    X, y = _binary_data()
    clf = MLPClassifier(n_iterations=5, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X[:15])
    assert proba.shape == (15, 2)


def test_classifier_predict_proba_sums_to_one():
    X, y = _binary_data()
    clf = MLPClassifier(n_iterations=10, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_predict_labels_in_classes():
    X, y = _binary_data()
    clf = MLPClassifier(n_iterations=5, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({0, 1})


def test_classifier_high_accuracy_binary():
    X, y = _binary_data(n=400)
    split = 300
    clf = MLPClassifier(
        hidden_layer_sizes=(32,), n_iterations=300,
        learning_rate=1e-2, random_state=0,
    )
    clf.fit(X[:split], y[:split])
    acc = accuracy(y[split:], clf.predict(X[split:]))
    assert acc >= 0.92, f"Expected ≥0.92, got {acc:.4f}"


def test_classifier_predict_before_fit_raises():
    clf = MLPClassifier()
    with pytest.raises(RuntimeError, match="fit"):
        clf.predict(np.zeros((5, 2)))


def test_classifier_invalid_activation_raises():
    with pytest.raises(ValueError, match="activation"):
        MLPClassifier(activation="elu")


# ---------------------------------------------------------------------------
# MLPClassifier — multi-class
# ---------------------------------------------------------------------------

def test_classifier_multiclass_classes_attribute():
    X, y = _multiclass_data()
    clf = MLPClassifier(hidden_layer_sizes=(16,), n_iterations=5, random_state=0)
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1, 2]


def test_classifier_multiclass_proba_shape():
    X, y = _multiclass_data()
    clf = MLPClassifier(n_iterations=5, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X[:10])
    assert proba.shape == (10, 3)


def test_classifier_multiclass_accuracy():
    X, y = _multiclass_data()
    split = int(0.75 * len(y))
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16), n_iterations=400,
        learning_rate=1e-2, random_state=7,
    )
    clf.fit(X[:split], y[:split])
    acc = accuracy(y[split:], clf.predict(X[split:]))
    assert acc >= 0.88, f"Expected ≥0.88, got {acc:.4f}"


def test_classifier_multiclass_proba_sums_to_one():
    X, y = _multiclass_data()
    clf = MLPClassifier(n_iterations=5, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# MLPClassifier — activation functions
# ---------------------------------------------------------------------------

def test_classifier_tanh_activation():
    X, y = _binary_data()
    clf = MLPClassifier(activation="tanh", n_iterations=5, random_state=0)
    clf.fit(X, y)
    assert clf.predict(X[:5]).shape == (5,)


def test_classifier_sigmoid_activation():
    X, y = _binary_data()
    clf = MLPClassifier(activation="sigmoid", n_iterations=5, random_state=0)
    clf.fit(X, y)
    assert clf.predict(X[:5]).shape == (5,)


def test_classifier_fullbatch_mode():
    X, y = _binary_data()
    clf = MLPClassifier(
        n_iterations=5, batch_size=None, random_state=0
    )
    clf.fit(X, y)
    assert len(clf.loss_history_) == 5


# ---------------------------------------------------------------------------
# MLPRegressor — lifecycle
# ---------------------------------------------------------------------------

def test_regressor_fit_returns_self():
    X, y = _regression_data()
    reg = MLPRegressor(hidden_layer_sizes=(16,), n_iterations=5, random_state=0)
    assert reg.fit(X, y) is reg


def test_regressor_attributes_after_fit():
    X, y = _regression_data()
    reg = MLPRegressor(hidden_layer_sizes=(16,), n_iterations=5, random_state=0)
    reg.fit(X, y)
    assert reg._fitted
    # input(3) → hidden(16) → output(1): two weight matrices
    assert len(reg.weights_) == 2
    assert reg.weights_[0].shape == (3, 16)
    assert reg.weights_[1].shape == (16, 1)


def test_regressor_predict_shape_1d():
    X, y = _regression_data()
    reg = MLPRegressor(n_iterations=5, random_state=0)
    reg.fit(X, y)
    preds = reg.predict(X[:25])
    assert preds.shape == (25,)


def test_regressor_loss_decreases():
    X, y = _regression_data(n=400)
    reg = MLPRegressor(
        hidden_layer_sizes=(32,), n_iterations=300,
        learning_rate=1e-2, random_state=0,
    )
    reg.fit(X, y)
    assert reg.loss_history_[-1] < reg.loss_history_[0]


def test_regressor_r2_score():
    X, y = _regression_data(n=400)
    split = 300
    reg = MLPRegressor(
        hidden_layer_sizes=(32,), n_iterations=500,
        learning_rate=1e-2, random_state=0,
    )
    reg.fit(X[:split], y[:split])
    r2 = r2_score(y[split:], reg.predict(X[split:]))
    assert r2 >= 0.90, f"Expected R²≥0.90, got {r2:.4f}"


def test_regressor_predict_before_fit_raises():
    reg = MLPRegressor()
    with pytest.raises(RuntimeError, match="fit"):
        reg.predict(np.zeros((5, 3)))


def test_regressor_loss_history_length():
    X, y = _regression_data()
    n_iter = 8
    reg = MLPRegressor(n_iterations=n_iter, random_state=0)
    reg.fit(X, y)
    assert len(reg.loss_history_) == n_iter


def test_regressor_reproducible_with_seed():
    X, y = _regression_data()
    reg1 = MLPRegressor(hidden_layer_sizes=(16,), n_iterations=10, random_state=99)
    reg2 = MLPRegressor(hidden_layer_sizes=(16,), n_iterations=10, random_state=99)
    reg1.fit(X, y)
    reg2.fit(X, y)
    np.testing.assert_array_equal(reg1.predict(X[:10]), reg2.predict(X[:10]))
