"""Tests for GaussianNB classifier."""

import numpy as np
import pytest

from ml_scratch.naive_bayes import GaussianNB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def binary_dataset():
    """Well-separated binary Gaussian clusters."""
    rng = np.random.default_rng(0)
    X_a = rng.normal(loc=[3.0, 3.0], scale=0.6, size=(80, 2))
    X_b = rng.normal(loc=[-3.0, -3.0], scale=0.6, size=(80, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * 80 + [1] * 80)
    return X, y


@pytest.fixture()
def multiclass_dataset():
    """Four-class 3-D Gaussian mixture."""
    rng = np.random.default_rng(17)
    centres = np.array([[4, 0, 0], [-4, 0, 0], [0, 4, 0], [0, -4, 0]], dtype=float)
    X = np.vstack([rng.normal(c, 0.8, (50, 3)) for c in centres])
    y = np.repeat([0, 1, 2, 3], 50)
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGaussianNB:
    def test_perfect_binary(self, binary_dataset):
        X, y = binary_dataset
        clf = GaussianNB().fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.99

    def test_multiclass_accuracy(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = GaussianNB().fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.95

    def test_predict_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = GaussianNB().fit(X, y)
        assert clf.predict(X[:20]).shape == (20,)

    def test_predict_proba_shape(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = GaussianNB().fit(X, y)
        proba = clf.predict_proba(X[:15])
        assert proba.shape == (15, 4)

    def test_predict_proba_sums_to_one(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = GaussianNB().fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-12)

    def test_predict_proba_non_negative(self, binary_dataset):
        X, y = binary_dataset
        proba = GaussianNB().fit(X, y).predict_proba(X)
        assert np.all(proba >= 0)

    def test_classes_attribute(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = GaussianNB().fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2, 3])

    def test_log_proba_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = GaussianNB().fit(X, y)
        log_p = clf.predict_log_proba(X[:10])
        assert log_p.shape == (10, 2)

    def test_predict_consistent_with_proba(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = GaussianNB().fit(X, y)
        preds = clf.predict(X)
        proba_preds = clf.classes_[np.argmax(clf.predict_proba(X), axis=1)]
        np.testing.assert_array_equal(preds, proba_preds)

    def test_var_smoothing_constant_feature(self):
        """Constant feature column must not cause division-by-zero."""
        X = np.column_stack([np.ones(40), np.random.default_rng(5).normal(size=40)])
        y = np.array([0] * 20 + [1] * 20)
        clf = GaussianNB(var_smoothing=1e-9).fit(X, y)
        assert clf.predict(X).shape == (40,)

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            GaussianNB().predict(np.array([[1.0, 2.0]]))
