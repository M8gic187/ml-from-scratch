"""Tests for RandomForestClassifier."""

import numpy as np
import pytest

from ml_scratch.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def binary_dataset():
    """Linearly separable binary data."""
    rng = np.random.default_rng(42)
    X_a = rng.normal(loc=[2.0, 2.0], scale=0.7, size=(60, 2))
    X_b = rng.normal(loc=[-2.0, -2.0], scale=0.7, size=(60, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


@pytest.fixture()
def multiclass_dataset():
    """Three-class dataset in 2-D."""
    rng = np.random.default_rng(99)
    centres = np.array([[0, 4], [-4, -2], [4, -2]], dtype=float)
    X = np.vstack([rng.normal(c, 0.6, (40, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 40)
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRandomForestClassifier:
    def test_high_train_accuracy_binary(self, binary_dataset):
        X, y = binary_dataset
        clf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.97

    def test_high_train_accuracy_multiclass(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = RandomForestClassifier(n_estimators=50, random_state=7).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.95

    def test_predict_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        assert clf.predict(X[:25]).shape == (25,)

    def test_predict_proba_shape(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        proba = clf.predict_proba(X[:12])
        assert proba.shape == (12, 3)

    def test_predict_proba_sums_to_one(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = RandomForestClassifier(n_estimators=20, random_state=1).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-12)

    def test_classes_attribute(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_n_estimators_stored(self, binary_dataset):
        X, y = binary_dataset
        clf = RandomForestClassifier(n_estimators=17, random_state=0).fit(X, y)
        assert len(clf.estimators_) == 17

    def test_reproducible_with_seed(self, binary_dataset):
        X, y = binary_dataset
        preds_a = RandomForestClassifier(n_estimators=20, random_state=5).fit(X, y).predict(X)
        preds_b = RandomForestClassifier(n_estimators=20, random_state=5).fit(X, y).predict(X)
        np.testing.assert_array_equal(preds_a, preds_b)

    def test_no_bootstrap_mode(self, binary_dataset):
        X, y = binary_dataset
        clf = RandomForestClassifier(n_estimators=10, bootstrap=False, random_state=2).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.90

    def test_max_depth_respected(self, multiclass_dataset):
        """Shallow trees can still beat a trivial baseline."""
        X, y = multiclass_dataset
        clf = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.75

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            RandomForestClassifier().predict(np.array([[1.0, 2.0]]))
