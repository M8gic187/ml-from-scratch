"""Tests for DecisionTreeClassifier."""

import numpy as np
import pytest

from ml_scratch.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def binary_dataset():
    """Linearly separable binary data."""
    rng = np.random.default_rng(7)
    X_a = rng.normal(loc=[2, 2], scale=0.5, size=(60, 2))
    X_b = rng.normal(loc=[-2, -2], scale=0.5, size=(60, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


@pytest.fixture()
def multiclass_dataset():
    """Three-class dataset arranged in 2-D."""
    rng = np.random.default_rng(99)
    centres = np.array([[0, 4], [-4, -2], [4, -2]], dtype=float)
    X = np.vstack([rng.normal(c, 0.6, (40, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 40)
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDecisionTreeClassifier:
    def test_memorises_binary(self, binary_dataset):
        X, y = binary_dataset
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X, y)
        assert np.mean(clf.predict(X) == y) == 1.0

    def test_memorises_multiclass(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.95

    def test_max_depth_limits_tree(self, binary_dataset):
        X, y = binary_dataset
        # depth=1 creates a single decision stump
        clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
        # stump should still do better than random on this easy dataset
        assert np.mean(clf.predict(X) == y) > 0.7

    def test_predict_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = DecisionTreeClassifier(random_state=0).fit(X, y)
        assert clf.predict(X[:15]).shape == (15,)

    def test_predict_proba_shape(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = DecisionTreeClassifier(random_state=0).fit(X, y)
        proba = clf.predict_proba(X[:10])
        assert proba.shape == (10, 3)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(10))

    def test_classes_attribute(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = DecisionTreeClassifier().fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_feature_subspace(self, binary_dataset):
        X, y = binary_dataset
        clf = DecisionTreeClassifier(n_features=1, random_state=3).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.75

    def test_min_samples_leaf(self, binary_dataset):
        X, y = binary_dataset
        clf = DecisionTreeClassifier(min_samples_leaf=10, random_state=0).fit(X, y)
        # Larger leaf constraint should still produce reasonable accuracy
        assert np.mean(clf.predict(X) == y) > 0.85

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            DecisionTreeClassifier().predict(np.array([[1.0, 2.0]]))
