"""Tests for AdaBoostClassifier."""

import numpy as np
import pytest

from ml_scratch.ensemble import AdaBoostClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def binary_dataset():
    """Linearly separable binary data."""
    rng = np.random.default_rng(42)
    X_a = rng.normal(loc=[3.0, 3.0], scale=0.5, size=(60, 2))
    X_b = rng.normal(loc=[-3.0, -3.0], scale=0.5, size=(60, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


@pytest.fixture()
def string_labels():
    """Binary dataset with string class labels."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((100, 2))
    y = np.where(X[:, 0] > 0, "cat", "dog")
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdaBoostClassifier:
    def test_high_train_accuracy(self, binary_dataset):
        X, y = binary_dataset
        clf = AdaBoostClassifier(n_estimators=50).fit(X, y)
        assert (clf.predict(X) == y).mean() > 0.97

    def test_predict_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = AdaBoostClassifier(n_estimators=10).fit(X, y)
        assert clf.predict(X[:20]).shape == (20,)

    def test_predict_values_in_classes(self, binary_dataset):
        X, y = binary_dataset
        clf = AdaBoostClassifier(n_estimators=10).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_classes_attribute(self, binary_dataset):
        X, y = binary_dataset
        clf = AdaBoostClassifier().fit(X, y)
        assert list(clf.classes_) == [0, 1]

    def test_estimator_weights_positive(self, binary_dataset):
        X, y = binary_dataset
        clf = AdaBoostClassifier(n_estimators=20).fit(X, y)
        assert (clf.estimator_weights_ > 0).all()

    def test_estimator_errors_in_range(self, binary_dataset):
        X, y = binary_dataset
        clf = AdaBoostClassifier(n_estimators=20).fit(X, y)
        assert (clf.estimator_errors_ >= 0).all()
        assert (clf.estimator_errors_ <= 0.5 + 1e-6).all()

    def test_number_of_estimators(self, binary_dataset):
        X, y = binary_dataset
        clf = AdaBoostClassifier(n_estimators=15).fit(X, y)
        assert len(clf.estimators_) == 15
        assert len(clf.estimator_weights_) == 15

    def test_decision_function_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = AdaBoostClassifier(n_estimators=10).fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(X),)

    def test_string_labels(self, string_labels):
        """Classifier must work with non-integer label types."""
        X, y = string_labels
        clf = AdaBoostClassifier(n_estimators=30).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({"cat", "dog"})
        assert (preds == y).mean() > 0.80

    def test_multiclass_raises(self, binary_dataset):
        X, _ = binary_dataset
        y_multi = np.array([0, 1, 2] * 40)
        with pytest.raises(ValueError, match="binary"):
            AdaBoostClassifier().fit(X, y_multi)

    def test_not_fitted_raises(self, binary_dataset):
        X, _ = binary_dataset
        with pytest.raises(RuntimeError, match="fit"):
            AdaBoostClassifier().predict(X)
