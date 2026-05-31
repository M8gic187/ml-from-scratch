"""Tests for ExtraTreesClassifier."""

import numpy as np
import pytest

from ml_scratch.ensemble import ExtraTreesClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def binary_dataset():
    """Well-separated binary classification dataset."""
    rng = np.random.default_rng(0)
    X_a = rng.normal(loc=[3.0, 3.0], scale=0.7, size=(60, 2))
    X_b = rng.normal(loc=[-3.0, -3.0], scale=0.7, size=(60, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


@pytest.fixture()
def multiclass_dataset():
    """Three-cluster dataset in 2-D."""
    rng = np.random.default_rng(7)
    centres = np.array([[0, 5], [-5, -2], [5, -2]], dtype=float)
    X = np.vstack([rng.normal(c, 0.6, (50, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 50)
    return X, y


@pytest.fixture()
def four_feature_dataset():
    """Four-feature binary dataset for n_features subspace tests."""
    rng = np.random.default_rng(99)
    X_a = rng.normal([1, 1, 1, 1], 0.5, (50, 4))
    X_b = rng.normal([-1, -1, -1, -1], 0.5, (50, 4))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


# ---------------------------------------------------------------------------
# Construction and parameter validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_params(self):
        clf = ExtraTreesClassifier()
        assert clf.n_estimators == 100
        assert clf.bootstrap is False
        assert clf.n_features is None
        assert clf.max_depth is None

    def test_custom_params_stored(self):
        clf = ExtraTreesClassifier(n_estimators=25, max_depth=3, bootstrap=True, random_state=1)
        assert clf.n_estimators == 25
        assert clf.max_depth == 3
        assert clf.bootstrap is True
        assert clf.random_state == 1

    def test_not_fitted_raises(self, binary_dataset):
        X, _ = binary_dataset
        clf = ExtraTreesClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(X)

    def test_not_fitted_proba_raises(self, binary_dataset):
        X, _ = binary_dataset
        clf = ExtraTreesClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Post-fit attributes
# ---------------------------------------------------------------------------


class TestFitAttributes:
    def test_estimators_count(self, binary_dataset):
        X, y = binary_dataset
        clf = ExtraTreesClassifier(n_estimators=17, random_state=0).fit(X, y)
        assert len(clf.estimators_) == 17

    def test_classes_binary(self, binary_dataset):
        X, y = binary_dataset
        clf = ExtraTreesClassifier(n_estimators=10, random_state=0).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_classes_multiclass(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = ExtraTreesClassifier(n_estimators=10, random_state=0).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_n_features_in(self, four_feature_dataset):
        X, y = four_feature_dataset
        clf = ExtraTreesClassifier(n_estimators=5, random_state=0).fit(X, y)
        assert clf.n_features_in_ == 4


# ---------------------------------------------------------------------------
# Prediction shapes and types
# ---------------------------------------------------------------------------


class TestPredictShapes:
    def test_predict_shape_binary(self, binary_dataset):
        X, y = binary_dataset
        clf = ExtraTreesClassifier(n_estimators=20, random_state=0).fit(X, y)
        preds = clf.predict(X[:30])
        assert preds.shape == (30,)

    def test_predict_shape_multiclass(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = ExtraTreesClassifier(n_estimators=20, random_state=0).fit(X, y)
        preds = clf.predict(X[:15])
        assert preds.shape == (15,)

    def test_predict_labels_in_classes(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = ExtraTreesClassifier(n_estimators=10, random_state=0).fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1, 2})

    def test_proba_shape_binary(self, binary_dataset):
        X, y = binary_dataset
        clf = ExtraTreesClassifier(n_estimators=20, random_state=0).fit(X, y)
        proba = clf.predict_proba(X[:25])
        assert proba.shape == (25, 2)

    def test_proba_shape_multiclass(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = ExtraTreesClassifier(n_estimators=20, random_state=0).fit(X, y)
        proba = clf.predict_proba(X[:12])
        assert proba.shape == (12, 3)

    def test_proba_sums_to_one(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = ExtraTreesClassifier(n_estimators=30, random_state=3).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-12)

    def test_proba_non_negative(self, binary_dataset):
        X, y = binary_dataset
        clf = ExtraTreesClassifier(n_estimators=20, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        assert (proba >= 0).all()

    def test_predict_consistent_with_proba(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = ExtraTreesClassifier(n_estimators=20, random_state=0).fit(X, y)
        preds_direct = clf.predict(X)
        preds_from_proba = clf.classes_[np.argmax(clf.predict_proba(X), axis=1)]
        np.testing.assert_array_equal(preds_direct, preds_from_proba)


# ---------------------------------------------------------------------------
# Classification quality
# ---------------------------------------------------------------------------


class TestClassificationQuality:
    def test_high_accuracy_binary(self, binary_dataset):
        X, y = binary_dataset
        clf = ExtraTreesClassifier(n_estimators=50, random_state=0).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.97

    def test_high_accuracy_multiclass(self, multiclass_dataset):
        X, y = multiclass_dataset
        clf = ExtraTreesClassifier(n_estimators=50, random_state=0).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.95

    def test_accuracy_improves_with_more_trees(self, binary_dataset):
        X, y = binary_dataset
        acc5 = np.mean(ExtraTreesClassifier(n_estimators=5, random_state=0).fit(X, y).predict(X) == y)
        acc50 = np.mean(ExtraTreesClassifier(n_estimators=50, random_state=0).fit(X, y).predict(X) == y)
        assert acc50 >= acc5


# ---------------------------------------------------------------------------
# Parameter effects
# ---------------------------------------------------------------------------


class TestParameterEffects:
    def test_n_features_subspace(self, four_feature_dataset):
        X, y = four_feature_dataset
        clf = ExtraTreesClassifier(n_estimators=20, n_features=2, random_state=0).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.90

    def test_bootstrap_mode(self, binary_dataset):
        X, y = binary_dataset
        clf = ExtraTreesClassifier(n_estimators=30, bootstrap=True, random_state=5).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.90

    def test_max_depth_limits_trees(self, binary_dataset):
        X, y = binary_dataset
        clf = ExtraTreesClassifier(n_estimators=10, max_depth=1, random_state=0).fit(X, y)
        # Stumps still achieve positive accuracy on well-separated data
        assert np.mean(clf.predict(X) == y) > 0.70

    def test_min_samples_split(self, binary_dataset):
        X, y = binary_dataset
        clf = ExtraTreesClassifier(n_estimators=20, min_samples_split=10, random_state=0).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.90


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_predictions(self, binary_dataset):
        X, y = binary_dataset
        preds_a = ExtraTreesClassifier(n_estimators=20, random_state=42).fit(X, y).predict(X)
        preds_b = ExtraTreesClassifier(n_estimators=20, random_state=42).fit(X, y).predict(X)
        np.testing.assert_array_equal(preds_a, preds_b)

    def test_different_seeds_differ(self):
        # Shallow trees (max_depth=2) on overlapping data produce fractional
        # vote distributions that change with the random threshold seed.
        rng = np.random.default_rng(0)
        X_a = rng.normal([1.0, 1.0], 1.2, (80, 2))
        X_b = rng.normal([-1.0, -1.0], 1.2, (80, 2))
        X = np.vstack([X_a, X_b])
        y = np.array([0] * 80 + [1] * 80)
        proba_a = ExtraTreesClassifier(
            n_estimators=30, max_depth=2, random_state=1
        ).fit(X, y).predict_proba(X)
        proba_b = ExtraTreesClassifier(
            n_estimators=30, max_depth=2, random_state=999
        ).fit(X, y).predict_proba(X)
        assert not np.allclose(proba_a, proba_b)
