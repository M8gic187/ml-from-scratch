"""Tests for IsolationForest anomaly detection."""

import numpy as np
import pytest

from ml_scratch.ensemble import IsolationForest
from ml_scratch.ensemble.isolation_forest import _c, _IsolationTree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def inlier_blob():
    """200 normally-distributed inliers in 2-D."""
    rng = np.random.default_rng(0)
    return rng.normal(loc=[0.0, 0.0], scale=1.0, size=(200, 2))


@pytest.fixture()
def mixed_dataset():
    """200 inliers + 10 clear outliers far from the cluster."""
    rng = np.random.default_rng(42)
    inliers = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(200, 2))
    outliers = rng.normal(loc=[12.0, 12.0], scale=0.2, size=(10, 2))
    X = np.vstack([inliers, outliers])
    y_true = np.array([1] * 200 + [-1] * 10)
    return X, y_true


@pytest.fixture()
def fitted_forest(mixed_dataset):
    X, _ = mixed_dataset
    return IsolationForest(n_estimators=60, contamination=0.05, random_state=7).fit(X)


# ---------------------------------------------------------------------------
# _c correction factor
# ---------------------------------------------------------------------------

class TestCFactor:
    def test_zero_for_n_le_1(self):
        assert _c(0) == 0.0
        assert _c(1) == 0.0

    def test_one_for_n_2(self):
        assert _c(2) == 1.0

    def test_increases_with_n(self):
        values = [_c(n) for n in [2, 5, 10, 50, 200]]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def test_positive_for_large_n(self):
        assert _c(1000) > 0


# ---------------------------------------------------------------------------
# _IsolationTree
# ---------------------------------------------------------------------------

class TestIsolationTree:
    def test_fit_returns_self(self, inlier_blob):
        tree = _IsolationTree(max_depth=8, random_state=0)
        result = tree.fit(inlier_blob)
        assert result is tree

    def test_path_length_shape(self, inlier_blob):
        tree = _IsolationTree(max_depth=8, random_state=0).fit(inlier_blob)
        lengths = tree.path_length(inlier_blob[:20])
        assert lengths.shape == (20,)

    def test_path_length_positive(self, inlier_blob):
        tree = _IsolationTree(max_depth=8, random_state=1).fit(inlier_blob)
        lengths = tree.path_length(inlier_blob)
        assert np.all(lengths >= 0)

    def test_single_sample_fit(self):
        X = np.array([[1.0, 2.0]])
        tree = _IsolationTree(max_depth=4, random_state=0).fit(X)
        lengths = tree.path_length(X)
        assert lengths.shape == (1,)


# ---------------------------------------------------------------------------
# IsolationForest construction
# ---------------------------------------------------------------------------

class TestIsolationForestConstruction:
    def test_default_params(self):
        iso = IsolationForest()
        assert iso.n_estimators == 100
        assert iso.max_samples == "auto"
        assert iso.contamination == 0.1

    def test_invalid_contamination_zero(self):
        with pytest.raises(ValueError):
            IsolationForest(contamination=0.0)

    def test_invalid_contamination_above_half(self):
        with pytest.raises(ValueError):
            IsolationForest(contamination=0.51)

    def test_valid_contamination_boundary(self):
        iso = IsolationForest(contamination=0.5)
        assert iso.contamination == 0.5


# ---------------------------------------------------------------------------
# IsolationForest.fit
# ---------------------------------------------------------------------------

class TestIsolationForestFit:
    def test_fit_returns_self(self, inlier_blob):
        iso = IsolationForest(n_estimators=10, random_state=0)
        assert iso.fit(inlier_blob) is iso

    def test_estimators_count(self, inlier_blob):
        iso = IsolationForest(n_estimators=25, random_state=0).fit(inlier_blob)
        assert len(iso.estimators_) == 25

    def test_n_features_in(self, inlier_blob):
        iso = IsolationForest(n_estimators=10, random_state=0).fit(inlier_blob)
        assert iso.n_features_in_ == 2

    def test_offset_set_after_fit(self, inlier_blob):
        iso = IsolationForest(n_estimators=10, random_state=0).fit(inlier_blob)
        assert isinstance(iso.offset_, float)

    def test_max_samples_auto_caps_at_256(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1000, 3))
        iso = IsolationForest(n_estimators=5, max_samples="auto", random_state=0).fit(X)
        assert iso._max_samples_actual == 256

    def test_max_samples_auto_uses_n_when_small(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        iso = IsolationForest(n_estimators=5, max_samples="auto", random_state=0).fit(X)
        assert iso._max_samples_actual == 50

    def test_explicit_max_samples(self, inlier_blob):
        iso = IsolationForest(n_estimators=5, max_samples=100, random_state=0).fit(inlier_blob)
        assert iso._max_samples_actual == 100


# ---------------------------------------------------------------------------
# IsolationForest.score_samples
# ---------------------------------------------------------------------------

class TestScoreSamples:
    def test_score_shape(self, fitted_forest, mixed_dataset):
        X, _ = mixed_dataset
        scores = fitted_forest.score_samples(X)
        assert scores.shape == (len(X),)

    def test_scores_negative(self, fitted_forest, mixed_dataset):
        X, _ = mixed_dataset
        scores = fitted_forest.score_samples(X)
        assert np.all(scores <= 0)

    def test_outliers_score_lower_than_inliers(self, mixed_dataset):
        X, y_true = mixed_dataset
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=0).fit(X)
        scores = iso.score_samples(X)
        mean_inlier = scores[y_true == 1].mean()
        mean_outlier = scores[y_true == -1].mean()
        assert mean_outlier < mean_inlier

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            IsolationForest().score_samples(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# IsolationForest.predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_shape(self, fitted_forest, mixed_dataset):
        X, _ = mixed_dataset
        preds = fitted_forest.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_labels(self, fitted_forest, mixed_dataset):
        X, _ = mixed_dataset
        preds = fitted_forest.predict(X)
        assert set(preds).issubset({-1, 1})

    def test_detects_clear_outliers(self, mixed_dataset):
        X, y_true = mixed_dataset
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=0).fit(X)
        preds = iso.predict(X)
        # All 10 obvious outliers should be flagged
        outlier_recall = (preds[y_true == -1] == -1).mean()
        assert outlier_recall >= 0.8

    def test_inlier_precision(self, mixed_dataset):
        X, y_true = mixed_dataset
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=0).fit(X)
        preds = iso.predict(X)
        inlier_accuracy = (preds[y_true == 1] == 1).mean()
        assert inlier_accuracy >= 0.88

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            IsolationForest().predict(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# IsolationForest.decision_function
# ---------------------------------------------------------------------------

class TestDecisionFunction:
    def test_decision_shape(self, fitted_forest, mixed_dataset):
        X, _ = mixed_dataset
        dec = fitted_forest.decision_function(X)
        assert dec.shape == (len(X),)

    def test_decision_consistent_with_predict(self, fitted_forest, mixed_dataset):
        X, _ = mixed_dataset
        dec = fitted_forest.decision_function(X)
        preds = fitted_forest.predict(X)
        assert np.all((dec >= 0) == (preds == 1))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_scores(self, inlier_blob):
        a = IsolationForest(n_estimators=30, random_state=99).fit(inlier_blob)
        b = IsolationForest(n_estimators=30, random_state=99).fit(inlier_blob)
        np.testing.assert_array_equal(a.score_samples(inlier_blob), b.score_samples(inlier_blob))

    def test_different_seed_different_scores(self, inlier_blob):
        a = IsolationForest(n_estimators=30, random_state=1).fit(inlier_blob)
        b = IsolationForest(n_estimators=30, random_state=2).fit(inlier_blob)
        assert not np.array_equal(
            a.score_samples(inlier_blob), b.score_samples(inlier_blob)
        )
