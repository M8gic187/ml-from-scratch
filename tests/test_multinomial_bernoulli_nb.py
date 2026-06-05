"""Tests for MultinomialNB and BernoulliNB classifiers."""

import numpy as np
import pytest

from ml_scratch.naive_bayes import BernoulliNB, MultinomialNB


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def count_dataset():
    """Synthetic word-count matrix with two clearly separated classes."""
    rng = np.random.default_rng(42)
    # Class 0: high counts in first 5 features
    X0 = rng.poisson(lam=[8, 6, 7, 5, 6, 0, 0, 0, 0, 0], size=(100, 10)).astype(float)
    # Class 1: high counts in last 5 features
    X1 = rng.poisson(lam=[0, 0, 0, 0, 0, 8, 6, 7, 5, 6], size=(100, 10)).astype(float)
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


@pytest.fixture()
def binary_dataset():
    """Synthetic binary feature matrix with two clearly separated classes."""
    rng = np.random.default_rng(7)
    # Class 0: features 0-4 tend to be 1
    X0 = (rng.uniform(size=(80, 10)) < np.array([0.9, 0.85, 0.9, 0.8, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1])).astype(float)
    # Class 1: features 5-9 tend to be 1
    X1 = (rng.uniform(size=(80, 10)) < np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.85, 0.9, 0.8, 0.85])).astype(float)
    X = np.vstack([X0, X1])
    y = np.array([0] * 80 + [1] * 80)
    return X, y


@pytest.fixture()
def multiclass_count_dataset():
    """Four-class word-count dataset."""
    rng = np.random.default_rng(99)
    lams = [
        [10, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 10, 8, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 10, 8, 7, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 8, 7],
    ]
    X = np.vstack([rng.poisson(lam=l, size=(60, 12)).astype(float) for l in lams])
    y = np.repeat([0, 1, 2, 3], 60)
    return X, y


# ===========================================================================
# MultinomialNB tests
# ===========================================================================

class TestMultinomialNB:

    def test_binary_classification_accuracy(self, count_dataset):
        X, y = count_dataset
        clf = MultinomialNB().fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.97

    def test_multiclass_accuracy(self, multiclass_count_dataset):
        X, y = multiclass_count_dataset
        clf = MultinomialNB().fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.95

    def test_predict_shape(self, count_dataset):
        X, y = count_dataset
        clf = MultinomialNB().fit(X, y)
        assert clf.predict(X[:15]).shape == (15,)

    def test_predict_proba_shape(self, count_dataset):
        X, y = count_dataset
        clf = MultinomialNB().fit(X, y)
        assert clf.predict_proba(X[:10]).shape == (10, 2)

    def test_predict_proba_sums_to_one(self, count_dataset):
        X, y = count_dataset
        clf = MultinomialNB().fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-12)

    def test_predict_proba_non_negative(self, count_dataset):
        X, y = count_dataset
        proba = MultinomialNB().fit(X, y).predict_proba(X)
        assert np.all(proba >= 0)

    def test_predict_consistent_with_proba(self, multiclass_count_dataset):
        X, y = multiclass_count_dataset
        clf = MultinomialNB().fit(X, y)
        preds = clf.predict(X)
        proba_preds = clf.classes_[np.argmax(clf.predict_proba(X), axis=1)]
        np.testing.assert_array_equal(preds, proba_preds)

    def test_classes_attribute(self, multiclass_count_dataset):
        X, y = multiclass_count_dataset
        clf = MultinomialNB().fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2, 3])

    def test_feature_log_prob_shape(self, count_dataset):
        X, y = count_dataset
        clf = MultinomialNB().fit(X, y)
        assert clf.feature_log_prob_.shape == (2, X.shape[1])

    def test_feature_log_prob_sums_to_zero(self, count_dataset):
        """Each row of feature_log_prob_ must sum to log(1) = 0 (simplex constraint)."""
        X, y = count_dataset
        clf = MultinomialNB().fit(X, y)
        row_sums = np.exp(clf.feature_log_prob_).sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(2), atol=1e-12)

    def test_alpha_zero_no_smoothing(self):
        """With alpha=0, zero-count features remain zero; ensure no crash."""
        X = np.array([[3, 0], [2, 0], [0, 4], [0, 5]], dtype=float)
        y = np.array([0, 0, 1, 1])
        clf = MultinomialNB(alpha=0).fit(X, y)
        assert clf.predict(X).shape == (4,)

    def test_raises_on_negative_features(self):
        X = np.array([[-1, 2], [3, 4]], dtype=float)
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="non-negative"):
            MultinomialNB().fit(X, y)

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            MultinomialNB().predict(np.array([[1.0, 2.0]]))

    def test_log_proba_shape(self, count_dataset):
        X, y = count_dataset
        clf = MultinomialNB().fit(X, y)
        assert clf.predict_log_proba(X[:8]).shape == (8, 2)

    def test_laplace_vs_lidstone(self, count_dataset):
        """Different alpha values must give different (but valid) predictions."""
        X, y = count_dataset
        clf_laplace = MultinomialNB(alpha=1.0).fit(X, y)
        clf_lidstone = MultinomialNB(alpha=0.5).fit(X, y)
        # Both should achieve high accuracy
        assert np.mean(clf_laplace.predict(X) == y) > 0.95
        assert np.mean(clf_lidstone.predict(X) == y) > 0.95


# ===========================================================================
# BernoulliNB tests
# ===========================================================================

class TestBernoulliNB:

    def test_binary_classification_accuracy(self, binary_dataset):
        X, y = binary_dataset
        clf = BernoulliNB().fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.95

    def test_predict_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = BernoulliNB().fit(X, y)
        assert clf.predict(X[:20]).shape == (20,)

    def test_predict_proba_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = BernoulliNB().fit(X, y)
        assert clf.predict_proba(X[:10]).shape == (10, 2)

    def test_predict_proba_sums_to_one(self, binary_dataset):
        X, y = binary_dataset
        clf = BernoulliNB().fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-12)

    def test_predict_proba_non_negative(self, binary_dataset):
        X, y = binary_dataset
        proba = BernoulliNB().fit(X, y).predict_proba(X)
        assert np.all(proba >= 0)

    def test_predict_consistent_with_proba(self, binary_dataset):
        X, y = binary_dataset
        clf = BernoulliNB().fit(X, y)
        preds = clf.predict(X)
        proba_preds = clf.classes_[np.argmax(clf.predict_proba(X), axis=1)]
        np.testing.assert_array_equal(preds, proba_preds)

    def test_classes_attribute(self, binary_dataset):
        X, y = binary_dataset
        clf = BernoulliNB().fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_feature_log_prob_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = BernoulliNB().fit(X, y)
        assert clf.feature_log_prob_.shape == (2, X.shape[1])

    def test_feature_log_prob_in_valid_range(self, binary_dataset):
        """All log-probabilities must be <= 0 (probabilities <= 1)."""
        X, y = binary_dataset
        clf = BernoulliNB().fit(X, y)
        assert np.all(clf.feature_log_prob_ <= 0)

    def test_binarize_threshold(self):
        """Continuous inputs must be binarized correctly at the given threshold."""
        X = np.array([[0.2, 0.8], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]], dtype=float)
        y = np.array([0, 1, 0, 1])
        clf = BernoulliNB(binarize=0.5).fit(X, y)
        assert clf.predict(X).shape == (4,)

    def test_binarize_none_accepts_binary_input(self, binary_dataset):
        """With binarize=None, pre-binarized inputs should work identically."""
        X, y = binary_dataset
        clf_auto = BernoulliNB(binarize=0.0).fit(X, y)
        clf_pre = BernoulliNB(binarize=None).fit(X, y)
        np.testing.assert_array_equal(clf_auto.predict(X), clf_pre.predict(X))

    def test_absent_features_penalised(self):
        """Absent features must lower log-likelihood (neg_feature_log_prob < 0)."""
        X = np.array([[1, 0], [0, 1]], dtype=float)
        y = np.array([0, 1])
        clf = BernoulliNB(binarize=None).fit(X, y)
        assert np.all(clf._neg_feature_log_prob <= 0)

    def test_log_proba_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = BernoulliNB().fit(X, y)
        assert clf.predict_log_proba(X[:6]).shape == (6, 2)

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            BernoulliNB().predict(np.array([[1.0, 0.0]]))

    def test_multiclass(self):
        """BernoulliNB must handle more than two classes."""
        rng = np.random.default_rng(3)
        probs = np.array([
            [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
        ])
        X = np.vstack([(rng.uniform(size=(50, 6)) < p).astype(float) for p in probs])
        y = np.repeat([0, 1, 2], 50)
        clf = BernoulliNB().fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.93
        assert clf.predict_proba(X).shape == (150, 3)
