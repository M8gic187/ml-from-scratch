"""Tests for LinearDiscriminantAnalysis."""

import numpy as np
import pytest

from ml_scratch.decomposition import LinearDiscriminantAnalysis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def three_class_data():
    """Well-separated 3-class 2-D dataset."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal([0, 0], 0.8, (60, 2)),
        rng.normal([5, 5], 0.8, (60, 2)),
        rng.normal([10, 0], 0.8, (60, 2)),
    ])
    y = np.array([0] * 60 + [1] * 60 + [2] * 60)
    return X, y


@pytest.fixture()
def binary_data():
    """Binary classification data in 4-D feature space."""
    rng = np.random.default_rng(7)
    X = np.vstack([
        rng.normal([0, 0, 0, 0], 1, (80, 4)),
        rng.normal([3, 3, 3, 3], 1, (80, 4)),
    ])
    y = np.array([0] * 80 + [1] * 80)
    return X, y


@pytest.fixture()
def high_dim_data():
    """High-dimensional data with 5 classes."""
    rng = np.random.default_rng(42)
    centers = rng.normal(0, 5, (5, 20))
    X = np.vstack([rng.normal(c, 1, (30, 20)) for c in centers])
    y = np.repeat(np.arange(5), 30)
    return X, y


# ---------------------------------------------------------------------------
# Shape and attribute tests
# ---------------------------------------------------------------------------

class TestLDAShapes:
    def test_transform_output_shape(self, three_class_data):
        X, y = three_class_data
        X_t = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
        assert X_t.shape == (180, 2)

    def test_max_components_binary(self, binary_data):
        X, y = binary_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        assert lda.n_components_ == 1
        assert lda.components_.shape == (1, 4)

    def test_max_components_three_class(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        assert lda.n_components_ == 2
        assert lda.components_.shape == (2, 2)

    def test_n_components_clipped_to_max(self, binary_data):
        X, y = binary_data
        lda = LinearDiscriminantAnalysis(n_components=10).fit(X, y)
        assert lda.n_components_ == 1

    def test_scalings_shape(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
        assert lda.scalings_.shape == (2, 2)

    def test_means_shape(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        assert lda.means_.shape == (3, 2)

    def test_classes_detected(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        np.testing.assert_array_equal(lda.classes_, [0, 1, 2])

    def test_priors_sum_to_one(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        assert abs(lda.priors_.sum() - 1.0) < 1e-10

    def test_explained_variance_ratio_sums_to_one(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
        assert abs(lda.explained_variance_ratio_.sum() - 1.0) < 1e-9

    def test_partial_components_ratio_lt_one(self, high_dim_data):
        X, y = high_dim_data
        lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
        assert lda.explained_variance_ratio_.sum() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Solver equivalence
# ---------------------------------------------------------------------------

class TestSolvers:
    def test_eigen_and_svd_same_accuracy(self, three_class_data):
        X, y = three_class_data
        acc_e = (LinearDiscriminantAnalysis(solver='eigen').fit(X, y).predict(X) == y).mean()
        acc_s = (LinearDiscriminantAnalysis(solver='svd').fit(X, y).predict(X) == y).mean()
        assert abs(acc_e - acc_s) < 0.05

    def test_invalid_solver_raises(self, three_class_data):
        with pytest.raises(ValueError, match="solver"):
            LinearDiscriminantAnalysis(solver='bad')

    def test_svd_solver_transform_shape(self, three_class_data):
        X, y = three_class_data
        X_t = LinearDiscriminantAnalysis(n_components=2, solver='svd').fit_transform(X, y)
        assert X_t.shape == (180, 2)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestLDAClassification:
    def test_high_accuracy_well_separated(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
        assert (lda.predict(X) == y).mean() > 0.95

    def test_predict_proba_shape(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        proba = lda.predict_proba(X)
        assert proba.shape == (180, 3)

    def test_predict_proba_sums_to_one(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        proba = lda.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_in_range(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        proba = lda.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_consistent_with_proba(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        pred = lda.predict(X)
        proba = lda.predict_proba(X)
        pred_from_proba = lda.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(pred, pred_from_proba)

    def test_binary_classification(self, binary_data):
        X, y = binary_data
        lda = LinearDiscriminantAnalysis().fit(X, y)
        assert (lda.predict(X) == y).mean() > 0.95

    def test_custom_priors_affect_predictions(self, binary_data):
        X, y = binary_data
        lda_uniform = LinearDiscriminantAnalysis(priors=[0.5, 0.5]).fit(X, y)
        lda_biased = LinearDiscriminantAnalysis(priors=[0.99, 0.01]).fit(X, y)
        pred_uniform = lda_uniform.predict(X)
        pred_biased = lda_biased.predict(X)
        # Biased prior towards class 0 should predict more class 0
        assert (pred_biased == 0).sum() >= (pred_uniform == 0).sum()

    def test_wrong_priors_length_raises(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError, match="priors"):
            LinearDiscriminantAnalysis(priors=[0.3, 0.3, 0.4]).fit(X, y)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestLDAErrors:
    def test_not_fitted_transform_raises(self, three_class_data):
        X, _ = three_class_data
        with pytest.raises(RuntimeError, match="fit"):
            LinearDiscriminantAnalysis().transform(X)

    def test_not_fitted_predict_raises(self, three_class_data):
        X, _ = three_class_data
        with pytest.raises(RuntimeError, match="fit"):
            LinearDiscriminantAnalysis().predict(X)

    def test_fit_transform_matches_transform(self, three_class_data):
        X, y = three_class_data
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_ft = lda.fit_transform(X, y)
        X_t = lda.transform(X)
        np.testing.assert_allclose(X_ft, X_t, atol=1e-12)
