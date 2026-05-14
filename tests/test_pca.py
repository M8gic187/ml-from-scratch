"""Tests for PCA dimensionality reduction."""

import numpy as np
import pytest

from ml_scratch.decomposition import PCA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def correlated_data():
    """2-D dataset with strong correlation (principal axis ≈ [1, 1] / sqrt(2))."""
    rng = np.random.default_rng(0)
    t = rng.standard_normal(200)
    X = np.column_stack([t + 0.1 * rng.standard_normal(200),
                          t + 0.1 * rng.standard_normal(200)])
    return X


@pytest.fixture()
def standard_data():
    """Uncorrelated 5-D Gaussian data."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((150, 5))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPCA:
    def test_components_shape(self, standard_data):
        X = standard_data
        pca = PCA(n_components=3).fit(X)
        assert pca.components_.shape == (3, 5)

    def test_transform_shape(self, standard_data):
        X = standard_data
        X_t = PCA(n_components=3).fit_transform(X)
        assert X_t.shape == (150, 3)

    def test_inverse_transform_shape(self, standard_data):
        X = standard_data
        pca = PCA(n_components=3).fit(X)
        X_back = pca.inverse_transform(pca.transform(X))
        assert X_back.shape == X.shape

    def test_explained_variance_ratio_sums_le_one(self, standard_data):
        pca = PCA(n_components=4).fit(standard_data)
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9

    def test_full_components_ratio_sums_to_one(self, standard_data):
        pca = PCA().fit(standard_data)
        assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-9

    def test_explained_variance_descending(self, standard_data):
        pca = PCA().fit(standard_data)
        ev = pca.explained_variance_
        assert all(ev[i] >= ev[i + 1] for i in range(len(ev) - 1))

    def test_first_component_captures_most_variance(self, correlated_data):
        """First PC should explain > 90 % of strongly correlated 2-D data."""
        pca = PCA(n_components=1).fit(correlated_data)
        assert pca.explained_variance_ratio_[0] > 0.90

    def test_zero_mean_after_transform(self, standard_data):
        """Projected data should be centred."""
        X_t = PCA(n_components=3).fit_transform(standard_data)
        assert np.abs(X_t.mean(axis=0)).max() < 1e-10

    def test_reconstruction_error_decreases_with_more_components(self, standard_data):
        X = standard_data
        errors = []
        for k in (1, 2, 3, 4, 5):
            pca = PCA(n_components=k).fit(X)
            X_back = pca.inverse_transform(pca.transform(X))
            errors.append(np.mean((X - X_back) ** 2))
        assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))

    def test_fit_transform_equals_transform(self, standard_data):
        X = standard_data
        pca = PCA(n_components=2)
        X_ft = pca.fit_transform(X)
        X_t = pca.transform(X)
        assert np.allclose(X_ft, X_t)

    def test_not_fitted_raises(self, standard_data):
        with pytest.raises(RuntimeError, match="fit"):
            PCA(n_components=2).transform(standard_data)

    def test_n_components_none_keeps_all(self, standard_data):
        pca = PCA().fit(standard_data)
        assert pca.n_components_ == 5
        assert pca.components_.shape == (5, 5)
