"""Tests for FastICA (Independent Component Analysis)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_scratch.decomposition import FastICA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mixed_signals():
    """Three independent Laplace sources mixed by a random matrix."""
    rng = np.random.default_rng(0)
    n = 600
    S = np.column_stack([
        rng.laplace(size=n),
        rng.laplace(size=n),
        rng.laplace(size=n),
    ])
    A = np.array([[1.0, 0.5, 0.2],
                  [0.3, 1.0, 0.4],
                  [0.1, 0.6, 1.0]])
    return S @ A.T


@pytest.fixture()
def simple_data():
    """Generic 100-sample, 4-feature Gaussian dataset."""
    return np.random.default_rng(7).standard_normal((100, 4))


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_invalid_algorithm_raises(self, simple_data):
        with pytest.raises(ValueError, match="algorithm"):
            FastICA(algorithm="unknown").fit(simple_data)

    def test_invalid_fun_raises(self, simple_data):
        with pytest.raises(ValueError, match="fun"):
            FastICA(fun="sigmoid").fit(simple_data)

    def test_n_components_too_large_raises(self, simple_data):
        with pytest.raises(ValueError, match="n_components"):
            FastICA(n_components=10).fit(simple_data)

    def test_unfitted_transform_raises(self, simple_data):
        with pytest.raises(RuntimeError, match="fit"):
            FastICA().transform(simple_data)

    def test_unfitted_inverse_transform_raises(self, simple_data):
        with pytest.raises(RuntimeError, match="fit"):
            FastICA().inverse_transform(simple_data[:, :2])

    def test_default_params(self):
        ica = FastICA()
        assert ica.n_components is None
        assert ica.algorithm == "parallel"
        assert ica.fun == "logcosh"
        assert ica.whiten is True


# ---------------------------------------------------------------------------
# Fit attributes
# ---------------------------------------------------------------------------

class TestFitAttributes:
    def test_components_shape(self, simple_data):
        ica = FastICA(n_components=2, random_state=0).fit(simple_data)
        assert ica.components_.shape == (2, 4)

    def test_mixing_shape(self, simple_data):
        ica = FastICA(n_components=2, random_state=0).fit(simple_data)
        assert ica.mixing_.shape == (4, 2)

    def test_mean_shape_when_whitened(self, simple_data):
        ica = FastICA(n_components=2, random_state=0).fit(simple_data)
        assert ica.mean_.shape == (4,)

    def test_whitening_shape(self, simple_data):
        ica = FastICA(n_components=2, random_state=0).fit(simple_data)
        assert ica.whitening_.shape == (2, 4)

    def test_mean_is_none_without_whitening(self, simple_data):
        ica = FastICA(whiten=False, random_state=0).fit(simple_data)
        assert ica.mean_ is None

    def test_n_iter_list(self, simple_data):
        ica = FastICA(n_components=2, random_state=0).fit(simple_data)
        assert isinstance(ica.n_iter_, list)
        assert len(ica.n_iter_) == 1  # parallel: single entry

    def test_n_iter_list_deflation(self, simple_data):
        ica = FastICA(n_components=3, algorithm="deflation", random_state=0).fit(simple_data)
        assert len(ica.n_iter_) == 3  # one entry per component

    def test_components_none_before_fit(self):
        assert FastICA().components_ is None


# ---------------------------------------------------------------------------
# Transform shapes
# ---------------------------------------------------------------------------

class TestTransformShapes:
    def test_transform_shape(self, simple_data):
        S = FastICA(n_components=2, random_state=0).fit_transform(simple_data)
        assert S.shape == (100, 2)

    def test_full_components_shape(self, simple_data):
        S = FastICA(random_state=0).fit_transform(simple_data)
        assert S.shape == (100, 4)

    def test_inverse_transform_shape(self, simple_data):
        ica = FastICA(n_components=3, random_state=0).fit(simple_data)
        X_back = ica.inverse_transform(ica.transform(simple_data))
        assert X_back.shape == simple_data.shape

    def test_fit_transform_equals_transform(self, simple_data):
        ica = FastICA(n_components=2, random_state=0)
        S1 = ica.fit_transform(simple_data)
        S2 = ica.transform(simple_data)
        assert np.allclose(S1, S2)

    def test_transform_separate_data(self, simple_data):
        ica = FastICA(n_components=2, random_state=0).fit(simple_data)
        X_new = np.random.default_rng(99).standard_normal((50, 4))
        S = ica.transform(X_new)
        assert S.shape == (50, 2)


# ---------------------------------------------------------------------------
# Algorithm variants
# ---------------------------------------------------------------------------

class TestAlgorithms:
    @pytest.mark.parametrize("algorithm", ["parallel", "deflation"])
    def test_both_algorithms_run(self, simple_data, algorithm):
        S = FastICA(n_components=2, algorithm=algorithm, random_state=0).fit_transform(simple_data)
        assert S.shape == (100, 2)

    @pytest.mark.parametrize("fun", ["logcosh", "exp", "cube"])
    def test_all_contrast_functions(self, simple_data, fun):
        S = FastICA(n_components=2, fun=fun, random_state=0).fit_transform(simple_data)
        assert S.shape == (100, 2)

    def test_deflation_components_orthogonal_in_whitened_space(self, simple_data):
        ica = FastICA(n_components=3, algorithm="deflation", random_state=0).fit(simple_data)
        # Recover weight vectors in whitened space: W = components_ @ pinv(whitening_)
        # whitening_ is (n_components, n_features), pinv is (n_features, n_components)
        W = ica.components_ @ np.linalg.pinv(ica.whitening_)  # (n_components, n_components)
        gram = W @ W.T
        off_diag = gram - np.diag(np.diag(gram))
        assert np.abs(off_diag).max() < 1e-6


# ---------------------------------------------------------------------------
# Source separation quality
# ---------------------------------------------------------------------------

class TestSourceSeparation:
    def _max_abs_corr(self, S_true, S_hat):
        """Max absolute correlation between each recovered and any true source."""
        correlations = []
        for j in range(S_hat.shape[1]):
            col = []
            for i in range(S_true.shape[1]):
                a = S_true[:, i] - S_true[:, i].mean()
                b = S_hat[:, j] - S_hat[:, j].mean()
                r = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
                col.append(r)
            correlations.append(max(col))
        return correlations

    def test_recovers_independent_sources(self, mixed_signals):
        """Each recovered component should correlate strongly with a true source."""
        rng = np.random.default_rng(0)
        n = mixed_signals.shape[0]
        S_true = np.column_stack([rng.laplace(size=n), rng.laplace(size=n), rng.laplace(size=n)])
        A = np.array([[1.0, 0.5, 0.2], [0.3, 1.0, 0.4], [0.1, 0.6, 1.0]])
        S_true = mixed_signals @ np.linalg.pinv(A)

        ica = FastICA(n_components=3, random_state=0, max_iter=500).fit(mixed_signals)
        S_hat = ica.transform(mixed_signals)
        best_corrs = self._max_abs_corr(S_true, S_hat)
        assert all(c > 0.75 for c in best_corrs), f"Low correlations: {best_corrs}"

    def test_reproducibility(self, mixed_signals):
        ica1 = FastICA(n_components=2, random_state=42).fit_transform(mixed_signals)
        ica2 = FastICA(n_components=2, random_state=42).fit_transform(mixed_signals)
        assert np.allclose(ica1, ica2)

    def test_different_seeds_differ(self, mixed_signals):
        ica1 = FastICA(n_components=2, random_state=0).fit_transform(mixed_signals)
        ica2 = FastICA(n_components=2, random_state=99).fit_transform(mixed_signals)
        # Not identical — sign or ordering may differ
        assert not np.allclose(ica1, ica2)
