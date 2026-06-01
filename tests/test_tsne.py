"""Unit tests for the TSNE implementation."""

from __future__ import annotations

import numpy as np
import pytest

from ml_scratch.decomposition import TSNE
from ml_scratch.decomposition.tsne import (
    _binary_search_sigma,
    _gradient,
    _kl_divergence,
    _low_dim_affinities,
    _squared_euclidean,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_blobs(n=60, d=6, k=3, seed=0):
    """Return a simple k-cluster dataset."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-5, 5, (k, d))
    X = np.vstack([rng.standard_normal((n // k, d)) + c for c in centres])
    return X


# ---------------------------------------------------------------------------
# _squared_euclidean
# ---------------------------------------------------------------------------

class TestSquaredEuclidean:
    def test_shape(self):
        X = np.eye(5)
        D = _squared_euclidean(X)
        assert D.shape == (5, 5)

    def test_diagonal_zero(self):
        X = np.random.default_rng(0).standard_normal((10, 4))
        D = _squared_euclidean(X)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_symmetry(self):
        X = np.random.default_rng(1).standard_normal((8, 3))
        D = _squared_euclidean(X)
        np.testing.assert_allclose(D, D.T)

    def test_non_negative(self):
        X = np.random.default_rng(2).standard_normal((15, 5))
        assert (_squared_euclidean(X) >= 0).all()

    def test_known_values(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = _squared_euclidean(X)
        assert D[0, 1] == pytest.approx(25.0)
        assert D[1, 0] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# _binary_search_sigma
# ---------------------------------------------------------------------------

class TestBinarySearchSigma:
    def test_returns_positive(self):
        rng = np.random.default_rng(0)
        dists = rng.uniform(0, 4, 20)
        dists[5] = 0.0  # self-distance
        sigma = _binary_search_sigma(dists, np.log(10), 5)
        assert sigma > 0

    def test_entropy_matches_perplexity(self):
        """After calibration, P should have entropy close to log(perplexity)."""
        rng = np.random.default_rng(42)
        dists = rng.uniform(0, 2, 30)
        idx = 0
        dists[idx] = 0.0
        target = np.log(15)
        sigma = _binary_search_sigma(dists, target, idx)

        exp_v = np.exp(-dists / (2 * sigma))
        exp_v[idx] = 0.0
        p = exp_v / exp_v.sum()
        mask = p > 0
        H = -float(np.sum(p[mask] * np.log(p[mask])))
        assert abs(H - target) < 0.05

    def test_minimum_clamp(self):
        dists = np.zeros(10)
        sigma = _binary_search_sigma(dists, np.log(5), 0)
        assert sigma >= 1e-10


# ---------------------------------------------------------------------------
# _low_dim_affinities
# ---------------------------------------------------------------------------

class TestLowDimAffinities:
    def test_shapes(self):
        Y = np.random.default_rng(0).standard_normal((20, 2))
        num, Q = _low_dim_affinities(Y)
        assert num.shape == (20, 20)
        assert Q.shape == (20, 20)

    def test_diagonal_zero(self):
        Y = np.random.default_rng(1).standard_normal((15, 2))
        num, _ = _low_dim_affinities(Y)
        np.testing.assert_allclose(np.diag(num), 0.0, atol=1e-12)

    def test_Q_sums_to_one(self):
        Y = np.random.default_rng(2).standard_normal((12, 2))
        _, Q = _low_dim_affinities(Y)
        assert Q.sum() == pytest.approx(1.0, abs=1e-6)

    def test_Q_non_negative(self):
        Y = np.random.default_rng(3).standard_normal((10, 2))
        _, Q = _low_dim_affinities(Y)
        assert (Q >= 0).all()


# ---------------------------------------------------------------------------
# _kl_divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_zero_when_equal(self):
        n = 5
        P = np.full((n, n), 1.0 / n**2)
        assert _kl_divergence(P, P) == pytest.approx(0.0, abs=1e-10)

    def test_positive(self):
        rng = np.random.default_rng(0)
        P = rng.dirichlet(np.ones(25)).reshape(5, 5)
        Q = rng.dirichlet(np.ones(25)).reshape(5, 5)
        assert _kl_divergence(P, Q) >= 0


# ---------------------------------------------------------------------------
# TSNE class
# ---------------------------------------------------------------------------

class TestTSNEShapeAndType:
    def test_output_shape_2d(self):
        X = make_blobs(n=60, d=8)
        Y = TSNE(n_components=2, perplexity=10, n_iterations=100, random_state=0).fit_transform(X)
        assert Y.shape == (60, 2)

    def test_output_shape_3d(self):
        X = make_blobs(n=60, d=8)
        Y = TSNE(n_components=3, perplexity=10, n_iterations=100, random_state=0).fit_transform(X)
        assert Y.shape == (60, 3)

    def test_output_dtype_float(self):
        X = make_blobs(n=40, d=5).astype(np.float32)
        Y = TSNE(n_components=2, perplexity=8, n_iterations=50, random_state=1).fit_transform(X)
        assert Y.dtype == np.float64

    def test_no_nan_or_inf(self):
        X = make_blobs(n=45, d=6)
        Y = TSNE(n_components=2, perplexity=10, n_iterations=150, random_state=2).fit_transform(X)
        assert np.isfinite(Y).all()


class TestTSNEAttributes:
    def test_embedding_stored(self):
        tsne = TSNE(n_components=2, perplexity=10, n_iterations=80, random_state=0)
        X = make_blobs(n=40, d=5)
        Y = tsne.fit_transform(X)
        assert tsne.embedding_ is not None
        np.testing.assert_array_equal(tsne.embedding_, Y)

    def test_kl_divergence_non_negative(self):
        tsne = TSNE(n_components=2, perplexity=10, n_iterations=100, random_state=0)
        tsne.fit_transform(make_blobs(n=50, d=6))
        # KL divergence is non-negative; allow tiny floating-point slack
        assert tsne.kl_divergence_ >= -1e-6

    def test_n_iter_set(self):
        tsne = TSNE(n_components=2, perplexity=8, n_iterations=120, random_state=0)
        tsne.fit_transform(make_blobs(n=40, d=4))
        assert tsne.n_iter_ == 120


class TestTSNEReproducibility:
    def test_same_seed_same_result(self):
        X = make_blobs(n=50, d=6)
        Y1 = TSNE(n_components=2, perplexity=10, n_iterations=100, random_state=7).fit_transform(X)
        Y2 = TSNE(n_components=2, perplexity=10, n_iterations=100, random_state=7).fit_transform(X)
        np.testing.assert_array_equal(Y1, Y2)

    def test_different_seeds_differ(self):
        X = make_blobs(n=50, d=6)
        Y1 = TSNE(n_components=2, perplexity=10, n_iterations=100, random_state=0).fit_transform(X)
        Y2 = TSNE(n_components=2, perplexity=10, n_iterations=100, random_state=99).fit_transform(X)
        assert not np.allclose(Y1, Y2)


class TestTSNEValidation:
    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            TSNE(perplexity=5).fit_transform(np.array([[1.0, 2.0]]))

    def test_perplexity_too_large_raises(self):
        X = np.random.default_rng(0).standard_normal((10, 3))
        with pytest.raises(ValueError, match="perplexity"):
            TSNE(perplexity=15).fit_transform(X)


class TestTSNEAutoLR:
    def test_auto_learning_rate(self):
        X = make_blobs(n=60, d=5)
        Y = TSNE(
            n_components=2, perplexity=10, learning_rate="auto",
            n_iterations=100, random_state=0,
        ).fit_transform(X)
        assert Y.shape == (60, 2)
        assert np.isfinite(Y).all()
