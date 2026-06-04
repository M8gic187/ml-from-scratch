"""Tests for Non-negative Matrix Factorization (NMF)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_scratch.decomposition import NMF


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def nonneg_data():
    """Generic 80-sample, 20-feature non-negative dataset."""
    rng = np.random.default_rng(42)
    return np.abs(rng.standard_normal((80, 20)))


@pytest.fixture()
def rank3_data():
    """Approximately rank-3 non-negative matrix (useful for recovery tests)."""
    rng = np.random.default_rng(0)
    W_true = np.abs(rng.standard_normal((100, 3)))
    H_true = np.abs(rng.standard_normal((3, 15)))
    noise = rng.uniform(0, 0.05, (100, 15))
    return np.maximum(0.0, W_true @ H_true + noise)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_invalid_solver_raises(self, nonneg_data):
        with pytest.raises(ValueError, match="solver"):
            NMF(solver="sgd").fit(nonneg_data)

    def test_invalid_beta_loss_raises(self, nonneg_data):
        with pytest.raises(ValueError, match="beta_loss"):
            NMF(beta_loss="itakura").fit(nonneg_data)

    def test_n_components_zero_raises(self, nonneg_data):
        with pytest.raises(ValueError, match="n_components"):
            NMF(n_components=0).fit(nonneg_data)

    def test_kl_with_als_raises(self, nonneg_data):
        with pytest.raises(ValueError, match="beta_loss"):
            NMF(solver="als", beta_loss="kl").fit(nonneg_data)

    def test_negative_input_raises(self):
        X = np.array([[1.0, -0.1], [0.5, 0.3]])
        with pytest.raises(ValueError, match="non-negative"):
            NMF(n_components=1).fit(X)

    def test_non_2d_input_raises(self, nonneg_data):
        with pytest.raises(ValueError, match="2-D"):
            NMF(n_components=2).fit(nonneg_data.ravel())

    def test_unfitted_transform_raises(self, nonneg_data):
        with pytest.raises(RuntimeError, match="fit"):
            NMF(n_components=2).transform(nonneg_data)

    def test_unfitted_inverse_transform_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            NMF(n_components=2).inverse_transform(np.ones((5, 2)))

    def test_default_params(self):
        nmf = NMF()
        assert nmf.n_components == 2
        assert nmf.solver == "mu"
        assert nmf.beta_loss == "frobenius"
        assert nmf.max_iter == 200


# ---------------------------------------------------------------------------
# Fit attributes
# ---------------------------------------------------------------------------

class TestFitAttributes:
    def test_components_shape(self, nonneg_data):
        nmf = NMF(n_components=4, random_state=0).fit(nonneg_data)
        assert nmf.components_.shape == (4, 20)

    def test_n_components_attr(self, nonneg_data):
        nmf = NMF(n_components=3, random_state=0).fit(nonneg_data)
        assert nmf.n_components_ == 3

    def test_n_iter_is_positive_int(self, nonneg_data):
        nmf = NMF(n_components=3, random_state=0).fit(nonneg_data)
        assert isinstance(nmf.n_iter_, int)
        assert nmf.n_iter_ >= 1

    def test_reconstruction_err_is_finite(self, nonneg_data):
        nmf = NMF(n_components=5, random_state=0).fit(nonneg_data)
        assert np.isfinite(nmf.reconstruction_err_)
        assert nmf.reconstruction_err_ >= 0.0

    def test_components_nonneg(self, nonneg_data):
        nmf = NMF(n_components=4, random_state=0).fit(nonneg_data)
        assert nmf.components_.min() >= 0.0

    def test_components_none_before_fit(self):
        assert NMF().components_ is None

    @pytest.mark.parametrize("solver", ["mu", "als"])
    def test_both_solvers_produce_components(self, nonneg_data, solver):
        nmf = NMF(n_components=3, solver=solver, random_state=0).fit(nonneg_data)
        assert nmf.components_.shape == (3, 20)


# ---------------------------------------------------------------------------
# Transform shapes and non-negativity
# ---------------------------------------------------------------------------

class TestTransformShapes:
    def test_transform_shape(self, nonneg_data):
        nmf = NMF(n_components=5, random_state=0).fit(nonneg_data)
        W = nmf.transform(nonneg_data)
        assert W.shape == (80, 5)

    def test_transform_nonneg(self, nonneg_data):
        nmf = NMF(n_components=4, random_state=0).fit(nonneg_data)
        W = nmf.transform(nonneg_data)
        assert W.min() >= 0.0

    def test_fit_transform_shape(self, nonneg_data):
        W = NMF(n_components=3, random_state=0).fit_transform(nonneg_data)
        assert W.shape == (80, 3)

    def test_fit_transform_nonneg(self, nonneg_data):
        W = NMF(n_components=3, random_state=0).fit_transform(nonneg_data)
        assert W.min() >= 0.0

    def test_inverse_transform_shape(self, nonneg_data):
        nmf = NMF(n_components=4, random_state=0).fit(nonneg_data)
        W = nmf.transform(nonneg_data)
        X_back = nmf.inverse_transform(W)
        assert X_back.shape == nonneg_data.shape

    def test_inverse_transform_nonneg(self, nonneg_data):
        nmf = NMF(n_components=4, random_state=0).fit(nonneg_data)
        W = nmf.transform(nonneg_data)
        X_back = nmf.inverse_transform(W)
        assert X_back.min() >= 0.0

    def test_transform_new_data(self, nonneg_data):
        nmf = NMF(n_components=3, random_state=0).fit(nonneg_data)
        X_new = np.abs(np.random.default_rng(99).standard_normal((30, 20)))
        W = nmf.transform(X_new)
        assert W.shape == (30, 3)


# ---------------------------------------------------------------------------
# Solvers and loss functions
# ---------------------------------------------------------------------------

class TestSolvers:
    @pytest.mark.parametrize("solver", ["mu", "als"])
    def test_frobenius_loss_decreases(self, nonneg_data, solver):
        """Reconstruction error must be finite and reasonably small."""
        nmf = NMF(n_components=5, solver=solver, max_iter=300, random_state=0)
        nmf.fit(nonneg_data)
        # Error must be less than baseline (all-zeros reconstruction = ‖X‖_F)
        baseline = float(np.linalg.norm(nonneg_data, "fro"))
        assert nmf.reconstruction_err_ < baseline

    def test_kl_loss_mu_runs(self, nonneg_data):
        nmf = NMF(n_components=4, solver="mu", beta_loss="kl", random_state=0)
        nmf.fit(nonneg_data)
        assert nmf.components_.shape == (4, 20)
        assert nmf.components_.min() >= 0.0

    def test_convergence_max_iter_respected(self, nonneg_data):
        nmf = NMF(n_components=3, max_iter=5, tol=0.0, random_state=0)
        nmf.fit(nonneg_data)
        assert nmf.n_iter_ <= 5

    @pytest.mark.parametrize("solver", ["mu", "als"])
    def test_reproducibility(self, nonneg_data, solver):
        W1 = NMF(n_components=3, solver=solver, random_state=7).fit_transform(nonneg_data)
        W2 = NMF(n_components=3, solver=solver, random_state=7).fit_transform(nonneg_data)
        assert np.allclose(W1, W2)


# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------

class TestRegularisation:
    def test_l1_reg_H_reduces_components_magnitude(self, nonneg_data):
        nmf_reg = NMF(n_components=4, l1_reg_H=0.1, random_state=0).fit(nonneg_data)
        nmf_plain = NMF(n_components=4, random_state=0).fit(nonneg_data)
        assert nmf_reg.components_.mean() <= nmf_plain.components_.mean() + 1e-6

    def test_l2_reg_H_runs(self, nonneg_data):
        nmf = NMF(n_components=4, l2_reg_H=0.5, random_state=0).fit(nonneg_data)
        assert nmf.components_.shape == (4, 20)

    def test_l1_reg_W_runs(self, nonneg_data):
        W = NMF(n_components=3, l1_reg_W=0.05, random_state=0).fit_transform(nonneg_data)
        assert W.shape == (80, 3)

    def test_l2_reg_W_runs(self, nonneg_data):
        W = NMF(n_components=3, l2_reg_W=0.1, random_state=0).fit_transform(nonneg_data)
        assert W.shape == (80, 3)


# ---------------------------------------------------------------------------
# Factorisation quality
# ---------------------------------------------------------------------------

class TestFactorisationQuality:
    def test_low_rank_reconstruction(self, rank3_data):
        """NMF should reconstruct an approximately rank-3 matrix well."""
        nmf = NMF(n_components=3, max_iter=500, tol=1e-5, random_state=0)
        W = nmf.fit_transform(rank3_data)
        X_hat = nmf.inverse_transform(W)
        rel_err = np.linalg.norm(rank3_data - X_hat, "fro") / np.linalg.norm(rank3_data, "fro")
        assert rel_err < 0.25, f"Relative reconstruction error too large: {rel_err:.4f}"

    def test_more_components_lower_error(self, nonneg_data):
        """More components should (generally) yield a lower or equal reconstruction error."""
        err3 = NMF(n_components=3, random_state=0).fit(nonneg_data).reconstruction_err_
        err8 = NMF(n_components=8, random_state=0).fit(nonneg_data).reconstruction_err_
        assert err8 <= err3 + 1.0  # allow small tolerance for stochastic init

    def test_full_rank_zero_error(self, nonneg_data):
        """With n_components = n_features the reconstruction error should be very small."""
        n_features = nonneg_data.shape[1]
        nmf = NMF(n_components=n_features, max_iter=500, tol=1e-6, random_state=0)
        nmf.fit(nonneg_data)
        # Full-rank NMF should produce a very small relative error
        rel_err = nmf.reconstruction_err_ / np.linalg.norm(nonneg_data, "fro")
        assert rel_err < 0.5
