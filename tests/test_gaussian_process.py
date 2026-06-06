"""Tests for GaussianProcessRegressor and kernel functions."""

import numpy as np
import pytest

from ml_scratch.gaussian_process import GaussianProcessRegressor
from ml_scratch.gaussian_process.kernels import (
    linear_kernel,
    matern52_kernel,
    periodic_kernel,
    rbf_kernel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sin_data():
    """Noisy sine-wave training set (1-D inputs)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-3.0, 3.0, (40, 1))
    y = np.sin(X.ravel()) + rng.normal(0, 0.05, 40)
    return X, y


@pytest.fixture()
def linear_data():
    """Noisy linear training set (1-D inputs)."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-2.0, 2.0, (30, 1))
    y = 2.0 * X.ravel() - 1.0 + rng.normal(0, 0.1, 30)
    return X, y


@pytest.fixture()
def fitted_gpr(sin_data):
    """Pre-fitted RBF GPR on sin data."""
    X, y = sin_data
    gpr = GaussianProcessRegressor(kernel="rbf", length_scale=1.0, noise_var=0.01)
    return gpr.fit(X, y)


@pytest.fixture()
def X_test():
    return np.linspace(-3.0, 3.0, 15).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------

class TestKernels:
    def test_rbf_shape(self):
        X1 = np.random.default_rng(0).standard_normal((5, 2))
        X2 = np.random.default_rng(1).standard_normal((7, 2))
        K = rbf_kernel(X1, X2)
        assert K.shape == (5, 7)

    def test_rbf_symmetric(self):
        X = np.random.default_rng(0).standard_normal((8, 3))
        K = rbf_kernel(X, X)
        assert np.allclose(K, K.T)

    def test_rbf_diagonal_equals_signal_var(self):
        X = np.random.default_rng(0).standard_normal((6, 2))
        K = rbf_kernel(X, X, signal_var=2.5)
        assert np.allclose(np.diag(K), 2.5)

    def test_rbf_all_positive(self):
        X1 = np.random.default_rng(0).standard_normal((10, 2))
        X2 = np.random.default_rng(1).standard_normal((10, 2))
        assert np.all(rbf_kernel(X1, X2) > 0)

    def test_matern52_shape(self):
        X1 = np.random.default_rng(0).standard_normal((6, 3))
        X2 = np.random.default_rng(1).standard_normal((4, 3))
        assert matern52_kernel(X1, X2).shape == (6, 4)

    def test_matern52_symmetric(self):
        X = np.random.default_rng(0).standard_normal((7, 2))
        K = matern52_kernel(X, X)
        assert np.allclose(K, K.T)

    def test_matern52_diagonal_equals_signal_var(self):
        X = np.random.default_rng(0).standard_normal((5, 2))
        K = matern52_kernel(X, X, signal_var=3.0)
        assert np.allclose(np.diag(K), 3.0)

    def test_periodic_shape(self):
        X1 = np.random.default_rng(2).standard_normal((5, 1))
        X2 = np.random.default_rng(3).standard_normal((8, 1))
        assert periodic_kernel(X1, X2).shape == (5, 8)

    def test_periodic_symmetric(self):
        X = np.random.default_rng(0).standard_normal((6, 1))
        K = periodic_kernel(X, X)
        assert np.allclose(K, K.T)

    def test_linear_kernel_shape(self):
        X1 = np.ones((4, 3))
        X2 = np.ones((6, 3))
        assert linear_kernel(X1, X2).shape == (4, 6)

    def test_linear_kernel_symmetric(self):
        X = np.random.default_rng(0).standard_normal((5, 2))
        K = linear_kernel(X, X)
        assert np.allclose(K, K.T)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestGPRConstruction:
    def test_default_kernel(self):
        gpr = GaussianProcessRegressor()
        assert gpr.kernel == "rbf"

    def test_custom_length_scale(self):
        gpr = GaussianProcessRegressor(length_scale=2.5)
        assert gpr.length_scale == 2.5

    def test_custom_noise_var(self):
        gpr = GaussianProcessRegressor(noise_var=0.5)
        assert gpr.noise_var == 0.5

    def test_repr(self):
        gpr = GaussianProcessRegressor(kernel="matern52", length_scale=1.5, noise_var=0.05)
        r = repr(gpr)
        assert "GaussianProcessRegressor" in r
        assert "matern52" in r

    def test_unfitted_attributes_are_none(self):
        gpr = GaussianProcessRegressor()
        assert gpr.X_train_ is None
        assert gpr.alpha_ is None
        assert gpr.L_ is None

    def test_predict_before_fit_raises(self, X_test):
        gpr = GaussianProcessRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            gpr.predict(X_test)


# ---------------------------------------------------------------------------
# Fit tests
# ---------------------------------------------------------------------------

class TestGPRFit:
    def test_returns_self(self, sin_data):
        X, y = sin_data
        gpr = GaussianProcessRegressor()
        assert gpr.fit(X, y) is gpr

    def test_stores_X_train(self, sin_data):
        X, y = sin_data
        gpr = GaussianProcessRegressor().fit(X, y)
        assert gpr.X_train_.shape == X.shape

    def test_cholesky_factor_lower_triangular(self, sin_data):
        X, y = sin_data
        gpr = GaussianProcessRegressor().fit(X, y)
        L = gpr.L_
        assert np.allclose(np.triu(L, k=1), 0)

    def test_alpha_shape(self, sin_data):
        X, y = sin_data
        gpr = GaussianProcessRegressor().fit(X, y)
        assert gpr.alpha_.shape == (len(y),)

    def test_log_marginal_likelihood_finite(self, sin_data):
        X, y = sin_data
        gpr = GaussianProcessRegressor().fit(X, y)
        assert np.isfinite(gpr.log_marginal_likelihood_)

    def test_mismatched_X_y_raises(self):
        X = np.ones((10, 1))
        y = np.ones(8)
        with pytest.raises(ValueError, match="same number"):
            GaussianProcessRegressor().fit(X, y)

    def test_invalid_kernel_raises(self, sin_data):
        X, y = sin_data
        gpr = GaussianProcessRegressor(kernel="bogus_kernel")
        with pytest.raises(ValueError, match="Unknown kernel"):
            gpr.fit(X, y)


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------

class TestGPRPredict:
    def test_predict_shape(self, fitted_gpr, X_test):
        mean = fitted_gpr.predict(X_test)
        assert mean.shape == (len(X_test),)

    def test_predict_std_shape(self, fitted_gpr, X_test):
        mean, std = fitted_gpr.predict(X_test, return_std=True)
        assert mean.shape == std.shape == (len(X_test),)

    def test_predict_cov_shape(self, fitted_gpr, X_test):
        mean, cov = fitted_gpr.predict(X_test, return_cov=True)
        n = len(X_test)
        assert mean.shape == (n,)
        assert cov.shape == (n, n)

    def test_std_non_negative(self, fitted_gpr, X_test):
        _, std = fitted_gpr.predict(X_test, return_std=True)
        assert np.all(std >= 0)

    def test_cov_symmetric(self, fitted_gpr, X_test):
        _, cov = fitted_gpr.predict(X_test, return_cov=True)
        assert np.allclose(cov, cov.T, atol=1e-9)

    def test_posterior_interpolates_noiseless(self):
        """With tiny noise, predictions should be close to training values."""
        rng = np.random.default_rng(99)
        X = rng.uniform(-2, 2, (20, 1))
        y = np.sin(X.ravel())
        gpr = GaussianProcessRegressor(kernel="rbf", noise_var=1e-8).fit(X, y)
        pred = gpr.predict(X)
        assert np.allclose(pred, y, atol=1e-4)

    def test_return_std_and_cov_mutually_exclusive(self, fitted_gpr, X_test):
        with pytest.raises(ValueError, match="mutually exclusive"):
            fitted_gpr.predict(X_test, return_std=True, return_cov=True)

    def test_matern52_predict(self, sin_data, X_test):
        X, y = sin_data
        gpr = GaussianProcessRegressor(kernel="matern52", noise_var=0.01).fit(X, y)
        mean = gpr.predict(X_test)
        assert mean.shape == (len(X_test),)

    def test_periodic_predict(self, sin_data, X_test):
        X, y = sin_data
        gpr = GaussianProcessRegressor(
            kernel="periodic", period=np.pi, noise_var=0.01
        ).fit(X, y)
        mean = gpr.predict(X_test)
        assert mean.shape == (len(X_test),)

    def test_linear_kernel_predict(self, linear_data, X_test):
        X, y = linear_data
        gpr = GaussianProcessRegressor(kernel="linear", noise_var=0.05).fit(X, y)
        mean = gpr.predict(X_test)
        assert mean.shape == (len(X_test),)

    def test_callable_kernel(self, sin_data, X_test):
        """A user-supplied kernel callable should work transparently."""
        from ml_scratch.gaussian_process.kernels import rbf_kernel

        def my_kernel(X1, X2):
            return rbf_kernel(X1, X2, length_scale=1.0, signal_var=1.0)

        X, y = sin_data
        gpr = GaussianProcessRegressor(kernel=my_kernel, noise_var=0.01).fit(X, y)
        mean = gpr.predict(X_test)
        assert mean.shape == (len(X_test),)


# ---------------------------------------------------------------------------
# sample_y tests
# ---------------------------------------------------------------------------

class TestGPRSampleY:
    def test_sample_shape_single(self, fitted_gpr, X_test):
        samples = fitted_gpr.sample_y(X_test, n_samples=1, random_state=0)
        assert samples.shape == (len(X_test), 1)

    def test_sample_shape_multiple(self, fitted_gpr, X_test):
        samples = fitted_gpr.sample_y(X_test, n_samples=5, random_state=0)
        assert samples.shape == (len(X_test), 5)

    def test_samples_are_reproducible(self, fitted_gpr, X_test):
        s1 = fitted_gpr.sample_y(X_test, n_samples=3, random_state=42)
        s2 = fitted_gpr.sample_y(X_test, n_samples=3, random_state=42)
        assert np.allclose(s1, s2)


# ---------------------------------------------------------------------------
# Score and LML tests
# ---------------------------------------------------------------------------

class TestGPRScore:
    def test_r2_close_to_one_on_training_signal(self, sin_data):
        """Low-noise GPR should nearly interpolate → R² ≈ 1 on noiseless signal."""
        X, y = sin_data
        gpr = GaussianProcessRegressor(kernel="rbf", noise_var=1e-6).fit(X, y)
        r2 = gpr.score(X, np.sin(X.ravel()))
        assert r2 > 0.98

    def test_score_range(self, sin_data, X_test):
        X, y = sin_data
        gpr = GaussianProcessRegressor(kernel="rbf", noise_var=0.01).fit(X, y)
        r2 = gpr.score(X_test, np.sin(X_test.ravel()))
        assert r2 <= 1.0

    def test_lml_method_no_args(self, fitted_gpr):
        lml = fitted_gpr.log_marginal_likelihood()
        assert np.isfinite(lml)
        assert lml == fitted_gpr.log_marginal_likelihood_

    def test_lml_method_with_data(self, sin_data):
        X, y = sin_data
        gpr = GaussianProcessRegressor(kernel="rbf", noise_var=0.01).fit(X, y)
        lml = gpr.log_marginal_likelihood(X, y)
        assert np.isfinite(lml)

    def test_lml_method_requires_both_X_and_y(self, fitted_gpr, sin_data):
        X, _ = sin_data
        with pytest.raises(ValueError):
            fitted_gpr.log_marginal_likelihood(X=X)
