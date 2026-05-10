"""Tests for KNNClassifier and KNNRegressor."""

import numpy as np
import pytest

from ml_scratch.neighbors import KNNClassifier, KNNRegressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def linearly_separable():
    """Two well-separated classes along x-axis."""
    rng = np.random.default_rng(42)
    X_neg = rng.normal(loc=-3.0, scale=0.5, size=(50, 2))
    X_pos = rng.normal(loc=+3.0, scale=0.5, size=(50, 2))
    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture()
def sine_regression():
    """1-D sine targets with moderate noise."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2 * np.pi, size=(120, 1))
    y = np.sin(X.ravel()) + rng.normal(0, 0.1, size=120)
    return X, y


# ---------------------------------------------------------------------------
# KNNClassifier tests
# ---------------------------------------------------------------------------

class TestKNNClassifier:
    def test_perfect_separation(self, linearly_separable):
        X, y = linearly_separable
        clf = KNNClassifier(k=3)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.mean(preds == y) > 0.95

    def test_returns_correct_shape(self, linearly_separable):
        X, y = linearly_separable
        clf = KNNClassifier(k=5).fit(X, y)
        assert clf.predict(X[:10]).shape == (10,)

    def test_distance_weights(self, linearly_separable):
        X, y = linearly_separable
        clf = KNNClassifier(k=5, weights="distance").fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.95

    def test_manhattan_metric(self, linearly_separable):
        X, y = linearly_separable
        clf = KNNClassifier(k=3, metric="manhattan").fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.90

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            KNNClassifier().predict(np.array([[1.0, 2.0]]))

    def test_invalid_metric(self, linearly_separable):
        X, y = linearly_separable
        clf = KNNClassifier(metric="cosine").fit(X, y)
        with pytest.raises(ValueError, match="Unknown metric"):
            clf.predict(X[:1])


# ---------------------------------------------------------------------------
# KNNRegressor tests
# ---------------------------------------------------------------------------

class TestKNNRegressor:
    def test_close_to_mean_neighbors(self, sine_regression):
        X, y = sine_regression
        reg = KNNRegressor(k=5).fit(X, y)
        preds = reg.predict(X)
        # Should reconstruct the sine wave reasonably well on training data
        residuals = np.abs(preds - y)
        assert np.mean(residuals) < 0.3

    def test_output_shape(self, sine_regression):
        X, y = sine_regression
        reg = KNNRegressor(k=3).fit(X, y)
        assert reg.predict(X[:7]).shape == (7,)

    def test_distance_weighted(self, sine_regression):
        X, y = sine_regression
        reg = KNNRegressor(k=5, weights="distance").fit(X, y)
        assert np.mean(np.abs(reg.predict(X) - y)) < 0.3

    def test_exact_match_returns_target(self):
        """Exact training point must return its own target."""
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([10.0, 20.0, 30.0])
        reg = KNNRegressor(k=1, weights="distance").fit(X, y)
        np.testing.assert_allclose(reg.predict(X), y, atol=1e-10)

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            KNNRegressor().predict(np.array([[1.0]]))
