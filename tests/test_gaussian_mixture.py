"""Tests for GaussianMixture clustering."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import GaussianMixture


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_blobs(k: int = 3, n_per: int = 100, seed: int = 0):
    """Return well-separated Gaussian blobs with known cluster structure."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]])[:k]
    X = np.vstack([
        rng.multivariate_normal(c, np.eye(2) * 0.4, size=n_per)
        for c in centers
    ])
    return X, centers


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

def test_invalid_n_components_raises():
    try:
        GaussianMixture(n_components=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_covariance_type_raises():
    try:
        GaussianMixture(covariance_type="bad")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_predict_before_fit_raises():
    gmm = GaussianMixture(n_components=2)
    try:
        gmm.predict(np.ones((5, 2)))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_score_before_fit_raises():
    gmm = GaussianMixture(n_components=2)
    try:
        gmm.score(np.ones((5, 2)))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_too_few_samples_raises():
    gmm = GaussianMixture(n_components=10)
    try:
        gmm.fit(np.ones((5, 2)))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# fit() return value and attributes
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    assert gmm.fit(X) is gmm


def test_attributes_set_after_fit():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    assert gmm.weights_ is not None
    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert isinstance(gmm.converged_, bool)
    assert isinstance(gmm.n_iter_, int)
    assert isinstance(gmm.lower_bound_, float)


def test_weights_sum_to_one():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    assert abs(gmm.weights_.sum() - 1.0) < 1e-9


def test_n_iter_positive():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    assert gmm.n_iter_ >= 1


# ---------------------------------------------------------------------------
# Covariance shapes
# ---------------------------------------------------------------------------

def test_covariance_shape_full():
    X, _ = _make_blobs(k=2)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(X)
    assert gmm.covariances_.shape == (2, 2, 2)


def test_covariance_shape_diag():
    X, _ = _make_blobs(k=2)
    gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=0)
    gmm.fit(X)
    assert gmm.covariances_.shape == (2, 2)


def test_covariance_shape_spherical():
    X, _ = _make_blobs(k=2)
    gmm = GaussianMixture(n_components=2, covariance_type="spherical", random_state=0)
    gmm.fit(X)
    assert gmm.covariances_.shape == (2,)


def test_covariances_positive_full():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    gmm.fit(X)
    for cov in gmm.covariances_:
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0), "Full covariance not positive definite"


def test_covariances_positive_diag():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0)
    gmm.fit(X)
    assert np.all(gmm.covariances_ > 0)


def test_covariances_positive_spherical():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, covariance_type="spherical", random_state=0)
    gmm.fit(X)
    assert np.all(gmm.covariances_ > 0)


# ---------------------------------------------------------------------------
# predict / predict_proba
# ---------------------------------------------------------------------------

def test_predict_returns_labels_in_range():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    labels = gmm.fit_predict(X)
    assert labels.shape == (X.shape[0],)
    assert np.all(labels >= 0) and np.all(labels < 3)


def test_predict_proba_shape():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    proba = gmm.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)


def test_predict_proba_rows_sum_to_one():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    row_sums = gmm.predict_proba(X).sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_predict_proba_values_in_01():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    proba = gmm.predict_proba(X)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_predict_consistent_with_predict_proba():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    labels = gmm.predict(X)
    proba_labels = np.argmax(gmm.predict_proba(X), axis=1)
    assert np.array_equal(labels, proba_labels)


def test_fit_predict_equivalent_to_fit_then_predict():
    X, _ = _make_blobs()
    gmm1 = GaussianMixture(n_components=3, random_state=7)
    labels1 = gmm1.fit_predict(X)

    gmm2 = GaussianMixture(n_components=3, random_state=7)
    gmm2.fit(X)
    labels2 = gmm2.predict(X)

    assert np.array_equal(labels1, labels2)


# ---------------------------------------------------------------------------
# Clustering quality
# ---------------------------------------------------------------------------

def test_recovers_three_clusters_full():
    X, _ = _make_blobs(k=3, n_per=150, seed=1)
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    labels = gmm.fit_predict(X)
    # Each true cluster should map predominantly to one GMM component
    counts = np.zeros((3, 3), dtype=int)
    for true_k in range(3):
        for gmm_k in range(3):
            counts[true_k, gmm_k] = np.sum(labels[true_k * 150: (true_k + 1) * 150] == gmm_k)
    # Each row should have at least one dominant column (>= 90 % of samples)
    assert np.any(counts >= 130, axis=1).all()


def test_means_close_to_true_centers():
    X, centers = _make_blobs(k=3, n_per=200, seed=2)
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    for learned_mu in gmm.means_:
        min_dist = np.min(np.linalg.norm(centers - learned_mu, axis=1))
        assert min_dist < 1.5, f"Mean {learned_mu} too far from true centers"


# ---------------------------------------------------------------------------
# score / AIC / BIC
# ---------------------------------------------------------------------------

def test_score_returns_float():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    assert isinstance(gmm.score(X), float)


def test_score_negative_log_likelihood():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    # Mean log-likelihood should be finite and negative for typical data
    s = gmm.score(X)
    assert np.isfinite(s)


def test_aic_bic_positive_and_finite():
    X, _ = _make_blobs()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(X)
    assert np.isfinite(gmm.aic(X)) and gmm.aic(X) > 0
    assert np.isfinite(gmm.bic(X)) and gmm.bic(X) > 0


def test_bic_penalises_underfitting():
    """A single component should have higher BIC than the true k on clear blobs."""
    X, _ = _make_blobs(k=3, n_per=200, seed=5)
    bic_1 = GaussianMixture(n_components=1, random_state=0).fit(X).bic(X)
    bic_3 = GaussianMixture(n_components=3, random_state=0).fit(X).bic(X)
    # k=1 cannot capture the three-blob structure, so its BIC must be worse
    assert bic_1 > bic_3, "BIC should penalise the severely underfitting k=1 model"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_component_recovers_mean():
    rng = np.random.default_rng(99)
    X = rng.multivariate_normal([3.0, -1.0], np.eye(2) * 0.1, size=200)
    gmm = GaussianMixture(n_components=1, random_state=0)
    gmm.fit(X)
    assert np.linalg.norm(gmm.means_[0] - np.array([3.0, -1.0])) < 0.2


def test_high_dimensional_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 20))
    gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0)
    labels = gmm.fit_predict(X)
    assert labels.shape == (100,)


def test_all_covariance_types_run():
    X, _ = _make_blobs()
    for cov_type in ("full", "diag", "spherical"):
        gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=0)
        labels = gmm.fit_predict(X)
        assert labels.shape == (X.shape[0],)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
