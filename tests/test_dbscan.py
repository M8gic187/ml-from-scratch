"""Tests for DBSCAN clustering and silhouette_score metric."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import DBSCAN
from ml_scratch.utils.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _two_blobs(n: int = 80, seed: int = 7):
    """Two clearly separated Gaussian blobs."""
    rng = np.random.default_rng(seed)
    A = rng.multivariate_normal([-5.0, 0.0], np.eye(2) * 0.2, n)
    B = rng.multivariate_normal([5.0, 0.0], np.eye(2) * 0.2, n)
    return np.vstack([A, B])


def _three_blobs(n: int = 60, seed: int = 7):
    """Three clearly separated Gaussian blobs."""
    rng = np.random.default_rng(seed)
    A = rng.multivariate_normal([-8.0, 0.0], np.eye(2) * 0.2, n)
    B = rng.multivariate_normal([8.0, 0.0], np.eye(2) * 0.2, n)
    C = rng.multivariate_normal([0.0, 8.0], np.eye(2) * 0.2, n)
    return np.vstack([A, B, C])


def _with_outlier(seed: int = 3):
    """Tight cluster plus one obvious outlier point."""
    rng = np.random.default_rng(seed)
    cluster = rng.multivariate_normal([0.0, 0.0], np.eye(2) * 0.1, 30)
    outlier = np.array([[50.0, 50.0]])
    return np.vstack([cluster, outlier])


# ---------------------------------------------------------------------------
# fit / fit_predict API
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    X = _two_blobs()
    model = DBSCAN(eps=1.5, min_samples=5)
    assert model.fit(X) is model


def test_fit_predict_shape():
    X = _two_blobs()
    labels = DBSCAN(eps=1.5, min_samples=5).fit_predict(X)
    assert labels.shape == (X.shape[0],)


def test_labels_dtype_int():
    X = _two_blobs()
    labels = DBSCAN(eps=1.5, min_samples=5).fit_predict(X)
    assert np.issubdtype(labels.dtype, np.integer)


# ---------------------------------------------------------------------------
# Cluster count correctness
# ---------------------------------------------------------------------------

def test_two_blobs_two_clusters():
    X = _two_blobs()
    db = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert db.n_clusters_ == 2


def test_three_blobs_three_clusters():
    X = _three_blobs()
    db = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert db.n_clusters_ == 3


def test_all_noise_when_eps_tiny():
    X = _two_blobs()
    db = DBSCAN(eps=1e-6, min_samples=5).fit(X)
    assert db.n_clusters_ == 0
    assert np.all(db.labels_ == -1)


def test_single_cluster_when_eps_huge():
    X = _two_blobs()
    db = DBSCAN(eps=1000.0, min_samples=5).fit(X)
    assert db.n_clusters_ == 1


# ---------------------------------------------------------------------------
# Noise detection
# ---------------------------------------------------------------------------

def test_outlier_labelled_noise():
    X = _with_outlier()
    db = DBSCAN(eps=0.5, min_samples=5).fit(X)
    # Last point is [50, 50] — must be noise
    assert db.labels_[-1] == -1


def test_noise_label_is_minus_one():
    X = _with_outlier()
    db = DBSCAN(eps=0.5, min_samples=5).fit(X)
    assert DBSCAN.NOISE == -1
    assert -1 in db.labels_


def test_no_noise_on_dense_grid():
    """All points on a regular 5×5 grid at spacing 0.4 should form 1 cluster."""
    coords = np.array([[x, y] for x in np.arange(5) * 0.4 for y in np.arange(5) * 0.4])
    db = DBSCAN(eps=0.5, min_samples=3).fit(coords)
    assert db.n_clusters_ == 1
    assert (db.labels_ == -1).sum() == 0


# ---------------------------------------------------------------------------
# Core sample indices
# ---------------------------------------------------------------------------

def test_core_sample_indices_non_empty():
    X = _two_blobs()
    db = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert len(db.core_sample_indices_) > 0


def test_core_sample_indices_are_valid():
    X = _two_blobs()
    db = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert np.all(db.core_sample_indices_ >= 0)
    assert np.all(db.core_sample_indices_ < X.shape[0])


def test_all_noise_means_no_core_points():
    X = _two_blobs()
    db = DBSCAN(eps=1e-6, min_samples=5).fit(X)
    assert len(db.core_sample_indices_) == 0


# ---------------------------------------------------------------------------
# Manhattan distance metric
# ---------------------------------------------------------------------------

def test_manhattan_metric_finds_clusters():
    X = _two_blobs()
    db = DBSCAN(eps=1.5, min_samples=5, metric="manhattan").fit(X)
    assert db.n_clusters_ == 2


def test_unknown_metric_raises():
    try:
        DBSCAN(metric="cosine")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_negative_eps_raises():
    try:
        DBSCAN(eps=-0.1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_zero_eps_raises():
    try:
        DBSCAN(eps=0.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_zero_min_samples_raises():
    try:
        DBSCAN(min_samples=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# silhouette_score
# ---------------------------------------------------------------------------

def test_silhouette_perfect_separation():
    X = _two_blobs()
    db = DBSCAN(eps=1.0, min_samples=5).fit(X)
    score = silhouette_score(X, db.labels_)
    assert score > 0.8, f"Expected high silhouette, got {score:.4f}"


def test_silhouette_excludes_noise():
    X = _with_outlier()
    db = DBSCAN(eps=0.4, min_samples=5).fit(X)
    # Should not raise even if noise points are present
    score = silhouette_score(X, db.labels_)
    assert isinstance(score, float)


def test_silhouette_returns_zero_for_single_cluster():
    X = _two_blobs()
    labels = np.zeros(X.shape[0], dtype=int)
    score = silhouette_score(X, labels)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
