"""Tests for SpectralClustering."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import SpectralClustering


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _three_blobs(n: int = 60, seed: int = 7):
    """Three clearly separated Gaussian blobs."""
    rng = np.random.default_rng(seed)
    centres = [[-8.0, 0.0], [8.0, 0.0], [0.0, 8.0]]
    return np.vstack([
        rng.multivariate_normal(c, np.eye(2) * 0.3, n) for c in centres
    ])


def _two_blobs(n: int = 80, seed: int = 7):
    """Two clearly separated Gaussian blobs."""
    rng = np.random.default_rng(seed)
    A = rng.multivariate_normal([-6.0, 0.0], np.eye(2) * 0.2, n)
    B = rng.multivariate_normal([6.0, 0.0], np.eye(2) * 0.2, n)
    return np.vstack([A, B])


def _concentric_rings(n: int = 100, seed: int = 42):
    """Two concentric rings — non-convex, cannot be separated by K-Means alone."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n)
    inner = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, 0.05, (n, 2))
    outer = np.column_stack([3 * np.cos(t), 3 * np.sin(t)]) + rng.normal(0, 0.05, (n, 2))
    return np.vstack([inner, outer])


# ---------------------------------------------------------------------------
# fit / fit_predict API
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    X = _two_blobs()
    sc = SpectralClustering(n_clusters=2, random_state=0)
    assert sc.fit(X) is sc


def test_fit_predict_shape():
    X = _two_blobs()
    labels = SpectralClustering(n_clusters=2, random_state=0).fit_predict(X)
    assert labels.shape == (X.shape[0],)


def test_labels_dtype_int():
    X = _two_blobs()
    labels = SpectralClustering(n_clusters=2, random_state=0).fit_predict(X)
    assert np.issubdtype(labels.dtype, np.integer)


def test_labels_range():
    X = _three_blobs()
    labels = SpectralClustering(n_clusters=3, random_state=0).fit_predict(X)
    assert set(labels.tolist()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Attributes after fit
# ---------------------------------------------------------------------------

def test_embedding_shape():
    X = _three_blobs()
    sc = SpectralClustering(n_clusters=3, random_state=0).fit(X)
    assert sc.embedding_.shape == (X.shape[0], 3)


def test_affinity_matrix_shape():
    X = _two_blobs()
    sc = SpectralClustering(n_clusters=2, random_state=0).fit(X)
    n = X.shape[0]
    assert sc.affinity_matrix_.shape == (n, n)


def test_affinity_matrix_symmetric():
    X = _two_blobs()
    sc = SpectralClustering(n_clusters=2, affinity="rbf", random_state=0).fit(X)
    assert np.allclose(sc.affinity_matrix_, sc.affinity_matrix_.T)


def test_affinity_matrix_non_negative():
    X = _two_blobs()
    sc = SpectralClustering(n_clusters=2, affinity="rbf", random_state=0).fit(X)
    assert np.all(sc.affinity_matrix_ >= 0)


def test_rbf_diagonal_is_one():
    """RBF affinity of a point with itself is exp(0) = 1."""
    X = _two_blobs()
    sc = SpectralClustering(n_clusters=2, affinity="rbf", gamma=1.0, random_state=0).fit(X)
    assert np.allclose(np.diag(sc.affinity_matrix_), 1.0)


# ---------------------------------------------------------------------------
# Cluster quality — well-separated blobs
# ---------------------------------------------------------------------------

def test_two_blobs_two_clusters_rbf():
    X = _two_blobs()
    labels = SpectralClustering(n_clusters=2, affinity="rbf", gamma=1.0, random_state=0).fit_predict(X)
    assert len(set(labels.tolist())) == 2


def test_three_blobs_three_clusters_rbf():
    X = _three_blobs()
    labels = SpectralClustering(n_clusters=3, affinity="rbf", gamma=2.0, random_state=0).fit_predict(X)
    assert len(set(labels.tolist())) == 3


def test_two_blobs_two_clusters_knn():
    X = _two_blobs()
    labels = SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors", n_neighbors=8, random_state=0
    ).fit_predict(X)
    assert len(set(labels.tolist())) == 2


def test_three_blobs_three_clusters_knn():
    X = _three_blobs()
    labels = SpectralClustering(
        n_clusters=3, affinity="nearest_neighbors", n_neighbors=8, random_state=0
    ).fit_predict(X)
    assert len(set(labels.tolist())) == 3


# ---------------------------------------------------------------------------
# Non-convex cluster (concentric rings) — k-NN affinity
# ---------------------------------------------------------------------------

def test_rings_separated_by_knn():
    """Spectral clustering with k-NN affinity correctly splits concentric rings."""
    X = _concentric_rings()
    sc = SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors", n_neighbors=10, random_state=0
    )
    labels = sc.fit_predict(X)
    inner_label = labels[0]
    outer_label = labels[100]
    assert inner_label != outer_label
    # All inner-ring points must share one label
    assert len(set(labels[:100].tolist())) == 1
    # All outer-ring points must share the other label
    assert len(set(labels[100:].tolist())) == 1


# ---------------------------------------------------------------------------
# k-NN symmetry
# ---------------------------------------------------------------------------

def test_knn_affinity_matrix_binary():
    """k-NN affinity contains only 0 and 1 entries."""
    X = _two_blobs()
    sc = SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors", n_neighbors=5, random_state=0
    ).fit(X)
    unique_vals = set(np.unique(sc.affinity_matrix_).tolist())
    assert unique_vals <= {0.0, 1.0}


def test_knn_affinity_matrix_symmetric():
    X = _two_blobs()
    sc = SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors", n_neighbors=5, random_state=0
    ).fit(X)
    assert np.allclose(sc.affinity_matrix_, sc.affinity_matrix_.T)


# ---------------------------------------------------------------------------
# gamma effect on RBF affinity
# ---------------------------------------------------------------------------

def test_small_gamma_sparse_affinity():
    """Very small gamma → long-range weights near zero."""
    X = _two_blobs()
    sc_small = SpectralClustering(n_clusters=2, affinity="rbf", gamma=0.01, random_state=0).fit(X)
    off_diag = sc_small.affinity_matrix_[0, 80:]   # cross-cluster affinities
    assert off_diag.max() < 1e-3


def test_large_gamma_dense_affinity():
    """Very large gamma → nearly uniform affinity matrix (all ≈ 1)."""
    X = _two_blobs()
    sc_large = SpectralClustering(n_clusters=2, affinity="rbf", gamma=1e6, random_state=0).fit(X)
    assert sc_large.affinity_matrix_.min() > 0.99


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_n_clusters_one_raises():
    try:
        SpectralClustering(n_clusters=1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_affinity_raises():
    try:
        SpectralClustering(affinity="cosine")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_negative_gamma_raises():
    try:
        SpectralClustering(gamma=-1.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_zero_gamma_raises():
    try:
        SpectralClustering(gamma=0.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_zero_n_neighbors_raises():
    try:
        SpectralClustering(n_neighbors=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_n_samples_less_than_n_clusters_raises():
    X = np.eye(3)
    try:
        SpectralClustering(n_clusters=5, random_state=0).fit(X)
        assert False, "Expected ValueError"
    except ValueError:
        pass


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
