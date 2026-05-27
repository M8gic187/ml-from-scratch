"""Tests for AgglomerativeClustering."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blobs(k: int = 3, n_per: int = 50, seed: int = 0, spread: float = 0.4):
    """Create k well-separated Gaussian clusters."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-6, -6], [6, 6], [-6, 6], [6, -6]], dtype=float)[:k]
    X = np.vstack([
        rng.multivariate_normal(c, np.eye(2) * spread, n_per) for c in centers
    ])
    true = np.repeat(np.arange(k), n_per)
    return X, true


def _label_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of samples where two labellings agree (up to permutation)."""
    n = len(a)
    from itertools import permutations
    ka = int(a.max()) + 1
    kb = int(b.max()) + 1
    if ka != kb:
        return 0.0
    best = 0
    for perm in permutations(range(ka)):
        mapping = np.array(perm)
        hits = np.sum(mapping[a] == b)
        if hits > best:
            best = hits
    return best / n


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_invalid_n_clusters_raises():
    try:
        AgglomerativeClustering(n_clusters=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_linkage_raises():
    try:
        AgglomerativeClustering(linkage="bogus")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_metric_raises():
    try:
        AgglomerativeClustering(metric="cosine")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_ward_requires_euclidean():
    try:
        AgglomerativeClustering(linkage="ward", metric="manhattan")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_n_samples_less_than_n_clusters_raises():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    try:
        AgglomerativeClustering(n_clusters=5).fit(X)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Return types & shapes
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    X, _ = _make_blobs(k=2)
    model = AgglomerativeClustering(n_clusters=2)
    assert model.fit(X) is model


def test_fit_predict_returns_ndarray():
    X, _ = _make_blobs(k=2)
    labels = AgglomerativeClustering(n_clusters=2).fit_predict(X)
    assert isinstance(labels, np.ndarray)


def test_labels_shape():
    X, _ = _make_blobs(k=3, n_per=40)
    labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    assert labels.shape == (X.shape[0],)


def test_n_unique_labels_equals_n_clusters():
    for k in (2, 3, 4):
        X, _ = _make_blobs(k=k)
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        assert len(np.unique(labels)) == k, f"Expected {k} unique labels for k={k}"


def test_labels_are_zero_indexed():
    X, _ = _make_blobs(k=3)
    labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    assert set(np.unique(labels)) == {0, 1, 2}


def test_n_clusters_attribute_set():
    model = AgglomerativeClustering(n_clusters=4)
    X, _ = _make_blobs(k=4)
    model.fit(X)
    assert model.n_clusters_ == 4


# ---------------------------------------------------------------------------
# Clustering quality
# ---------------------------------------------------------------------------

def test_two_blobs_ward_separates_perfectly():
    X, true = _make_blobs(k=2)
    labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)
    assert _label_agreement(labels, true) == 1.0


def test_three_blobs_all_linkages():
    X, true = _make_blobs(k=3)
    for linkage in ("single", "complete", "average", "ward"):
        labels = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit_predict(X)
        agreement = _label_agreement(labels, true)
        assert agreement == 1.0, f"linkage={linkage}: agreement={agreement:.2f}"


def test_single_cluster_all_zero():
    X, _ = _make_blobs(k=3)
    labels = AgglomerativeClustering(n_clusters=1).fit_predict(X)
    assert np.all(labels == 0)


def test_n_clusters_equals_n_samples_trivial():
    X = np.array([[1.0], [2.0], [3.0]])
    labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    assert len(np.unique(labels)) == 3


# ---------------------------------------------------------------------------
# Metric variants
# ---------------------------------------------------------------------------

def test_manhattan_metric_single_linkage():
    X, true = _make_blobs(k=2)
    labels = AgglomerativeClustering(
        n_clusters=2, linkage="single", metric="manhattan"
    ).fit_predict(X)
    assert _label_agreement(labels, true) == 1.0


def test_manhattan_metric_complete_linkage():
    X, true = _make_blobs(k=2)
    labels = AgglomerativeClustering(
        n_clusters=2, linkage="complete", metric="manhattan"
    ).fit_predict(X)
    assert _label_agreement(labels, true) == 1.0


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def test_fit_predict_equals_fit_labels():
    X, _ = _make_blobs(k=3)
    model = AgglomerativeClustering(n_clusters=3)
    model.fit(X)
    assert np.array_equal(model.fit_predict(X), model.labels_)


def test_deterministic_output():
    """Agglomerative clustering is deterministic (no random state)."""
    X, _ = _make_blobs(k=3)
    model = AgglomerativeClustering(n_clusters=3)
    labels_a = model.fit_predict(X)
    labels_b = model.fit_predict(X)
    assert np.array_equal(labels_a, labels_b)


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
