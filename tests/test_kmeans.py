"""Tests for KMeans clustering."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import KMeans


def _make_blobs(k: int = 3, n: int = 150, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = np.array([[-5, -5], [0, 5], [5, -5]][:k], dtype=float)
    X = np.vstack([
        rng.multivariate_normal(c, np.eye(2) * 0.5, size=n // k)
        for c in centers
    ])
    return X, centers


def test_fit_returns_self():
    X, _ = _make_blobs()
    model = KMeans(k=3, random_state=0)
    assert model.fit(X) is model


def test_fit_predict_returns_labels():
    X, _ = _make_blobs()
    labels = KMeans(k=3, random_state=0).fit_predict(X)
    assert labels.shape == (X.shape[0],)
    assert set(np.unique(labels)) == {0, 1, 2}


def test_centroids_close_to_true_centers():
    X, centers = _make_blobs()
    model = KMeans(k=3, random_state=0)
    model.fit(X)
    # Each learned centroid should be within 1.0 of a true center
    for c in model.centroids_:
        min_dist = np.min(np.linalg.norm(centers - c, axis=1))
        assert min_dist < 1.0, f"Centroid {c} too far from true centers"


def test_inertia_is_positive():
    X, _ = _make_blobs()
    model = KMeans(k=3, random_state=0)
    model.fit(X)
    assert model.inertia_ > 0


def test_predict_consistent_with_fit_labels():
    X, _ = _make_blobs()
    model = KMeans(k=3, random_state=0)
    model.fit(X)
    assert np.array_equal(model.predict(X), model.labels_)


def test_predict_before_fit_raises():
    model = KMeans(k=3)
    try:
        model.predict(np.array([[1.0, 2.0]]))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


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
