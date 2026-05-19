"""Tests for KFold, StratifiedKFold, cross_val_score, and new metrics."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.utils.cross_validation import KFold, StratifiedKFold, cross_val_score
from ml_scratch.utils.metrics import confusion_matrix, classification_report, log_loss
from ml_scratch.linear_models import LogisticRegression


# ---------------------------------------------------------------------------
# KFold tests
# ---------------------------------------------------------------------------

def _dataset(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = rng.integers(0, 2, n)
    return X, y


def test_kfold_number_of_splits():
    X, _ = _dataset()
    kf = KFold(n_splits=5)
    splits = list(kf.split(X))
    assert len(splits) == 5


def test_kfold_indices_partition_dataset():
    X, _ = _dataset(n=97)
    kf = KFold(n_splits=5)
    all_val = np.concatenate([val for _, val in kf.split(X)])
    assert len(np.unique(all_val)) == 97


def test_kfold_train_val_disjoint():
    X, _ = _dataset()
    kf = KFold(n_splits=5)
    for train, val in kf.split(X):
        assert len(np.intersect1d(train, val)) == 0


def test_kfold_sizes_roughly_equal():
    X, _ = _dataset(n=103)
    kf = KFold(n_splits=5)
    sizes = [len(val) for _, val in kf.split(X)]
    assert max(sizes) - min(sizes) <= 1


def test_kfold_shuffle_changes_order():
    X, _ = _dataset(n=50)
    kf1 = KFold(n_splits=5, shuffle=False)
    kf2 = KFold(n_splits=5, shuffle=True, random_state=99)
    vals1 = [val[0] for _, val in kf1.split(X)]
    vals2 = [val[0] for _, val in kf2.split(X)]
    assert vals1 != vals2


def test_kfold_get_n_splits():
    assert KFold(n_splits=7).get_n_splits() == 7


def test_kfold_invalid_n_splits():
    try:
        KFold(n_splits=1)
        assert False
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# StratifiedKFold tests
# ---------------------------------------------------------------------------

def _imbalanced(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = np.array([0] * (n // 5) + [1] * (4 * n // 5))
    return X, y


def test_stratified_kfold_number_of_splits():
    X, y = _imbalanced()
    skf = StratifiedKFold(n_splits=5)
    splits = list(skf.split(X, y))
    assert len(splits) == 5


def test_stratified_kfold_indices_partition_dataset():
    X, y = _imbalanced(n=100)
    skf = StratifiedKFold(n_splits=5)
    all_val = np.concatenate([val for _, val in skf.split(X, y)])
    assert len(np.unique(all_val)) == 100


def test_stratified_kfold_train_val_disjoint():
    X, y = _imbalanced()
    skf = StratifiedKFold(n_splits=5)
    for train, val in skf.split(X, y):
        assert len(np.intersect1d(train, val)) == 0


def test_stratified_kfold_preserves_class_ratio():
    n = 100
    X, y = _imbalanced(n=n)
    skf = StratifiedKFold(n_splits=5)
    global_ratio = np.mean(y == 1)
    for _, val in skf.split(X, y):
        fold_ratio = np.mean(y[val] == 1)
        assert abs(fold_ratio - global_ratio) < 0.15


def test_stratified_kfold_multiclass():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((90, 3))
    y = np.repeat([0, 1, 2], 30)
    skf = StratifiedKFold(n_splits=3)
    for _, val in skf.split(X, y):
        classes_in_fold = np.unique(y[val])
        assert len(classes_in_fold) == 3


# ---------------------------------------------------------------------------
# cross_val_score tests
# ---------------------------------------------------------------------------

def _blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal([0, 0], np.eye(2), 100)
    X1 = rng.multivariate_normal([5, 5], np.eye(2), 100)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(100), np.ones(100)]).astype(int)
    return X, y


def test_cross_val_score_returns_array():
    X, y = _blobs()
    scores = cross_val_score(LogisticRegression(n_iterations=300), X, y, cv=5)
    assert scores.shape == (5,)


def test_cross_val_score_all_in_unit_interval():
    X, y = _blobs()
    scores = cross_val_score(LogisticRegression(n_iterations=300), X, y, cv=5)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_cross_val_score_high_accuracy_on_separable():
    X, y = _blobs()
    scores = cross_val_score(LogisticRegression(n_iterations=500), X, y, cv=5)
    assert scores.mean() >= 0.90


def test_cross_val_score_custom_scoring():
    X, y = _blobs()
    from ml_scratch.utils.metrics import precision_recall_f1
    scoring = lambda yt, yp: precision_recall_f1(yt, yp)[2]  # F1
    scores = cross_val_score(LogisticRegression(n_iterations=300), X, y, cv=3, scoring=scoring)
    assert scores.shape == (3,)
    assert scores.mean() >= 0.85


def test_cross_val_score_with_kfold_instance():
    X, y = _blobs()
    kf = KFold(n_splits=4, shuffle=True, random_state=1)
    scores = cross_val_score(LogisticRegression(n_iterations=300), X, y, cv=kf)
    assert scores.shape == (4,)


# ---------------------------------------------------------------------------
# confusion_matrix tests
# ---------------------------------------------------------------------------

def test_confusion_matrix_binary():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm[1, 1] == 2  # TP
    assert cm[0, 0] == 2  # TN
    assert cm[0, 1] == 1  # FP
    assert cm[1, 0] == 1  # FN


def test_confusion_matrix_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    cm = confusion_matrix(y, y)
    assert np.all(cm == np.diag(np.diag(cm)))
    assert cm.sum() == len(y)


def test_confusion_matrix_shape_multiclass():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 0, 1, 1])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.sum() == 6


def test_confusion_matrix_rows_sum_to_support():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 2])
    cm = confusion_matrix(y_true, y_pred)
    expected_row_sums = np.array([3, 2, 1])
    np.testing.assert_array_equal(cm.sum(axis=1), expected_row_sums)


# ---------------------------------------------------------------------------
# classification_report tests
# ---------------------------------------------------------------------------

def test_classification_report_returns_string():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    report = classification_report(y_true, y_pred)
    assert isinstance(report, str)


def test_classification_report_contains_class_names():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 0, 2])
    report = classification_report(y_true, y_pred)
    assert "0" in report and "1" in report and "2" in report


def test_classification_report_contains_averages():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    report = classification_report(y_true, y_pred)
    assert "macro avg" in report
    assert "weighted avg" in report


# ---------------------------------------------------------------------------
# log_loss tests
# ---------------------------------------------------------------------------

def test_log_loss_perfect():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert log_loss(y, p) < 1e-10


def test_log_loss_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    p = rng.uniform(0.1, 0.9, 50)
    loss = log_loss(y, p)
    assert 0 < loss < 5


def test_log_loss_worse_than_perfect():
    y = np.array([1, 0, 1, 0])
    p_perfect = np.array([0.99, 0.01, 0.99, 0.01])
    p_random = np.array([0.5, 0.5, 0.5, 0.5])
    assert log_loss(y, p_perfect) < log_loss(y, p_random)


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
