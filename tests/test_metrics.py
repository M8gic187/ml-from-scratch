"""Unit tests for evaluation metrics."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.utils.metrics import mse, rmse, mae, r2_score, accuracy, precision_recall_f1


def test_mse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


def test_mse_known():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    assert abs(mse(y_true, y_pred) - 0.375) < 1e-9


def test_rmse_is_sqrt_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.5])
    assert abs(rmse(y_true, y_pred) - np.sqrt(mse(y_true, y_pred))) < 1e-9


def test_r2_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == 1.0


def test_r2_baseline():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, np.mean(y_true))
    assert abs(r2_score(y_true, y_pred)) < 1e-9


def test_accuracy():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    assert accuracy(y_true, y_pred) == 0.8


def test_precision_recall_f1():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    assert abs(p - 1.0) < 1e-9
    assert abs(r - 0.75) < 1e-9


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
