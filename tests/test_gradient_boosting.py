"""Unit tests for GradientBoostingClassifier and GradientBoostingRegressor."""

import numpy as np
import pytest

from ml_scratch.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def binary_dataset():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture()
def regression_dataset():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0, 0.2, 200)
    return X, y


# ---------------------------------------------------------------------------
# GradientBoostingClassifier
# ---------------------------------------------------------------------------

class TestGradientBoostingClassifier:

    def test_fit_returns_self(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(n_estimators=10)
        assert clf.fit(X, y) is clf

    def test_predict_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_labels_from_classes(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        assert set(np.unique(clf.predict(X))).issubset(set(clf.classes_))

    def test_predict_proba_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_training_accuracy(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1).fit(X, y)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.90, f"Expected >0.90 accuracy, got {acc:.3f}"

    def test_train_loss_decreases(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1).fit(X, y)
        assert clf.train_loss_[0] > clf.train_loss_[-1], "Loss should decrease over iterations"

    def test_n_estimators_attribute(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(n_estimators=15).fit(X, y)
        assert len(clf.estimators_) == 15

    def test_subsample(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, subsample=0.8, random_state=42
        ).fit(X, y)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.85

    def test_rejects_multiclass(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 2))
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        with pytest.raises(ValueError, match="binary"):
            GradientBoostingClassifier().fit(X, y)

    def test_predict_before_fit_raises(self):
        clf = GradientBoostingClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.zeros((5, 3)))

    def test_string_labels(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 2))
        y = np.where(X[:, 0] > 0, "cat", "dog")
        clf = GradientBoostingClassifier(n_estimators=20).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({"cat", "dog"})

    def test_decision_function_shape(self, binary_dataset):
        X, y = binary_dataset
        clf = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(X),)


# ---------------------------------------------------------------------------
# GradientBoostingRegressor
# ---------------------------------------------------------------------------

class TestGradientBoostingRegressor:

    def test_fit_returns_self(self, regression_dataset):
        X, y = regression_dataset
        reg = GradientBoostingRegressor(n_estimators=10)
        assert reg.fit(X, y) is reg

    def test_predict_shape(self, regression_dataset):
        X, y = regression_dataset
        reg = GradientBoostingRegressor(n_estimators=10).fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (len(X),)

    def test_mse_loss_low(self, regression_dataset):
        X, y = regression_dataset
        reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1).fit(X, y)
        mse = np.mean((reg.predict(X) - y) ** 2)
        assert mse < 0.15, f"Expected MSE < 0.15, got {mse:.4f}"

    def test_mae_loss(self, regression_dataset):
        X, y = regression_dataset
        reg = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, loss="mae"
        ).fit(X, y)
        mae = np.mean(np.abs(reg.predict(X) - y))
        assert mae < 0.5

    def test_train_loss_decreases(self, regression_dataset):
        X, y = regression_dataset
        reg = GradientBoostingRegressor(n_estimators=50).fit(X, y)
        assert reg.train_loss_[0] > reg.train_loss_[-1]

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="loss"):
            GradientBoostingRegressor(loss="huber")

    def test_predict_before_fit_raises(self):
        reg = GradientBoostingRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict(np.zeros((5, 3)))

    def test_subsample_stochastic(self, regression_dataset):
        X, y = regression_dataset
        reg = GradientBoostingRegressor(
            n_estimators=80, subsample=0.7, random_state=0
        ).fit(X, y)
        mse = np.mean((reg.predict(X) - y) ** 2)
        assert mse < 0.5

    def test_n_estimators_attribute(self, regression_dataset):
        X, y = regression_dataset
        reg = GradientBoostingRegressor(n_estimators=20).fit(X, y)
        assert len(reg.estimators_) == 20

    def test_init_value_is_mean_for_mse(self, regression_dataset):
        X, y = regression_dataset
        reg = GradientBoostingRegressor(n_estimators=5, loss="mse").fit(X, y)
        np.testing.assert_allclose(reg.init_value_, y.mean(), rtol=1e-6)

    def test_init_value_is_median_for_mae(self, regression_dataset):
        X, y = regression_dataset
        reg = GradientBoostingRegressor(n_estimators=5, loss="mae").fit(X, y)
        np.testing.assert_allclose(reg.init_value_, np.median(y), rtol=1e-6)
