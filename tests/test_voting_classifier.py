"""Tests for VotingClassifier."""

import numpy as np
import pytest

from ml_scratch.ensemble import VotingClassifier, RandomForestClassifier
from ml_scratch.tree import DecisionTreeClassifier
from ml_scratch.neighbors import KNNClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def binary_dataset():
    rng = np.random.default_rng(0)
    X_a = rng.normal(loc=[3.0, 3.0], scale=0.7, size=(60, 2))
    X_b = rng.normal(loc=[-3.0, -3.0], scale=0.7, size=(60, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


@pytest.fixture()
def multiclass_dataset():
    rng = np.random.default_rng(7)
    centres = np.array([[0, 5], [-5, -2], [5, -2]], dtype=float)
    X = np.vstack([rng.normal(c, 0.6, (50, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 50)
    return X, y


@pytest.fixture()
def hard_estimators():
    return [
        ("dt", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("knn", KNNClassifier(k=5)),
    ]


@pytest.fixture()
def soft_estimators():
    return [
        ("dt", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=20, random_state=0)),
    ]


# ---------------------------------------------------------------------------
# Construction and parameter validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_voting(self):
        clf = VotingClassifier([("dt", DecisionTreeClassifier())])
        assert clf.voting == "hard"

    def test_default_weights_is_none(self):
        clf = VotingClassifier([("dt", DecisionTreeClassifier())])
        assert clf.weights is None

    def test_custom_voting_stored(self):
        clf = VotingClassifier([("dt", DecisionTreeClassifier())], voting="soft")
        assert clf.voting == "soft"

    def test_custom_weights_stored(self):
        clf = VotingClassifier([("dt", DecisionTreeClassifier())], weights=[2.0])
        assert clf.weights == [2.0]

    def test_not_fitted_raises(self, binary_dataset, hard_estimators):
        X, _ = binary_dataset
        clf = VotingClassifier(hard_estimators)
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(X)

    def test_empty_estimators_raises(self, binary_dataset):
        X, y = binary_dataset
        with pytest.raises(ValueError, match="empty"):
            VotingClassifier([]).fit(X, y)

    def test_invalid_voting_raises(self, binary_dataset):
        X, y = binary_dataset
        with pytest.raises(ValueError, match="voting"):
            VotingClassifier([("dt", DecisionTreeClassifier())], voting="mean").fit(X, y)

    def test_duplicate_names_raises(self, binary_dataset):
        X, y = binary_dataset
        with pytest.raises(ValueError, match="unique"):
            VotingClassifier(
                [("dt", DecisionTreeClassifier()), ("dt", DecisionTreeClassifier())]
            ).fit(X, y)

    def test_weights_wrong_length_raises(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        with pytest.raises(ValueError, match="weights"):
            VotingClassifier(hard_estimators, weights=[1.0]).fit(X, y)


# ---------------------------------------------------------------------------
# Post-fit attributes
# ---------------------------------------------------------------------------


class TestFitAttributes:
    def test_estimators_count(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(hard_estimators).fit(X, y)
        assert len(clf.estimators_) == 2

    def test_classes_binary(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(hard_estimators).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_classes_multiclass(self, multiclass_dataset, soft_estimators):
        X, y = multiclass_dataset
        clf = VotingClassifier(soft_estimators).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_fit_returns_self(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(hard_estimators)
        assert clf.fit(X, y) is clf


# ---------------------------------------------------------------------------
# Hard voting
# ---------------------------------------------------------------------------


class TestHardVoting:
    def test_predict_shape(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(hard_estimators, voting="hard").fit(X, y)
        assert clf.predict(X[:20]).shape == (20,)

    def test_predict_labels_in_classes(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(hard_estimators, voting="hard").fit(X, y)
        assert set(clf.predict(X)).issubset({0, 1})

    def test_high_accuracy_binary(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(hard_estimators, voting="hard").fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.90

    def test_high_accuracy_multiclass(self, multiclass_dataset):
        X, y = multiclass_dataset
        est = [
            ("dt", DecisionTreeClassifier(max_depth=4, random_state=0)),
            ("knn", KNNClassifier(k=3)),
        ]
        clf = VotingClassifier(est, voting="hard").fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.90

    def test_weighted_hard_predict_shape(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(hard_estimators, voting="hard", weights=[2.0, 1.0]).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_weighted_hard_labels_in_classes(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(hard_estimators, voting="hard", weights=[3.0, 1.0]).fit(X, y)
        assert set(clf.predict(X)).issubset({0, 1})

    def test_proba_raises_for_hard(self, binary_dataset, hard_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(hard_estimators, voting="hard").fit(X, y)
        with pytest.raises(AttributeError):
            clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Soft voting
# ---------------------------------------------------------------------------


class TestSoftVoting:
    def test_predict_shape(self, binary_dataset, soft_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(soft_estimators, voting="soft").fit(X, y)
        assert clf.predict(X[:25]).shape == (25,)

    def test_proba_shape_binary(self, binary_dataset, soft_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(soft_estimators, voting="soft").fit(X, y)
        assert clf.predict_proba(X).shape == (len(X), 2)

    def test_proba_sums_to_one_binary(self, binary_dataset, soft_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(soft_estimators, voting="soft").fit(X, y)
        np.testing.assert_allclose(
            clf.predict_proba(X).sum(axis=1), np.ones(len(X)), atol=1e-12
        )

    def test_proba_non_negative(self, binary_dataset, soft_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(soft_estimators, voting="soft").fit(X, y)
        assert (clf.predict_proba(X) >= 0).all()

    def test_predict_consistent_with_proba(self, binary_dataset, soft_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(soft_estimators, voting="soft").fit(X, y)
        preds_direct = clf.predict(X)
        preds_from_proba = clf.classes_[np.argmax(clf.predict_proba(X), axis=1)]
        np.testing.assert_array_equal(preds_direct, preds_from_proba)

    def test_high_accuracy_soft(self, binary_dataset, soft_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(soft_estimators, voting="soft").fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.90

    def test_weighted_proba_sums_to_one(self, binary_dataset, soft_estimators):
        X, y = binary_dataset
        clf = VotingClassifier(soft_estimators, voting="soft", weights=[3.0, 1.0]).fit(X, y)
        np.testing.assert_allclose(
            clf.predict_proba(X).sum(axis=1), np.ones(len(X)), atol=1e-12
        )

    def test_proba_shape_multiclass(self, multiclass_dataset, soft_estimators):
        X, y = multiclass_dataset
        clf = VotingClassifier(soft_estimators, voting="soft").fit(X, y)
        assert clf.predict_proba(X).shape == (len(X), 3)

    def test_proba_sums_to_one_multiclass(self, multiclass_dataset, soft_estimators):
        X, y = multiclass_dataset
        clf = VotingClassifier(soft_estimators, voting="soft").fit(X, y)
        np.testing.assert_allclose(
            clf.predict_proba(X).sum(axis=1), np.ones(len(X)), atol=1e-12
        )

    def test_single_estimator_proba_passthrough(self, binary_dataset):
        X, y = binary_dataset
        dt = DecisionTreeClassifier(max_depth=4, random_state=0)
        dt.fit(X, y)
        solo_proba = dt.predict_proba(X)

        clf = VotingClassifier(
            [("dt", DecisionTreeClassifier(max_depth=4, random_state=0))], voting="soft"
        ).fit(X, y)
        ensemble_proba = clf.predict_proba(X)
        np.testing.assert_allclose(ensemble_proba, solo_proba, atol=1e-12)
