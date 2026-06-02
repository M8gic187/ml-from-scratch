"""Demonstration of VotingClassifier: hard and soft voting ensembles.

Run from the repository root:
    python examples/voting_classifier_demo.py
"""

import numpy as np

from ml_scratch.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from ml_scratch.tree import DecisionTreeClassifier
from ml_scratch.neighbors import KNNClassifier

rng = np.random.default_rng(42)


def make_binary(n=200):
    X_a = rng.normal([2.5, 2.5], 0.8, (n // 2, 2))
    X_b = rng.normal([-2.5, -2.5], 0.8, (n // 2, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def make_multiclass(n_per_class=60):
    centres = [[0, 4], [-4, -2], [4, -2]]
    X = np.vstack([rng.normal(c, 0.7, (n_per_class, 2)) for c in centres])
    y = np.repeat([0, 1, 2], n_per_class)
    return X, y


def acc(clf, X, y):
    return np.mean(clf.predict(X) == y)


# ---------------------------------------------------------------------------
# Example 1: hard voting, binary
# ---------------------------------------------------------------------------
print("=" * 60)
print("Example 1 — Hard Voting (binary)")
print("=" * 60)

X, y = make_binary()

dt = DecisionTreeClassifier(max_depth=3, random_state=0)
knn = KNNClassifier(k=7)
rf = RandomForestClassifier(n_estimators=30, random_state=0)

hard_clf = VotingClassifier(
    [("dt", dt), ("knn", knn), ("rf", rf)],
    voting="hard",
)
hard_clf.fit(X, y)
print(f"  DecisionTree alone : {acc(DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y), X, y):.3f}")
print(f"  KNN alone          : {acc(KNNClassifier(k=7).fit(X, y), X, y):.3f}")
print(f"  RandomForest alone : {acc(RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y), X, y):.3f}")
print(f"  Hard Voting        : {acc(hard_clf, X, y):.3f}")

# ---------------------------------------------------------------------------
# Example 2: soft voting, binary — average posterior probabilities
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Example 2 — Soft Voting (binary)")
print("=" * 60)

soft_clf = VotingClassifier(
    [
        ("dt", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=30, random_state=0)),
    ],
    voting="soft",
)
soft_clf.fit(X, y)
proba = soft_clf.predict_proba(X)
print(f"  Soft Voting accuracy     : {acc(soft_clf, X, y):.3f}")
print(f"  Mean P(class=1) for y=1  : {proba[y == 1, 1].mean():.3f}")
print(f"  Mean P(class=1) for y=0  : {proba[y == 0, 1].mean():.3f}")

# ---------------------------------------------------------------------------
# Example 3: weighted hard voting — trust RandomForest more
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Example 3 — Weighted Hard Voting")
print("=" * 60)

weighted_clf = VotingClassifier(
    [
        ("dt", DecisionTreeClassifier(max_depth=2, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=50, random_state=0)),
    ],
    voting="hard",
    weights=[1, 3],
)
weighted_clf.fit(X, y)
print(f"  Weighted (dt=1, rf=3)    : {acc(weighted_clf, X, y):.3f}")

# ---------------------------------------------------------------------------
# Example 4: soft voting, multiclass
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Example 4 — Soft Voting (multiclass, 3 classes)")
print("=" * 60)

X3, y3 = make_multiclass()

mc_clf = VotingClassifier(
    [
        ("dt", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=40, random_state=0)),
    ],
    voting="soft",
)
mc_clf.fit(X3, y3)
proba3 = mc_clf.predict_proba(X3)
print(f"  Soft Voting accuracy     : {acc(mc_clf, X3, y3):.3f}")
print(f"  proba shape              : {proba3.shape}")
print(f"  Row sums (first 5)       : {proba3[:5].sum(axis=1)}")

# ---------------------------------------------------------------------------
# Example 5: weighted soft voting — boost GradientBoosting contribution
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Example 5 — Weighted Soft Voting with GradientBoosting")
print("=" * 60)

wsoft_clf = VotingClassifier(
    [
        ("dt", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("gb", GradientBoostingClassifier(n_estimators=40, random_state=0)),
    ],
    voting="soft",
    weights=[1, 2],
)
wsoft_clf.fit(X, y)
print(f"  Weighted soft (dt=1, gb=2) : {acc(wsoft_clf, X, y):.3f}")

# ---------------------------------------------------------------------------
# Example 6: single-estimator voting (sanity check)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Example 6 — Single Estimator (sanity check)")
print("=" * 60)

solo_dt = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
solo_voting = VotingClassifier(
    [("dt", DecisionTreeClassifier(max_depth=4, random_state=0))],
    voting="soft",
).fit(X, y)

preds_match = np.array_equal(solo_dt.predict(X), solo_voting.predict(X))
print(f"  Predictions match solo DT : {preds_match}")
print(f"  Max proba diff            : {np.abs(solo_dt.predict_proba(X) - solo_voting.predict_proba(X)).max():.2e}")
