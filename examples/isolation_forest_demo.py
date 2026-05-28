"""Isolation Forest demo: unsupervised anomaly detection.

Demonstrates four real-world-style scenarios:
  1. Basic 2-D blob with injected point outliers
  2. Effect of contamination rate on the decision threshold
  3. High-dimensional data (20 features)
  4. score_samples() for ranking anomalies by severity
"""

from __future__ import annotations

import numpy as np

from ml_scratch.ensemble import IsolationForest

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp = ((y_pred == -1) & (y_true == -1)).sum()
    fp = ((y_pred == -1) & (y_true == 1)).sum()
    fn = ((y_pred == 1) & (y_true == -1)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ---------------------------------------------------------------------------
# Demo 1 — Basic 2-D blob with injected outliers
# ---------------------------------------------------------------------------

def demo_basic() -> None:
    _sep("Demo 1: Basic 2-D blob with 10 injected outliers")

    inliers = RNG.normal(loc=[0.0, 0.0], scale=1.0, size=(300, 2))
    outliers = RNG.uniform(low=6.0, high=10.0, size=(10, 2))
    X = np.vstack([inliers, outliers])
    y_true = np.array([1] * 300 + [-1] * 10)

    iso = IsolationForest(n_estimators=100, contamination=0.04, random_state=0)
    iso.fit(X)
    y_pred = iso.predict(X)

    precision, recall = _precision_recall(y_true, y_pred)
    n_flagged = (y_pred == -1).sum()

    print(f"  Forest size       : {iso.n_estimators} trees")
    print(f"  Contamination     : {iso.contamination}")
    print(f"  Decision offset   : {iso.offset_:.4f}")
    print(f"  Samples flagged   : {n_flagged}")
    print(f"  Outlier precision : {precision:.2%}")
    print(f"  Outlier recall    : {recall:.2%}")


# ---------------------------------------------------------------------------
# Demo 2 — Effect of contamination rate
# ---------------------------------------------------------------------------

def demo_contamination_sweep() -> None:
    _sep("Demo 2: Contamination rate sweep (0.02 → 0.15)")

    inliers = RNG.normal(loc=[0.0, 0.0], scale=1.0, size=(400, 2))
    outliers = RNG.normal(loc=[7.0, 7.0], scale=0.3, size=(20, 2))
    X = np.vstack([inliers, outliers])
    y_true = np.array([1] * 400 + [-1] * 20)

    print(f"  {'contamination':>15}  {'flagged':>8}  {'precision':>10}  {'recall':>8}")
    print(f"  {'-' * 47}")
    for c in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
        iso = IsolationForest(n_estimators=80, contamination=c, random_state=1).fit(X)
        y_pred = iso.predict(X)
        prec, rec = _precision_recall(y_true, y_pred)
        print(f"  {c:>15.2f}  {(y_pred==-1).sum():>8}  {prec:>10.2%}  {rec:>8.2%}")


# ---------------------------------------------------------------------------
# Demo 3 — High-dimensional data (20 features)
# ---------------------------------------------------------------------------

def demo_high_dimensional() -> None:
    _sep("Demo 3: High-dimensional data (20 features, 500 inliers, 15 outliers)")

    n_features = 20
    inliers = RNG.normal(loc=0.0, scale=1.0, size=(500, n_features))
    outliers = RNG.normal(loc=5.0, scale=0.5, size=(15, n_features))
    X = np.vstack([inliers, outliers])
    y_true = np.array([1] * 500 + [-1] * 15)

    iso = IsolationForest(n_estimators=150, max_samples=256, contamination=0.03, random_state=2)
    iso.fit(X)
    y_pred = iso.predict(X)
    precision, recall = _precision_recall(y_true, y_pred)

    print(f"  Features          : {n_features}")
    print(f"  Training samples  : {len(X)}")
    print(f"  Max samples / tree: {iso._max_samples_actual}")
    print(f"  Outlier precision : {precision:.2%}")
    print(f"  Outlier recall    : {recall:.2%}")


# ---------------------------------------------------------------------------
# Demo 4 — Ranking anomalies by score severity
# ---------------------------------------------------------------------------

def demo_score_ranking() -> None:
    _sep("Demo 4: Ranking top-10 anomalies by score_samples()")

    inliers = RNG.normal(loc=[0.0, 0.0], scale=1.0, size=(200, 2))
    mild_outliers = RNG.normal(loc=[3.5, 3.5], scale=0.4, size=(5, 2))
    severe_outliers = RNG.normal(loc=[9.0, 9.0], scale=0.2, size=(5, 2))
    X = np.vstack([inliers, mild_outliers, severe_outliers])
    labels = ["inlier"] * 200 + ["mild outlier"] * 5 + ["severe outlier"] * 5

    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=3).fit(X)
    scores = iso.score_samples(X)

    ranked = np.argsort(scores)[:10]
    print(f"  {'rank':>4}  {'score':>8}  label")
    print(f"  {'-' * 35}")
    for rank, idx in enumerate(ranked, 1):
        print(f"  {rank:>4}  {scores[idx]:>8.4f}  {labels[idx]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_basic()
    demo_contamination_sweep()
    demo_high_dimensional()
    demo_score_ranking()
    print("\nAll demos completed successfully.")
