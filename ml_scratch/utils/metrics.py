"""Evaluation metrics for regression and classification tasks."""

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²).

    Returns 1.0 for a perfect predictor and can be negative for arbitrarily
    bad models. A baseline model predicting the mean always scores 0.0.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1-score for binary classification.

    Returns:
        (precision, recall, f1) as a tuple of floats.
    """
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean Silhouette Coefficient for a clustering result.

    Measures how similar each point is to its own cluster compared to
    other clusters.  Score in [-1, 1]; higher is better.  Noise points
    (label == -1) are excluded from the computation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
        Cluster labels; -1 is treated as noise and ignored.

    Returns
    -------
    float
        Mean silhouette coefficient, or 0.0 when fewer than 2 valid
        clusters exist.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    mask = labels != -1
    X_valid, l_valid = X[mask], labels[mask]
    unique = np.unique(l_valid)

    if len(unique) < 2:
        return 0.0

    n = X_valid.shape[0]
    sq = np.sum(X_valid ** 2, axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X_valid @ X_valid.T), 0.0))

    scores = np.empty(n)
    for i in range(n):
        ci = l_valid[i]
        same = l_valid == ci
        other_clusters = unique[unique != ci]

        # a(i): mean intra-cluster distance
        intra = dist[i, same]
        a = intra[intra != 0].mean() if same.sum() > 1 else 0.0

        # b(i): mean distance to the nearest other cluster
        b = min(
            dist[i, l_valid == c].mean() for c in other_clusters
        )

        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return float(scores.mean())
