"""Evaluation metrics for regression and classification tasks."""

from __future__ import annotations
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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the confusion matrix for a multi-class classification result.

    Rows correspond to true classes, columns to predicted classes.
    Classes are inferred from the union of ``y_true`` and ``y_pred`` and
    sorted in ascending order.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)

    Returns
    -------
    cm : ndarray of shape (n_classes, n_classes)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    label_to_idx = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    digits: int = 3,
) -> str:
    """Build a text report with per-class and macro-averaged metrics.

    Reports precision, recall, F1-score, and support for every class,
    plus macro-averaged and weighted-average summary rows.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)
    digits : int
        Number of decimal places in the output.

    Returns
    -------
    str
        Formatted classification report.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    rows: list[tuple[str, float, float, float, int]] = []
    for cls in classes:
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append((str(cls), prec, rec, f1, support))

    total = sum(r[4] for r in rows)
    macro_p = float(np.mean([r[1] for r in rows]))
    macro_r = float(np.mean([r[2] for r in rows]))
    macro_f = float(np.mean([r[3] for r in rows]))
    w_p = float(sum(r[1] * r[4] for r in rows) / total) if total > 0 else 0.0
    w_r = float(sum(r[2] * r[4] for r in rows) / total) if total > 0 else 0.0
    w_f = float(sum(r[3] * r[4] for r in rows) / total) if total > 0 else 0.0

    header = f"{'':>12}  {'precision':>{digits+5}}  {'recall':>{digits+5}}  {'f1-score':>{digits+5}}  {'support':>7}"
    sep = "-" * len(header)
    lines = [header, sep]
    fmt = f"{{:>12}}  {{:{digits+5}.{digits}f}}  {{:{digits+5}.{digits}f}}  {{:{digits+5}.{digits}f}}  {{:>7}}"
    for name, p, r, f, sup in rows:
        lines.append(fmt.format(name, p, r, f, sup))
    lines.append(sep)
    lines.append(fmt.format("macro avg", macro_p, macro_r, macro_f, total))
    lines.append(fmt.format("weighted avg", w_p, w_r, w_f, total))
    return "\n".join(lines)


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy loss averaged over samples.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,), values in {0, 1}
    y_prob : ndarray of shape (n_samples,)
        Predicted probabilities for the positive class.
    eps : float
        Clipping value to avoid log(0).

    Returns
    -------
    float
        Mean log-loss (lower is better).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


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
