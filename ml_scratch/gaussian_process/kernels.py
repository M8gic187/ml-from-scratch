"""Kernel functions for Gaussian Process models."""

from __future__ import annotations

import numpy as np


def _sq_dists(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Return matrix of squared Euclidean distances, shape (n1, n2)."""
    # ||x - y||^2 = ||x||^2 - 2 x·y + ||y||^2
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)   # (n1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)   # (n2, 1)
    return np.maximum(X1_sq + X2_sq.T - 2.0 * X1 @ X2.T, 0.0)


def rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: float = 1.0,
    signal_var: float = 1.0,
) -> np.ndarray:
    """Radial Basis Function (squared-exponential) kernel.

    k(x, x') = signal_var * exp(-||x - x'||^2 / (2 * length_scale^2))

    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
    X2 : ndarray of shape (n2, d)
    length_scale : float
        Controls the smoothness / width of the kernel.
    signal_var : float
        Output variance (amplitude squared).

    Returns
    -------
    K : ndarray of shape (n1, n2)
    """
    sq_d = _sq_dists(X1, X2)
    return signal_var * np.exp(-0.5 * sq_d / (length_scale ** 2))


def matern52_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: float = 1.0,
    signal_var: float = 1.0,
) -> np.ndarray:
    """Matérn 5/2 kernel.

    k(x, x') = signal_var * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)

    where r = ||x - x'||.

    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
    X2 : ndarray of shape (n2, d)
    length_scale : float
    signal_var : float

    Returns
    -------
    K : ndarray of shape (n1, n2)
    """
    r = np.sqrt(np.maximum(_sq_dists(X1, X2), 0.0))
    z = np.sqrt(5.0) * r / length_scale
    return signal_var * (1.0 + z + z ** 2 / 3.0) * np.exp(-z)


def periodic_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: float = 1.0,
    period: float = 1.0,
    signal_var: float = 1.0,
) -> np.ndarray:
    """Periodic (exp-sin-squared) kernel.

    k(x, x') = signal_var * exp(-2 * sin^2(pi * ||x - x'|| / period) / length_scale^2)

    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
    X2 : ndarray of shape (n2, d)
    length_scale : float
    period : float
        Periodicity of the kernel.
    signal_var : float

    Returns
    -------
    K : ndarray of shape (n1, n2)
    """
    r = np.sqrt(np.maximum(_sq_dists(X1, X2), 0.0))
    sin2 = np.sin(np.pi * r / period) ** 2
    return signal_var * np.exp(-2.0 * sin2 / (length_scale ** 2))


def linear_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    signal_var: float = 1.0,
    bias: float = 0.0,
) -> np.ndarray:
    """Linear (dot-product) kernel.

    k(x, x') = signal_var * (x · x') + bias

    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
    X2 : ndarray of shape (n2, d)
    signal_var : float
    bias : float
        Constant offset (c in sklearn's DotProduct).

    Returns
    -------
    K : ndarray of shape (n1, n2)
    """
    return signal_var * (X1 @ X2.T) + bias
