"""t-SNE: t-Distributed Stochastic Neighbor Embedding (van der Maaten & Hinton, 2008)."""

from __future__ import annotations

import numpy as np


class TSNE:
    """Nonlinear dimensionality reduction via t-Distributed Stochastic Neighbor Embedding.

    Preserves local neighbourhood structure by modelling pairwise similarities in
    high-dimensional space as a Gaussian probability distribution and minimising
    the KL divergence to a matching Student-t distribution in the low-dimensional
    embedding.  The heavy tail of the Student-t kernel alleviates the crowding
    problem that arises when compressing distances from high to low dimensions.

    Algorithm outline
    -----------------
    1. Compute pairwise squared Euclidean distances in input space.
    2. For each point *i*, find σ_i via binary search so that the conditional
       distribution P(·|i) achieves the target entropy log(perplexity).
    3. Symmetrise: P_{ij} = (P_{j|i} + P_{i|j}) / 2n.
    4. Initialise Y ~ N(0, 10⁻⁴).
    5. For *n_iterations* steps:
       a. Compute Q via Student-t kernel in low-D space.
       b. Compute ∇_Y KL(P ∥ Q).
       c. Update Y with momentum gradient descent and per-component adaptive gains.
    6. Apply early exaggeration (multiply P by a constant) during the first
       *exaggeration_iter* steps to encourage well-separated initial clusters.

    Parameters
    ----------
    n_components : int
        Dimension of the embedded space.  Typically 2 (visualisation) or 3.
    perplexity : float
        Effective number of neighbours considered per point.  Values in [5, 50]
        work well for most datasets; larger datasets tolerate higher perplexity.
    learning_rate : float | "auto"
        Gradient-descent step size.  ``"auto"`` sets ``max(200, n_samples / 12)``.
    n_iterations : int
        Total number of optimisation iterations.
    early_exaggeration : float
        Multiplier applied to P during the first *exaggeration_iter* iterations.
        Larger values push embeddings into tighter clusters early.
    exaggeration_iter : int
        Number of iterations to run with early exaggeration.
    momentum : float
        Momentum coefficient used in the first *momentum_switch_iter* steps.
    final_momentum : float
        Momentum coefficient used after *momentum_switch_iter*.
    momentum_switch_iter : int
        Iteration at which momentum switches from *momentum* to *final_momentum*.
    min_gain : float
        Minimum per-component adaptive gain to prevent stalling.
    random_state : int | None
        Seed for reproducible random initialisation.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Final low-dimensional coordinates.
    kl_divergence_ : float
        KL(P ∥ Q) evaluated at the end of optimisation (without exaggeration).
    n_iter_ : int
        Number of gradient-descent iterations completed.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_scratch.decomposition import TSNE
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((120, 10))
    >>> tsne = TSNE(n_components=2, perplexity=20, n_iterations=300, random_state=0)
    >>> Y = tsne.fit_transform(X)
    >>> Y.shape
    (120, 2)
    >>> tsne.kl_divergence_ > 0
    True

    References
    ----------
    van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE.
    *Journal of Machine Learning Research*, 9, 2579–2605.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float | str = 200.0,
        n_iterations: int = 1000,
        early_exaggeration: float = 12.0,
        exaggeration_iter: int = 250,
        momentum: float = 0.5,
        final_momentum: float = 0.8,
        momentum_switch_iter: int = 250,
        min_gain: float = 0.01,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.early_exaggeration = early_exaggeration
        self.exaggeration_iter = exaggeration_iter
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.momentum_switch_iter = momentum_switch_iter
        self.min_gain = min_gain
        self.random_state = random_state

        self.embedding_: np.ndarray | None = None
        self.kl_divergence_: float = 0.0
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit t-SNE and return the low-dimensional embedding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            High-dimensional input data.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
        """
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        if n_samples < 2:
            raise ValueError("t-SNE requires at least 2 samples.")
        if self.perplexity >= n_samples:
            raise ValueError(
                f"perplexity ({self.perplexity}) must be less than n_samples ({n_samples})."
            )

        lr = (
            max(200.0, n_samples / 12.0)
            if self.learning_rate == "auto"
            else float(self.learning_rate)
        )

        rng = np.random.default_rng(self.random_state)

        # High-dimensional joint probability matrix
        sq_dists = _squared_euclidean(X)
        P = self._joint_probabilities(sq_dists)

        # Random initialisation — small scale to stay near origin
        Y = rng.standard_normal((n_samples, self.n_components)) * 1e-4
        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)

        P_exag = P * self.early_exaggeration

        for it in range(1, self.n_iterations + 1):
            P_cur = P_exag if it <= self.exaggeration_iter else P
            mom = (
                self.momentum if it <= self.momentum_switch_iter else self.final_momentum
            )

            num, Q = _low_dim_affinities(Y)
            grad = _gradient(P_cur, Q, Y, num)

            # Adaptive per-component gains (Jacobs, 1988)
            sign_same = np.sign(grad) == np.sign(velocity)
            gains = np.where(sign_same, gains * 0.8, gains + 0.2)
            gains = np.clip(gains, self.min_gain, None)

            velocity = mom * velocity - lr * gains * grad
            Y = Y + velocity
            Y -= Y.mean(axis=0)  # keep embedding centred

        # Record final KL divergence without exaggeration
        _, Q_final = _low_dim_affinities(Y)
        self.kl_divergence_ = _kl_divergence(P, Q_final)
        self.embedding_ = Y
        self.n_iter_ = self.n_iterations
        return Y

    # ------------------------------------------------------------------
    # Internal: high-dimensional probability matrix
    # ------------------------------------------------------------------

    def _joint_probabilities(self, sq_dists: np.ndarray) -> np.ndarray:
        """Build symmetric joint-probability matrix P.

        For each row i, calibrate σ_i so that H(P(·|i)) = log(perplexity),
        then symmetrise: P_{ij} = (P_{j|i} + P_{i|j}) / (2n).
        """
        n = sq_dists.shape[0]
        target_H = np.log(self.perplexity)
        P_cond = np.zeros((n, n))

        for i in range(n):
            dists_i = sq_dists[i].copy()
            dists_i[i] = 0.0
            sigma_sq = _binary_search_sigma(dists_i, target_H, i)
            exp_v = np.exp(-dists_i / (2.0 * sigma_sq))
            exp_v[i] = 0.0
            s = exp_v.sum()
            P_cond[i] = exp_v / s if s > 0 else exp_v

        P = (P_cond + P_cond.T) / (2.0 * n)
        return np.maximum(P, 1e-12)


# ------------------------------------------------------------------
# Module-level helpers (no self state needed)
# ------------------------------------------------------------------

def _squared_euclidean(X: np.ndarray) -> np.ndarray:
    """Return (n, n) matrix of squared Euclidean distances."""
    sq = np.sum(X ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.maximum(D, 0.0)


def _binary_search_sigma(
    dists: np.ndarray,
    target_H: float,
    idx: int,
    n_steps: int = 50,
    tol: float = 1e-5,
) -> float:
    """Binary-search for σ² achieving Shannon entropy H = log(perplexity).

    Parameters
    ----------
    dists     : squared distances from point *idx* to all others.
    target_H  : log(perplexity).
    idx       : self-index (excluded from conditional probability).

    Returns
    -------
    sigma_sq : float — calibrated bandwidth variance.
    """
    lo, hi = 1e-10, 1e10
    sigma_sq = 1.0

    for _ in range(n_steps):
        exp_v = np.exp(-dists / (2.0 * sigma_sq))
        exp_v[idx] = 0.0
        s = exp_v.sum()
        if s == 0.0:
            break
        p = exp_v / s
        mask = p > 0
        H = -float(np.sum(p[mask] * np.log(p[mask])))

        diff = H - target_H
        if abs(diff) < tol:
            break
        # H is monotonically increasing in sigma_sq:
        #   diff > 0 (H too high) → sigma too large → shrink upper bound
        #   diff < 0 (H too low)  → sigma too small → raise lower bound
        if diff > 0:
            hi = sigma_sq
            sigma_sq = (sigma_sq + lo) / 2.0
        else:
            lo = sigma_sq
            sigma_sq = (sigma_sq + hi) / 2.0

    return max(sigma_sq, 1e-10)


def _low_dim_affinities(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Student-t (df=1) affinities in the embedding space.

    Returns
    -------
    num : (n, n) — un-normalised kernel values  (1 + ||y_i − y_j||²)^{-1}.
    Q   : (n, n) — row-normalised matrix, floored at 1e-12.
    """
    sq = np.sum(Y ** 2, axis=1)
    sq_dists = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Y @ Y.T), 0.0)
    num = 1.0 / (1.0 + sq_dists)
    np.fill_diagonal(num, 0.0)
    total = num.sum()
    Q = num / total if total > 0 else num
    return num, np.maximum(Q, 1e-12)


def _gradient(
    P: np.ndarray, Q: np.ndarray, Y: np.ndarray, num: np.ndarray
) -> np.ndarray:
    """Gradient of KL(P ∥ Q) with respect to embedding Y.

    ∂C/∂y_i = 4 Σ_j (P_{ij} − Q_{ij}) (y_i − y_j) (1 + ||y_i − y_j||²)^{-1}
    """
    PQ = (P - Q) * num                                   # (n, n)
    diff = Y[:, None, :] - Y[None, :, :]                 # (n, n, d)
    return 4.0 * (PQ[:, :, None] * diff).sum(axis=1)     # (n, d)


def _kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """KL(P ∥ Q) = Σ_{i,j} P_{ij} log(P_{ij} / Q_{ij})."""
    mask = P > 0
    return float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))
