"""Support Vector Machine classifiers.

LinearSVC  -- primal sub-gradient descent (Pegasos algorithm).
SVC        -- kernel SVM via a simplified SMO solver in the dual.
"""

import numpy as np


class LinearSVC:
    """Binary SVM trained via Pegasos (primal sub-gradient descent).

    Minimises the L2-regularised hinge loss::

        (λ/2)||w||² + (1/n) Σᵢ max(0, 1 − yᵢ(wᵀxᵢ + b))

    where λ = 1 / (C · n_samples).

    Parameters
    ----------
    C : float
        Regularisation strength; larger C → softer margin.
    n_iterations : int
        Total number of stochastic sub-gradient steps.
    """

    def __init__(self, C: float = 1.0, n_iterations: int = 2000) -> None:
        self.C = C
        self.n_iterations = n_iterations

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVC":
        """Fit the SVM to binary-labelled training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,), values in {0, 1} or {-1, 1}

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y_pm = _to_pm(np.asarray(y))
        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        lam = 1.0 / (self.C * n_samples)

        rng = np.random.default_rng(0)
        for t in range(1, self.n_iterations + 1):
            lr = 1.0 / (lam * t)
            idx = int(rng.integers(n_samples))
            xi, yi = X[idx], y_pm[idx]

            margin = yi * (float(np.dot(self.weights_, xi)) + self.bias_)
            if margin < 1:
                self.weights_ = (1 - lr * lam) * self.weights_ + lr * self.C * yi * xi
                self.bias_ += lr * self.C * yi
            else:
                self.weights_ = (1 - lr * lam) * self.weights_

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to the decision hyperplane."""
        self._check_fitted()
        return np.asarray(X, dtype=float) @ self.weights_ + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels in {0, 1}."""
        return (self.decision_function(X) >= 0).astype(int)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")


class SVC:
    """Kernel Support Vector Classifier via a simplified SMO dual solver.

    Solves the soft-margin dual::

        max  Σᵢ αᵢ − ½ Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
        s.t. 0 ≤ αᵢ ≤ C,   Σᵢ αᵢ yᵢ = 0

    using coordinate-wise (pairwise) analytical updates identical to the
    original SMO paper (Platt 1998), with the max-|Ei − Ej| heuristic for
    the second working-set element.

    Parameters
    ----------
    C : float
        Regularisation strength; larger C → stricter constraint on errors.
    kernel : {'linear', 'rbf', 'poly'}
        Kernel function.
    gamma : float or 'scale'
        Kernel coefficient for 'rbf' and 'poly'.
        ``'scale'`` sets γ = 1 / (n_features × Var(X)).
    degree : int
        Degree for the polynomial kernel.
    n_iterations : int
        Maximum number of full passes over the training set.
    tol : float
        KKT tolerance; smaller → tighter convergence.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: float | str = "scale",
        degree: int = 3,
        n_iterations: int = 100,
        tol: float = 1e-3,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.n_iterations = n_iterations
        self.tol = tol

        self.alpha_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.support_vectors_: np.ndarray | None = None
        self.support_labels_: np.ndarray | None = None
        self.support_alpha_: np.ndarray | None = None
        self._gamma_fit: float = 1.0

    # ------------------------------------------------------------------
    # Kernels
    # ------------------------------------------------------------------

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, str) and self.gamma == "scale":
            var = float(X.var())
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        return float(self.gamma)

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
        if self.kernel == "linear":
            return X1 @ X2.T
        if self.kernel == "rbf":
            sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
            sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
            dist2 = sq1 + sq2.T - 2.0 * (X1 @ X2.T)
            return np.exp(-gamma * np.maximum(dist2, 0.0))
        if self.kernel == "poly":
            return (gamma * (X1 @ X2.T) + 1.0) ** self.degree
        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        """Fit the kernel SVM to binary-labelled training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,), values in {0, 1} or {-1, 1}

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y_pm = _to_pm(np.asarray(y))
        n = len(y_pm)
        gamma = self._resolve_gamma(X)
        self._gamma_fit = gamma

        # Pre-compute full kernel matrix — O(n²) but enables O(1) updates.
        K = self._kernel_matrix(X, X, gamma)

        alpha = np.zeros(n)
        # f[i] = sum_j alpha[j]*y[j]*K[i,j]  (without bias, maintained incrementally)
        f = np.zeros(n)
        bias = 0.0

        for _ in range(self.n_iterations):
            n_changed = 0

            for i in range(n):
                Ei = f[i] + bias - y_pm[i]

                # KKT violation check
                cond_upper = y_pm[i] * Ei < -self.tol and alpha[i] < self.C
                cond_lower = y_pm[i] * Ei > self.tol and alpha[i] > 0
                if not (cond_upper or cond_lower):
                    continue

                # Second element: maximise |Ej − Ei| (heuristic)
                errors = f + bias - y_pm
                j = int(np.argmax(np.abs(errors - Ei)))
                if j == i:
                    j = (i + 1) % n
                Ej = errors[j]

                ai_old, aj_old = alpha[i], alpha[j]

                # Clip bounds for alpha[j]
                if y_pm[i] == y_pm[j]:
                    L = max(0.0, aj_old + ai_old - self.C)
                    H = min(self.C, aj_old + ai_old)
                else:
                    L = max(0.0, aj_old - ai_old)
                    H = min(self.C, self.C + aj_old - ai_old)
                if L >= H:
                    continue

                eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
                if eta <= 1e-12:
                    continue

                alpha_j_new = float(np.clip(aj_old + y_pm[j] * (Ei - Ej) / eta, L, H))
                if abs(alpha_j_new - aj_old) < self.tol * 1e-1:
                    continue

                alpha_i_new = float(
                    np.clip(ai_old + y_pm[i] * y_pm[j] * (aj_old - alpha_j_new), 0.0, self.C)
                )

                # Incremental decision function update
                di = (alpha_i_new - ai_old) * y_pm[i]
                dj = (alpha_j_new - aj_old) * y_pm[j]
                f += K[:, i] * di + K[:, j] * dj

                # Bias update (SMO rule).
                # di already carries the y_i factor, so no extra y_pm[i] here.
                b1 = bias - Ei - di * K[i, i] - dj * K[i, j]
                b2 = bias - Ej - di * K[i, j] - dj * K[j, j]
                if 0.0 < alpha_i_new < self.C:
                    bias = b1
                elif 0.0 < alpha_j_new < self.C:
                    bias = b2
                else:
                    bias = (b1 + b2) / 2.0

                alpha[i] = alpha_i_new
                alpha[j] = alpha_j_new
                n_changed += 1

            if n_changed == 0:
                break

        self.alpha_ = alpha
        self.bias_ = bias
        self._X_train = X
        self._y_train = y_pm

        # Cache support vectors (alpha > threshold)
        sv_mask = alpha > 1e-6
        self.support_vectors_ = X[sv_mask]
        self.support_labels_ = y_pm[sv_mask]
        self.support_alpha_ = alpha[sv_mask]

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to the decision hyperplane."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        K = self._kernel_matrix(X, self.support_vectors_, self._gamma_fit)
        return K @ (self.support_alpha_ * self.support_labels_) + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels in {0, 1}."""
        return (self.decision_function(X) >= 0).astype(int)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.alpha_ is None:
            raise RuntimeError("Call fit() before predict().")


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _to_pm(y: np.ndarray) -> np.ndarray:
    """Map labels to {-1, +1}. Accepts both {0, 1} and {-1, 1} input."""
    y = np.asarray(y, dtype=float)
    # If any label is already negative the array is already in ±1 form.
    if np.any(y < 0):
        return y
    return np.where(y == 0, -1.0, 1.0)
