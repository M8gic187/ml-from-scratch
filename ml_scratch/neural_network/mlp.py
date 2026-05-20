"""Multilayer Perceptron (MLP) for classification and regression."""

import numpy as np


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def _tanh_deriv(z: np.ndarray) -> np.ndarray:
    t = np.tanh(z)
    return 1.0 - t ** 2


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s * (1.0 - s)


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


_ACTIVATIONS = {
    "relu": (_relu, _relu_deriv),
    "tanh": (np.tanh, _tanh_deriv),
    "sigmoid": (_sigmoid, _sigmoid_deriv),
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _MLPBase:
    """Shared weight initialisation and forward/backward pass logic.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Number of neurons in each hidden layer.
    activation : {"relu", "tanh", "sigmoid"}
        Activation function for hidden layers.
    learning_rate : float
        Constant step size for gradient updates.
    n_iterations : int
        Number of full-batch gradient descent steps.
    batch_size : int or None
        Samples per mini-batch. ``None`` uses the full dataset.
    random_state : int or None
        Seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: str = "relu",
        learning_rate: float = 1e-3,
        n_iterations: int = 500,
        batch_size: int | None = 32,
        random_state: int | None = None,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS)}")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.loss_history_: list[float] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, layer_sizes: list[int]) -> None:
        rng = np.random.default_rng(self.random_state)
        self.weights_ = []
        self.biases_ = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He-initialisation for ReLU; Xavier for tanh / sigmoid
            if self.activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.weights_.append(rng.normal(0, scale, (fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> tuple[list, list]:
        """Return (pre_activations, activations) for every layer."""
        act_fn, _ = _ACTIVATIONS[self.activation]
        zs: list[np.ndarray] = []
        activations: list[np.ndarray] = [X]
        a = X
        # Hidden layers
        for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
            z = a @ W + b
            zs.append(z)
            a = act_fn(z)
            activations.append(a)
        # Output layer (no hidden activation; subclass handles output)
        z_out = a @ self.weights_[-1] + self.biases_[-1]
        zs.append(z_out)
        return zs, activations

    # ------------------------------------------------------------------
    # Backward pass (chain rule)
    # ------------------------------------------------------------------

    def _backward(
        self,
        zs: list[np.ndarray],
        activations: list[np.ndarray],
        delta_out: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return weight and bias gradients for all layers."""
        _, act_deriv = _ACTIVATIONS[self.activation]
        n = activations[0].shape[0]

        grad_W = [None] * len(self.weights_)
        grad_b = [None] * len(self.biases_)

        delta = delta_out
        grad_W[-1] = activations[-1].T @ delta / n
        grad_b[-1] = delta.mean(axis=0)

        for layer in range(len(self.weights_) - 2, -1, -1):
            delta = (delta @ self.weights_[layer + 1].T) * act_deriv(zs[layer])
            grad_W[layer] = activations[layer].T @ delta / n
            grad_b[layer] = delta.mean(axis=0)

        return grad_W, grad_b

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    def _update(
        self, grad_W: list[np.ndarray], grad_b: list[np.ndarray]
    ) -> None:
        for i in range(len(self.weights_)):
            self.weights_[i] -= self.learning_rate * grad_W[i]
            self.biases_[i] -= self.learning_rate * grad_b[i]

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def _iter_batches(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ):
        n = X.shape[0]
        bs = n if self.batch_size is None else self.batch_size
        idx = rng.permutation(n)
        for start in range(0, n, bs):
            batch = idx[start : start + bs]
            yield X[batch], y[batch]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")


# ---------------------------------------------------------------------------
# MLPClassifier
# ---------------------------------------------------------------------------

class MLPClassifier(_MLPBase):
    """MLP for single-label classification via cross-entropy loss.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Number of neurons per hidden layer.
    activation : {"relu", "tanh", "sigmoid"}
        Hidden-layer activation function.
    learning_rate : float
        Step size for gradient descent.
    n_iterations : int
        Training epochs.
    batch_size : int or None
        Mini-batch size; ``None`` uses full-batch GD.
    random_state : int or None
        Seed for weight initialisation and mini-batch shuffling.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels encountered during ``fit``.
    loss_history_ : list[float]
        Mean cross-entropy loss recorded after each epoch.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        """Train the network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[c] for c in y])

        # One-hot encode
        Y = np.zeros((len(y_idx), n_classes))
        Y[np.arange(len(y_idx)), y_idx] = 1.0

        layer_sizes = [X.shape[1], *self.hidden_layer_sizes, n_classes]
        self._init_weights(layer_sizes)
        self.loss_history_ = []

        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_iterations):
            epoch_losses: list[float] = []
            for X_b, Y_b in self._iter_batches(X, Y, rng):
                zs, activations = self._forward(X_b)
                probs = _softmax(zs[-1])
                # Cross-entropy delta at output layer
                delta_out = probs - Y_b
                # Loss
                log_probs = np.log(np.clip(probs, 1e-15, 1.0))
                epoch_losses.append(-np.mean(np.sum(Y_b * log_probs, axis=1)))

                grad_W, grad_b = self._backward(zs, activations, delta_out)
                self._update(grad_W, grad_b)
            self.loss_history_.append(float(np.mean(epoch_losses)))

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        zs, _ = self._forward(np.asarray(X, dtype=float))
        return _softmax(zs[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# MLPRegressor
# ---------------------------------------------------------------------------

class MLPRegressor(_MLPBase):
    """MLP for regression via mean squared error loss.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Number of neurons per hidden layer.
    activation : {"relu", "tanh", "sigmoid"}
        Hidden-layer activation function.
    learning_rate : float
        Step size for gradient descent.
    n_iterations : int
        Training epochs.
    batch_size : int or None
        Mini-batch size; ``None`` uses full-batch GD.
    random_state : int or None
        Seed for weight initialisation and mini-batch shuffling.

    Attributes
    ----------
    loss_history_ : list[float]
        Mean MSE loss recorded after each epoch.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        """Train the network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        n_outputs = y.shape[1]

        layer_sizes = [X.shape[1], *self.hidden_layer_sizes, n_outputs]
        self._init_weights(layer_sizes)
        self.loss_history_ = []
        self._n_outputs = n_outputs

        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_iterations):
            epoch_losses: list[float] = []
            for X_b, y_b in self._iter_batches(X, y, rng):
                zs, activations = self._forward(X_b)
                # Linear output; delta = dMSE/dz_out = y_hat - y
                delta_out = zs[-1] - y_b
                epoch_losses.append(float(np.mean(delta_out ** 2)))

                grad_W, grad_b = self._backward(zs, activations, delta_out)
                self._update(grad_W, grad_b)
            self.loss_history_.append(float(np.mean(epoch_losses)))

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted target values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_outputs)
        """
        self._check_fitted()
        zs, _ = self._forward(np.asarray(X, dtype=float))
        out = zs[-1]
        return out.ravel() if self._n_outputs == 1 else out
