"""Demo: fit LinearRegression on synthetic data and report metrics."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.linear_models import LinearRegression
from ml_scratch.utils.metrics import mse, r2_score

rng = np.random.default_rng(42)

# y = 3x + 7 + noise
X = rng.uniform(-3, 3, size=(200, 1))
y = 3.0 * X.squeeze() + 7.0 + rng.normal(0, 0.5, size=200)

split = 160
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LinearRegression(learning_rate=0.05, n_iterations=500)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"Learned weight : {model.weights_[0]:.4f}  (true: 3.0)")
print(f"Learned bias   : {model.bias_:.4f}  (true: 7.0)")
print(f"Test MSE       : {mse(y_test, preds):.4f}")
print(f"Test R²        : {r2_score(y_test, preds):.4f}")
