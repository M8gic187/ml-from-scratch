"""Demo: binary classification on a 2-feature synthetic dataset."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.linear_models import LogisticRegression
from ml_scratch.utils.metrics import accuracy, precision_recall_f1

rng = np.random.default_rng(0)

# Two Gaussian blobs
X0 = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], size=150)
X1 = rng.multivariate_normal([3, 3], [[1, 0.3], [0.3, 1]], size=150)
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(150), np.ones(150)])

# Shuffle and split
idx = rng.permutation(300)
X, y = X[idx], y[idx]
split = 240
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LogisticRegression(learning_rate=0.1, n_iterations=500)
model.fit(X_train, y_train)
preds = model.predict(X_test)

p, r, f1 = precision_recall_f1(y_test.astype(int), preds)
print(f"Accuracy  : {accuracy(y_test.astype(int), preds):.4f}")
print(f"Precision : {p:.4f}")
print(f"Recall    : {r:.4f}")
print(f"F1-Score  : {f1:.4f}")
