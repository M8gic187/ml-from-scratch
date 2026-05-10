# ml-from-scratch

A clean, educational implementation of core machine learning algorithms using only NumPy.
No scikit-learn, no PyTorch — just math and code.

## Algorithms

| Category        | Algorithm              | Status |
|-----------------|------------------------|--------|
| Linear Models   | Linear Regression      | ✅     |
| Linear Models   | Logistic Regression    | ✅     |
| Clustering      | K-Means                | ✅     |
| Neighbors       | K-Nearest Neighbors    | ✅     |
| Tree            | Decision Tree (CART)   | ✅     |

## Structure

```
ml_scratch/
├── utils/          # Metrics and evaluation helpers
├── linear_models/  # Gradient-descent regression & classification
├── clustering/     # Unsupervised learning
├── neighbors/      # Instance-based learning (KNN)
└── tree/           # Tree-based models (CART)
```

## Quick Start

```python
from ml_scratch.linear_models import LinearRegression, LogisticRegression
from ml_scratch.clustering import KMeans
from ml_scratch.neighbors import KNNClassifier, KNNRegressor
from ml_scratch.tree import DecisionTreeClassifier
from ml_scratch.utils.metrics import r2_score, accuracy
```

### Linear Regression

```python
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
print(f"R²: {r2_score(y_test, model.predict(X_test)):.4f}")
```

### K-Nearest Neighbors

```python
# Classifier with distance weighting
clf = KNNClassifier(k=5, weights="distance")
clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy(y_test, clf.predict(X_test)):.4f}")

# Regressor
reg = KNNRegressor(k=7, metric="euclidean")
reg.fit(X_train, y_train)
```

### Decision Tree

```python
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)
tree.fit(X_train, y_train)
print(f"Accuracy: {accuracy(y_test, tree.predict(X_test)):.4f}")
```

## Running Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Requirements

- Python 3.9+
- NumPy >= 1.21

## Installation

```bash
pip install -r requirements.txt
```
