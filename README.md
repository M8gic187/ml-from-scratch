# ml-from-scratch

A clean, educational implementation of core machine learning algorithms using only NumPy.
No scikit-learn, no PyTorch — just math and code.

## Algorithms

| Category        | Algorithm              | Status |
|-----------------|------------------------|--------|
| Linear Models   | Linear Regression      | ✅     |
| Linear Models   | Logistic Regression    | ✅     |
| Clustering      | K-Means                | ✅     |

## Structure

```
ml_scratch/
├── utils/          # Metrics, data preprocessing helpers
├── linear_models/  # Regression algorithms
└── clustering/     # Unsupervised learning
```

## Usage

```python
from ml_scratch.linear_models import LinearRegression
from ml_scratch.utils.metrics import mse, r2_score

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"R²: {r2_score(y_test, predictions):.4f}")
```

## Requirements

- Python 3.9+
- NumPy >= 1.21

## Installation

```bash
pip install -r requirements.txt
```
