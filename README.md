# ml-from-scratch

A clean, educational implementation of core machine learning algorithms using only NumPy.
No scikit-learn, no PyTorch — just math and code.

## Algorithms

| Category           | Algorithm              | Status |
|--------------------|------------------------|--------|
| Linear Models      | Linear Regression      | ✅     |
| Linear Models      | Logistic Regression    | ✅     |
| SVM                | LinearSVC (Pegasos)    | ✅     |
| SVM                | SVC (kernel SMO)       | ✅     |
| Clustering         | K-Means                | ✅     |
| Clustering         | DBSCAN                 | ✅     |
| Neighbors          | K-Nearest Neighbors    | ✅     |
| Tree               | Decision Tree (CART)   | ✅     |
| Naive Bayes        | Gaussian Naive Bayes   | ✅     |
| Ensemble           | Random Forest          | ✅     |
| Ensemble           | AdaBoost               | ✅     |
| Ensemble           | Gradient Boosting      | ✅     |
| Decomposition      | PCA                    | ✅     |

## Structure

```
ml_scratch/
├── utils/          # Metrics, cross-validation, and evaluation helpers
├── linear_models/  # Gradient-descent regression & classification
├── svm/            # LinearSVC (Pegasos) and SVC (kernel SMO)
├── clustering/     # Unsupervised learning
├── neighbors/      # Instance-based learning (KNN)
├── tree/           # Tree-based models (CART)
├── naive_bayes/    # Gaussian Naive Bayes classifier
├── ensemble/       # Random Forest, AdaBoost, Gradient Boosting
└── decomposition/  # PCA dimensionality reduction
```

## Quick Start

```python
from ml_scratch.linear_models import LinearRegression, LogisticRegression
from ml_scratch.svm import LinearSVC, SVC
from ml_scratch.clustering import KMeans, DBSCAN
from ml_scratch.neighbors import KNNClassifier, KNNRegressor
from ml_scratch.tree import DecisionTreeClassifier
from ml_scratch.naive_bayes import GaussianNB
from ml_scratch.ensemble import RandomForestClassifier, AdaBoostClassifier
from ml_scratch.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from ml_scratch.decomposition import PCA
from ml_scratch.utils import (
    accuracy, r2_score, confusion_matrix, classification_report, log_loss,
    KFold, StratifiedKFold, cross_val_score,
)
```

### Linear Regression

```python
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
print(f"R²: {r2_score(y_test, model.predict(X_test)):.4f}")
```

### Support Vector Machine

```python
# LinearSVC — Pegasos primal sub-gradient (fast, works great on large datasets)
clf = LinearSVC(C=1.0, n_iterations=3000)
clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy(y_test, clf.predict(X_test)):.4f}")
scores = clf.decision_function(X_test)  # signed margin distance

# SVC — kernel SVM via simplified SMO in the dual
# linear kernel
svc = SVC(kernel="linear", C=1.0)
svc.fit(X_train, y_train)

# RBF kernel (handles non-linearly separable data)
svc = SVC(kernel="rbf", C=5.0, gamma="scale", n_iterations=100)
svc.fit(X_train, y_train)
print(f"Accuracy         : {accuracy(y_test, svc.predict(X_test)):.4f}")
print(f"Support vectors  : {len(svc.support_vectors_)}")

# Polynomial kernel
svc = SVC(kernel="poly", degree=3, C=1.0, gamma="scale")
svc.fit(X_train, y_train)
```

### Cross-Validation

```python
from ml_scratch.utils import KFold, StratifiedKFold, cross_val_score

# K-Fold (plain split)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    ...

# Stratified K-Fold (preserves class ratio per fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(SVC(kernel="rbf"), X, y, cv=skf)
print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Extended Metrics

```python
from ml_scratch.utils import confusion_matrix, classification_report, log_loss

cm = confusion_matrix(y_test, y_pred)   # integer matrix, rows=true, cols=pred
print(classification_report(y_test, y_pred))
loss = log_loss(y_test, y_prob)         # binary cross-entropy
```

### DBSCAN

```python
# Density-based clustering — no need to specify k, finds arbitrarily shaped clusters
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)           # noise points receive label -1

print(f"Clusters found  : {db.n_clusters_}")
print(f"Noise points    : {(labels == -1).sum()}")
print(f"Core points     : {len(db.core_sample_indices_)}")

# Evaluate cluster quality (noise excluded automatically)
score = silhouette_score(X, labels)
print(f"Silhouette score: {score:.4f}")

# Manhattan distance variant
db_l1 = DBSCAN(eps=0.8, min_samples=5, metric="manhattan")
db_l1.fit(X)
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

### Gaussian Naive Bayes

```python
# Multi-class classification with posterior probabilities
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy(y_test, clf.predict(X_test)):.4f}")

# Retrieve normalised class posteriors
proba = clf.predict_proba(X_test)   # shape (n_samples, n_classes)
```

### Random Forest

```python
# 100 trees, sqrt(n_features) feature subspace per split
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
print(f"Accuracy: {accuracy(y_test, rf.predict(X_test)):.4f}")

# Soft vote probabilities (averaged across trees)
proba = rf.predict_proba(X_test)    # shape (n_samples, n_classes)
```

### AdaBoost

```python
# 50 boosting rounds, binary classification only
clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy(y_test, clf.predict(X_test)):.4f}")

# Raw boosted scores (positive → class 1)
scores = clf.decision_function(X_test)  # shape (n_samples,)
```

### Gradient Boosting

```python
# Binary classification with log-loss (stochastic gradient boosting)
clf = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8
)
clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy(y_test, clf.predict(X_test)):.4f}")

# Class probabilities and raw log-odds scores
proba  = clf.predict_proba(X_test)       # shape (n_samples, 2)
scores = clf.decision_function(X_test)   # shape (n_samples,)

# Regression with MSE or MAE loss
reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, loss="mse")
reg.fit(X_train, y_train)
print(f"R²: {r2_score(y_test, reg.predict(X_test)):.4f}")
```

### PCA

```python
# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train)   # shape (n_samples, 2)
X_back    = pca.inverse_transform(X_reduced)  # approximate reconstruction

print("Explained variance:", pca.explained_variance_ratio_.round(3))
print(f"Cumulative: {pca.explained_variance_ratio_.sum():.4f}")
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
