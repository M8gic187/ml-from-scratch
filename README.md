# ml-from-scratch

A clean, educational implementation of core machine learning algorithms using only NumPy.
No scikit-learn, no PyTorch — just math and code.

## Algorithms

| Category           | Algorithm              | Status |
|--------------------|------------------------|--------|
| Linear Models      | Linear Regression      | ✅     |
| Linear Models      | Logistic Regression    | ✅     |
| Linear Models      | Ridge Regression       | ✅     |
| Linear Models      | Lasso Regression       | ✅     |
| Linear Models      | ElasticNet             | ✅     |
| SVM                | LinearSVC (Pegasos)    | ✅     |
| SVM                | SVC (kernel SMO)       | ✅     |
| Clustering         | K-Means                | ✅     |
| Clustering         | DBSCAN                 | ✅     |
| Clustering         | Gaussian Mixture (GMM) | ✅     |
| Clustering         | Agglomerative          | ✅     |
| Clustering         | Spectral Clustering    | ✅     |
| Neighbors          | K-Nearest Neighbors    | ✅     |
| Tree               | Decision Tree (CART)   | ✅     |
| Naive Bayes        | Gaussian Naive Bayes   | ✅     |
| Ensemble           | Random Forest          | ✅     |
| Ensemble           | Extra Trees            | ✅     |
| Ensemble           | AdaBoost               | ✅     |
| Ensemble           | Gradient Boosting      | ✅     |
| Ensemble           | Isolation Forest       | ✅     |
| Ensemble           | Voting Classifier      | ✅     |
| Decomposition      | PCA                    | ✅     |
| Decomposition      | LDA                    | ✅     |
| Decomposition      | t-SNE                  | ✅     |
| Decomposition      | FastICA                | ✅     |
| Decomposition      | NMF                    | ✅     |
| Neural Network     | MLP Classifier         | ✅     |
| Neural Network     | MLP Regressor          | ✅     |
| Preprocessing      | StandardScaler         | ✅     |
| Preprocessing      | MinMaxScaler           | ✅     |
| Preprocessing      | LabelEncoder           | ✅     |
| Preprocessing      | OneHotEncoder          | ✅     |

## Structure

```
ml_scratch/
├── utils/           # Metrics, cross-validation, and evaluation helpers
├── linear_models/   # Gradient-descent regression & classification
├── svm/             # LinearSVC (Pegasos) and SVC (kernel SMO)
├── clustering/      # Unsupervised learning
├── neighbors/       # Instance-based learning (KNN)
├── tree/            # Tree-based models (CART)
├── naive_bayes/     # Gaussian Naive Bayes classifier
├── ensemble/        # Random Forest, Extra Trees, AdaBoost, Gradient Boosting, Isolation Forest, Voting
├── decomposition/   # PCA, LDA, t-SNE, FastICA, NMF dimensionality reduction and factorisation
├── neural_network/  # MLP (Multilayer Perceptron) classifier and regressor
└── preprocessing/   # Feature scalers and categorical encoders
```

## Quick Start

```python
from ml_scratch.linear_models import LinearRegression, LogisticRegression
from ml_scratch.linear_models import Ridge, Lasso, ElasticNet
from ml_scratch.svm import LinearSVC, SVC
from ml_scratch.clustering import KMeans, DBSCAN, GaussianMixture, AgglomerativeClustering, SpectralClustering
from ml_scratch.neighbors import KNNClassifier, KNNRegressor
from ml_scratch.tree import DecisionTreeClassifier
from ml_scratch.naive_bayes import GaussianNB
from ml_scratch.ensemble import RandomForestClassifier, ExtraTreesClassifier
from ml_scratch.ensemble import AdaBoostClassifier
from ml_scratch.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from ml_scratch.ensemble import IsolationForest, VotingClassifier
from ml_scratch.decomposition import PCA, LinearDiscriminantAnalysis, TSNE, FastICA, NMF
from ml_scratch.neural_network import MLPClassifier, MLPRegressor
from ml_scratch.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder,
)
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

### Regularised Linear Models

```python
# Ridge — L2 penalty (gradient descent); shrinks all weights but keeps them non-zero
ridge = Ridge(alpha=1.0, learning_rate=0.01, n_iterations=1000)
ridge.fit(X_train, y_train)
print(f"R²: {r2_score(y_test, ridge.predict(X_test)):.4f}")
print(f"||w||: {np.linalg.norm(ridge.weights_):.4f}")

# Lasso — L1 penalty (coordinate descent); drives irrelevant weights to exactly zero
lasso = Lasso(alpha=0.1, n_iterations=1000)
lasso.fit(X_train, y_train)
n_zeros = (np.abs(lasso.weights_) < 1e-6).sum()
print(f"R²: {r2_score(y_test, lasso.predict(X_test)):.4f}")
print(f"Sparse weights (zero / total): {n_zeros} / {len(lasso.weights_)}")

# ElasticNet — blended L1+L2 (coordinate descent); best for correlated features
# l1_ratio=1.0 → pure Lasso, l1_ratio=0.0 → pure Ridge
en = ElasticNet(alpha=0.1, l1_ratio=0.5, n_iterations=1000)
en.fit(X_train, y_train)
print(f"R²: {r2_score(y_test, en.predict(X_test)):.4f}")
print(f"Converged in {en.n_iter_} iterations")
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

### Gaussian Mixture Model (GMM)

```python
# Soft clustering via Expectation-Maximisation over Gaussian components
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm.fit(X)

# Hard cluster labels (most likely component)
labels = gmm.predict(X)

# Soft posterior probabilities — rows sum to 1
proba = gmm.predict_proba(X)   # shape (n_samples, n_components)

# Model quality
print(f"Converged       : {gmm.converged_} ({gmm.n_iter_} iterations)")
print(f"Log-likelihood  : {gmm.score(X):.4f}")
print(f"AIC             : {gmm.aic(X):.2f}")
print(f"BIC             : {gmm.bic(X):.2f}")
print(f"Mixture weights : {gmm.weights_.round(3)}")

# Model selection: pick k that minimises BIC
bics = [GaussianMixture(n_components=k, random_state=0).fit(X).bic(X) for k in range(1, 8)]
best_k = int(np.argmin(bics)) + 1
print(f"BIC-optimal k   : {best_k}")

# Diagonal covariance (faster, assumes feature independence)
gmm_diag = GaussianMixture(n_components=3, covariance_type="diag")
gmm_diag.fit(X)

# Spherical covariance (single variance per component)
gmm_sph = GaussianMixture(n_components=3, covariance_type="spherical")
gmm_sph.fit(X)
```

### Agglomerative Clustering

```python
# Ward linkage (default) — minimises within-cluster variance, tends to produce
# equally-sized, spherical clusters; requires Euclidean distance
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = agg.fit_predict(X)
print(f"Cluster sizes: {[int((labels == l).sum()) for l in range(3)]}")

# Compare linkage strategies on the same data
for linkage in ("single", "complete", "average", "ward"):
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    model.fit(X)
    sil = silhouette_score(X, model.labels_)
    print(f"  {linkage:<10}: silhouette={sil:.4f}")

# Manhattan metric (available for single / complete / average)
agg_l1 = AgglomerativeClustering(n_clusters=2, linkage="complete", metric="manhattan")
agg_l1.fit(X)

# Merge to a single cluster (returns all zeros)
agg_1 = AgglomerativeClustering(n_clusters=1).fit_predict(X)
```

### Spectral Clustering

```python
from ml_scratch.clustering import SpectralClustering

# RBF affinity — works well for compact, well-separated blobs
sc = SpectralClustering(n_clusters=3, affinity="rbf", gamma=2.0, random_state=0)
labels = sc.fit_predict(X)

# k-NN affinity — handles non-convex shapes (rings, crescents)
sc_knn = SpectralClustering(
    n_clusters=2, affinity="nearest_neighbors", n_neighbors=10, random_state=0
)
labels_knn = sc_knn.fit_predict(X)

# Inspect spectral embedding and affinity matrix
print(sc.embedding_.shape)          # (n_samples, n_clusters)
print(sc.affinity_matrix_.shape)    # (n_samples, n_samples)
```

**When to use Spectral Clustering:**
- Data has non-convex cluster shapes that K-Means cannot separate.
- Use `affinity="nearest_neighbors"` for concentric rings or interlocking crescents.
- Use `affinity="rbf"` for smooth, blob-like clusters; tune `gamma` to control the
  affinity bandwidth (smaller = more local, larger = more global).

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

### Extra Trees

Extremely Randomised Trees (Geurts et al., 2006) extends Random Forest by
choosing split thresholds *uniformly at random* from each feature's observed
range instead of searching for the optimal Gini split.  The extra randomness
makes each tree construction faster and the ensemble more diverse, typically
matching or exceeding Random Forest test accuracy while being cheaper to train.

Key differences from `RandomForestClassifier`:

| Property | Random Forest | Extra Trees |
|---|---|---|
| Threshold selection | optimal (best Gini) | random (uniform in range) |
| `bootstrap` default | `True` | `False` |
| `n_features` default | `sqrt(p)` | all features |

```python
from ml_scratch.ensemble import ExtraTreesClassifier

# Basic usage — mirrors the Random Forest API
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
et.fit(X_train, y_train)
print(f"Accuracy: {accuracy(y_test, et.predict(X_test)):.4f}")

# Per-class probability estimates (soft vote)
proba = et.predict_proba(X_test)   # shape (n_samples, n_classes)

# Restrict feature candidates per split (can help on high-dimensional data)
et_sub = ExtraTreesClassifier(n_estimators=100, n_features=5, random_state=0)
et_sub.fit(X_train, y_train)

# Enable bootstrap sampling (closer to a Random Forest in training variance)
et_boot = ExtraTreesClassifier(n_estimators=100, bootstrap=True, random_state=0)
et_boot.fit(X_train, y_train)

# Control tree complexity
et_shallow = ExtraTreesClassifier(
    n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=0
)
et_shallow.fit(X_train, y_train)
```

**When to prefer Extra Trees over Random Forest:**
- Training speed is a priority (random thresholds avoid the O(n log n) split search).
- The dataset is large and the optimal split threshold is unlikely to overfit.
- Used as a fast baseline before tuning a more expensive ensemble.

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

### Isolation Forest

Isolation Forest is an efficient, tree-based anomaly detector.  Instead of
modelling normality, it explicitly isolates outliers by randomly partitioning
the feature space.  Anomalous points (in sparse regions) are separated near
the root of each tree and therefore have shorter average path lengths.

```python
from ml_scratch.ensemble import IsolationForest

# Basic usage: fit on unlabelled data, predict inlier (+1) / outlier (-1)
iso = IsolationForest(
    n_estimators=100,      # number of isolation trees
    max_samples="auto",    # subsample size per tree (default: min(256, n_samples))
    contamination=0.05,    # expected fraction of outliers → sets decision threshold
    random_state=42,
)
iso.fit(X_train)
labels = iso.predict(X_test)          # +1 inlier, -1 outlier

# Raw anomaly scores — more negative = more anomalous
scores = iso.score_samples(X_test)    # shape (n_samples,)

# Offset-corrected signed distance; positive → inlier, negative → outlier
decisions = iso.decision_function(X_test)

# Rank the 10 most anomalous samples
top_outliers = X_test[np.argsort(scores)[:10]]
```

### Voting Classifier

`VotingClassifier` combines any number of heterogeneous base estimators via
majority vote (hard) or probability averaging (soft).  It is especially useful
as a final layer after tuning several independent models: combining diverse
learners typically reduces variance while preserving each model's strengths.

| Mode | Aggregation | Requires `predict_proba` |
|------|-------------|--------------------------|
| `'hard'` | strict (weighted) majority vote | no |
| `'soft'` | (weighted) average of posterior probabilities | yes |

```python
from ml_scratch.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from ml_scratch.tree import DecisionTreeClassifier
from ml_scratch.neighbors import KNNClassifier

# Hard voting — combine any estimators
hard_clf = VotingClassifier(
    estimators=[
        ("dt",  DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("knn", KNNClassifier(k=7)),
        ("rf",  RandomForestClassifier(n_estimators=50, random_state=0)),
    ],
    voting="hard",
)
hard_clf.fit(X_train, y_train)
print(f"Hard voting accuracy: {accuracy(y_test, hard_clf.predict(X_test)):.4f}")

# Soft voting — average class probabilities (estimators must expose predict_proba)
soft_clf = VotingClassifier(
    estimators=[
        ("dt", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=50, random_state=0)),
        ("gb", GradientBoostingClassifier(n_estimators=50, random_state=0)),
    ],
    voting="soft",
)
soft_clf.fit(X_train, y_train)
proba = soft_clf.predict_proba(X_test)   # shape (n_samples, n_classes)
print(f"Soft voting accuracy: {accuracy(y_test, soft_clf.predict(X_test)):.4f}")

# Weighted voting — trust some estimators more than others
weighted = VotingClassifier(
    estimators=[
        ("dt", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=0)),
    ],
    voting="hard",
    weights=[1, 3],   # RandomForest gets 3× the vote weight
)
weighted.fit(X_train, y_train)
```

**When to use VotingClassifier:**
- Combining models with complementary strengths (e.g. a tree, a KNN, and a linear model).
- As a lightweight alternative to stacking when a meta-learner is too expensive.
- Use `voting='soft'` when the base estimators are well-calibrated — averaged probabilities
  carry more signal than hard votes and usually outperform `voting='hard'`.

### PCA

```python
# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train)   # shape (n_samples, 2)
X_back    = pca.inverse_transform(X_reduced)  # approximate reconstruction

print("Explained variance:", pca.explained_variance_ratio_.round(3))
print(f"Cumulative: {pca.explained_variance_ratio_.sum():.4f}")
```

### Linear Discriminant Analysis (LDA)

LDA is a supervised dimensionality reduction technique that finds the axes
maximising between-class separation relative to within-class variance.  It
also doubles as a Gaussian classifier via nearest-centroid prediction in the
discriminant subspace.

```python
from ml_scratch.decomposition import LinearDiscriminantAnalysis

# --- Dimensionality reduction (supervised) ---
# At most min(n_features, n_classes − 1) components are available
lda = LinearDiscriminantAnalysis(n_components=2)
X_reduced = lda.fit_transform(X_train, y_train)   # shape (n_samples, 2)

print("Explained variance:", lda.explained_variance_ratio_.round(3))
print(f"Cumulative: {lda.explained_variance_ratio_.sum():.4f}")

# Project new data (uses same axes from training)
X_test_reduced = lda.transform(X_test)

# --- Classification (nearest centroid in discriminant space) ---
lda_clf = LinearDiscriminantAnalysis(n_components=2)
lda_clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy(y_test, lda_clf.predict(X_test)):.4f}")

# Class posterior probabilities (softmax of log-posteriors)
proba = lda_clf.predict_proba(X_test)   # shape (n_samples, n_classes)

# --- Custom class priors ---
# Useful for imbalanced datasets or when deployment priors differ from training
lda_prior = LinearDiscriminantAnalysis(priors=[0.9, 0.1])
lda_prior.fit(X_train, y_train)

# --- Numerical solver ---
# 'svd' (default-stable) whitens the data first; 'eigen' solves SW⁻¹ SB directly
lda_svd = LinearDiscriminantAnalysis(n_components=2, solver='svd')
lda_svd.fit(X_train, y_train)
```

### t-SNE

t-SNE (van der Maaten & Hinton, 2008) maps high-dimensional data to a
low-dimensional (typically 2-D) embedding that preserves local neighbourhood
structure.  Pairwise similarities in high-D space are modelled with Gaussian
kernels calibrated per-point to achieve a target *perplexity* (effective number
of neighbours).  The low-D affinities use a Student-t kernel, whose heavier
tail relieves the crowding problem.  Optimisation minimises KL(P ∥ Q) via
momentum gradient descent with adaptive per-component gains.

```python
from ml_scratch.decomposition import TSNE

# --- Basic 2-D visualisation ---
tsne = TSNE(n_components=2, perplexity=30, n_iterations=1000, random_state=42)
Y = tsne.fit_transform(X)           # shape (n_samples, 2)

print(f"KL divergence : {tsne.kl_divergence_:.4f}")
print(f"Iterations    : {tsne.n_iter_}")

# --- 3-D embedding ---
tsne_3d = TSNE(n_components=3, perplexity=20, n_iterations=800, random_state=0)
Y_3d = tsne_3d.fit_transform(X)    # shape (n_samples, 3)

# --- Tune perplexity (typical range 5 – 50) ---
# Low perplexity → captures very local structure, may fragment clusters
tsne_local = TSNE(perplexity=5, n_iterations=800, random_state=0)

# High perplexity → more global view, smoother layout
tsne_global = TSNE(perplexity=50, n_iterations=800, random_state=0)

# --- Auto learning rate (scales with n_samples, recommended for large data) ---
tsne_auto = TSNE(perplexity=30, learning_rate="auto", n_iterations=1000, random_state=0)
Y_auto = tsne_auto.fit_transform(X)

# --- PCA pre-whitening pipeline (standard practice for high-D data) ---
from ml_scratch.decomposition import PCA

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)           # reduce to 50-D first
Y = TSNE(perplexity=30, random_state=0).fit_transform(X_pca)
```

**When to use t-SNE:**
- Exploratory data visualisation of high-dimensional datasets.
- Inspecting cluster quality before applying a clustering algorithm.
- Understanding the manifold structure of embeddings (e.g. neural-network features).

**Practical tips:**
- Always run with at least 500–1000 iterations for stable layouts.
- Pre-reduce to ~50 PCA components before t-SNE when `n_features > 100`.
- Different random seeds produce topologically similar but geometrically rotated layouts — distances between clusters are not globally meaningful.

### FastICA

FastICA (Hyvärinen & Oja, 2000) recovers statistically independent source signals from a
set of mixed observations — the classic *blind source separation* problem.  Unlike PCA
(which finds uncorrelated directions that maximise variance), ICA maximises
non-Gaussianity of the projections as a proxy for independence.

```python
from ml_scratch.decomposition import FastICA

# --- Blind source separation -------------------------------------------------
# Separate three mixed audio-like signals into their independent sources
ica = FastICA(n_components=3, random_state=0)
S_recovered = ica.fit_transform(X_mixed)   # shape (n_samples, 3)

# --- Feature extraction / dimensionality reduction ---------------------------
# Extract 10 independent features from a 50-dimensional dataset
ica = FastICA(n_components=10, random_state=0)
X_ica = ica.fit_transform(X)               # shape (n_samples, 10)

# Reconstruct approximate original signals
X_approx = ica.inverse_transform(X_ica)   # shape (n_samples, 50)

# --- Algorithm variants ------------------------------------------------------
# Deflation: extract components one at a time (more stable, slower)
ica_def = FastICA(n_components=5, algorithm="deflation", random_state=0)
S = ica_def.fit_transform(X)

# Different contrast functions
ica_exp  = FastICA(fun="exp",  random_state=0)   # super-Gaussian sources
ica_cube = FastICA(fun="cube", random_state=0)   # fast, less robust
S = ica_exp.fit_transform(X)

# --- PCA pre-whitening pipeline ----------------------------------------------
from ml_scratch.decomposition import PCA
# First reduce noise with PCA, then extract independent components
X_pca  = PCA(n_components=20).fit_transform(X)
S_ica  = FastICA(n_components=10, random_state=0).fit_transform(X_pca)
```

| Parameter   | Default      | Options                        | Effect                                    |
|-------------|--------------|--------------------------------|-------------------------------------------|
| `algorithm` | `'parallel'` | `'parallel'`, `'deflation'`    | Deflation is more robust on ill-conditioned data |
| `fun`       | `'logcosh'`  | `'logcosh'`, `'exp'`, `'cube'` | `'logcosh'` is safest; `'cube'` is fastest |
| `whiten`    | `True`       | `True`, `False`                | PCA whitening before extraction (recommended) |
| `max_iter`  | `200`        | int                            | Increase to 500+ for hard mixing problems |

**When to use FastICA:**
- Recovering independent audio, image, or sensor sources from mixed measurements.
- Feature extraction when sources are non-Gaussian (finance, biomedical signals).
- Post-processing PCA components to remove residual dependencies.
- `fun='logcosh'` (default) is the most robust; switch to `'exp'` for sparse/sparse-spike sources.

### NMF

NMF (Non-negative Matrix Factorization, Lee & Seung 2001) decomposes a non-negative
matrix `V` (shape `n_samples × n_features`) into two non-negative factors `W` and `H`
such that `V ≈ W @ H`.  Because all entries remain non-negative, the learned components
are additive parts — making them inherently interpretable (topics in text, spectral
components in audio, localised features in images).

Two solvers are available:

| Solver | Algorithm | Beta-loss support | Typical use |
|--------|-----------|-------------------|-------------|
| `'mu'` | Multiplicative Update (Lee & Seung) | `'frobenius'`, `'kl'` | General; use `'kl'` for count data |
| `'als'`| Alternating Least Squares | `'frobenius'` only | Faster convergence on dense data |

```python
from ml_scratch.decomposition import NMF

# --- Basic factorisation -------------------------------------------------------
nmf = NMF(n_components=5, random_state=0)
W = nmf.fit_transform(X)          # (n_samples, 5) — activation matrix
H = nmf.components_               # (5, n_features) — basis / dictionary

print(f"Reconstruction error : {nmf.reconstruction_err_:.4f}")
print(f"Converged in         : {nmf.n_iter_} iterations")

# Reconstruct the original matrix
X_approx = nmf.inverse_transform(W)   # shape (n_samples, n_features), all ≥ 0

# Encode new (held-out) data with the fitted H
W_new = nmf.transform(X_test)         # (n_test, 5)

# --- Solver comparison ---------------------------------------------------------
nmf_mu  = NMF(n_components=5, solver="mu",  random_state=0).fit(X)
nmf_als = NMF(n_components=5, solver="als", random_state=0).fit(X)
print(f"MU  err={nmf_mu.reconstruction_err_:.3f}  iters={nmf_mu.n_iter_}")
print(f"ALS err={nmf_als.reconstruction_err_:.3f}  iters={nmf_als.n_iter_}")

# --- KL-divergence loss (good for count / histogram data) ---------------------
# 'kl' beta-loss is only available with solver='mu'
nmf_kl = NMF(n_components=5, solver="mu", beta_loss="kl", random_state=0)
nmf_kl.fit(X_counts)               # X_counts should be non-negative integers

# --- Regularisation -----------------------------------------------------------
# L1 on H → sparse dictionary components (topic-like sparsity)
nmf_l1 = NMF(n_components=5, l1_reg_H=0.05, random_state=0)
W_l1 = nmf_l1.fit_transform(X)
print(f"H sparsity: {(nmf_l1.components_ < 1e-4).mean():.2%}")

# L2 on W → smoother activations
nmf_l2 = NMF(n_components=5, l2_reg_W=0.1, random_state=0)
W_l2 = nmf_l2.fit_transform(X)

# --- Topic modelling analogy ---------------------------------------------------
# Each row of H is a latent topic; top features per row are the topic keywords.
top_per_topic = [
    vocab[i] for i in nmf.components_[topic_id].argsort()[::-1][:5]
    for topic_id in range(nmf.n_components_)
]
```

| Parameter    | Default       | Options                       | Effect                                              |
|--------------|---------------|-------------------------------|-----------------------------------------------------|
| `solver`     | `'mu'`        | `'mu'`, `'als'`               | ALS converges faster; MU supports KL loss           |
| `beta_loss`  | `'frobenius'` | `'frobenius'`, `'kl'`         | KL suits sparse count data (requires `solver='mu'`) |
| `l1_reg_H`   | `0.0`         | float ≥ 0                     | Encourages sparse columns in H                      |
| `l2_reg_W`   | `0.0`         | float ≥ 0                     | Regularises activation magnitudes                   |
| `max_iter`   | `200`         | int                           | Increase for difficult / high-rank factorisations   |

**When to use NMF:**
- Text / topic modelling: rows of `W` are document–topic mixtures; columns of `H` are topic–word distributions.
- Spectral analysis: decompose spectrograms into additive spectral components.
- Image parts: learn localised, additive filters (faces → eyes + nose + mouth).
- Collaborative filtering: users and items share a latent non-negative factor space.
- Use `beta_loss='kl'` for count matrices (word frequencies, pixel histograms); `'frobenius'` otherwise.

### MLP Neural Network

```python
from ml_scratch.neural_network import MLPClassifier, MLPRegressor

# --- Classification (multi-class, softmax + cross-entropy) ---
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),   # two hidden layers
    activation="relu",
    learning_rate=1e-3,
    n_iterations=300,
    batch_size=32,
    random_state=42,
)
clf.fit(X_train, y_train)
print(f"Accuracy : {accuracy(y_test, clf.predict(X_test)):.4f}")

# Per-class probabilities (shape: n_samples × n_classes)
proba = clf.predict_proba(X_test)

# Training curve
print(f"Final loss: {clf.loss_history_[-1]:.4f}")

# --- Regression (linear output, MSE loss) ---
reg = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="tanh",
    learning_rate=5e-4,
    n_iterations=500,
    batch_size=64,
    random_state=0,
)
reg.fit(X_train, y_train)
print(f"R²: {r2_score(y_test, reg.predict(X_test)):.4f}")
```

### Preprocessing

Preprocessing transformers follow the same `fit` / `transform` / `fit_transform` API as all other estimators in this library and compose naturally in manual pipelines.

#### StandardScaler

Standardises each feature to zero mean and unit variance.  Features with zero variance are left unchanged (scale replaced by 1 to avoid `NaN`).

```python
from ml_scratch.preprocessing import StandardScaler

ss = StandardScaler()
X_train_s = ss.fit_transform(X_train)   # learn mean/std and apply
X_test_s  = ss.transform(X_test)        # apply learned parameters
X_orig    = ss.inverse_transform(X_train_s)   # undo scaling
```

#### MinMaxScaler

Scales features into a fixed range (default `[0, 1]`).

```python
from ml_scratch.preprocessing import MinMaxScaler

# Scale pixel intensities to [-1, 1] for neural-net input
mms = MinMaxScaler(feature_range=(-1, 1))
X_scaled = mms.fit_transform(X_train)
X_orig   = mms.inverse_transform(X_scaled)
```

#### LabelEncoder

Encodes a 1-D label array as consecutive integers `[0, n_classes − 1]`.

```python
from ml_scratch.preprocessing import LabelEncoder

le = LabelEncoder()
y_enc = le.fit_transform(y)               # e.g. ['cat','dog'] → [0, 1]
print(le.classes_, le.n_classes_)
y_dec = le.inverse_transform(y_enc)       # back to original labels
```

#### OneHotEncoder

Expands categorical columns into binary indicator columns.  Pass `drop='first'` (default) to remove the reference category and avoid multicollinearity, or `drop=None` to keep all columns.

```python
from ml_scratch.preprocessing import OneHotEncoder

# Mixed categorical design matrix
X_cat = np.array([['male', 'urban'], ['female', 'rural'], ['female', 'urban']])

ohe = OneHotEncoder(drop='first')         # drop reference category
X_enc = ohe.fit_transform(X_cat)
print(ohe.get_feature_names_out())        # ['x0_male', 'x1_urban']

ohe_full = OneHotEncoder(drop=None)       # keep all columns
X_full = ohe_full.fit_transform(X_cat)   # shape (3, 4)
```

#### Combining scalers and encoders

```python
import numpy as np
from ml_scratch.preprocessing import StandardScaler, OneHotEncoder

# Numeric columns
ss = StandardScaler()
X_num = ss.fit_transform(X_numeric_train)

# Categorical columns
ohe = OneHotEncoder(drop='first')
X_cat = ohe.fit_transform(X_categorical_train)

# Single feature matrix ready for any estimator
X_ready = np.hstack([X_num, X_cat])
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
