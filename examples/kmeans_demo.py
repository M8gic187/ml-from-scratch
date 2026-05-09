"""Demo: K-Means on a 3-cluster synthetic dataset."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_scratch.clustering import KMeans

rng = np.random.default_rng(1)

centers = [[-4, -4], [0, 4], [4, -4]]
X = np.vstack([
    rng.multivariate_normal(c, np.eye(2) * 0.8, size=100)
    for c in centers
])

model = KMeans(k=3, random_state=42)
labels = model.fit_predict(X)

print(f"Converged in  : {model.n_iter_} iterations")
print(f"Inertia       : {model.inertia_:.2f}")
print(f"Learned centers:")
for i, c in enumerate(model.centroids_):
    true_c = centers[i]
    print(f"  Cluster {i}: [{c[0]:+.3f}, {c[1]:+.3f}]  (true: {true_c})")
