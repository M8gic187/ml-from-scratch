"""Preprocessing demo: StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder.

Shows end-to-end usage of all four transformers in realistic scenarios:
  1. Standardise a mixed-scale numeric dataset before feeding it to a model.
  2. Min-max-scale features into a custom range for neural-net input.
  3. Encode a multi-class label vector with LabelEncoder.
  4. One-hot-encode a categorical design matrix with OneHotEncoder.
"""

import numpy as np

from ml_scratch.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. StandardScaler — numeric features with very different scales
# ---------------------------------------------------------------------------
section("1. StandardScaler")

rng = np.random.default_rng(42)
# Age (20-60), Salary (20k-80k), Hours/week (10-60)
X_train = rng.uniform([20, 20_000, 10], [60, 80_000, 60], size=(200, 3))
X_test = rng.uniform([20, 20_000, 10], [60, 80_000, 60], size=(50, 3))

ss = StandardScaler()
X_train_s = ss.fit_transform(X_train)
X_test_s = ss.transform(X_test)

print(f"Train — mean before scaling: {X_train.mean(axis=0).round(1)}")
print(f"Train — mean after  scaling: {X_train_s.mean(axis=0).round(3)}")
print(f"Train — std  after  scaling: {X_train_s.std(axis=0).round(3)}")
print(f"Test  — std  after  scaling: {X_test_s.std(axis=0).round(3)}")

# Inverse transform restores original values
X_restored = ss.inverse_transform(X_train_s)
print(f"Inverse-transform error (max abs): {np.abs(X_restored - X_train).max():.2e}")

# ---------------------------------------------------------------------------
# 2. MinMaxScaler — scaling pixel intensities to [-1, 1] for a CNN
# ---------------------------------------------------------------------------
section("2. MinMaxScaler  →  range [-1, 1]")

pixels = rng.integers(0, 256, size=(500, 784)).astype(float)  # 500 fake 28×28 images

mms = MinMaxScaler(feature_range=(-1, 1))
pixels_scaled = mms.fit_transform(pixels)

print(f"Original pixel range : [{pixels.min():.0f},  {pixels.max():.0f}]")
print(f"Scaled   pixel range : [{pixels_scaled.min():.2f}, {pixels_scaled.max():.2f}]")
print(f"All values in [-1,1] : {bool(np.all(pixels_scaled >= -1) and np.all(pixels_scaled <= 1))}")

# ---------------------------------------------------------------------------
# 3. LabelEncoder — multi-class target variable
# ---------------------------------------------------------------------------
section("3. LabelEncoder")

species = np.array(
    ["setosa", "versicolor", "virginica", "setosa", "virginica", "versicolor"]
)
le = LabelEncoder()
y_encoded = le.fit_transform(species)
y_decoded = le.inverse_transform(y_encoded)

print(f"Original  : {species}")
print(f"Classes   : {le.classes_}  (n={le.n_classes_})")
print(f"Encoded   : {y_encoded}")
print(f"Decoded   : {y_decoded}")
print(f"Round-trip: {np.all(y_decoded == species)}")

# ---------------------------------------------------------------------------
# 4. OneHotEncoder — categorical design matrix
# ---------------------------------------------------------------------------
section("4. OneHotEncoder")

X_cat = np.array(
    [
        ["male",   "high",   "urban"],
        ["female", "low",    "rural"],
        ["female", "medium", "urban"],
        ["male",   "low",    "suburban"],
        ["female", "high",   "rural"],
    ]
)

# drop='first' removes the reference category per column
ohe_drop = OneHotEncoder(drop="first")
X_ohe_drop = ohe_drop.fit_transform(X_cat)
print("drop='first'  →  shape:", X_ohe_drop.shape)
print("Feature names:", ohe_drop.get_feature_names_out())
print(X_ohe_drop)

# drop=None keeps all columns
ohe_full = OneHotEncoder(drop=None)
X_ohe_full = ohe_full.fit_transform(X_cat)
print("\ndrop=None    →  shape:", X_ohe_full.shape)
print("Feature names:", ohe_full.get_feature_names_out())
print(X_ohe_full)

# ---------------------------------------------------------------------------
# 5. Combining scalers and encoders in a simple pipeline
# ---------------------------------------------------------------------------
section("5. Manual pipeline: numeric + categorical features")

# 50 samples with mixed features
n = 50
ages = rng.uniform(18, 70, size=(n, 1))
incomes = rng.uniform(10_000, 100_000, size=(n, 1))
genders = rng.choice(["male", "female"], size=(n, 1))
cities = rng.choice(["urban", "suburban", "rural"], size=(n, 1))

# Scale numeric columns
ss_pipe = StandardScaler()
X_num = ss_pipe.fit_transform(np.hstack([ages, incomes]))

# Encode categorical columns
ohe_pipe = OneHotEncoder(drop="first")
X_cat_part = ohe_pipe.fit_transform(np.hstack([genders, cities]))

# Concatenate into a single feature matrix
X_final = np.hstack([X_num, X_cat_part])
print(f"Final feature matrix shape: {X_final.shape}")
print(f"  Numeric  columns : 2  (age, income — standardised)")
print(f"  Categoric columns: {X_cat_part.shape[1]}  (gender, city — OHE, drop first)")
print(f"  Total            : {X_final.shape[1]}")
