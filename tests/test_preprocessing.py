"""Unit tests for ml_scratch.preprocessing (scalers and encoders)."""

import numpy as np
import pytest

from ml_scratch.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_X():
    return np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])


@pytest.fixture
def constant_col_X():
    """Array where the second column is constant (edge case for scalers)."""
    return np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])


# ===========================================================================
# StandardScaler
# ===========================================================================

class TestStandardScaler:
    def test_fit_stores_mean_and_scale(self, simple_X):
        ss = StandardScaler()
        ss.fit(simple_X)
        expected_mean = simple_X.mean(axis=0)
        expected_std = simple_X.std(axis=0)
        np.testing.assert_allclose(ss.mean_, expected_mean)
        np.testing.assert_allclose(ss.scale_, expected_std)

    def test_transform_zero_mean(self, simple_X):
        ss = StandardScaler()
        Xs = ss.fit_transform(simple_X)
        np.testing.assert_allclose(Xs.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_transform_unit_std(self, simple_X):
        ss = StandardScaler()
        Xs = ss.fit_transform(simple_X)
        np.testing.assert_allclose(Xs.std(axis=0), [1.0, 1.0], atol=1e-10)

    def test_inverse_transform_round_trip(self, simple_X):
        ss = StandardScaler()
        X_back = ss.fit_transform(simple_X)
        X_orig = ss.inverse_transform(X_back)
        np.testing.assert_allclose(X_orig, simple_X)

    def test_constant_feature_no_nan(self, constant_col_X):
        ss = StandardScaler()
        Xs = ss.fit_transform(constant_col_X)
        assert not np.any(np.isnan(Xs)), "NaN produced for zero-variance feature."
        # constant column should map to zeros
        np.testing.assert_array_equal(Xs[:, 1], [0.0, 0.0, 0.0])

    def test_n_features_in(self, simple_X):
        ss = StandardScaler()
        ss.fit(simple_X)
        assert ss.n_features_in_ == 2

    def test_transform_before_fit_raises(self):
        ss = StandardScaler()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ss.transform(np.array([[1.0, 2.0]]))

    def test_wrong_n_features_raises(self, simple_X):
        ss = StandardScaler()
        ss.fit(simple_X)
        with pytest.raises(ValueError, match="2 features"):
            ss.transform(np.array([[1.0, 2.0, 3.0]]))

    def test_1d_input_raises(self):
        ss = StandardScaler()
        with pytest.raises(ValueError, match="2-D"):
            ss.fit(np.array([1.0, 2.0, 3.0]))

    def test_fit_returns_self(self, simple_X):
        ss = StandardScaler()
        assert ss.fit(simple_X) is ss


# ===========================================================================
# MinMaxScaler
# ===========================================================================

class TestMinMaxScaler:
    def test_transform_range_0_1(self, simple_X):
        mms = MinMaxScaler()
        Xm = mms.fit_transform(simple_X)
        np.testing.assert_allclose(Xm.min(axis=0), [0.0, 0.0])
        np.testing.assert_allclose(Xm.max(axis=0), [1.0, 1.0])

    def test_custom_feature_range(self, simple_X):
        mms = MinMaxScaler(feature_range=(-1, 1))
        Xm = mms.fit_transform(simple_X)
        np.testing.assert_allclose(Xm.min(axis=0), [-1.0, -1.0])
        np.testing.assert_allclose(Xm.max(axis=0), [1.0, 1.0])

    def test_inverse_transform_round_trip(self, simple_X):
        mms = MinMaxScaler()
        Xm = mms.fit_transform(simple_X)
        X_orig = mms.inverse_transform(Xm)
        np.testing.assert_allclose(X_orig, simple_X)

    def test_constant_feature_no_nan(self, constant_col_X):
        mms = MinMaxScaler()
        Xm = mms.fit_transform(constant_col_X)
        assert not np.any(np.isnan(Xm)), "NaN produced for constant feature."

    def test_invalid_feature_range_raises(self):
        with pytest.raises(ValueError, match="low < high"):
            MinMaxScaler(feature_range=(1, 0))

    def test_transform_before_fit_raises(self):
        mms = MinMaxScaler()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            mms.transform(np.array([[1.0, 2.0]]))

    def test_n_features_mismatch_raises(self, simple_X):
        mms = MinMaxScaler()
        mms.fit(simple_X)
        with pytest.raises(ValueError, match="2 features"):
            mms.transform(np.ones((3, 3)))

    def test_fit_returns_self(self, simple_X):
        mms = MinMaxScaler()
        assert mms.fit(simple_X) is mms


# ===========================================================================
# LabelEncoder
# ===========================================================================

class TestLabelEncoder:
    def test_integer_encoding(self):
        le = LabelEncoder()
        y = np.array(["cat", "dog", "bird", "cat"])
        y_enc = le.fit_transform(y)
        # sorted unique: bird=0, cat=1, dog=2
        expected = np.array([1, 2, 0, 1])
        np.testing.assert_array_equal(y_enc, expected)

    def test_classes_sorted(self):
        le = LabelEncoder()
        le.fit(np.array([3, 1, 2, 1]))
        np.testing.assert_array_equal(le.classes_, [1, 2, 3])

    def test_inverse_transform_round_trip(self):
        le = LabelEncoder()
        y = np.array(["a", "b", "c", "a", "c"])
        y_enc = le.fit_transform(y)
        y_dec = le.inverse_transform(y_enc)
        np.testing.assert_array_equal(y_dec, y)

    def test_n_classes(self):
        le = LabelEncoder()
        le.fit(np.array([0, 1, 2, 3]))
        assert le.n_classes_ == 4

    def test_unseen_label_raises(self):
        le = LabelEncoder()
        le.fit(np.array(["a", "b"]))
        with pytest.raises(ValueError, match="Unknown label"):
            le.transform(np.array(["c"]))

    def test_transform_before_fit_raises(self):
        le = LabelEncoder()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            le.transform(np.array(["x"]))

    def test_2d_input_raises(self):
        le = LabelEncoder()
        with pytest.raises(ValueError, match="1-D"):
            le.fit(np.array([[1, 2], [3, 4]]))

    def test_numeric_labels(self):
        le = LabelEncoder()
        y = np.array([10, 20, 30, 10])
        y_enc = le.fit_transform(y)
        np.testing.assert_array_equal(y_enc, [0, 1, 2, 0])


# ===========================================================================
# OneHotEncoder
# ===========================================================================

class TestOneHotEncoder:
    def _make_X(self):
        return np.array([["red", "S"], ["blue", "M"], ["green", "L"], ["red", "L"]])

    def test_output_shape_drop_first(self):
        ohe = OneHotEncoder(drop="first")
        X = self._make_X()
        X_enc = ohe.fit_transform(X)
        # col0: 3 cats -> 2 cols; col1: 3 cats -> 2 cols => 4 total
        assert X_enc.shape == (4, 4)

    def test_output_shape_no_drop(self):
        ohe = OneHotEncoder(drop=None)
        X = self._make_X()
        X_enc = ohe.fit_transform(X)
        # col0: 3 cats; col1: 3 cats => 6 total
        assert X_enc.shape == (4, 6)

    def test_binary_values(self):
        ohe = OneHotEncoder(drop=None)
        X = self._make_X()
        X_enc = ohe.fit_transform(X)
        unique_vals = np.unique(X_enc)
        np.testing.assert_array_equal(unique_vals, [0.0, 1.0])

    def test_row_sums_drop_first(self):
        """Each row should have exactly one 1 per original feature."""
        ohe = OneHotEncoder(drop=None)
        X = np.array([["a"], ["b"], ["c"]])
        X_enc = ohe.fit_transform(X)
        np.testing.assert_array_equal(X_enc.sum(axis=1), [1, 1, 1])

    def test_get_feature_names_out(self):
        ohe = OneHotEncoder(drop=None)
        ohe.fit(np.array([["red"], ["blue"], ["green"]]))
        names = ohe.get_feature_names_out()
        assert names == ["x0_blue", "x0_green", "x0_red"]

    def test_unseen_category_raises(self):
        ohe = OneHotEncoder(drop=None)
        ohe.fit(np.array([["a"], ["b"]]))
        with pytest.raises(ValueError, match="unknown category"):
            ohe.transform(np.array([["c"]]))

    def test_transform_before_fit_raises(self):
        ohe = OneHotEncoder()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ohe.transform(np.array([["a"]]))

    def test_invalid_drop_arg_raises(self):
        with pytest.raises(ValueError, match="drop"):
            OneHotEncoder(drop="last")
