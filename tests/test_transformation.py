"""Tests for transformation modules."""

import pandas as pd
import pytest

from src.transformation.encoding import encode
from src.transformation.scaling import scale


# --- scaling --------------------------------------------------------------------

def test_scale_standard():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = scale(df, method="standard")
    assert out["a"].mean() == pytest.approx(0.0)


def test_scale_minmax_range():
    df = pd.DataFrame({"a": [1.0, 5.0, 10.0]})
    out = scale(df, method="minmax")
    assert out["a"].min() == pytest.approx(0.0)
    assert out["a"].max() == pytest.approx(1.0)


def test_scale_robust_centers_on_median():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0]})
    out = scale(df, method="robust")
    assert out["a"].median() == pytest.approx(0.0)


def test_scale_invalid_method():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="Invalid scaling method"):
        scale(df, method="zscore")


def test_scale_unknown_column():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="not found"):
        scale(df, columns=["missing"])


def test_scale_non_numeric_column():
    df = pd.DataFrame({"name": ["a", "b"]})
    with pytest.raises(ValueError, match="non-numeric"):
        scale(df, columns=["name"])


def test_scale_invalid_columns_type():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="list of column names"):
        scale(df, columns="a")


def test_scale_empty_dataframe():
    empty = pd.DataFrame({"a": pd.Series([], dtype=float)})
    out = scale(empty)
    assert out.empty and list(out.columns) == ["a"]


# --- encoding -------------------------------------------------------------------

def test_encode_onehot():
    df = pd.DataFrame({"c": ["x", "y", "x"]})
    out = encode(df, method="onehot")
    assert set(out.columns) == {"c_x", "c_y"}


def test_encode_ordinal():
    df = pd.DataFrame({"c": ["x", "y", "x"]})
    out = encode(df, method="ordinal")
    assert out["c"].dtype.kind in "iu"


def test_encode_invalid_method():
    df = pd.DataFrame({"c": ["x"]})
    with pytest.raises(ValueError, match="Invalid encoding method"):
        encode(df, method="target")


def test_encode_unknown_column():
    df = pd.DataFrame({"c": ["x"]})
    with pytest.raises(ValueError, match="not found"):
        encode(df, columns=["missing"])


def test_encode_invalid_columns_type():
    df = pd.DataFrame({"c": ["x"]})
    with pytest.raises(ValueError, match="list of column names"):
        encode(df, columns="c")


def test_encode_empty_dataframe_preserves_columns():
    empty = pd.DataFrame({"c": pd.Series([], dtype=object)})
    out = encode(empty)
    assert out.empty and list(out.columns) == ["c"]


def test_scale_empty_dataframe_still_validates_columns():
    """Column validation runs even when there are no rows (L1)."""
    empty = pd.DataFrame({"a": pd.Series([], dtype=float)})
    with pytest.raises(ValueError, match="not found"):
        scale(empty, columns=["nope"])


def test_encode_empty_dataframe_still_validates_columns():
    empty = pd.DataFrame({"c": pd.Series([], dtype=object)})
    with pytest.raises(ValueError, match="not found"):
        encode(empty, columns=["nope"])


def test_encode_category_onehot():
    df = pd.DataFrame({"c": pd.Series(["x", "y", "x"], dtype="category")})
    out = encode(df, method="onehot")
    assert set(out.columns) == {"c_x", "c_y"}


def test_encode_category_ordinal():
    df = pd.DataFrame({"c": pd.Series(["x", "y", "x"], dtype="category")})
    out = encode(df, method="ordinal")
    assert out["c"].tolist() == [0, 1, 0]
