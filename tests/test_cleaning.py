"""Tests for cleaning modules."""

import pandas as pd
import pytest

from src.cleaning.duplicates import drop_duplicates
from src.cleaning.inconsistent_values import normalize_strings
from src.cleaning.missing_values import drop_high_missing, impute


# --- missing values -------------------------------------------------------------

def test_drop_high_missing():
    df = pd.DataFrame({"a": [1, None, None], "b": [1, 2, 3]})
    out = drop_high_missing(df, threshold=0.5)
    assert "a" not in out.columns and "b" in out.columns


def test_drop_high_missing_all_missing_column_dropped():
    df = pd.DataFrame({"empty": [None, None], "ok": [1, 2]})
    out = drop_high_missing(df, threshold=0.5)
    assert list(out.columns) == ["ok"]


def test_drop_high_missing_empty_dataframe_keeps_columns():
    empty = pd.DataFrame({"a": pd.Series([], dtype=float)})
    out = drop_high_missing(empty, threshold=0.5)
    assert list(out.columns) == ["a"]


@pytest.mark.parametrize("bad", [1.5, -0.1, "high", None, True])
def test_drop_high_missing_invalid_threshold(bad):
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="threshold"):
        drop_high_missing(df, threshold=bad)


def test_impute_median():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = impute(df, strategy="median")
    assert out["a"].isna().sum() == 0


def test_impute_mean():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = impute(df, strategy="mean")
    assert out["a"].iloc[1] == pytest.approx(2.0)


def test_impute_mode():
    df = pd.DataFrame({"a": ["x", "x", None, "y"]})
    out = impute(df, strategy="mode")
    assert out["a"].isna().sum() == 0
    assert out["a"].iloc[2] == "x"


def test_impute_invalid_strategy():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="Invalid imputation strategy"):
        impute(df, strategy="constant")


def test_impute_invalid_strategy_on_empty_dataframe():
    with pytest.raises(ValueError, match="Invalid imputation strategy"):
        impute(pd.DataFrame(), strategy="average")


def test_impute_all_missing_column_raises():
    df = pd.DataFrame({"a": [None, None], "b": [1.0, 2.0]})
    for strategy in ("mean", "median", "mode"):
        with pytest.raises(ValueError, match="only missing values"):
            impute(df, strategy=strategy)


def test_impute_all_missing_drop_strategy_returns_empty():
    df = pd.DataFrame({"a": [None, None], "b": [1.0, 2.0]})
    out = impute(df, strategy="drop")
    assert out.empty


def test_impute_empty_dataframe():
    empty = pd.DataFrame({"a": pd.Series([], dtype=float)})
    out = impute(empty, strategy="median")
    assert out.empty and list(out.columns) == ["a"]


# --- duplicates -----------------------------------------------------------------

def test_drop_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 3]})
    out = drop_duplicates(df)
    assert len(out) == 2


def test_drop_duplicates_keep_last():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 5]})
    out = drop_duplicates(df, keep="last")
    # The second copy of the duplicated row survives (index 1, not 0).
    assert list(out.index) == [1, 2]


def test_drop_duplicates_invalid_keep():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="keep"):
        drop_duplicates(df, keep="middle")


# --- inconsistent values --------------------------------------------------------

def test_normalize_strings():
    df = pd.DataFrame({"a": ["  Hello ", "WORLD"]})
    out = normalize_strings(df)
    assert list(out["a"]) == ["hello", "world"]


def test_normalize_case_lower():
    df = pd.DataFrame({"a": ["HeLLO", "WORLD"]})
    out = normalize_strings(df, case="lower")
    assert list(out["a"]) == ["hello", "world"]


def test_normalize_case_upper():
    df = pd.DataFrame({"a": ["hello", "World"]})
    out = normalize_strings(df, case="upper")
    assert list(out["a"]) == ["HELLO", "WORLD"]


def test_normalize_case_false_keeps_case_but_strips():
    df = pd.DataFrame({"a": ["  Hello ", "WORLD"]})
    out = normalize_strings(df, case=False)
    assert list(out["a"]) == ["Hello", "WORLD"]


def test_normalize_case_none_keeps_case():
    df = pd.DataFrame({"a": ["Hello"]})
    out = normalize_strings(df, case=None)
    assert list(out["a"]) == ["Hello"]


def test_normalize_case_invalid_value():
    df = pd.DataFrame({"a": ["Hello"]})
    with pytest.raises(ValueError, match="normalize_case"):
        normalize_strings(df, case="title")


def test_normalize_case_true_is_not_supported():
    df = pd.DataFrame({"a": ["Hello"]})
    with pytest.raises(ValueError, match="normalize_case"):
        normalize_strings(df, case=True)


def test_normalize_preserves_non_string_values():
    df = pd.DataFrame({"mixed": pd.Series([" Alice ", 42, 3.5, True, None], dtype=object)})
    out = normalize_strings(df)
    result = list(out["mixed"])
    assert result[0] == "alice"
    assert result[1] == 42 and isinstance(result[1], int)
    assert result[2] == 3.5
    assert result[3] is True
    assert result[4] is None


def test_normalize_leaves_numeric_columns_untouched():
    df = pd.DataFrame({"n": [1, 2], "f": [0.1, 0.2]})
    out = normalize_strings(df)
    pd.testing.assert_frame_equal(out, df)


def test_normalize_category_column():
    df = pd.DataFrame({"c": pd.Series([" X ", "y", None], dtype="category")})
    out = normalize_strings(df)
    assert out["c"].dropna().tolist() == ["x", "y"]
    assert out["c"].isna().sum() == 1


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_normalize_invalid_strip_type(bad):
    df = pd.DataFrame({"a": ["x"]})
    with pytest.raises(ValueError, match="strip_whitespace"):
        normalize_strings(df, strip=bad)
