"""Tests for data validation."""

import pandas as pd
import pytest

from src.data.validate_data import (
    validate_dataframe,
    validate_dtypes,
    validate_ranges,
    validate_schema,
)


# --- schema ---------------------------------------------------------------------

def test_validate_schema_valid():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert validate_schema(df, ["a", "b"]) == []


def test_validate_schema_missing_columns():
    df = pd.DataFrame({"a": [1]})
    problems = validate_schema(df, ["a", "b"])
    assert problems and "b" in problems[0]


def test_validate_schema_single_column_as_string():
    df = pd.DataFrame({"a": [1]})
    assert validate_schema(df, "a") == []
    assert validate_schema(df, "b")


def test_validate_schema_invalid_rule_type():
    with pytest.raises(ValueError, match="required_columns"):
        validate_schema(pd.DataFrame({"a": [1]}), {"a": True})


# --- data types -----------------------------------------------------------------

def test_validate_dtypes_pass():
    df = pd.DataFrame({"age": [30.0], "name": ["alice"], "active": [True]})
    rules = {"age": "numeric", "name": "string", "active": "boolean"}
    assert validate_dtypes(df, rules) == []


def test_validate_dtypes_violations():
    df = pd.DataFrame({"age": ["thirty"], "name": [42]})
    problems = validate_dtypes(df, {"age": "numeric", "name": "string"})
    assert len(problems) == 2
    assert "age" in problems[0] and "numeric" in problems[0]


def test_validate_dtypes_group_names_are_case_insensitive():
    df = pd.DataFrame({"age": [1.0]})
    assert validate_dtypes(df, {"age": "NUMERIC"}) == []


def test_validate_dtypes_missing_column_left_to_schema_check():
    df = pd.DataFrame({"a": [1]})
    assert validate_dtypes(df, {"b": "numeric"}) == []


def test_validate_dtypes_pandas_dtype_name():
    df = pd.DataFrame({"a": [1, 2]})  # int64
    assert validate_dtypes(df, {"a": "int64"}) == []
    assert validate_dtypes(df, {"a": "datetime64[ns]"})


def test_validate_dtypes_boolean_is_not_numeric():
    df = pd.DataFrame({"flag": [True, False]})
    assert validate_dtypes(df, {"flag": "boolean"}) == []
    assert validate_dtypes(df, {"flag": "numeric"})


def test_validate_dtypes_unknown_group():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Invalid expected dtype"):
        validate_dtypes(df, {"a": "numericish"})


def test_validate_dtypes_non_string_rule_value():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Invalid expected dtype"):
        validate_dtypes(df, {"a": 64})


def test_validate_dtypes_invalid_rules_type():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="expected_dtypes"):
        validate_dtypes(df, ["age"])


# --- ranges ---------------------------------------------------------------------

def test_validate_ranges_pass():
    df = pd.DataFrame({"age": [25, 60]})
    assert validate_ranges(df, {"age": {"min": 0, "max": 120}}) == []


def test_validate_ranges_single_bound():
    df = pd.DataFrame({"age": [25]})
    assert validate_ranges(df, {"age": {"min": 0}}) == []
    assert validate_ranges(df, {"age": {"max": 120}}) == []


def test_validate_ranges_violations():
    df = pd.DataFrame({"age": [-5, 30, 200]})
    problems = validate_ranges(df, {"age": {"min": 0, "max": 120}})
    assert len(problems) == 1
    assert "below min" in problems[0] and "above max" in problems[0]
    assert "age" in problems[0]


def test_validate_ranges_ignores_missing_values():
    df = pd.DataFrame({"age": [25.0, None, 60.0]})
    assert validate_ranges(df, {"age": {"min": 0, "max": 120}}) == []


def test_validate_ranges_non_numeric_column():
    df = pd.DataFrame({"name": ["alice"]})
    problems = validate_ranges(df, {"name": {"min": 0}})
    assert problems and "not numeric" in problems[0]


def test_validate_ranges_boolean_column_reported_not_numeric():
    """Booleans are validated via dtype groups, not range bounds."""
    df = pd.DataFrame({"flag": [True, False]})
    problems = validate_ranges(df, {"flag": {"min": 0, "max": 1}})
    assert len(problems) == 1 and "not numeric" in problems[0]


def test_validate_ranges_missing_column_left_to_schema_check():
    df = pd.DataFrame({"a": [1]})
    assert validate_ranges(df, {"b": {"min": 0}}) == []


def test_validate_ranges_min_greater_than_max():
    df = pd.DataFrame({"age": [30]})
    with pytest.raises(ValueError, match="greater than"):
        validate_ranges(df, {"age": {"min": 100, "max": 10}})


def test_validate_ranges_non_numeric_bound():
    df = pd.DataFrame({"age": [30]})
    with pytest.raises(ValueError, match="must be a number"):
        validate_ranges(df, {"age": {"min": "young"}})


def test_validate_ranges_unknown_bound():
    df = pd.DataFrame({"age": [30]})
    with pytest.raises(ValueError, match="Supported bounds"):
        validate_ranges(df, {"age": {"minimum": 0}})


def test_validate_ranges_invalid_rule_structure():
    df = pd.DataFrame({"age": [30]})
    with pytest.raises(ValueError, match="range rule"):
        validate_ranges(df, {"age": 0})


# --- combined rules -------------------------------------------------------------

def test_validate_dataframe_no_rules():
    df = pd.DataFrame({"a": [1]})
    assert validate_dataframe(df, None) == []
    assert validate_dataframe(df, {}) == []


def test_validate_dataframe_empty_list_rules_rejected():
    """Empty-but-wrong-typed rules fail loudly instead of being ignored."""
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="expected_dtypes"):
        validate_dataframe(df, {"expected_dtypes": []})
    with pytest.raises(ValueError, match="ranges"):
        validate_dataframe(df, {"ranges": []})


def test_validate_dataframe_unknown_rule():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Unknown validation rule"):
        validate_dataframe(df, {"columns": ["a"]})


def test_validate_dataframe_combines_all_rules():
    df = pd.DataFrame({"age": [200]})
    rules = {
        "required_columns": ["id"],
        "expected_dtypes": {"age": "numeric"},
        "ranges": {"age": {"min": 0, "max": 120}},
    }
    problems = validate_dataframe(df, rules)
    assert len(problems) == 2  # missing 'id' + age above max
    assert any("id" in p for p in problems)
    assert any("age" in p for p in problems)


def test_validate_dataframe_empty_dataframe_with_rules():
    empty = pd.DataFrame({"age": pd.Series([], dtype=float)})
    rules = {
        "required_columns": ["age", "id"],
        "expected_dtypes": {"age": "numeric"},
        "ranges": {"age": {"min": 0, "max": 120}},
    }
    problems = validate_dataframe(empty, rules)
    assert len(problems) == 1  # only the missing 'id'
