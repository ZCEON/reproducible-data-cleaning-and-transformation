"""Validate a DataFrame against configurable rules.

All ``validate_*`` functions return a list of human-readable problems;
an empty list means the data passed the check. Invalid *rule definitions*
(not data problems) raise ``ValueError``.
"""

import pandas as pd

#: Logical data-type groups accepted in ``validation.expected_dtypes``.
DTYPE_GROUPS = ("numeric", "boolean", "string", "datetime")

#: Top-level rules accepted in the ``validation`` section of config.yaml.
SUPPORTED_RULES = ("required_columns", "expected_dtypes", "ranges")


def _matches_group(series: pd.Series, group: str) -> bool:
    """Return True if ``series`` matches the logical type ``group``."""
    if group == "numeric":
        return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    if group == "boolean":
        return pd.api.types.is_bool_dtype(series)
    if group == "string":
        # "empty" (all-NaN / zero-row) columns cannot be disproved as strings.
        return pd.api.types.infer_dtype(series, skipna=True) in ("string", "empty")
    if group == "datetime":
        return pd.api.types.is_datetime64_any_dtype(series)
    raise ValueError(  # pragma: no cover - guarded by validate_dtypes
        f"Unknown data-type group: {group!r}. Supported groups: {DTYPE_GROUPS}."
    )


def validate_schema(df: pd.DataFrame, expected_columns) -> list[str]:
    """Return schema violations: expected columns missing from ``df``.

    ``expected_columns`` may be a list or a single column name string.

    Raises:
        ValueError: if ``expected_columns`` has an invalid type.
    """
    if expected_columns is None:
        expected_columns = []
    if isinstance(expected_columns, str):
        expected_columns = [expected_columns]
    if not isinstance(expected_columns, (list, tuple, set)):
        raise ValueError(
            f"Invalid 'required_columns' rule: {expected_columns!r}. "
            "Expected a list of column names."
        )
    problems = []
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        problems.append(f"Missing required column(s): {missing}.")
    return problems


def validate_dtypes(df: pd.DataFrame, expected_dtypes: dict) -> list[str]:
    """Return data-type violations for the listed columns.

    Rules map a column name to either a logical group (``'numeric'``,
    ``'boolean'``, ``'string'``, ``'datetime'``) or a pandas dtype name
    such as ``'int64'`` or ``'datetime64[ns]'``. Columns not present in
    ``df`` are left for :func:`validate_schema` to report.

    Raises:
        ValueError: if a rule definition is invalid.
    """
    if not isinstance(expected_dtypes, dict):
        raise ValueError(
            f"Invalid 'expected_dtypes' rule: {expected_dtypes!r}. "
            "Expected a mapping of column -> dtype."
        )
    problems = []
    for col, expected in expected_dtypes.items():
        if col not in df.columns:
            continue
        if not isinstance(expected, str):
            raise ValueError(
                f"Invalid expected dtype for column {col!r}: {expected!r} "
                f"(expected a string: one of {DTYPE_GROUPS} or a pandas dtype name)."
            )
        group = expected.lower()
        if group in DTYPE_GROUPS:
            if not _matches_group(df[col], group):
                problems.append(
                    f"Column {col!r} has dtype {df[col].dtype} but expected {group}."
                )
            continue
        try:
            target = pd.api.types.pandas_dtype(expected)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid expected dtype for column {col!r}: {expected!r}. "
                f"Supported groups: {DTYPE_GROUPS}; or a pandas dtype name "
                "(e.g. 'float64', 'datetime64[ns]')."
            ) from None
        if not pd.api.types.is_dtype_equal(df[col].dtype, target):
            problems.append(
                f"Column {col!r} has dtype {df[col].dtype} but expected {expected}."
            )
    return problems


def validate_ranges(df: pd.DataFrame, ranges: dict) -> list[str]:
    """Return range violations for numeric columns.

    Rules map a column name to inclusive bounds, e.g.
    ``{'age': {'min': 0, 'max': 120}}``; either bound may be omitted.
    Missing values are ignored (they are a separate concern). Boolean
    columns are not considered numeric for range checks (they are checked
    via the ``boolean`` dtype group instead).
    Columns not present in ``df`` are left for :func:`validate_schema`.

    Raises:
        ValueError: if a rule definition is invalid.
    """
    if not isinstance(ranges, dict):
        raise ValueError(
            f"Invalid 'ranges' rule: {ranges!r}. "
            "Expected a mapping of column -> {min, max}."
        )
    problems = []
    for col, bounds in ranges.items():
        if col not in df.columns:
            continue
        if not isinstance(bounds, dict) or not bounds:
            raise ValueError(
                f"Invalid range rule for column {col!r}: {bounds!r}. "
                "Expected a mapping like {'min': 0, 'max': 120}."
            )
        unknown = sorted(set(bounds) - {"min", "max"})
        if unknown:
            raise ValueError(
                f"Invalid range bound(s) for column {col!r}: {unknown}. "
                "Supported bounds: 'min', 'max'."
            )
        lo, hi = bounds.get("min"), bounds.get("max")
        for name, value in (("min", lo), ("max", hi)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(
                    f"Invalid {name!r} bound for column {col!r}: {value!r} must be a number."
                )
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"Invalid range for column {col!r}: min ({lo}) is greater than max ({hi})."
            )
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            problems.append(
                f"Column {col!r} has a range rule but is not numeric (dtype {series.dtype})."
            )
            continue
        values = series.dropna()
        violations = []
        if lo is not None and (values < lo).any():
            below = values[values < lo]
            violations.append(f"{len(below)} value(s) below min {lo} (e.g. {below.iloc[0]})")
        if hi is not None and (values > hi).any():
            above = values[values > hi]
            violations.append(f"{len(above)} value(s) above max {hi} (e.g. {above.iloc[0]})")
        if violations:
            problems.append(f"Column {col!r} fails range check: {'; '.join(violations)}.")
    return problems


def validate_dataframe(df: pd.DataFrame, rules: dict | None) -> list[str]:
    """Run every rule configured under the ``validation`` config section.

    Supported keys: ``required_columns``, ``expected_dtypes``, ``ranges``.

    Rule values must be either ``None`` (rule disabled) or a valid
    definition; empty-but-wrong-typed values (e.g. a list where a mapping
    is required) raise instead of being silently ignored.

    Raises:
        ValueError: if the rules mapping contains unknown or invalid entries.
    """
    rules = rules or {}
    if not isinstance(rules, dict):
        raise ValueError(
            f"Invalid 'validation' section: {rules!r}. Expected a mapping of rules."
        )
    unknown = sorted(set(rules) - set(SUPPORTED_RULES))
    if unknown:
        raise ValueError(
            f"Unknown validation rule(s): {unknown}. Supported rules: {sorted(SUPPORTED_RULES)}."
        )
    dtype_rules = rules.get("expected_dtypes")
    range_rules = rules.get("ranges")
    problems: list[str] = []
    problems += validate_schema(df, rules.get("required_columns"))
    problems += validate_dtypes(df, {} if dtype_rules is None else dtype_rules)
    problems += validate_ranges(df, {} if range_rules is None else range_rules)
    return problems
