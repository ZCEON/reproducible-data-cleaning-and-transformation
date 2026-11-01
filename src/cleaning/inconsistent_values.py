"""Normalize inconsistent values (whitespace, casing)."""

from functools import partial

import pandas as pd

SUPPORTED_CASES = ("lower", "upper")


def _validate_case(case: str | bool | None) -> str | None:
    """Return the normalized case setting (``None`` = leave case unchanged).

    Raises:
        ValueError: if ``case`` is not 'lower', 'upper', false or None.
    """
    if case is False or case is None:
        return None
    if case in SUPPORTED_CASES:
        return case
    raise ValueError(
        f"Invalid normalize_case value: {case!r}. "
        "Supported values: 'lower', 'upper', or false."
    )


def _normalize_value(value, strip: bool, case: str | None):
    """Normalize a single value; non-string values are returned unchanged."""
    if not isinstance(value, str):
        return value
    if strip:
        value = value.strip()
    if case == "lower":
        value = value.lower()
    elif case == "upper":
        value = value.upper()
    return value


def normalize_strings(
    df: pd.DataFrame,
    strip: bool = True,
    case: str | bool | None = "lower",
) -> pd.DataFrame:
    """Strip whitespace and/or change the case of string values.

    Only actual string values inside object/string/category columns are
    modified; non-string data (numbers, booleans, NaN) is preserved as-is.

    Args:
        df: Input data frame.
        strip: Remove leading/trailing whitespace when True.
        case: ``'lower'``, ``'upper'``, or ``false``/``None`` to keep the
            original case.

    Raises:
        ValueError: if ``strip`` is not a boolean or ``case`` is not a
            supported value.
    """
    if not isinstance(strip, bool):
        raise ValueError(
            f"Invalid strip_whitespace value: {strip!r}. Expected true or false."
        )
    case = _validate_case(case)
    out = df.copy()
    normalizer = partial(_normalize_value, strip=strip, case=case)
    for col in out.select_dtypes(include=["object", "string", "category"]).columns:
        out[col] = out[col].map(normalizer, na_action="ignore")
    return out
