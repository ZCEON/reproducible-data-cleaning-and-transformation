"""Encode categorical features."""

import pandas as pd

SUPPORTED_METHODS = ("onehot", "ordinal")


def encode(
    df: pd.DataFrame,
    method: str = "onehot",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Encode categorical columns via one-hot or ordinal encoding.

    Args:
        df: Input data frame.
        method: ``'onehot'`` or ``'ordinal'``.
        columns: Columns to encode. Empty/None = all object/string/category
            columns.

    Raises:
        ValueError: if ``method`` is unknown, ``columns`` is not a list of
            existing column names.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Invalid encoding method: {method!r}. "
            f"Supported methods: {', '.join(repr(m) for m in SUPPORTED_METHODS)}."
        )
    if columns is not None and not isinstance(columns, (list, tuple)):
        raise ValueError(
            f"Invalid 'columns' value for encoding: {columns!r}. "
            "Expected a list of column names."
        )
    cols = list(columns) if columns else list(
        df.select_dtypes(include=["object", "string", "category"]).columns
    )
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot encode column(s) not found in the data: {missing}.")
    if not cols or df.empty:
        return df.copy()
    if method == "onehot":
        return pd.get_dummies(df, columns=cols, dtype=int)
    out = df.copy()
    for col in cols:
        out[col] = out[col].astype("category").cat.codes
    return out
