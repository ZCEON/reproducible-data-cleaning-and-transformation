"""Handle missing values."""

import pandas as pd

SUPPORTED_STRATEGIES = ("mean", "median", "mode", "drop")


def _validate_threshold(threshold: float) -> float:
    """Return a valid threshold or raise a clear ValueError."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"Invalid missing-value threshold: {threshold!r}. "
            "Expected a number between 0 and 1."
        )
    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Invalid missing-value threshold: {threshold}. "
            "Expected a number between 0 and 1."
        )
    return float(threshold)


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns whose missing fraction exceeds ``threshold``.

    Empty data frames are returned unchanged (missing fractions are
    undefined for zero rows).

    Raises:
        ValueError: if ``threshold`` is not a number in [0, 1].
    """
    threshold = _validate_threshold(threshold)
    if df.empty:
        return df
    missing_fraction = df.isna().mean().fillna(0.0)
    return df.loc[:, missing_fraction <= threshold]


def impute(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute missing values using the configured strategy.

    Strategies:
        * ``mean`` / ``median`` - fill numeric columns with that statistic.
        * ``mode`` - fill every column with its most frequent value.
        * ``drop`` - remove rows containing any missing value.

    Raises:
        ValueError: if ``strategy`` is unknown, or if any column contains
            only missing values (nothing can be imputed from).
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Invalid imputation strategy: {strategy!r}. "
            f"Supported strategies: {', '.join(repr(s) for s in SUPPORTED_STRATEGIES)}."
        )
    if strategy == "drop":
        return df.dropna()
    if df.empty:
        return df
    all_missing = [col for col in df.columns if df[col].isna().all()]
    if all_missing:
        raise ValueError(
            f"Cannot impute missing values with strategy {strategy!r}: "
            f"column(s) {all_missing} contain only missing values. "
            "Drop them first (e.g. lower 'cleaning.missing_values.threshold') "
            "or use strategy 'drop'."
        )
    out = df.copy()
    if strategy in ("mean", "median"):
        numeric = out.select_dtypes(include="number").columns
        out[numeric] = out[numeric].fillna(getattr(out[numeric], strategy)())
    else:  # mode
        for col in out.columns:
            out[col] = out[col].fillna(out[col].mode().iloc[0])
    return out
