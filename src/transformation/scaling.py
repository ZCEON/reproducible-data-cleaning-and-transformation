"""Scale numeric features."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def scale(
    df: pd.DataFrame,
    method: str = "standard",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Scale numeric columns with the chosen scaler.

    Args:
        df: Input data frame.
        method: One of ``'standard'``, ``'minmax'``, ``'robust'``.
        columns: Columns to scale. Empty/None = all numeric columns.

    Raises:
        ValueError: if ``method`` is unknown, ``columns`` is not a list of
            existing column names, or a requested column is not numeric.
    """
    if method not in _SCALERS:
        raise ValueError(
            f"Invalid scaling method: {method!r}. "
            f"Supported methods: {', '.join(repr(m) for m in _SCALERS)}."
        )
    if columns is not None and not isinstance(columns, (list, tuple)):
        raise ValueError(
            f"Invalid 'columns' value for scaling: {columns!r}. "
            "Expected a list of column names."
        )
    cols = list(columns) if columns else list(df.select_dtypes("number").columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot scale column(s) not found in the data: {missing}.")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Cannot scale non-numeric column(s): {non_numeric}.")
    if not cols or df.empty:
        return df.copy()
    out = df.copy()
    scaler = _SCALERS[method]()
    out[cols] = scaler.fit_transform(out[cols])
    return out
