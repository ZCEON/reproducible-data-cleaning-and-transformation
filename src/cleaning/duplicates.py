"""Remove duplicate rows."""

import pandas as pd


def drop_duplicates(df: pd.DataFrame, keep: str | bool = "first") -> pd.DataFrame:
    """Drop exact duplicate rows (run after string normalization).

    Raises:
        ValueError: if ``keep`` is not 'first', 'last' or false.
    """
    if keep not in ("first", "last") and keep is not False:
        raise ValueError(
            f"Invalid duplicate keep value: {keep!r}. "
            "Supported values: 'first', 'last', false."
        )
    return df.drop_duplicates(keep=keep)
