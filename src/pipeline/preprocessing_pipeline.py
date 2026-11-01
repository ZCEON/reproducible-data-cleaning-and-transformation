"""Preprocessing pipeline: configuration-driven cleaning and transformation.

Execution order:
    ``clean()``      - drop high-missing columns -> normalize strings
                       -> impute -> drop duplicates
    ``transform()``  - scale numeric columns -> encode categorical columns

Normalizing before deduplicating ensures case/whitespace variants of the
same record are recognized as duplicates.
"""

import pandas as pd

from src.cleaning.duplicates import drop_duplicates
from src.cleaning.inconsistent_values import normalize_strings
from src.cleaning.missing_values import drop_high_missing, impute
from src.transformation.encoding import encode
from src.transformation.scaling import scale


def clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cleaning stage: missing values, normalization, duplicates.

    Raises:
        ValueError: if the missing-value threshold would remove every
            column of a non-empty input.
    """
    cfg = config.get("cleaning", {})

    mv = cfg.get("missing_values", {})
    threshold = mv.get("threshold", 0.5)
    input_columns = len(df.columns)
    df = drop_high_missing(df, threshold)
    if input_columns and not len(df.columns):
        raise ValueError(
            "Cleaning removed every column: all columns exceed "
            f"'cleaning.missing_values.threshold' ({threshold}). "
            "Lower the threshold or check the input data."
        )

    iv = cfg.get("inconsistent_values", {})
    df = normalize_strings(
        df,
        strip=iv.get("strip_whitespace", True),
        case=iv.get("normalize_case", "lower"),
    )

    df = impute(df, mv.get("strategy", "median"))
    df = drop_duplicates(df, cfg.get("duplicates", {}).get("keep", "first"))
    return df


def transform(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Transformation stage: scaling and encoding."""
    cfg = config.get("transformation", {})

    sc = cfg.get("scaling", {})
    df = scale(df, sc.get("method", "standard"), sc.get("columns") or None)

    enc = cfg.get("encoding", {})
    df = encode(df, enc.get("method", "onehot"), enc.get("columns") or None)
    return df


def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the full pipeline: ``clean()`` followed by ``transform()``."""
    return transform(clean(df, config), config)
