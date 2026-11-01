# Methodology

This document describes the cleaning and transformation methodology as
**implemented** in this project. Everything below corresponds to existing
code; see `config/config.yaml` for the configurable options.

## 1. Data Loading

- Raw data is read from `data/raw/` with `pandas.read_csv`; files are
  matched case-insensitively on the `.csv` extension (`.CSV`, `.Csv`
  all work, on every platform).
- The raw directory is never written to. Before any file is processed,
  the runner verifies that `raw_data`, `interim_data` and `processed_data`
  are three distinct directories (no equality, no nesting) and aborts
  with exit code 2 otherwise, so outputs can never overwrite raw input.
- The pipeline writes only to `data/interim/` and `data/processed/`.
- **Type-inference limitation:** `read_csv` infers dtypes, so ID columns
  with leading zeros (`00123`) are read as integers and lose their zeros.
  Pass explicit `dtype=` arguments to `load_csv` when identifiers must be
  preserved as strings.

## 2. Validation (before cleaning)

Validation runs on every file before any cleaning step. Each check returns
a list of human-readable problems; rule *definitions* that are themselves
invalid raise `ValueError`.

Configured under `validation:` in `config/config.yaml`:

- **Schema checks** (`required_columns`): every listed column must exist.
- **Data-type checks** (`expected_dtypes`): a column must match a logical
  group (`numeric`, `boolean`, `string`, `datetime`) or an exact pandas
  dtype (e.g. `int64`).
- **Range checks** (`ranges`): numeric values must lie within inclusive
  `min`/`max` bounds; missing values are ignored. Boolean columns are not
  treated as numeric for range checks - use the `boolean` dtype group to
  validate them instead.

Files with validation problems are logged and skipped, as are files that
fail to load or fail during processing (the process exits with code 1 if
any file failed); a failing file does not stop the rest of the batch.

## 3. Cleaning

Steps run in this exact order (`src/pipeline/preprocessing_pipeline.clean`):

1. **Missing-value column drop** — columns whose missing fraction exceeds
   `missing_values.threshold` are removed. If this would remove *every*
   column of a non-empty input, cleaning fails with a `ValueError` rather
   than silently emitting an empty table.
2. **String normalization** — leading/trailing whitespace is stripped and
   case is normalized (`normalize_case: lower | upper | false`) for string
   values in object/string/category columns. Non-string values (numbers,
   booleans, NaN) are preserved untouched.
3. **Imputation** — remaining missing values are filled using
   `missing_values.strategy`:
   - `mean` / `median`: fill **numeric columns only** (this is the actual
     behavior — these statistics are undefined for text). Missing values in
     non-numeric columns are *not* filled at this stage; they carry into
     transformation, where one-hot encoding represents them as an all-zero
     row and ordinal encoding as code `-1`. If a non-numeric column is
     excluded from encoding, its missing values reach `data/processed/`
     unchanged. Use `mode` to fill text/categorical columns as well.
   - `mode`: fills every column with its most frequent value.
   - `drop`: removes rows containing any missing value.

   Columns containing *only* missing values cannot be imputed and raise a
   `ValueError` explaining how to fix it (lower the threshold or use
   `drop`).
4. **Duplicate removal** — exact duplicate rows are removed using
   `duplicates.keep` (`first`, `last`, `false`).

Normalization runs **before** duplicate removal so that case/whitespace
variants of the same record (`' Alice '` vs `'alice'`) are recognized as
duplicates.

## 4. Transformation

Run after cleaning (`src/pipeline.preprocessing_pipeline.transform`):

1. **Scaling** — numeric columns are scaled with `standard`
   (z-score), `minmax` (0–1), or `robust` (median/IQR) scaling. An empty
   `columns` list means "all numeric columns".
2. **Encoding** — object/string/category columns are encoded with
   `onehot` or `ordinal` encoding. An empty `columns` list means "all
   object/string/category columns". For one-hot encoding, unobserved
   levels of a pandas `category` column still produce a dummy column.

Non-numeric columns listed for scaling, or unknown column names in either
step, raise `ValueError`.

## 5. Outputs

| Path | Contents |
|------|----------|
| `data/interim/` | Cleaned data (post-cleaning, pre-transformation) |
| `data/processed/` | Final data (cleaned + scaled + encoded) |

`results/figures/`, `results/reports/`, and `results/metrics/` are
reserved for future use; the current pipeline does not write them.

## 6. Reproducibility

- The pipeline contains no randomness: the same input files and the same
  `config/config.yaml` always produce the same outputs.
- The raw data in `data/raw/` is treated as immutable; re-running the
  pipeline overwrites only `data/interim/` and `data/processed/`.
