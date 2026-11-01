# Reproducible Data Cleaning and Transformation

A configuration-driven pipeline that validates, cleans, and transforms
tabular CSV data reproducibly. Every behavior — validation rules,
imputation strategy, scaling, encoding, paths — is declared in
`config/config.yaml`, so the same input files and the same configuration
always produce the same outputs.

## Main Features

- **Configuration-driven** — switch imputation, scaling, encoding, or
  validation rules without touching code (`config/config.yaml`).
- **Validation gate before cleaning** — optional schema, data-type, and
  numeric-range rules; files that fail (or fail to load) are logged and
  skipped without stopping the batch.
- **Cleaning** — drop high-missing columns → normalize strings
  (leading/trailing whitespace + case) → impute (`mean|median|mode|drop`)
  → deduplicate. Normalization runs *before* dedup, so `' Alice '` and
  `'alice'` are recognized as the same record.
- **Transformation** — scale numerics (`standard|minmax|robust`), encode
  categoricals (`onehot|ordinal`), including pandas `category` columns.
- **Raw-data protection** — the runner verifies before any I/O that the
  raw, interim, and processed directories are distinct; `data/raw/` can
  never be overwritten.
- **Reproducible** — no randomness anywhere in the pipeline; same inputs
  + same config ⇒ byte-identical outputs (same platform; CSV line
  endings follow `os.linesep`).
- **Tested** — 123 pytest tests, including end-to-end runner tests that
  write only to temporary directories.
- **Demonstrated end-to-end** — ships a clearly labelled synthetic
  example dataset and a fully executed EDA notebook.

## Architecture and Project Structure

```
config/config.yaml ──┐ (rules & options)
                     ▼
data/raw/*.csv → scripts/run_pipeline.py
                     │
        ┌────────────▼────────────┐
        │ 1. src/data             │  load_csv → validate_dataframe
        └────────────┬────────────┘  (schema / dtypes / ranges)
                     ▼
        ┌────────────────────────┐
        │ 2. clean()             │  src/cleaning: drop high-missing cols
        │    src/pipeline        │  → normalize strings → impute → dedup
        └────────────┬───────────┘
                     ▼  data/interim/<name>.csv
        ┌────────────────────────┐
        │ 3. transform()         │  src/transformation: scale numerics
        │    src/pipeline        │  → encode categoricals
        └────────────┬───────────┘
                     ▼  data/processed/<name>.csv
```

| Path | Responsibility |
|------|----------------|
| `src/data/` | `load_csv` (CSV → DataFrame), `validate_dataframe` (required columns, dtype groups, inclusive ranges) |
| `src/cleaning/` | `drop_high_missing`, `impute`, `normalize_strings`, `drop_duplicates` |
| `src/transformation/` | `scale` (Standard/MinMax/Robust), `encode` (one-hot / ordinal) |
| `src/pipeline/` | `clean()` → `transform()` orchestration and `preprocess()` |
| `src/utils/` | logging setup |
| `scripts/run_pipeline.py` | CLI entry point: file discovery, validation gate, stage execution, output writing, exit codes |
| `tests/` | unit tests per module + integration tests for the pipeline and runner |
| `notebooks/` | executed walkthrough on the synthetic example |
| `config/config.yaml` | the single source of runtime behavior |
| `docs/methodology.md` | precise description of every stage as implemented |

```
.
├── config/
│   └── config.yaml      # paths, validation rules, cleaning & transformation options
├── data/
│   ├── raw/             # input CSVs (read-only) + synthetic example dataset
│   ├── interim/         # output of the cleaning stage
│   └── processed/       # output of the transformation stage
├── notebooks/
│   └── exploratory_data_analysis.ipynb   # full workflow on the example data
├── src/
│   ├── data/            # loading & validation (schema, dtypes, ranges)
│   ├── cleaning/        # missing values, duplicates, string normalization
│   ├── transformation/  # scaling & encoding
│   ├── pipeline/        # clean() -> transform() orchestration
│   └── utils/           # logging
├── tests/               # unit + integration tests (pytest)
├── scripts/
│   └── run_pipeline.py  # CLI entry point
├── results/             # reserved for future figures/reports/metrics
├── docs/
│   └── methodology.md
├── pyproject.toml       # packaging, dependencies, pytest configuration
└── requirements.txt     # pinned, verified dependency versions
```

## Installation

Requirements: **Python >= 3.10** (verified on Python 3.13.7).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs the project in editable mode (all `src` imports resolve
without any `sys.path` tweaks) plus runtime dependencies
(`pandas`, `numpy`, `scikit-learn`, `PyYAML`) and the test tool
(`pytest`). Add the notebook toolchain with
`pip install -e ".[dev,notebook]"`.

For exact, verified versions instead of ranges:

```bash
pip install -r requirements.txt
```

`pyproject.toml` is the canonical dependency declaration;
`requirements.txt` pins the same core dependencies to the exact versions
this project was verified with.

## Adding a CSV Dataset

1. **Drop the file into `data/raw/`** — any name ending in `.csv` in any
   case (`.csv`, `.CSV`, `.Csv` all work, on every platform).
2. **Keep raw data read-only in practice** — the runner refuses to start
   (exit `2`) if any output path overlaps `data/raw/`, so inputs are
   never overwritten.
3. **Optionally enable validation rules** in `config/config.yaml`
   (`required_columns`, `expected_dtypes`, `ranges`). Files that violate
   them are skipped with a logged reason per problem; the run exits `1`.
   With the shipped config (all rules empty) every loadable file passes.
4. **Mind type inference** — `read_csv` guesses dtypes: leading zeros in
   ID columns are lost (`00123` → `123`), and a column containing any
   free text is treated as text (then normalized/encoded). See
   [Limitations](#limitations).
5. **Run the pipeline** (next section); your file appears under the same
   name in `data/interim/` and `data/processed/`.
6. **Git note** — `.gitignore` ignores everything in `data/raw/` except
   the shipped `synthetic_*.csv` example and its label, so your own
   datasets (and all outputs) stay out of the repository by default.

## Configuration

All behavior is driven by `config/config.yaml`:

- `paths`: `raw_data`, `interim_data`, `processed_data` — must be three
  distinct directories (checked before any I/O).
- `validation`: optional rules (`required_columns`, `expected_dtypes`,
  `ranges`) checked before cleaning; groups are
  `numeric | boolean | string | datetime` or an exact pandas dtype.
- `cleaning.missing_values`: strategy (`mean|median|mode|drop`) and
  column-drop threshold. `mean`/`median` fill **numeric columns only**;
  use `mode` to fill text columns too (see methodology).
- `cleaning.duplicates`: `keep: first|last|false`.
- `cleaning.inconsistent_values`: whitespace stripping and
  `normalize_case: lower|upper|false` (applies to object/string/category
  columns).
- `transformation.scaling`: `standard|minmax|robust`.
- `transformation.encoding`: `onehot|ordinal`.

Invalid configuration values raise `ValueError` with a message naming the
supported options; the CLI catches these up front and exits `2`.

## Running the Pipeline

```bash
python scripts/run_pipeline.py                 # default config (config/config.yaml)
python scripts/run_pipeline.py --help          # usage, options, exit codes
python scripts/run_pipeline.py --config PATH   # alternative configuration file
```

Stages:

1. Load every CSV from `data/raw/` (`.csv` matched case-insensitively).
2. Validate it against the `validation` rules in `config/config.yaml`
   (required columns, data types, numeric ranges).
3. **Clean** (drop high-missing columns → normalize strings → impute →
   deduplicate) → write to `data/interim/`.
4. **Transform** (scale → encode) → write to `data/processed/`.

Exit codes:

| Code | Meaning |
|------|---------|
| `0` | success (including: no input files) |
| `1` | one or more files failed validation or processing |
| `2` | usage or configuration error (bad arguments, missing/unparseable config file, invalid option values, overlapping paths, missing input directory) |

Failure handling (all verified):

- The configuration is validated **up front** (structure, option values,
  path overlaps): an invalid configuration exits `2` with a message naming
  the supported options, instead of failing identically on every file.
  Column lists that reference missing columns are data-dependent and
  surface per file (exit `1`).
- Files that fail validation **or fail to load/process** are logged and
  skipped; a single bad file does not stop the batch. The process exits
  with code `1` if any file failed, `0` otherwise.
- Before processing, the runner checks that `raw_data`, `interim_data`
  and `processed_data` are three distinct directories; any overlap aborts
  the run (exit `2`) so raw data can never be overwritten.
- A configured input directory that does not exist exits `2`.
- If `data/raw/` is empty it logs a warning and exits with code `0`.
- Cleaning fails with `ValueError` if the threshold would drop every
  column (no silently empty outputs).

## Example Input and Output

The shipped example — **`data/raw/synthetic_example_orders.csv`, synthetic
demonstration data, not real collected data** (full description in
`data/raw/README.md`) — contains 18 rows with missing values (9 cells),
one exact duplicate pair, one normalization-only duplicate pair,
inconsistent casing/whitespace, numeric and categorical columns, and two
planted out-of-range values for demonstrating validation.

Input (first 4 of 18 lines, verbatim):

```csv
customer_name,region,product,units,unit_price,rating
 Alice Smith ,North,Widget A,3,19.99,4.5
alice smith,NORTH,Widget A,3,19.99,4.5
bob jones,South,Widget A,2,19.99,
bob jones,South,Widget A,2,19.99,
```

Run result (real log, timestamps, logger prefix and output paths shortened):

```
INFO | Processing synthetic_example_orders.csv
INFO | Loaded 18 rows x 6 columns
INFO | Cleaned: 16 rows x 6 columns -> data/interim/synthetic_example_orders.csv
INFO | Transformed: 16 rows x 16 columns -> data/processed/synthetic_example_orders.csv
INFO | Processed 1 file(s) successfully.
exit: 0
```

Interim output (first 4 rows): whitespace/case normalized, duplicates
collapsed, missing `rating` filled with the median (4.0):

```csv
customer_name,region,product,units,unit_price,rating
alice smith,north,widget a,3.0,19.99,4.5
bob jones,south,widget a,2.0,19.99,4.0
carol lee,east,widget b,4.0,9.99,5.0
carol lee,east,gadget c,1.0,149.5,2.5
```

Processed output (first 3 of 16 rows, 5 of 16 columns): z-scored
numerics + one-hot indicators (`1`/`0`):

```csv
units,unit_price,rating,customer_name_alice smith,customer_name_bob jones
0.07886548694517029,-0.5153349534262319,0.6024158104133038,1,0
-0.34175044342907124,-0.5153349534262319,0.042029010028835245,0,1
0.4994814173194118,-0.6885422312772467,1.1628026107977725,0,0
```

Verified before/after summary (computed values):

| | raw | interim (cleaned) | processed |
|---|---|---|---|
| rows | 18 | 16 | 16 |
| columns | 6 | 6 | 16 |
| missing cells | 9 | 3 | 0 |
| duplicate rows | 1 | 0 | 0 |

The 3 remaining missing cells after cleaning are text values (filled
neither by `median` nor dropped); one-hot encoding later represents them
as all-zero rows — hence 0 in the processed output.

## Running the Tests

```bash
pytest
# or: python -m pytest tests/
```

Verified result: **123 passed** (unit tests per module plus integration
tests covering the pipeline and the runner). Tests are configured in
`pyproject.toml` (`testpaths`, `pythonpath`) and use an inline fixture
config, so they pass regardless of your local `config/config.yaml` and
write only to temporary directories.

Continuous integration (`.github/workflows/ci.yml`) runs the same suite
on every push and pull request across Python 3.10–3.13.

## Running the Notebook

`notebooks/exploratory_data_analysis.ipynb` demonstrates the complete
workflow on the synthetic example dataset: load → shape and columns →
missing values and duplicates → dtypes and basic statistics → validation
(default config and stricter example rules) → preprocessing →
before/after comparison. It states explicitly that the data is synthetic.

After installation with the notebook extra
(`pip install -e ".[dev,notebook]"`) every cell runs top to bottom —
interactively, or headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/exploratory_data_analysis.ipynb
```

All 12 code cells (21 total) execute without errors; executed outputs
are stored in the file.

## Limitations

- **Leading zeros in ID columns** are lost to `read_csv` type inference
  (`00123` → `123`); pass explicit `dtype=` to `load_csv` when
  identifiers must remain strings.
- **`mean`/`median` imputation fills numeric columns only** (the
  documented default behavior). Text missing values become all-zero
  one-hot rows or ordinal code `-1`; if a text column is *excluded* from
  encoding, its missing values reach `data/processed/` unchanged. Use
  `mode` to fill text columns.
- **Normalization is leading/trailing-whitespace + case only** — internal
  whitespace runs are not collapsed and no Unicode accent folding is
  applied.
- **Row-wise normalization** (`.map`) is Python-level and O(n); very
  large datasets will feel it.
- **Single-process, in-memory** — datasets larger than available RAM are
  not supported.
- **No load-time dtype control from config** — read options can only be
  passed programmatically via `load_csv(..., **kwargs)`.

## Future Improvements

- Write figures/reports/metrics into the reserved `results/` directories.
- Configurable load-time dtypes (would also remove the leading-zero
  caveat) and per-column imputation strategies.
- Vectorized normalization for large frames; optional parallel per-file
  processing.
- A console-script entry point (`pip install`-provided `run-pipeline`).

## Methodology

See [docs/methodology.md](docs/methodology.md) for the precise,
code-accurate description of every stage.

## License

This project is released under the [MIT License](LICENSE) —
Copyright (c) 2026 ZCEON. The synthetic example dataset in
`data/raw/` is original demonstration data covered by the same
license; the pipeline uses no third-party datasets or copied code.
Its dependencies (pandas, scikit-learn, NumPy — BSD-3-Clause;
PyYAML — MIT) all ship permissive licenses compatible with MIT.
