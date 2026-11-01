# data/raw/

This directory holds **input CSV files (read-only)**. The pipeline never
writes here — `scripts/run_pipeline.py` verifies up front that the raw,
interim and processed directories are distinct and aborts otherwise.

## Example dataset

`synthetic_example_orders.csv` is **synthetic demonstration data** created
only to show this pipeline working end to end. It is **not real collected
data** and must not be treated as such.

It was hand-designed to contain, in one small file:

| Aspect | Where |
|---|---|
| Missing values | `rating` (4 rows), `region`, `product`, `units`, `unit_price`, `customer_name` (1 row each) |
| Duplicate rows | one byte-identical pair; one pair that only becomes duplicate *after* string normalization |
| Inconsistent text | leading/trailing/double spaces and mixed casing in `customer_name` and `region` |
| Numeric columns | `units`, `unit_price`, `rating` |
| Categorical columns | `region`, `product` (+ free-text `customer_name`) |
| Validation examples | `units = -1` and `rating = 6.4` violate the example rules used in the notebook |

Everything else in `data/raw/` (files not named `synthetic_*`) is ignored
by Git along with the outputs in `data/interim/` and `data/processed/`.

## License

`synthetic_example_orders.csv` was generated for this project (no
third-party data was copied into it) and is covered by the project's
[MIT License](../../LICENSE).
