"""Integration tests for the complete preprocessing pipeline and runner.

The fixture config below is defined inline (not read from
``config/config.yaml``) so these tests stay independent of the live
configuration file: tuning the config must never break the test suite.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.validate_data import validate_dataframe
from src.pipeline.preprocessing_pipeline import clean, preprocess, transform
from scripts import run_pipeline as run_module

# Default-equivalent fixture configuration (intentionally NOT the live file).
CONFIG = {
    "paths": {
        "raw_data": "data/raw/",
        "interim_data": "data/interim/",
        "processed_data": "data/processed/",
    },
    "validation": {
        "required_columns": [],
        "expected_dtypes": {},
        "ranges": {},
    },
    "cleaning": {
        "missing_values": {"strategy": "median", "threshold": 0.5},
        "duplicates": {"keep": "first"},
        "inconsistent_values": {"strip_whitespace": True, "normalize_case": "lower"},
    },
    "transformation": {
        "scaling": {"method": "standard", "columns": []},
        "encoding": {"method": "onehot", "columns": []},
    },
}


def _sample_frame() -> pd.DataFrame:
    """Dirty sample: case/whitespace variants, a missing value, duplicates."""
    return pd.DataFrame(
        {
            "name": [" Alice ", "alice", "bob"],
            "age": [25.0, 25.0, None],
            "city": ["NY", "ny", "LA"],
        }
    )


# --- pipeline behavior ----------------------------------------------------------

def test_normalized_duplicates_are_removed():
    """Case/whitespace variants become duplicates only after normalization."""
    df = _sample_frame()
    # Without normalization the three rows are all distinct:
    assert len(df.drop_duplicates()) == 3

    out = preprocess(df, CONFIG)
    # ' Alice ' and 'alice' collapse into one row after normalization.
    assert len(out) == 2


def test_preprocess_equals_clean_then_transform():
    df = _sample_frame()
    direct = preprocess(df, CONFIG)
    staged = transform(clean(df.copy(), CONFIG), CONFIG)
    pd.testing.assert_frame_equal(direct, staged)


def test_pipeline_no_missing_values_remain():
    out = preprocess(_sample_frame(), CONFIG)
    assert out.isna().sum().sum() == 0


def test_pipeline_output_is_numeric_after_encoding():
    out = preprocess(_sample_frame(), CONFIG)
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)


def test_pipeline_with_custom_config_order():
    """Normalization happens before dedup inside clean()."""
    cfg = {
        "cleaning": {
            "missing_values": {"strategy": "median", "threshold": 0.5},
            "duplicates": {"keep": "first"},
            "inconsistent_values": {"strip_whitespace": True, "normalize_case": "upper"},
        },
        "transformation": {
            "scaling": {"method": "minmax", "columns": []},
            "encoding": {"method": "ordinal", "columns": []},
        },
    }
    df = _sample_frame()
    out = preprocess(df, cfg)
    assert len(out) == 2


def test_empty_dataframe_end_to_end():
    empty = pd.DataFrame({"name": pd.Series([], dtype=object), "age": pd.Series([], dtype=float)})
    out = preprocess(empty, CONFIG)
    assert isinstance(out, pd.DataFrame) and out.empty


def test_category_dtype_end_to_end():
    """category columns are normalized, deduplicated and encoded (M4)."""
    df = pd.DataFrame(
        {
            "name": pd.Series([" Alice ", "alice", "bob"], dtype="category"),
            "age": [25.0, 25.0, 30.0],
        }
    )
    out = preprocess(df, CONFIG)
    assert len(out) == 2
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)


def test_all_missing_column_raises_in_clean_stage():
    """threshold=1.0 keeps an all-missing column; impute must refuse it."""
    df = pd.DataFrame({"age": [25.0, None], "empty": [None, None]})
    cfg = {
        "cleaning": {
            "missing_values": {"strategy": "mean", "threshold": 1.0},
            "duplicates": {"keep": "first"},
            "inconsistent_values": {"strip_whitespace": True, "normalize_case": False},
        },
        "transformation": {"scaling": {"method": "standard", "columns": []},
                           "encoding": {"method": "onehot", "columns": []}},
    }
    with pytest.raises(ValueError, match="only missing values"):
        preprocess(df, cfg)


def test_clean_raises_when_every_column_would_be_dropped():
    """A threshold that removes all columns fails clearly (L4)."""
    df = pd.DataFrame({"a": [None, 1.0], "b": [None, None]})
    cfg = {
        "cleaning": {
            "missing_values": {"strategy": "median", "threshold": 0.0},
            "duplicates": {"keep": "first"},
            "inconsistent_values": {"strip_whitespace": True, "normalize_case": False},
        },
    }
    with pytest.raises(ValueError, match="removed every column"):
        clean(df, cfg)


def test_clean_zero_column_input_does_not_raise():
    """A frame that starts with no columns is not a threshold failure."""
    assert clean(pd.DataFrame(), CONFIG).empty


# --- validation gate ------------------------------------------------------------

def test_validation_gate_passes_valid_data():
    df = _sample_frame()
    rules = {
        "required_columns": ["name", "age", "city"],
        "expected_dtypes": {"name": "string", "age": "numeric", "city": "string"},
        "ranges": {"age": {"min": 0, "max": 120}},
    }
    assert validate_dataframe(df, rules) == []


def test_validation_gate_rejects_out_of_range_data():
    df = _sample_frame()
    df.loc[0, "age"] = 200
    rules = {"ranges": {"age": {"min": 0, "max": 120}}}
    problems = validate_dataframe(df, rules)
    assert len(problems) == 1 and "age" in problems[0]


# --- runner helpers -------------------------------------------------------------

def test_validate_output_paths_accepts_distinct_dirs():
    assert run_module.validate_output_paths(CONFIG) is None


@pytest.mark.parametrize(
    "collision",
    [
        {"raw_data": "data/raw/", "interim_data": "data/raw/"},          # equal
        {"raw_data": "data/raw/", "processed_data": "data/raw/nested/"},  # nested in raw
        {"raw_data": "data/raw/", "interim_data": "data/"},               # raw nested in interim
        {"interim_data": "data/shared/", "processed_data": "data/shared/"},
    ],
)
def test_validate_output_paths_rejects_overlap(collision):
    """Raw, interim and processed paths must never overlap (M2)."""
    paths = {**CONFIG["paths"], **collision}
    msg = run_module.validate_output_paths({"paths": paths})
    assert msg is not None and "collision" in msg


def test_validate_output_paths_missing_entries():
    assert run_module.validate_output_paths({}) is not None
    assert run_module.validate_output_paths({"paths": {"raw_data": "x/"}}) is not None


def test_find_input_files_case_insensitive(tmp_path):
    """Uppercase .csv extensions are picked up on every platform (L5)."""
    (tmp_path / "UPPER.CSV").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "lower.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "Mixed.CsV").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "dir.csv").mkdir()  # directory, not a file
    # Compare as a set: sort order of Paths differs between platforms
    # (Windows path comparison is case-insensitive).
    found = {p.name for p in run_module.find_input_files(tmp_path)}
    assert found == {"Mixed.CsV", "UPPER.CSV", "lower.csv"}


def test_find_input_files_missing_dir(tmp_path):
    assert run_module.find_input_files(tmp_path / "nope") == []


# --- runner end-to-end (writes only to pytest tmp dirs) -------------------------

def _write_runner_env(tmp_path: Path, config: dict) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "data" / "raw").mkdir(parents=True)


def test_main_survives_malformed_csv(tmp_path, monkeypatch):
    """A file that fails to load does not stop the batch (M3)."""
    _write_runner_env(tmp_path, CONFIG)
    raw = tmp_path / "data" / "raw"
    (raw / "good.csv").write_text("name,age\nAlice,30\n", encoding="utf-8")
    # Extra field on line 2 -> pandas ParserError (unparseable):
    (raw / "BAD.CSV").write_text("name,age\n1,2\n3,4,5\n", encoding="utf-8")

    monkeypatch.setattr(run_module, "ROOT", tmp_path)
    rc = run_module.main()

    assert rc == 1  # reported failure -> non-zero exit
    assert (tmp_path / "data" / "interim" / "good.csv").exists()
    assert (tmp_path / "data" / "processed" / "good.csv").exists()
    assert not (tmp_path / "data" / "processed" / "BAD.CSV").exists()


def test_main_processes_uppercase_csv(tmp_path, monkeypatch):
    _write_runner_env(tmp_path, CONFIG)
    (tmp_path / "data" / "raw" / "SAMPLE.CSV").write_text(
        "name,age\nAlice,30\n", encoding="utf-8"
    )
    monkeypatch.setattr(run_module, "ROOT", tmp_path)
    assert run_module.main() == 0
    assert (tmp_path / "data" / "processed" / "SAMPLE.CSV").exists()


def test_main_rejects_colliding_output_paths(tmp_path, monkeypatch):
    """Configured output paths overlapping raw abort before any I/O (M2).

    A configuration error, hence exit code 2.
    """
    bad = {**CONFIG, "paths": {**CONFIG["paths"], "interim_data": "data/raw/"}}
    _write_runner_env(tmp_path, bad)
    (tmp_path / "data" / "raw" / "in.csv").write_text("a\n1\n", encoding="utf-8")

    monkeypatch.setattr(run_module, "ROOT", tmp_path)
    assert run_module.main() == 2
    # Nothing was written into the raw directory:
    assert (tmp_path / "data" / "raw" / "in.csv").read_text(encoding="utf-8") == "a\n1\n"
    assert not (tmp_path / "data" / "raw" / "in.csv".replace("in", "out")).exists()


def test_main_empty_raw_dir(tmp_path, monkeypatch):
    _write_runner_env(tmp_path, CONFIG)
    monkeypatch.setattr(run_module, "ROOT", tmp_path)
    assert run_module.main() == 0


# --- CLI behavior ---------------------------------------------------------------

def test_cli_help_shows_usage_options_and_exit_codes(capsys):
    """--help prints usage, the --config option and the exit-code table."""
    with pytest.raises(SystemExit) as exc:
        run_module.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "--config" in out
    assert "exit codes" in out


def test_cli_custom_config_is_used(tmp_path, monkeypatch):
    """--config selects an alternative file (different output paths)."""
    _write_runner_env(tmp_path, CONFIG)
    (tmp_path / "data" / "raw" / "in.csv").write_text("name,age\nAlice,30\n", encoding="utf-8")
    alt = {**CONFIG, "paths": {**CONFIG["paths"],
                               "interim_data": "data/cleaned/",
                               "processed_data": "data/final/"}}
    alt_path = tmp_path / "alt_config.yaml"
    alt_path.write_text(yaml.safe_dump(alt, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(run_module, "ROOT", tmp_path)
    assert run_module.main(["--config", str(alt_path)]) == 0
    assert (tmp_path / "data" / "cleaned" / "in.csv").exists()
    assert (tmp_path / "data" / "final" / "in.csv").exists()
    assert not (tmp_path / "data" / "interim").exists()  # default config unused


def test_cli_missing_config_file_returns_2(tmp_path, caplog):
    rc = run_module.main(["--config", str(tmp_path / "nope.yaml")])
    assert rc == 2
    assert "not found" in caplog.text


def test_cli_unparseable_config_returns_2(tmp_path, caplog):
    bad = tmp_path / "bad.yaml"
    bad.write_text("paths: [unterminated", encoding="utf-8")
    assert run_module.main(["--config", str(bad)]) == 2
    assert "Could not parse" in caplog.text


def test_cli_invalid_option_value_returns_2(tmp_path, monkeypatch, caplog):
    """Bad option values fail fast as configuration errors, not per file."""
    bad = {**CONFIG, "cleaning": {**CONFIG["cleaning"],
                                  "missing_values": {"strategy": "bogus", "threshold": 0.5}}}
    _write_runner_env(tmp_path, bad)
    (tmp_path / "data" / "raw" / "in.csv").write_text("a\n1\n", encoding="utf-8")
    monkeypatch.setattr(run_module, "ROOT", tmp_path)
    assert run_module.main() == 2
    assert "bogus" in caplog.text


def test_cli_missing_input_directory_returns_2(tmp_path, monkeypatch, caplog):
    _write_runner_env(tmp_path, CONFIG)
    (tmp_path / "data" / "raw").rmdir()  # configured input dir does not exist
    monkeypatch.setattr(run_module, "ROOT", tmp_path)
    assert run_module.main() == 2
    assert "does not exist" in caplog.text


def test_cli_non_string_path_value_returns_2(tmp_path, caplog):
    """A non-string path entry is a config error (2), not a TypeError crash."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "paths: {raw_data: 123, interim_data: 'data/i/', processed_data: 'data/p/'}",
        encoding="utf-8",
    )
    assert run_module.main(["--config", str(bad)]) == 2
    assert "paths.raw_data" in caplog.text


def test_cli_non_utf8_config_returns_2(tmp_path, caplog):
    """A binary file passed as config exits 2 (UnicodeDecodeError is caught)."""
    bad = tmp_path / "binary.yaml"
    bad.write_bytes(b"\xff\xfe\x00\x01AB")
    assert run_module.main(["--config", str(bad)]) == 2
    assert "Could not read" in caplog.text


def test_validate_config_accepts_valid_config():
    assert run_module.validate_config(CONFIG) is None


@pytest.mark.parametrize(
    "bad, expected",
    [
        ("not a dict", "mapping"),
        ({"cleaning": {"missing_values": {"strategy": "bogus"}}}, "mapping"),  # checked first
        ({**CONFIG, "cleaning": {**CONFIG["cleaning"],
                                 "missing_values": {"strategy": "bogus", "threshold": 0.5}}}, "bogus"),
        ({**CONFIG, "transformation": {"scaling": {"method": "standard", "columns": "units"}}}, "columns"),
        ({**CONFIG, "validation": {"unknown_rule": {}}}, "unknown_rule"),
        ({"paths": {"raw_data": "data/raw/", "interim_data": "data/raw/",
                    "processed_data": "data/out/"}}, "collision"),
    ],
)
def test_validate_config_rejects_invalid(bad, expected):
    msg = run_module.validate_config(bad)
    assert msg is not None, f"expected rejection, got None for {bad!r}"
    assert expected.lower() in msg.lower(), msg


# --- reproducibility guarantees -------------------------------------------------

def _write_sample_raw(raw: Path) -> None:
    """One input file with a duplicate row and a missing value."""
    (raw / "sample.csv").write_text(
        "name,score\nAlpha,10\nAlpha,10\nBeta,\n", encoding="utf-8"
    )


def test_runner_outputs_have_correct_headers_and_no_index(tmp_path, monkeypatch):
    """Output CSVs keep the input headers and never gain an index column."""
    _write_runner_env(tmp_path, CONFIG)
    _write_sample_raw(tmp_path / "data" / "raw")
    monkeypatch.setattr(run_module, "ROOT", tmp_path)

    assert run_module.main() == 0
    input_header = (tmp_path / "data" / "raw" / "sample.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    interim_header = (tmp_path / "data" / "interim" / "sample.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    processed_header = (tmp_path / "data" / "processed" / "sample.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]

    # Cleaning preserves the input columns exactly.
    assert interim_header == input_header
    # No accidental index column anywhere; encoding keeps originals + dummies.
    for header in (interim_header, processed_header):
        assert "Unnamed" not in header
        assert not any(col.startswith("Unnamed") for col in header.split(","))
    processed_cols = processed_header.split(",")
    assert "score" in processed_cols
    assert any(col.startswith("name_") for col in processed_cols)


def test_repeated_runs_produce_byte_identical_outputs(tmp_path, monkeypatch):
    """Same input + same config => bit-for-bit reproducible outputs."""
    _write_runner_env(tmp_path, CONFIG)
    _write_sample_raw(tmp_path / "data" / "raw")
    monkeypatch.setattr(run_module, "ROOT", tmp_path)

    assert run_module.main() == 0
    first = {
        stage: (tmp_path / "data" / stage / "sample.csv").read_bytes()
        for stage in ("interim", "processed")
    }
    assert run_module.main() == 0
    for stage, content in first.items():
        assert (tmp_path / "data" / stage / "sample.csv").read_bytes() == content, stage


def test_raw_input_is_not_modified_by_run(tmp_path, monkeypatch):
    """The runner only reads from the raw directory."""
    _write_runner_env(tmp_path, CONFIG)
    _write_sample_raw(tmp_path / "data" / "raw")
    raw_file = tmp_path / "data" / "raw" / "sample.csv"
    before = raw_file.read_bytes()
    monkeypatch.setattr(run_module, "ROOT", tmp_path)

    assert run_module.main() == 0
    assert raw_file.read_bytes() == before
    assert sorted(p.name for p in (tmp_path / "data" / "raw").iterdir()) == ["sample.csv"]


def test_two_input_files_get_isolated_outputs(tmp_path, monkeypatch):
    """Each input file gets its own outputs; no shared or mixed content."""
    _write_runner_env(tmp_path, CONFIG)
    raw = tmp_path / "data" / "raw"
    (raw / "a.csv").write_text(
        "name,score\nAlpha,10\nAlpha,10\nBeta,\n", encoding="utf-8"
    )
    (raw / "b.csv").write_text("name,score\nGamma,30\nDelta,40\n", encoding="utf-8")
    monkeypatch.setattr(run_module, "ROOT", tmp_path)

    assert run_module.main() == 0
    interim = tmp_path / "data" / "interim"
    processed = tmp_path / "data" / "processed"
    assert sorted(p.name for p in interim.iterdir()) == ["a.csv", "b.csv"]
    assert sorted(p.name for p in processed.iterdir()) == ["a.csv", "b.csv"]

    a = pd.read_csv(interim / "a.csv")
    b = pd.read_csv(interim / "b.csv")
    # a: duplicate dropped (3 -> 2 rows), missing value imputed.
    assert len(a) == 2 and a.isna().sum().sum() == 0 and a.duplicated().sum() == 0
    # No content crosses between the two outputs.
    assert not a["name"].isin(["gamma", "delta"]).any()
    assert not b["name"].isin(["alpha", "beta"]).any()
    assert list(b["name"]) == ["gamma", "delta"]


def test_logs_describe_each_processing_run(tmp_path, monkeypatch, caplog):
    """A run logs its config, per-file progress, shapes, outputs and summary."""
    _write_runner_env(tmp_path, CONFIG)
    _write_sample_raw(tmp_path / "data" / "raw")
    monkeypatch.setattr(run_module, "ROOT", tmp_path)

    assert run_module.main() == 0
    text = caplog.text
    assert "Using configuration:" in text
    assert "Processing sample.csv" in text
    assert "Loaded 3 rows x 2 columns" in text
    assert "Cleaned: 2 rows x 2 columns" in text
    assert "Transformed: 2 rows x 3 columns" in text  # score + 2 name dummies
    assert "sample.csv" in text  # output paths name the file
    assert "Processed 1 file(s) successfully." in text
