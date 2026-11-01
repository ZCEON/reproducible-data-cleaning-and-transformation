"""Run the full preprocessing pipeline.

Pipeline stages (driven by the configuration file):
    1. Load CSV files from data/raw/ (case-insensitive .csv match)
    2. Validate against the ``validation`` rules (schema, dtypes, ranges)
    3. Clean     -> write to data/interim/
    4. Transform -> write to data/processed/

Usage:
    python scripts/run_pipeline.py [--config PATH]

The configuration is validated up front (structure, option values, path
overlaps) so misconfiguration fails fast with a clear message instead of
failing per file. Raw data is never written to.

Exit codes:
    0  success (including: no input files)
    1  one or more input files failed validation or processing
    2  usage or configuration error
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

try:
    import src  # noqa: F401
except ModuleNotFoundError:
    # Preferred setup is `pip install -e .` (see README), after which this
    # never triggers; the fallback keeps `python scripts/run_pipeline.py`
    # working from a fresh, non-installed clone as well.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_csv
from src.data.validate_data import validate_dataframe
from src.pipeline.preprocessing_pipeline import clean, transform
from src.transformation.encoding import encode
from src.transformation.scaling import scale
from src.utils.logger import get_logger

logger = get_logger("run_pipeline")

ROOT = Path(__file__).resolve().parents[1]

_PATH_KEYS = ("raw_data", "interim_data", "processed_data")

EXIT_OK = 0
EXIT_FILE_FAILURES = 1
EXIT_CONFIG_ERROR = 2


def validate_output_paths(config: dict) -> str | None:
    """Return an error message if raw/interim/processed paths overlap.

    All three directories must be distinct (no equality, no nesting in
    either direction) so that pipeline outputs can never overwrite the
    raw input data.
    """
    paths = config.get("paths") if isinstance(config, dict) else None
    if not isinstance(paths, dict):
        return "Invalid config: missing top-level 'paths' mapping."
    missing = [key for key in _PATH_KEYS if key not in paths]
    if missing:
        return f"Invalid config: 'paths' is missing entries {missing}."
    for key in _PATH_KEYS:
        value = paths[key]
        if not isinstance(value, str) or not value.strip():
            return (
                f"Invalid config: 'paths.{key}' must be a non-empty string, "
                f"got {value!r}."
            )
    resolved = {key: (ROOT / paths[key]).resolve() for key in _PATH_KEYS}
    names = list(_PATH_KEYS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = resolved[names[i]], resolved[names[j]]
            if a == b or a in b.parents or b in a.parents:
                return (
                    f"Path collision: '{names[i]}' ({a}) and '{names[j]}' ({b}) "
                    "overlap. Raw, interim and processed directories must be "
                    "distinct so raw data is never overwritten."
                )
    return None


def find_input_files(raw_dir: Path) -> list[Path]:
    """Return CSV files in ``raw_dir`` (case-insensitive .csv extension)."""
    if not raw_dir.is_dir():
        return []
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )


def validate_config(config) -> str | None:
    """Return an error message if the configuration itself is unusable.

    Checks the structure (mapping, ``paths``) and the option *values*
    (strategies, methods, flags, validation-rule shapes) on a tiny probe
    frame, so a bad configuration fails fast with exit code 2 instead of
    failing identically on every input file. Column *membership* lists
    depend on the actual data and are checked per file (exit code 1).

    Raises nothing; every failure is returned as a human-readable message.
    """
    if not isinstance(config, dict):
        return f"Invalid config: expected a mapping of settings, got {type(config).__name__}."
    path_error = validate_output_paths(config)
    if path_error:
        return path_error
    probe = pd.DataFrame({"num": [1.0], "txt": ["a"]})
    try:
        # Raises only on invalid rule *structure*; data-dependent problems
        # come back as a list and are intentionally ignored here.
        validate_dataframe(probe, config.get("validation"))
        # clean() validates threshold, strategy, strip, case and keep.
        cleaned = clean(probe, config)
        transformation = config.get("transformation", {})
        sc = transformation.get("scaling", {})
        en = transformation.get("encoding", {})
        for section, opts in (("scaling", sc), ("encoding", en)):
            columns = opts.get("columns")
            if columns is not None and not isinstance(columns, (list, tuple)):
                return (
                    f"Invalid config: transformation.{section}.columns must be "
                    f"a list of column names, got {columns!r}."
                )
        # Validate methods (defaults mirror transform()/config.yaml).
        scale(cleaned, sc.get("method", "standard"))
        encode(cleaned, en.get("method", "onehot"))
    except Exception as exc:
        return f"Invalid config: {exc}"
    return None


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser (usage, options, exit codes)."""
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "Validate, clean and transform every CSV file in the raw data "
            "directory (see config/config.yaml for stages and rules)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/run_pipeline.py\n"
            "  python scripts/run_pipeline.py --config path/to/config.yaml\n"
            "\n"
            "exit codes:\n"
            f"  {EXIT_OK}  success (including: no input files)\n"
            f"  {EXIT_FILE_FAILURES}  one or more files failed validation or processing\n"
            f"  {EXIT_CONFIG_ERROR}  usage or configuration error (bad arguments, missing\n"
            "     or invalid config file, overlapping paths, missing input directory)\n"
            "\n"
            "Relative data paths from the configuration are resolved from the\n"
            "project root; raw data is never modified."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        default=None,
        help="configuration file to use (default: config/config.yaml)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and return a process exit code.

    ``argv`` is the argument list to parse (e.g. ``sys.argv[1:]``); when
    ``None`` no arguments are parsed (library/test use).
    """
    args = build_parser().parse_args([] if argv is None else argv)

    if args.config is None:
        # Computed at call time so a monkeypatched ROOT (tests) is honoured.
        config_path = ROOT / "config" / "config.yaml"
    else:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
    if not config_path.is_file():
        logger.error("Configuration file not found: %s", config_path)
        return EXIT_CONFIG_ERROR
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        logger.error("Could not parse configuration file %s: %s", config_path, exc)
        return EXIT_CONFIG_ERROR
    except OSError as exc:
        logger.error("Could not read configuration file %s: %s", config_path, exc)
        return EXIT_CONFIG_ERROR
    except UnicodeError as exc:
        # read_text raises UnicodeDecodeError (a ValueError, not OSError)
        # when the file is not valid UTF-8 text.
        logger.error("Could not read configuration file %s: %s", config_path, exc)
        return EXIT_CONFIG_ERROR
    logger.info("Using configuration: %s", config_path)

    config_error = validate_config(config)
    if config_error:
        logger.error(config_error)
        return EXIT_CONFIG_ERROR

    raw_dir = ROOT / config["paths"]["raw_data"]
    if not raw_dir.is_dir():
        logger.error("Input directory does not exist: %s", raw_dir)
        return EXIT_CONFIG_ERROR
    input_files = find_input_files(raw_dir)
    if not input_files:
        logger.warning("No CSV files found in %s - nothing to do.", raw_dir)
        return EXIT_OK

    rejected = 0  # validation failures
    failed = 0    # load or processing failures
    for path in input_files:
        logger.info("Processing %s", path.name)
        try:
            df = load_csv(path)
            logger.info("Loaded %d rows x %d columns", *df.shape)

            problems = validate_dataframe(df, config.get("validation"))
            if problems:
                rejected += 1
                for problem in problems:
                    logger.error("Validation failed for %s: %s", path.name, problem)
                logger.error("Skipping %s (%d validation problem(s)).", path.name, len(problems))
                continue

            cleaned = clean(df, config)
            interim_path = ROOT / config["paths"]["interim_data"] / path.name
            interim_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned.to_csv(interim_path, index=False)
            logger.info("Cleaned: %d rows x %d columns -> %s", *cleaned.shape, interim_path)

            processed = transform(cleaned, config)
            processed_path = ROOT / config["paths"]["processed_data"] / path.name
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            processed.to_csv(processed_path, index=False)
            logger.info("Transformed: %d rows x %d columns -> %s", *processed.shape, processed_path)
        except Exception as exc:
            failed += 1
            logger.error(
                "Failed to process %s: %s: %s",
                path.name,
                type(exc).__name__,
                exc,
            )

    if rejected or failed:
        logger.error(
            "Finished with failures: %d rejected by validation, %d failed "
            "(of %d file(s)).",
            rejected,
            failed,
            len(input_files),
        )
        return EXIT_FILE_FAILURES
    logger.info("Processed %d file(s) successfully.", len(input_files))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
