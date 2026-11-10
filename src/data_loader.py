"""CSV loading, dataset inspection, and Phase 2 output generation."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import IO

import pandas as pd


class DatasetLoadError(ValueError):
    """Raised when a provided CSV cannot be loaded as a usable table."""


def load_lendingclub_csv(source: str | Path | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Load a LendingClub CSV from a path or a seekable uploaded file."""
    if isinstance(source, (str, Path)) and not Path(source).is_file():
        raise DatasetLoadError(f"Dataset file does not exist: {source}")
    try:
        skip_rows = _find_header_rows(source)
        dataframe = pd.read_csv(source, skiprows=skip_rows, low_memory=False)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise DatasetLoadError(f"Could not read the dataset CSV: {error}") from error

    if dataframe.empty or len(dataframe.columns) == 0:
        raise DatasetLoadError("The CSV is empty or contains no columns.")

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    if dataframe.columns.duplicated().any():
        raise DatasetLoadError("The CSV contains duplicate column names.")
    return dataframe


def _find_header_rows(source: str | Path | IO[bytes] | IO[str]) -> int:
    if isinstance(source, (str, Path)):
        with Path(source).open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as stream:
            lines = list(stream.readline() for _ in range(50))
    else:
        if not source.seekable():
            raise DatasetLoadError("CSV uploads must support seeking.")
        source.seek(0)
        lines = []
        for _ in range(50):
            line = source.readline()
            if not line:
                break
            lines.append(line.decode("utf-8-sig", errors="replace") if isinstance(line, bytes) else line)
        source.seek(0)

    for line_number, line in enumerate(lines):
        try:
            columns = {column.strip().casefold() for column in next(csv.reader([line]))}
        except csv.Error:
            continue
        if {"loan_status", "loan_amnt"}.issubset(columns):
            return line_number
    raise DatasetLoadError(
        "Could not locate a LendingClub CSV header containing loan_status and loan_amnt "
        "within the first 50 lines."
    )


def inspect_dataset(dataframe: pd.DataFrame) -> dict[str, object]:
    """Collect raw dimensions, dtypes, missingness, duplicates, and status counts."""
    status_column = next(
        (column for column in dataframe.columns if column.casefold() == "loan_status"),
        None,
    )
    status_counts = (
        dataframe[status_column]
        .astype("string")
        .fillna("<missing>")
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        if status_column is not None
        else {}
    )
    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": dataframe.columns.tolist(),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
        "missing_values": dataframe.isna().sum().astype(int).to_dict(),
        "unique_values": dataframe.nunique(dropna=True).astype(int).to_dict(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "loan_status_counts": status_counts,
    }


def write_dataset_reports(
    raw_dataframe: pd.DataFrame,
    cleaned_dataframe: pd.DataFrame,
    cleaning_report: Mapping[str, object],
    reports_dir: str | Path,
) -> dict[str, Path]:
    """Write truthful, data-derived summary and cleaning CSV/text reports."""
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = inspect_dataset(raw_dataframe)
    raw_missing = raw_dataframe.isna().sum().astype(int)
    cleaned_missing = cleaned_dataframe.isna().sum().astype(int)
    missing_report = pd.concat(
        [
            pd.DataFrame(
                {
                    "stage": "raw",
                    "column": raw_dataframe.columns,
                    "missing_count": raw_missing.to_numpy(),
                    "missing_pct": (
                        raw_missing.div(max(len(raw_dataframe), 1))
                        .mul(100)
                        .round(2)
                    ).to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "stage": "processed",
                    "column": cleaned_dataframe.columns,
                    "missing_count": cleaned_missing.to_numpy(),
                    "missing_pct": (
                        cleaned_missing.div(max(len(cleaned_dataframe), 1))
                        .mul(100)
                        .round(2)
                    ).to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )
    missing_path = output_dir / "missing_values.csv"
    missing_report.to_csv(missing_path, index=False)

    target_counts = cleaned_dataframe["target"].value_counts().sort_index()
    target_report = pd.DataFrame(
        {
            "target": target_counts.index.astype(int),
            "loan_status": ["Fully Paid" if value == 0 else "Charged Off" for value in target_counts.index],
            "count": target_counts.values,
            "share_pct": (
                target_counts.div(max(target_counts.sum(), 1)).mul(100).round(2).values
            ),
        }
    )
    target_path = output_dir / "target_distribution.csv"
    target_report.to_csv(target_path, index=False)

    feature_rows = []
    engineered = cleaning_report.get("engineered_features", {})
    for column in cleaned_dataframe.columns:
        feature_rows.append(
            {
                "column": column,
                "dtype": str(cleaned_dataframe[column].dtype),
                "description": (
                    str(engineered[column])
                    if isinstance(engineered, Mapping) and column in engineered
                    else "Binary target: 0 = Fully Paid, 1 = Charged Off."
                    if column == "target"
                    else "Retained LendingClub application-time feature."
                ),
            }
        )
    feature_path = output_dir / "feature_descriptions.csv"
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)

    summary_path = output_dir / "dataset_summary.txt"
    dtypes_report = pd.DataFrame(
        {
            "column": raw_dataframe.columns,
            "dtype": [str(raw_dataframe[column].dtype) for column in raw_dataframe.columns],
            "unique_values": [
                int(raw_dataframe[column].nunique(dropna=True))
                for column in raw_dataframe.columns
            ],
            "missing_values": [
                int(raw_dataframe[column].isna().sum())
                for column in raw_dataframe.columns
            ],
        }
    )
    status_counts = raw_summary["loan_status_counts"]
    summary_path.write_text(
        "\n".join(
            [
                "LendingClub dataset inspection",
                "==============================",
                f"Raw dimensions: {raw_summary['rows']} rows x {raw_summary['columns']} columns",
                f"Exact duplicate rows in raw file: {raw_summary['duplicate_rows']}",
                f"Processed dimensions: {len(cleaned_dataframe)} rows x {len(cleaned_dataframe.columns)} columns",
                "",
                "Raw loan_status counts:",
                _format_mapping(status_counts),
                "",
                "Raw columns, types, distinct non-null values, and missing values:",
                dtypes_report.to_string(index=False),
                "",
                "Processed target distribution:",
                target_report.to_string(index=False),
                "",
                "Source details and limitations: see README.md. Raw uploads are not "
                "persisted by the application; only processed data and reports are saved.",
            ]
        ),
        encoding="utf-8",
    )

    cleaning_path = output_dir / "data_cleaning_report.txt"
    cleaning_path.write_text(
        "\n".join(
            [
                "LendingClub data cleaning report",
                "================================",
                *[
                    f"{key}: {_format_value(value)}"
                    for key, value in cleaning_report.items()
                ],
                "",
                "Missing feature values are retained in this CSV and must be imputed "
                "inside a training-only preprocessing pipeline to avoid leakage.",
                "Extreme values are reported, not automatically removed.",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "dataset_summary": summary_path,
        "missing_values": missing_path,
        "target_distribution": target_path,
        "feature_descriptions": feature_path,
        "data_cleaning_report": cleaning_path,
    }


def _format_mapping(value: object) -> str:
    if not isinstance(value, Mapping) or not value:
        return "Not available"
    return "\n".join(f"  {key}: {item}" for key, item in value.items())


def _format_value(value: object) -> str:
    if isinstance(value, Mapping):
        return _format_mapping(value)
    return str(value)
