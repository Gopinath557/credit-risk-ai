from io import StringIO

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    DatasetLoadError,
    inspect_dataset,
    load_lendingclub_csv,
    write_dataset_reports,
)
from src.data_validation import validate_and_clean_dataset
from src.eda import save_dataset_charts
from src.preprocessing import build_preprocessor, feature_columns


def lendingclub_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_status": [
                "Fully Paid",
                "Charged Off",
                "Current",
                "In Grace Period",
                "Fully Paid",
                "Does not meet the credit policy. Status:Fully Paid",
                "Issued",
            ],
            "loan_amnt": [10000, 12000, 9000, 8000, 10000, 6000, 7000],
            "annual_inc": [50000, 40000, 60000, 30000, 50000, 40000, 50000],
            "int_rate": ["10.0%", "15.5%", "8.0%", "12.0%", "10.0%", "7.0%", "9.0%"],
            "dti": [12.0, 28.0, 5.0, 19.0, 12.0, 9.0, 11.0],
            "emp_length": ["2 years", "< 1 year", "5 years", "10+ years", "2 years", "1 year", "3 years"],
            "grade": ["B", "D", "A", "C", "B", "A", "B"],
            "id": [101, 102, 103, 104, 101, 105, 106],
            "total_pymnt": [10000, 1000, 500, 300, 10000, 0, 0],
            "last_pymnt_d": ["Jan-2018", "Feb-2018", "Mar-2018", None, "Jan-2018", None, None],
            "issue_d": ["Jan-2016", "Jan-2016", "Jan-2016", "Jan-2016", "Jan-2016", "Jan-2016", "Jan-2016"],
            "earliest_cr_line": ["Jan-2000", "Feb-2005", "Mar-2001", "Apr-1999", "Jan-2000", "May-2010", "Jun-2010"],
        }
    )


def test_load_csv_and_inspect_dataset() -> None:
    raw = load_lendingclub_csv(
        StringIO(
            "LendingClub historical loan data\n"
            "loan_status,loan_amnt\n"
            "Fully Paid,1000\n"
        )
    )

    assert raw.shape == (1, 2)
    inspection = inspect_dataset(raw)
    assert inspection["rows"] == 1
    assert inspection["loan_status_counts"] == {"Fully Paid": 1}


def test_missing_dataset_path_has_actionable_error(tmp_path) -> None:
    with pytest.raises(DatasetLoadError, match="does not exist"):
        load_lendingclub_csv(tmp_path / "missing.csv")


def test_cleaning_maps_target_removes_unresolved_duplicates_and_leakage() -> None:
    raw = lendingclub_frame()
    cleaned, report = validate_and_clean_dataset(raw)

    assert cleaned["target"].tolist() == [0, 1]
    assert report["exact_duplicates_removed"] == 1
    assert report["excluded_unresolved_status_rows"] == 4
    assert "loan_status" not in cleaned.columns
    assert "total_pymnt" not in cleaned.columns
    assert "last_pymnt_d" not in cleaned.columns
    assert "id" not in cleaned.columns
    assert "issue_d" not in cleaned.columns
    assert "earliest_cr_line" not in cleaned.columns
    assert cleaned["credit_history_months"].tolist() == [192.0, 131.0]
    assert cleaned["employment_length_years"].tolist() == [2.0, 0.5]
    assert np.isclose(cleaned.loc[0, "loan_to_income"], 0.2)


def test_cleaning_requires_resolved_target_rows() -> None:
    with pytest.raises(ValueError, match="No Fully Paid or Charged Off"):
        validate_and_clean_dataset(
            pd.DataFrame({"loan_status": ["Current"], "loan_amnt": [1000]})
        )


def test_cleaning_excludes_payment_and_account_identifiers() -> None:
    raw = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off"],
            "loan_amnt": [1000, 2000],
            "annual_inc": [30000, 40000],
            "member_id": [20, 21],
            "recoveries": [0, 500],
            "collection_recovery_fee": [0, 20],
            "total_rec_prncp": [1000, 100],
            "next_pymnt_d": [None, "Apr-2018"],
            "settlement_status": [None, "COMPLETE"],
        }
    )
    cleaned, _ = validate_and_clean_dataset(raw)

    assert not {
        "loan_status",
        "member_id",
        "recoveries",
        "collection_recovery_fee",
        "total_rec_prncp",
        "next_pymnt_d",
        "settlement_status",
    }.intersection(cleaned.columns)


def test_preprocessor_imputes_scales_encodes_and_ignores_unknown_categories() -> None:
    dataframe = pd.DataFrame(
        {
            "loan_amnt": [1000.0, np.nan, 3000.0],
            "grade": ["A", "B", "A"],
            "target": [0, 1, 0],
        }
    )
    numeric, categorical = feature_columns(dataframe)
    transformer = build_preprocessor(numeric, categorical)
    transformed = transformer.fit_transform(dataframe.drop(columns="target"))

    assert transformed.shape[0] == 3
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(dense).any()
    assert transformer.transform(
        pd.DataFrame({"loan_amnt": [2000.0], "grade": ["Z"]})
    ).shape[0] == 1


def test_preprocessor_rejects_non_binary_target() -> None:
    with pytest.raises(ValueError, match="binary"):
        feature_columns(pd.DataFrame({"feature": [1], "target": [2]}))


def test_reports_are_written_with_data_derived_counts(tmp_path) -> None:
    raw = lendingclub_frame()
    cleaned, report = validate_and_clean_dataset(raw)

    paths = write_dataset_reports(raw, cleaned, report, tmp_path)

    assert all(path.is_file() for path in paths.values())
    target_distribution = pd.read_csv(paths["target_distribution"])
    assert target_distribution["count"].sum() == len(cleaned)
    assert "loan_status" in paths["dataset_summary"].read_text(encoding="utf-8")
    missing = pd.read_csv(paths["missing_values"])
    assert set(missing["stage"]) == {"raw", "processed"}


def test_eda_saves_target_and_feature_distribution_charts(tmp_path) -> None:
    cleaned, _ = validate_and_clean_dataset(lendingclub_frame())

    paths = save_dataset_charts(cleaned, tmp_path)

    assert (tmp_path / "target_distribution.png") in paths
    assert (tmp_path / "missing_values.png") in paths
    assert (tmp_path / "numeric_distributions.png") in paths
    assert all(path.is_file() for path in paths)
