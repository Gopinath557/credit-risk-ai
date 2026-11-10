"""LendingClub status mapping, application-time filtering, and data checks."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.feature_engineering import engineer_application_features


TARGET_STATUS_MAP = {"fully paid": 0, "charged off": 1}
OUTCOME_COLUMNS = {
    "loan_status",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_low",
    "last_fico_range_high",
    "out_prncp",
    "out_prncp_inv",
    "funded_amnt",
    "funded_amnt_inv",
    "installment",
    "pymnt_plan",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_loan_status",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "debt_settlement_flag",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
}
IDENTIFIER_OR_SENSITIVE_COLUMNS = {
    "id",
    "member_id",
    "url",
    "zip_code",
    "emp_title",
    "title",
    "desc",
}
APPLICATION_DATE_COLUMNS = {"issue_d", "earliest_cr_line"}
POLICY_DISALLOWED_STATUSES = {
    "does not meet the credit policy. status:fully paid",
    "does not meet the credit policy. status:charged off",
}

NUMERIC_APPLICATION_COLUMNS = {
    "loan_amnt",
    "annual_inc",
    "int_rate",
    "dti",
    "revol_util",
    "revol_bal",
    "installment",
    "total_acc",
    "open_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "fico_range_low",
    "fico_range_high",
    "collections_12_mths_ex_med",
    "acc_now_delinq",
    "chargeoff_within_12_mths",
    "tax_liens",
}
NONNEGATIVE_APPLICATION_COLUMNS = {
    "loan_amnt",
    "annual_inc",
    "int_rate",
    "dti",
    "revol_util",
    "revol_bal",
    "total_acc",
    "open_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
}


def validate_and_clean_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create the target, remove unresolved loans/leakage, and engineer features."""
    if dataframe.empty:
        raise ValueError("The dataset has no rows.")
    status_column = next(
        (column for column in dataframe.columns if column.casefold() == "loan_status"),
        None,
    )
    if status_column is None:
        raise ValueError("Required LendingClub column 'loan_status' was not found.")

    raw_rows, raw_columns = dataframe.shape
    duplicate_rows = int(dataframe.duplicated().sum())
    cleaned = dataframe.drop_duplicates().copy()
    if status_column != "loan_status":
        cleaned = cleaned.rename(columns={status_column: "loan_status"})

    normalized_status = cleaned["loan_status"].astype("string").str.strip().str.casefold()
    status_counts = (
        cleaned["loan_status"]
        .astype("string")
        .fillna("<missing>")
        .value_counts(dropna=False)
        .to_dict()
    )
    target = normalized_status.map(TARGET_STATUS_MAP)
    eligible = target.notna() & ~normalized_status.isin(POLICY_DISALLOWED_STATUSES)
    excluded_statuses = Counter(
        cleaned.loc[~eligible, "loan_status"].fillna("<missing>").astype(str).tolist()
    )
    cleaned = cleaned.loc[eligible].copy()
    cleaned["target"] = target.loc[eligible].astype("int8")
    if cleaned.empty:
        raise ValueError(
            "No Fully Paid or Charged Off records remain after excluding unresolved statuses."
        )

    cleaned = _normalize_application_numeric_values(cleaned)
    invalid_loan_amounts = 0
    if "loan_amnt" in cleaned.columns:
        invalid_loan_amounts = int(
            (cleaned["loan_amnt"].notna() & (cleaned["loan_amnt"] <= 0)).sum()
        )
        cleaned = cleaned.loc[
            cleaned["loan_amnt"].isna() | (cleaned["loan_amnt"] > 0)
        ].copy()

    invalid_numeric_values: dict[str, int] = {}
    for column in NONNEGATIVE_APPLICATION_COLUMNS.intersection(cleaned.columns):
        invalid = cleaned[column].notna() & (cleaned[column] < 0)
        invalid_count = int(invalid.sum())
        if invalid_count:
            cleaned.loc[invalid, column] = float("nan")
            invalid_numeric_values[column] = invalid_count

    cleaned, engineered_features = engineer_application_features(cleaned)
    excluded_columns = _columns_to_exclude(cleaned.columns)
    cleaned = cleaned.drop(columns=excluded_columns)

    if cleaned["target"].isna().any() or not cleaned["target"].isin([0, 1]).all():
        raise ValueError("Target validation failed; expected only binary values 0 and 1.")
    if cleaned.columns.duplicated().any():
        raise ValueError("Cleaning produced duplicate feature names.")
    if not len(cleaned.columns.difference(["target"])):
        raise ValueError("No application-time features remain after cleaning.")

    outlier_counts = _iqr_outlier_counts(cleaned)
    report: dict[str, object] = {
        "raw_rows": int(raw_rows),
        "raw_columns": int(raw_columns),
        "exact_duplicates_removed": duplicate_rows,
        "rows_after_duplicate_removal": int(raw_rows - duplicate_rows),
        "excluded_unresolved_status_rows": int(sum(excluded_statuses.values())),
        "excluded_status_counts": dict(excluded_statuses),
        "observed_loan_status_counts": status_counts,
        "invalid_loan_amount_rows_removed": invalid_loan_amounts,
        "invalid_negative_numeric_values_set_missing": invalid_numeric_values,
        "excluded_leakage_or_identifier_columns": excluded_columns,
        "engineered_features": engineered_features,
        "iqr_outlier_counts_not_removed": outlier_counts,
        "final_rows": int(len(cleaned)),
        "final_columns": int(len(cleaned.columns)),
        "target_distribution": {
            int(key): int(value)
            for key, value in cleaned["target"].value_counts().sort_index().items()
        },
    }
    return cleaned.reset_index(drop=True), report


def _columns_to_exclude(columns: pd.Index) -> list[str]:
    excluded = []
    for column in columns:
        normalized = str(column).strip().casefold()
        if (
            normalized in OUTCOME_COLUMNS
            or normalized in IDENTIFIER_OR_SENSITIVE_COLUMNS
            or normalized in APPLICATION_DATE_COLUMNS
            or normalized.startswith(
                (
                    "total_pymnt",
                    "total_rec",
                    "hardship_",
                    "settlement_",
                    "last_pymnt",
                    "last_fico",
                )
            )
        ):
            excluded.append(column)
    return excluded


def _normalize_application_numeric_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    columns = NUMERIC_APPLICATION_COLUMNS.intersection(result.columns)
    for column in columns:
        values = result[column].astype("string").str.strip()
        values = values.str.replace(",", "", regex=False)
        values = values.str.replace("$", "", regex=False)
        if column in {"int_rate", "revol_util"}:
            values = values.str.replace("%", "", regex=False)
        result[column] = pd.to_numeric(values, errors="coerce")
    return result


def _iqr_outlier_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in dataframe.select_dtypes(include="number").columns:
        if column == "target":
            continue
        values = dataframe[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        if spread <= 0:
            continue
        count = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
        if count:
            counts[column] = count
    return counts
