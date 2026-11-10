"""Application-time LendingClub feature engineering."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def engineer_application_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add supported features without using post-origination information."""
    result = dataframe.copy()
    descriptions: dict[str, str] = {}

    if {"loan_amnt", "annual_inc"}.issubset(result.columns):
        income = result["annual_inc"].where(result["annual_inc"] > 0)
        result["loan_to_income"] = result["loan_amnt"].div(income)
        descriptions["loan_to_income"] = (
            "Application loan amount divided by positive annual income; missing when "
            "income is unavailable or non-positive."
        )

    if "emp_length" in result.columns:
        result["employment_length_years"] = result["emp_length"].map(
            _employment_length_years
        )
        descriptions["employment_length_years"] = (
            "Employment tenure in years parsed from LendingClub emp_length; '< 1 year' "
            "is represented as 0.5 and '10+ years' as 10."
        )

    if {"issue_d", "earliest_cr_line"}.issubset(result.columns):
        issue_dates = result["issue_d"].map(_parse_lendingclub_month)
        credit_dates = result["earliest_cr_line"].map(_parse_lendingclub_month)
        months = (
            (issue_dates.dt.year - credit_dates.dt.year) * 12
            + issue_dates.dt.month
            - credit_dates.dt.month
        )
        result["credit_history_months"] = months.where(
            issue_dates.notna() & credit_dates.notna() & (months >= 0)
        ).astype("float64")
        descriptions["credit_history_months"] = (
            "Months between the earliest reported credit line and loan issue month; "
            "negative or unparseable intervals are missing."
        )

    return result, descriptions


def _employment_length_years(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().casefold()
    if text in {"", "n/a", "na", "unknown"}:
        return None
    if text.startswith("<"):
        return 0.5
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        return None
    years = float(match.group())
    return min(years, 10.0)


def _parse_lendingclub_month(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    for date_format in ("%b-%y", "%b-%Y", "%Y-%m-%d", "%Y-%m"):
        parsed = pd.to_datetime(text, format=date_format, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    return pd.NaT
