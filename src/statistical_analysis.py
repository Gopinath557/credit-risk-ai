"""Statistical summaries and default-rate analysis for cleaned loan data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "target"
DEFAULT_TARGETS = {"Fully Paid": 0, "Charged Off": 1}
NUMERIC_FEATURES = {
    "loan_amnt": "Loan amount",
    "funded_amnt": "Funded amount",
    "annual_inc": "Annual income",
    "int_rate": "Interest rate",
    "dti": "Debt-to-income ratio",
    "revol_util": "Revolving utilization",
    "total_acc": "Total credit lines",
    "delinq_2yrs": "Delinquencies in past 2 years",
    "employment_length_years": "Employment length (years)",
    "loan_to_income": "Loan-to-income ratio",
    "credit_history_months": "Credit history (months)",
}
CATEGORICAL_FEATURES = {
    "grade": "Loan grade",
    "home_ownership": "Home ownership",
    "purpose": "Loan purpose",
    "verification_status": "Income verification status",
    "emp_length": "Employment length",
    "term": "Loan term",
}


def validate_eda_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a valid binary target; fail clearly on incompatible input."""
    if dataframe.empty:
        raise ValueError("The processed dataset contains no rows.")
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError("The processed dataset must contain a 'target' column.")

    result = dataframe.copy()
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
    result = result.loc[result[TARGET_COLUMN].isin([0, 1])].copy()
    if result.empty:
        raise ValueError("No rows have a valid target (0 = paid, 1 = charged off).")
    result[TARGET_COLUMN] = result[TARGET_COLUMN].astype("int8")
    return result


def describe_numeric_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate distribution statistics for available numeric features."""
    data = validate_eda_data(dataframe)
    columns = [
        column
        for column in data.select_dtypes(include="number").columns
        if column != TARGET_COLUMN
    ]
    rows = []
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            continue
        modes = values.mode(dropna=True)
        rows.append(
            {
                "feature": column,
                "description": NUMERIC_FEATURES.get(column, column.replace("_", " ").title()),
                "count": int(values.count()),
                "missing_count": int(data[column].isna().sum()),
                "mean": values.mean(),
                "median": values.median(),
                "mode": modes.iloc[0] if not modes.empty else np.nan,
                "std": values.std(ddof=1),
                "min": values.min(),
                "q1": values.quantile(0.25),
                "q2": values.quantile(0.50),
                "q3": values.quantile(0.75),
                "max": values.max(),
                "skewness": values.skew(),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "description",
            "count",
            "missing_count",
            "mean",
            "median",
            "mode",
            "std",
            "min",
            "q1",
            "q2",
            "q3",
            "max",
            "skewness",
        ],
    )


def numeric_correlations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlations for available numeric columns and target."""
    data = validate_eda_data(dataframe)
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame(index=numeric.columns, columns=numeric.columns, dtype=float)
    return numeric.corr(method="pearson", min_periods=2)


def default_rate_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate sample sizes and observed default rates for available features."""
    data = validate_eda_data(dataframe)
    results: list[pd.DataFrame] = []
    for column in CATEGORICAL_FEATURES:
        if column not in data.columns:
            continue
        values = data[column].astype("string").fillna("<missing>")
        grouped = data.assign(_group=values).groupby("_group", observed=True)
        summary = grouped[TARGET_COLUMN].agg(applications="size", defaults="sum")
        results.append(_format_rate_rows(column, summary))

    for column in NUMERIC_FEATURES:
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        valid = values.notna()
        unique_count = int(values.loc[valid].nunique())
        if unique_count < 2:
            continue
        bins = min(10, unique_count)
        try:
            categories = pd.qcut(values.loc[valid], q=bins, duplicates="drop")
        except ValueError:
            continue
        grouped = data.loc[valid].assign(_group=categories.astype(str)).groupby(
            "_group", observed=True
        )
        summary = grouped[TARGET_COLUMN].agg(applications="size", defaults="sum")
        rate_rows = _format_rate_rows(column, summary)
        rate_rows["feature_type"] = "numeric_quantile"
        results.append(rate_rows)

    if not results:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_type",
                "category",
                "applications",
                "defaults",
                "default_rate",
            ]
        )
    return pd.concat(results, ignore_index=True)


def _format_rate_rows(column: str, summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": column,
            "feature_type": (
                "categorical"
                if column in CATEGORICAL_FEATURES
                else "numeric_quantile"
            ),
            "category": summary.index.astype(str),
            "applications": summary["applications"].astype(int).to_numpy(),
            "defaults": summary["defaults"].astype(int).to_numpy(),
            "default_rate": (
                summary["defaults"].div(summary["applications"]).to_numpy()
            ),
        }
    )


def create_insights(
    dataframe: pd.DataFrame,
    default_rates: pd.DataFrame | None = None,
    correlations: pd.DataFrame | None = None,
) -> str:
    """Summarize observed associations with sample counts and causal caveat."""
    data = validate_eda_data(dataframe)
    rates = default_rate_analysis(data) if default_rates is None else default_rates
    corr = numeric_correlations(data) if correlations is None else correlations
    total = len(data)
    overall_rate = float(data[TARGET_COLUMN].mean())
    lines = [
        "Exploratory data analysis — descriptive insights",
        "===============================================",
        f"Eligible observations analyzed: {total:,}",
        f"Observed charged-off rate: {overall_rate:.2%}",
        "",
        "All rates and relationships below are descriptive associations in this "
        "historical dataset. They do not establish causation, are not model "
        "predictions, and may reflect selection, vintage, or data-quality effects.",
        "",
    ]

    for feature in ("grade", "purpose", "home_ownership"):
        feature_rows = rates.loc[
            (rates["feature"] == feature) & (rates["feature_type"] == "categorical")
        ].sort_values(["default_rate", "applications"], ascending=[False, False])
        if feature_rows.empty:
            lines.append(f"{CATEGORICAL_FEATURES[feature]}: not present in the dataset.")
            continue
        top = feature_rows.iloc[0]
        lines.append(
            f"Highest observed {CATEGORICAL_FEATURES[feature]} default rate: "
            f"{top['category']} ({top['default_rate']:.2%}; "
            f"{int(top['defaults'])}/{int(top['applications'])} applications)."
        )

    if "int_rate" in data.columns:
        interest = pd.to_numeric(data["int_rate"], errors="coerce")
        paid = interest.loc[data[TARGET_COLUMN] == 0].dropna()
        charged_off = interest.loc[data[TARGET_COLUMN] == 1].dropna()
        if not paid.empty and not charged_off.empty:
            lines.append(
                "Interest-rate medians: "
                f"Fully Paid {paid.median():.2f}; Charged Off {charged_off.median():.2f} "
                "(units follow the source column)."
            )
        else:
            lines.append("Interest-rate comparison: insufficient outcome-group data.")
    else:
        lines.append("Interest-rate comparison: interest rate is not present.")

    for feature in ("annual_inc", "loan_amnt", "dti"):
        if feature not in data.columns:
            lines.append(f"{NUMERIC_FEATURES[feature]}: not present in the dataset.")
            continue
        groups = data.groupby(TARGET_COLUMN)[feature].agg(["median", "count"])
        if {0, 1}.issubset(groups.index):
            lines.append(
                f"{NUMERIC_FEATURES[feature]} median by outcome — Fully Paid: "
                f"{groups.loc[0, 'median']:.3g} (n={int(groups.loc[0, 'count']):,}); "
                f"Charged Off: {groups.loc[1, 'median']:.3g} "
                f"(n={int(groups.loc[1, 'count']):,})."
            )
        else:
            lines.append(f"{NUMERIC_FEATURES[feature]}: one outcome group is absent.")

    if TARGET_COLUMN in corr.columns:
        associations = corr[TARGET_COLUMN].drop(labels=[TARGET_COLUMN], errors="ignore")
        associations = associations.dropna().sort_values(key=lambda values: values.abs(), ascending=False)
        if not associations.empty:
            lines.extend(
                [
                    "",
                    "Largest absolute Pearson correlations with the binary target "
                    "(not causal effects):",
                ]
            )
            for feature, value in associations.head(5).items():
                lines.append(f"  {feature}: {value:.3f}")

    if not rates.empty and (rates["applications"] < 30).any():
        lines.extend(
            [
                "",
                "Some reported segments contain fewer than 30 applications; their "
                "default rates are especially sensitive to individual observations.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_statistical_reports(
    dataframe: pd.DataFrame,
    reports_dir: str | Path,
) -> dict[str, Path]:
    """Write descriptive statistics, correlations, default rates, and insights."""
    data = validate_eda_data(dataframe)
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    statistics = describe_numeric_features(data)
    correlations = numeric_correlations(data)
    default_rates = default_rate_analysis(data)
    insights = create_insights(data, default_rates, correlations)

    paths = {
        "descriptive_statistics": output_dir / "descriptive_statistics.csv",
        "correlation_matrix": output_dir / "correlation_matrix.csv",
        "default_rate_analysis": output_dir / "default_rate_analysis.csv",
        "eda_insights": output_dir / "eda_insights.txt",
    }
    statistics.to_csv(paths["descriptive_statistics"], index=False)
    correlations.to_csv(paths["correlation_matrix"], index=True, index_label="feature")
    default_rates.to_csv(paths["default_rate_analysis"], index=False)
    paths["eda_insights"].write_text(insights, encoding="utf-8")
    return paths


def analyze_target_distribution(dataframe: pd.DataFrame) -> Mapping[str, int]:
    """Return observed counts for the two encoded outcomes."""
    target = validate_eda_data(dataframe)[TARGET_COLUMN]
    return {
        "Fully Paid": int((target == 0).sum()),
        "Charged Off": int((target == 1).sum()),
    }


def filter_eda_data(
    dataframe: pd.DataFrame,
    outcomes: set[int] | None = None,
    category_column: str | None = None,
    categories: set[str] | None = None,
    numeric_column: str | None = None,
    numeric_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Apply validated outcome, category, and inclusive numeric-range filters."""
    data = validate_eda_data(dataframe)
    if outcomes is not None:
        if not outcomes.issubset({0, 1}):
            raise ValueError("Outcome filters may contain only 0 and 1.")
        data = data.loc[data[TARGET_COLUMN].isin(outcomes)].copy()

    if category_column is not None:
        if category_column not in data.columns or category_column == TARGET_COLUMN:
            raise ValueError(f"Unknown categorical filter: {category_column!r}.")
        values = data[category_column].astype("string").fillna("<missing>")
        if categories is not None:
            data = data.loc[values.isin(categories)].copy()

    if numeric_range is not None:
        if numeric_column is None or numeric_column not in data.columns:
            raise ValueError("A valid numeric feature is required for a range filter.")
        lower, upper = numeric_range
        if lower > upper:
            raise ValueError("Numeric range lower bound must not exceed its upper bound.")
        values = pd.to_numeric(data[numeric_column], errors="coerce")
        data = data.loc[values.between(lower, upper)].copy()
    return data
