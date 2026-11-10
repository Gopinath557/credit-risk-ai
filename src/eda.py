"""Descriptive EDA tables, interactive charts, and privacy-conscious artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.statistical_analysis import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    default_rate_analysis,
    numeric_correlations,
    validate_eda_data,
    write_statistical_reports,
)


def save_dataset_charts(
    dataframe: pd.DataFrame, output_dir: str | Path
) -> list[Path]:
    """Save Phase 2 target, missingness, and representative feature charts."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    sns.set_theme(style="whitegrid")

    if "target" in dataframe.columns:
        figure, axis = plt.subplots(figsize=(7, 4))
        target_counts = dataframe["target"].map(
            {0: "Fully Paid", 1: "Charged Off"}
        ).value_counts()
        sns.barplot(
            x=target_counts.index,
            y=target_counts.values,
            hue=target_counts.index,
            legend=False,
            palette=["#11a8c4", "#e36f58"][: len(target_counts)],
            ax=axis,
        )
        axis.set(title="Loan status distribution", xlabel="Loan status", ylabel="Records")
        outputs.append(_save_figure(figure, directory / "target_distribution.png"))

    missing_pct = dataframe.isna().mean().mul(100)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False).head(30)
    if missing_pct.empty:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.text(
            0.5,
            0.5,
            "No missing feature values",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        outputs.append(_save_figure(figure, directory / "missing_values.png"))
    else:
        figure, axis = plt.subplots(figsize=(10, max(4, len(missing_pct) * 0.28)))
        sns.barplot(x=missing_pct.values, y=missing_pct.index, color="#11a8c4", ax=axis)
        axis.set(title="Missing values by feature (top 30)", xlabel="Missing (%)", ylabel="")
        outputs.append(_save_figure(figure, directory / "missing_values.png"))

    numeric_columns = [
        column
        for column in dataframe.select_dtypes(include="number").columns
        if column != "target"
    ][:4]
    if numeric_columns:
        figure, axes = plt.subplots(
            nrows=len(numeric_columns),
            ncols=1,
            figsize=(8, 3 * len(numeric_columns)),
            squeeze=False,
        )
        for axis, column in zip(axes[:, 0], numeric_columns):
            sns.histplot(dataframe[column].dropna(), bins=30, ax=axis, color="#11a8c4")
            axis.set(title=f"{column} distribution", xlabel=column)
        outputs.append(_save_figure(figure, directory / "numeric_distributions.png"))

    categorical = [
        column
        for column in dataframe.columns
        if (
            pd.api.types.is_object_dtype(dataframe[column].dtype)
            or pd.api.types.is_string_dtype(dataframe[column].dtype)
            or isinstance(dataframe[column].dtype, pd.CategoricalDtype)
        )
        if dataframe[column].nunique(dropna=True) <= 15
    ][:3]
    if categorical:
        figure, axes = plt.subplots(
            nrows=len(categorical),
            ncols=1,
            figsize=(9, 3 * len(categorical)),
            squeeze=False,
        )
        for axis, column in zip(axes[:, 0], categorical):
            counts = dataframe[column].fillna("<missing>").value_counts().head(15)
            sns.barplot(
                x=counts.values,
                y=counts.index,
                color="#11a8c4",
                ax=axis,
            )
            axis.set(title=f"{column} distribution", xlabel="Records", ylabel="")
        outputs.append(_save_figure(figure, directory / "categorical_distributions.png"))

    return outputs


def create_eda_figures(dataframe: pd.DataFrame) -> dict[str, go.Figure]:
    """Create interactive charts only for features present in the input data."""
    data = validate_eda_data(dataframe)
    figures: dict[str, go.Figure] = {}
    target_counts = (
        data[TARGET_COLUMN]
        .map({0: "Fully Paid", 1: "Charged Off"})
        .value_counts()
        .rename_axis("Loan status")
        .reset_index(name="Applications")
    )
    if not target_counts.empty:
        figures["target_distribution"] = px.bar(
            target_counts,
            x="Loan status",
            y="Applications",
            color="Loan status",
            color_discrete_map={
                "Fully Paid": "#11a8c4",
                "Charged Off": "#e36f58",
            },
            title="Loan outcome distribution",
            hover_data={"Applications": ":,"},
        )

    for column in NUMERIC_FEATURES:
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        if values.notna().sum() < 1:
            continue
        label = NUMERIC_FEATURES[column]
        hist_data = data.assign(_value=values).dropna(subset=["_value"])
        figures[f"histogram_{column}"] = px.histogram(
            hist_data,
            x="_value",
            color=TARGET_COLUMN,
            nbins=40,
            opacity=0.75,
            title=f"{label} distribution by loan outcome",
            labels={
                "_value": label,
                TARGET_COLUMN: "Outcome (0 = Fully Paid, 1 = Charged Off)",
            },
            color_discrete_map={0: "#11a8c4", 1: "#e36f58"},
            marginal="box",
        )

    for column in CATEGORICAL_FEATURES:
        if column not in data.columns:
            continue
        categories = data[column].astype("string").fillna("<missing>")
        counts = (
            data.assign(_category=categories)
            .groupby("_category", observed=True)[TARGET_COLUMN]
            .agg(applications="size", defaults="sum")
            .reset_index()
            .sort_values("applications", ascending=False)
            .head(30)
        )
        if counts.empty:
            continue
        label = CATEGORICAL_FEATURES[column]
        figures[f"count_{column}"] = px.bar(
            counts,
            x="_category",
            y="applications",
            title=f"Applications by {label.lower()} (top 30)",
            labels={"_category": label, "applications": "Applications"},
            hover_data={"defaults": True},
        )
        counts["default_rate"] = counts["defaults"] / counts["applications"]
        figures[f"default_rate_{column}"] = px.bar(
            counts.sort_values("default_rate", ascending=False),
            x="_category",
            y="default_rate",
            title=f"Observed default rate by {label.lower()}",
            labels={
                "_category": label,
                "default_rate": "Observed default rate",
                "applications": "Applications",
            },
            hover_data={"applications": ":,", "defaults": ":,"},
            color="default_rate",
            color_continuous_scale=["#11a8c4", "#e36f58"],
        )
        figures[f"default_rate_{column}"].update_yaxes(tickformat=".0%")

    correlations = numeric_correlations(data)
    if correlations.shape[0] >= 2:
        figures["correlation_heatmap"] = px.imshow(
            correlations,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="Pearson correlation of numeric features and binary outcome",
            labels={"color": "Correlation"},
            aspect="auto",
        )

    if {"annual_inc", "loan_amnt"}.issubset(data.columns):
        income_loan = data.copy()
        income_loan["annual_inc"] = pd.to_numeric(
            income_loan["annual_inc"], errors="coerce"
        )
        income_loan["loan_amnt"] = pd.to_numeric(
            income_loan["loan_amnt"], errors="coerce"
        )
        income_loan = income_loan.dropna(
            subset=["annual_inc", "loan_amnt", TARGET_COLUMN]
        )
        if not income_loan.empty:
            if len(income_loan) > 10_000:
                income_loan = income_loan.sample(n=10_000, random_state=42)
            figures["income_vs_loan_amount"] = px.scatter(
                income_loan,
                x="annual_inc",
                y="loan_amnt",
                color=TARGET_COLUMN,
                opacity=0.55,
                title="Annual income versus requested loan amount",
                labels={
                    "annual_inc": "Annual income",
                    "loan_amnt": "Loan amount",
                    TARGET_COLUMN: "Outcome (0 = Fully Paid, 1 = Charged Off)",
                },
                color_discrete_map={0: "#11a8c4", 1: "#e36f58"},
                hover_data=[TARGET_COLUMN],
            )

    if {"int_rate", "grade"}.issubset(data.columns):
        grade_data = data.copy()
        grade_data["int_rate"] = pd.to_numeric(
            grade_data["int_rate"], errors="coerce"
        )
        grade_data = grade_data.dropna(subset=["int_rate", "grade"])
        if not grade_data.empty:
            figures["interest_rate_by_grade"] = px.box(
                grade_data,
                x="grade",
                y="int_rate",
                color=TARGET_COLUMN,
                points=False,
                title="Interest-rate distribution by loan grade and outcome",
                labels={
                    "grade": "Loan grade",
                    "int_rate": "Interest rate (source units)",
                    TARGET_COLUMN: "Outcome (0 = Fully Paid, 1 = Charged Off)",
                },
                color_discrete_map={0: "#11a8c4", 1: "#e36f58"},
            )

    pairwise_columns = [
        column
        for column in ("loan_amnt", "annual_inc", "int_rate", "dti")
        if column in data.columns
        and pd.to_numeric(data[column], errors="coerce").notna().any()
    ]
    if len(pairwise_columns) >= 2:
        pairwise_data = data[pairwise_columns + [TARGET_COLUMN]].copy()
        for column in pairwise_columns:
            pairwise_data[column] = pd.to_numeric(
                pairwise_data[column], errors="coerce"
            )
        pairwise_data = pairwise_data.dropna()
        if len(pairwise_data) > 2_000:
            pairwise_data = pairwise_data.sample(n=2_000, random_state=42)
        if len(pairwise_data) >= 2:
            figures["pairwise_relationships"] = px.scatter_matrix(
                pairwise_data,
                dimensions=pairwise_columns,
                color=TARGET_COLUMN,
                title="Pairwise relationships among available financial features",
                color_discrete_map={0: "#11a8c4", 1: "#e36f58"},
                labels={
                    column: NUMERIC_FEATURES[column]
                    for column in pairwise_columns
                },
                opacity=0.55,
            )
            figures["pairwise_relationships"].update_traces(
                diagonal_visible=False
            )

    rate_table = default_rate_analysis(data)
    if not rate_table.empty:
        dti_rates = rate_table.loc[
            (rate_table["feature"] == "dti")
            & (rate_table["feature_type"] == "numeric_quantile")
        ].copy()
        if not dti_rates.empty:
            dti_rates["applications"] = pd.to_numeric(
                dti_rates["applications"], errors="coerce"
            )
            dti_rates = dti_rates.sort_values("category")
            figures["default_rate_by_dti"] = px.bar(
                dti_rates,
                x="category",
                y="default_rate",
                title="Observed default rate by DTI quantile",
                labels={
                    "category": "DTI quantile interval",
                    "default_rate": "Observed default rate",
                    "applications": "Applications",
                },
                hover_data={"applications": ":,", "defaults": ":,"},
                color="default_rate",
                color_continuous_scale=["#11a8c4", "#e36f58"],
            )
            figures["default_rate_by_dti"].update_yaxes(tickformat=".0%")

    return figures


def save_eda_visualizations(
    dataframe: pd.DataFrame, output_dir: str | Path
) -> list[Path]:
    """Save aggregated, non-row-level PNGs for sharing alongside reports."""
    data = validate_eda_data(dataframe)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    sns.set_theme(style="whitegrid")

    counts = data[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        ["Fully Paid", "Charged Off"],
        counts.to_numpy(),
        color=["#11a8c4", "#e36f58"],
    )
    axis.set(title="Loan outcome distribution", ylabel="Applications")
    outputs.append(_save_figure(figure, directory / "target_distribution.png"))

    for column in (
        "loan_amnt",
        "annual_inc",
        "int_rate",
        "dti",
        "revol_util",
        "employment_length_years",
        "total_acc",
        "delinq_2yrs",
    ):
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            continue
        label = NUMERIC_FEATURES[column]
        figure, axis = plt.subplots(figsize=(8, 4))
        sns.histplot(values, bins=40, ax=axis, color="#11a8c4")
        axis.set(title=f"{label} distribution", xlabel=label, ylabel="Applications")
        outputs.append(_save_figure(figure, directory / f"{column}_distribution.png"))

    correlations = numeric_correlations(data)
    if correlations.shape[0] >= 2:
        figure, axis = plt.subplots(
            figsize=(max(7, len(correlations.columns) * 0.8), 6)
        )
        image = axis.imshow(
            correlations,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            aspect="auto",
        )
        axis.set_xticks(range(len(correlations.columns)))
        axis.set_xticklabels(correlations.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(correlations.index)))
        axis.set_yticklabels(correlations.index)
        if len(correlations.columns) <= 12:
            for row in range(len(correlations.index)):
                for column in range(len(correlations.columns)):
                    value = correlations.iloc[row, column]
                    if pd.notna(value):
                        axis.text(
                            column,
                            row,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                        )
        figure.colorbar(image, ax=axis, label="Pearson correlation")
        axis.set(title="Numeric feature and target correlations")
        outputs.append(_save_figure(figure, directory / "correlation_heatmap.png"))

    rate_table = default_rate_analysis(data)
    for column in CATEGORICAL_FEATURES:
        grouped = rate_table.loc[
            (rate_table["feature"] == column)
            & (rate_table["feature_type"] == "categorical")
        ]
        if grouped.empty:
            continue
        distribution = grouped.sort_values(
            "applications", ascending=False
        ).head(20)
        figure, axis = plt.subplots(
            figsize=(9, max(4, len(distribution) * 0.3))
        )
        axis.barh(
            distribution["category"].astype(str).iloc[::-1],
            distribution["applications"].iloc[::-1],
            color="#11a8c4",
        )
        axis.set(
            title=f"Applications by {CATEGORICAL_FEATURES[column].lower()}",
            xlabel="Applications",
            ylabel=CATEGORICAL_FEATURES[column],
        )
        outputs.append(
            _save_figure(
                figure, directory / f"distribution_by_{column}.png"
            )
        )
        grouped = grouped.sort_values("default_rate", ascending=False).head(20)
        figure, axis = plt.subplots(figsize=(9, max(4, len(grouped) * 0.3)))
        axis.barh(
            grouped["category"].astype(str).iloc[::-1],
            grouped["default_rate"].iloc[::-1],
            color="#e36f58",
        )
        axis.set(
            title=f"Observed default rate by {CATEGORICAL_FEATURES[column].lower()}",
            xlabel="Observed default rate",
            ylabel=CATEGORICAL_FEATURES[column],
            xlim=(0, 1),
        )
        axis.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        )
        outputs.append(_save_figure(figure, directory / f"default_rate_by_{column}.png"))

    return outputs


def generate_eda_reports(
    dataframe: pd.DataFrame, reports_dir: str | Path
) -> tuple[dict[str, Path], list[Path]]:
    """Write all requested Phase 3 reports and aggregated visualizations."""
    data = validate_eda_data(dataframe)
    directory = Path(reports_dir)
    report_paths = write_statistical_reports(data, directory)
    visualization_paths = save_eda_visualizations(
        data, directory / "visualizations"
    )
    return report_paths, visualization_paths


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path
