import pandas as pd
import pytest

from src.eda import create_eda_figures, generate_eda_reports
from src.statistical_analysis import (
    analyze_target_distribution,
    create_insights,
    default_rate_analysis,
    describe_numeric_features,
    filter_eda_data,
    numeric_correlations,
    validate_eda_data,
)


@pytest.fixture
def eda_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [0, 0, 1, 1, 1, 0],
            "loan_amnt": [1000, 2000, 3000, 4000, 5000, 6000],
            "annual_inc": [30000, 40000, 35000, 45000, 55000, 65000],
            "int_rate": [7.0, 8.0, 15.0, 14.0, 18.0, 9.0],
            "dti": [5.0, 10.0, 20.0, 25.0, 30.0, 15.0],
            "grade": ["A", "B", "C", "C", "D", "A"],
            "purpose": ["car", "home", "car", "debt", "debt", "home"],
            "home_ownership": ["RENT", "OWN", "RENT", "MORTGAGE", "RENT", "OWN"],
            "verification_status": [
                "Verified",
                "Not Verified",
                "Verified",
                "Source Verified",
                "Verified",
                "Not Verified",
            ],
            "term": ["36 months", "36 months", "60 months", "36 months", "60 months", "36 months"],
            "employment_length_years": [2, 5, 1, 3, 10, 4],
        }
    )


def test_descriptive_statistics_match_known_values(eda_frame: pd.DataFrame) -> None:
    stats = describe_numeric_features(eda_frame).set_index("feature")

    assert stats.loc["loan_amnt", "count"] == 6
    assert stats.loc["loan_amnt", "mean"] == pytest.approx(3500)
    assert stats.loc["loan_amnt", "median"] == pytest.approx(3500)
    assert stats.loc["loan_amnt", "min"] == 1000
    assert stats.loc["loan_amnt", "q1"] == pytest.approx(2250)
    assert stats.loc["loan_amnt", "q3"] == pytest.approx(4750)
    assert stats.loc["loan_amnt", "max"] == 6000
    assert stats.loc["loan_amnt", "mode"] == 1000


def test_categorical_default_rates_use_observed_class_counts(
    eda_frame: pd.DataFrame,
) -> None:
    rates = default_rate_analysis(eda_frame)
    grade_c = rates.loc[
        (rates["feature"] == "grade")
        & (rates["category"] == "C")
    ].iloc[0]
    home_rent = rates.loc[
        (rates["feature"] == "home_ownership")
        & (rates["category"] == "RENT")
    ].iloc[0]

    assert grade_c["applications"] == 2
    assert grade_c["defaults"] == 2
    assert grade_c["default_rate"] == pytest.approx(1.0)
    assert home_rent["applications"] == 3
    assert home_rent["default_rate"] == pytest.approx(2 / 3)


def test_numeric_default_rate_analysis_has_only_valid_groups(
    eda_frame: pd.DataFrame,
) -> None:
    rates = default_rate_analysis(eda_frame)
    loan_amount_rates = rates.loc[rates["feature"] == "loan_amnt"]

    assert not loan_amount_rates.empty
    assert loan_amount_rates["default_rate"].between(0, 1).all()
    assert loan_amount_rates["applications"].sum() == len(eda_frame)


def test_correlation_includes_target_and_known_relationship(
    eda_frame: pd.DataFrame,
) -> None:
    correlations = numeric_correlations(eda_frame)

    assert correlations.loc["target", "target"] == pytest.approx(1)
    assert correlations.loc["loan_amnt", "annual_inc"] > 0


def test_missing_features_and_invalid_targets_are_handled(
    eda_frame: pd.DataFrame,
) -> None:
    reduced = eda_frame.drop(columns=["grade", "purpose", "dti"])
    figures = create_eda_figures(reduced)

    assert "target_distribution" in figures
    assert "histogram_loan_amnt" in figures
    assert "correlation_heatmap" in figures
    assert "count_grade" not in figures
    assert len(validate_eda_data(eda_frame)) == len(eda_frame)
    with pytest.raises(ValueError, match="target"):
        validate_eda_data(pd.DataFrame({"loan_amnt": [1]}))
    with pytest.raises(ValueError, match="valid target"):
        validate_eda_data(pd.DataFrame({"target": [2], "loan_amnt": [1]}))


def test_interactive_charts_cover_available_bivariate_and_multivariate_features(
    eda_frame: pd.DataFrame,
) -> None:
    figures = create_eda_figures(eda_frame)

    assert {
        "target_distribution",
        "histogram_loan_amnt",
        "count_grade",
        "default_rate_grade",
        "correlation_heatmap",
        "income_vs_loan_amount",
        "interest_rate_by_grade",
        "default_rate_by_dti",
        "pairwise_relationships",
    }.issubset(figures)


def test_reports_and_aggregated_visualizations_are_generated(
    eda_frame: pd.DataFrame, tmp_path
) -> None:
    report_paths, chart_paths = generate_eda_reports(eda_frame, tmp_path)

    assert {
        "descriptive_statistics",
        "correlation_matrix",
        "default_rate_analysis",
        "eda_insights",
    }.issubset(report_paths)
    assert all(path.is_file() for path in report_paths.values())
    assert chart_paths
    assert all(path.is_file() for path in chart_paths)
    chart_names = {path.name for path in chart_paths}
    assert "loan_amnt_distribution.png" in chart_names
    assert "distribution_by_grade.png" in chart_names
    assert "default_rate_by_grade.png" in chart_names
    assert "correlation_heatmap.png" in chart_names
    stats = pd.read_csv(report_paths["descriptive_statistics"])
    assert stats.loc[stats["feature"] == "loan_amnt", "mean"].iloc[0] == pytest.approx(3500)
    insights = report_paths["eda_insights"].read_text(encoding="utf-8")
    assert "do not establish causation" in insights


def test_target_distribution_is_truthful(eda_frame: pd.DataFrame) -> None:
    assert analyze_target_distribution(eda_frame) == {
        "Fully Paid": 3,
        "Charged Off": 3,
    }


def test_insights_handle_missing_candidate_features() -> None:
    data = pd.DataFrame({"target": [0, 1, 0, 1], "feature": [1, 2, 3, 4]})

    insights = create_insights(data)

    assert "Loan grade: not present" in insights
    assert "not present in the dataset" in insights


def test_interactive_filter_logic_combines_outcome_category_and_numeric_range(
    eda_frame: pd.DataFrame,
) -> None:
    filtered = filter_eda_data(
        eda_frame,
        outcomes={1},
        category_column="grade",
        categories={"C"},
        numeric_column="loan_amnt",
        numeric_range=(3000, 4000),
    )

    assert filtered["target"].tolist() == [1, 1]
    assert filtered["grade"].tolist() == ["C", "C"]
    assert filtered["loan_amnt"].tolist() == [3000, 4000]


def test_interactive_filters_reject_invalid_bounds_and_columns(
    eda_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="lower bound"):
        filter_eda_data(
            eda_frame, numeric_column="loan_amnt", numeric_range=(5, 1)
        )
    with pytest.raises(ValueError, match="Unknown categorical"):
        filter_eda_data(eda_frame, category_column="missing")
