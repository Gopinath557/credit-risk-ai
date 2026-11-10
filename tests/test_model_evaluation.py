from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.business_risk_analysis import calculate_business_cost, business_risk_table
from src.model_calibration import (
    apply_probability_calibrator,
    calibration_report,
    fit_probability_calibrator,
)
from src.model_evaluation import (
    binary_classification_metrics,
    confusion_matrix_figure,
    evaluate_models,
    generate_evaluation_reports,
    precision_recall_curve_table,
    roc_curve_table,
)
from src.threshold_optimization import (
    compare_thresholds,
    select_cost_threshold,
    select_cost_thresholds,
)


def test_classification_metrics_include_ranking_and_confusion_results() -> None:
    metrics = binary_classification_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.6, 0.4],
        threshold=0.5,
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(7 / 12)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.5)
    assert metrics["matthews_correlation_coefficient"] == pytest.approx(0)
    assert (
        metrics["true_negative"],
        metrics["false_positive"],
        metrics["false_negative"],
        metrics["true_positive"],
    ) == (1, 1, 1, 1)


@pytest.mark.parametrize(
    ("target", "probabilities", "threshold"),
    [
        ([0, 1], [0.2], 0.5),
        ([0, 0], [0.2, 0.3], 0.5),
        ([0, 1], [0.2, np.nan], 0.5),
        ([0, 1], [0.2, 1.1], 0.5),
        ([0, 1], [0.2, 0.8], 1.1),
        ([0, 2], [0.2, 0.8], 0.5),
    ],
)
def test_metrics_reject_invalid_evaluation_inputs(
    target: list[int], probabilities: list[float], threshold: float
) -> None:
    with pytest.raises(ValueError):
        binary_classification_metrics(target, probabilities, threshold)


def test_curves_and_confusion_chart_have_expected_models_and_counts() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = {
        "Tree": np.array([0.1, 0.8, 0.6, 0.4]),
        "Forest": np.array([0.1, 0.2, 0.8, 0.9]),
    }

    roc = roc_curve_table(target, probabilities)
    pr = precision_recall_curve_table(target, probabilities)
    confusion = confusion_matrix_figure(target, probabilities["Tree"], 0.5, "Tree")
    evaluated = evaluate_models(target, probabilities, {"Tree": 0.5, "Forest": 0.5})

    assert set(roc["model"]) == {"Tree", "Forest"}
    assert set(pr["model"]) == {"Tree", "Forest"}
    assert roc.groupby("model")["roc_auc"].first()["Forest"] == pytest.approx(1.0)
    assert len(confusion.data) == 1
    assert list(confusion.data[0].z[0]) == [1, 1]
    assert set(evaluated["model"]) == {"Tree", "Forest"}


def test_threshold_selection_uses_validation_costs_and_supported_values() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.4, 0.45, 0.8])

    selected, comparison = select_cost_threshold(
        target,
        probabilities,
        false_negative_cost=5,
        false_positive_cost=1,
        thresholds=[0.3, 0.5, 0.7],
    )
    selected_by_model, all_models = select_cost_thresholds(
        target,
        {"Logistic Regression": probabilities},
        thresholds=[0.3, 0.5, 0.7],
    )

    assert selected == 0.3
    assert comparison["threshold_source"].eq("validation_oof").all()
    assert comparison.set_index("threshold").loc[0.3, "total_relative_cost"] == 1
    assert selected_by_model["Logistic Regression"] == 0.3
    assert all_models["model"].eq("Logistic Regression").all()
    assert compare_thresholds(target, probabilities)["threshold"].tolist() == [
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
    ]


def test_threshold_optimization_rejects_bad_costs_and_probabilities() -> None:
    with pytest.raises(ValueError, match="Costs"):
        compare_thresholds([0, 1], [0.2, 0.8], 0, 0)
    with pytest.raises(ValueError, match="Probabilities"):
        compare_thresholds([0, 1], [0.2, 1.2])
    with pytest.raises(ValueError, match="Thresholds"):
        compare_thresholds([0, 1], [0.2, 0.8], thresholds=[-0.1])


def test_relative_cost_calculation_is_explicitly_weighted() -> None:
    target = [0, 0, 1, 1]
    probabilities = [0.1, 0.8, 0.6, 0.4]

    cost = calculate_business_cost(target, probabilities, 0.5, 5, 2)
    table = business_risk_table(
        target,
        {"Tree": probabilities},
        {"Tree": 0.5},
        false_negative_cost=5,
        false_positive_cost=2,
    )

    assert cost["false_negative_relative_cost"] == 5
    assert cost["false_positive_relative_cost"] == 2
    assert cost["total_relative_cost"] == 7
    assert cost["relative_cost_per_application"] == pytest.approx(1.75)
    assert table.loc[0, "total_relative_cost"] == 7
    with pytest.raises(ValueError, match="Costs"):
        calculate_business_cost(target, probabilities, 0.5, -1, 2)


def test_calibration_fit_and_report_use_validation_then_holdout() -> None:
    validation_target = np.array([0] * 6 + [1] * 6)
    validation_probabilities = np.array(
        [0.08, 0.12, 0.18, 0.22, 0.35, 0.4, 0.55, 0.62, 0.7, 0.78, 0.88, 0.95]
    )
    holdout_target = np.array([0] * 5 + [1] * 5)
    holdout_probabilities = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.55, 0.45, 0.6, 0.7, 0.8, 0.9]
    )

    calibrator = fit_probability_calibrator(
        validation_target, validation_probabilities
    )
    calibrated = apply_probability_calibrator(calibrator, holdout_probabilities)
    report, predictions = calibration_report(
        validation_target,
        {"Model": validation_probabilities},
        holdout_target,
        {"Model": holdout_probabilities},
        n_bins=4,
    )

    assert calibrated.shape == holdout_probabilities.shape
    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    assert set(report["probability_type"]) == {"raw", "calibrated"}
    assert report["brier_score"].between(0, 1).all()
    np.testing.assert_allclose(predictions["Model"], calibrated)
    with pytest.raises(ValueError, match="Only sigmoid"):
        fit_probability_calibrator(
            validation_target, validation_probabilities, "isotonic"
        )


def test_evaluation_report_generation_writes_metrics_curves_and_charts(
    tmp_path,
) -> None:
    validation_target = np.array([0] * 8 + [1] * 8)
    validation_probabilities = {
        "Model A": np.array(
            [
                0.05, 0.1, 0.2, 0.25, 0.4, 0.45, 0.55, 0.6,
                0.3, 0.5, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95,
            ]
        )
    }
    holdout_target = np.array([0] * 6 + [1] * 6)
    holdout_probabilities = {
        "Model A": np.array(
            [
                0.1, 0.2, 0.3, 0.4, 0.6, 0.7,
                0.25, 0.55, 0.65, 0.75, 0.85, 0.95,
            ]
        )
    }

    output_paths = generate_evaluation_reports(
        validation_target,
        validation_probabilities,
        holdout_target,
        holdout_probabilities,
        tmp_path,
    )

    required_reports = {
        "detailed_evaluation",
        "threshold_comparison",
        "calibration_report",
        "business_risk_report",
        "roc_curves",
        "precision_recall_curves",
    }
    assert required_reports.issubset(output_paths)
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in output_paths.values()
    )
    evaluation = pd.read_csv(output_paths["detailed_evaluation"])
    threshold_table = pd.read_csv(output_paths["threshold_comparison"])
    assert evaluation.loc[0, "model"] == "Model A"
    assert threshold_table["threshold_source"].eq("validation_oof").all()
    assert "not currency" in output_paths["business_risk_report"].read_text().lower()
    assert output_paths["confusion_matrix_visualization"].parent.name == "visualizations"
