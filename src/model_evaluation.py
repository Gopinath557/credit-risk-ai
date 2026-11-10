"""Classification evaluation, curves, reports, and visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.model_calibration import calibration_report
from src.threshold_optimization import select_cost_thresholds


def binary_classification_metrics(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate ranking and thresholded metrics, including confusion counts."""
    y_true, y_probability = _validate_inputs(target, probabilities)
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be finite and in [0, 1].")
    y_pred = (y_probability >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    false_negative_rate = float(fn / (fn + tp)) if fn + tp else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def evaluate_models(
    target: Sequence[int] | np.ndarray,
    model_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    thresholds: Mapping[str, float] | float = 0.5,
) -> pd.DataFrame:
    """Evaluate every model on the same untouched target vector."""
    y_true = np.asarray(target)
    rows = []
    for model_name, probabilities in model_probabilities.items():
        threshold = thresholds.get(model_name, 0.5) if isinstance(thresholds, Mapping) else thresholds
        rows.append(
            {
                "model": model_name,
                **binary_classification_metrics(
                    y_true, probabilities, float(threshold)
                ),
            }
        )
    if not rows:
        raise ValueError("No model probabilities were provided.")
    return pd.DataFrame(rows)


def roc_curve_table(
    target: Sequence[int] | np.ndarray,
    model_probabilities: Mapping[str, Sequence[float] | np.ndarray],
) -> pd.DataFrame:
    """Return long-form ROC coordinates and per-model ROC-AUC."""
    y_true = _validate_target(target)
    rows = []
    for model_name, probabilities in model_probabilities.items():
        scores = _validate_probabilities(probabilities, len(y_true))
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc = float(roc_auc_score(y_true, scores))
        rows.extend(
            {
                "model": model_name,
                "false_positive_rate": float(false_positive),
                "true_positive_rate": float(true_positive),
                "threshold": float(threshold),
                "roc_auc": auc,
            }
            for false_positive, true_positive, threshold in zip(fpr, tpr, thresholds)
        )
    if not rows:
        raise ValueError("No model probabilities were provided.")
    return pd.DataFrame(rows)


def precision_recall_curve_table(
    target: Sequence[int] | np.ndarray,
    model_probabilities: Mapping[str, Sequence[float] | np.ndarray],
) -> pd.DataFrame:
    """Return long-form precision-recall coordinates and average precision."""
    y_true = _validate_target(target)
    rows = []
    for model_name, probabilities in model_probabilities.items():
        scores = _validate_probabilities(probabilities, len(y_true))
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        average_precision = float(average_precision_score(y_true, scores))
        for index, (precision_value, recall_value) in enumerate(
            zip(precision, recall)
        ):
            threshold = float(thresholds[index]) if index < len(thresholds) else np.nan
            rows.append(
                {
                    "model": model_name,
                    "precision": float(precision_value),
                    "recall": float(recall_value),
                    "threshold": threshold,
                    "average_precision": average_precision,
                }
            )
    if not rows:
        raise ValueError("No model probabilities were provided.")
    return pd.DataFrame(rows)


def confusion_matrix_figure(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
    model_name: str,
) -> go.Figure:
    """Create interactive confusion matrix with financial error descriptions."""
    metrics = binary_classification_metrics(target, probabilities, threshold)
    z_values = [
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ]
    custom = [
        ["Correct: no default", "False alarm: credit may be declined"],
        ["Missed default: credit loss exposure", "Correctly flagged default"],
    ]
    figure = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=["Predicted Fully Paid", "Predicted Charged Off"],
            y=["Actual Fully Paid", "Actual Charged Off"],
            text=[[str(value) for value in row] for row in z_values],
            texttemplate="%{text}",
            customdata=custom,
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<br>%{customdata}<extra></extra>",
            colorscale="Blues",
            showscale=True,
        )
    )
    figure.update_layout(
        title=f"{model_name} confusion matrix at threshold {threshold:.2f}",
        xaxis_title="Prediction",
        yaxis_title="Observed outcome",
    )
    return figure


def roc_figure(
    target: Sequence[int] | np.ndarray,
    model_probabilities: Mapping[str, Sequence[float] | np.ndarray],
) -> go.Figure:
    """Build an interactive ROC comparison for all supplied models."""
    table = roc_curve_table(target, model_probabilities)
    figure = go.Figure()
    for model_name, group in table.groupby("model", sort=False):
        auc = float(group["roc_auc"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=group["false_positive_rate"],
                y=group["true_positive_rate"],
                mode="lines",
                name=f"{model_name} (AUC {auc:.3f})",
                hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#64748b"},
            name="Chance",
        )
    )
    figure.update_layout(
        title="ROC curves — untouched holdout",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate (recall)",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    return figure


def precision_recall_figure(
    target: Sequence[int] | np.ndarray,
    model_probabilities: Mapping[str, Sequence[float] | np.ndarray],
) -> go.Figure:
    """Build an interactive precision-recall comparison with prevalence baseline."""
    table = precision_recall_curve_table(target, model_probabilities)
    prevalence = float(np.mean(_validate_target(target)))
    figure = go.Figure()
    for model_name, group in table.groupby("model", sort=False):
        average_precision = float(group["average_precision"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=group["recall"],
                y=group["precision"],
                mode="lines",
                name=f"{model_name} (AP {average_precision:.3f})",
            )
        )
    figure.add_hline(
        y=prevalence,
        line_dash="dash",
        annotation_text=f"Default prevalence {prevalence:.3f}",
    )
    figure.update_layout(
        title="Precision-recall curves — untouched holdout",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    return figure


def threshold_figure(comparison: pd.DataFrame) -> go.Figure:
    """Plot recall, precision, and F1 over validation thresholds."""
    required = {"threshold", "recall", "precision", "f1"}
    if not required.issubset(comparison.columns):
        raise ValueError(f"Threshold table must include {sorted(required)}.")
    figure = go.Figure()
    for metric in ("recall", "precision", "f1"):
        figure.add_trace(
            go.Scatter(
                x=comparison["threshold"],
                y=comparison[metric],
                mode="lines+markers",
                name=metric.replace("_", " ").title(),
            )
        )
    figure.update_layout(
        title="Threshold trade-off on validation OOF predictions",
        xaxis_title="Default probability threshold",
        yaxis_title="Validation metric",
        yaxis={"range": [0, 1]},
    )
    return figure


def generate_evaluation_reports(
    validation_target: Sequence[int] | np.ndarray,
    validation_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    holdout_target: Sequence[int] | np.ndarray,
    holdout_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    reports_dir: str | Path,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    calibration_method: str = "sigmoid",
) -> dict[str, Path]:
    """Select thresholds on OOF only; evaluate/calibrate on untouched holdout."""
    thresholds, threshold_table = select_cost_thresholds(
        validation_target,
        validation_probabilities,
        false_negative_cost,
        false_positive_cost,
    )
    detailed = evaluate_models(holdout_target, holdout_probabilities, thresholds)
    calibration_rows, _ = calibration_report(
        validation_target,
        validation_probabilities,
        holdout_target,
        holdout_probabilities,
        method=calibration_method,
    )
    risk_lines = _business_risk_lines(
        detailed,
        false_negative_cost,
        false_positive_cost,
        thresholds,
    )

    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "detailed_evaluation": directory / "detailed_evaluation.csv",
        "threshold_comparison": directory / "threshold_comparison.csv",
        "calibration_report": directory / "calibration_report.csv",
        "business_risk_report": directory / "business_risk_report.txt",
        "roc_curves": directory / "roc_curves.csv",
        "precision_recall_curves": directory / "precision_recall_curves.csv",
    }
    detailed.to_csv(paths["detailed_evaluation"], index=False)
    threshold_table.to_csv(paths["threshold_comparison"], index=False)
    calibration_rows.to_csv(paths["calibration_report"], index=False)
    roc_curve_table(holdout_target, holdout_probabilities).to_csv(
        paths["roc_curves"], index=False
    )
    precision_recall_curve_table(
        holdout_target, holdout_probabilities
    ).to_csv(paths["precision_recall_curves"], index=False)
    paths["business_risk_report"].write_text(
        "\n".join(risk_lines), encoding="utf-8"
    )
    visualizations = save_evaluation_visualizations(
        holdout_target,
        holdout_probabilities,
        detailed,
        threshold_table,
        calibration_rows,
        directory / "visualizations",
    )
    for name, path in visualizations.items():
        paths[name] = path
    return paths


def save_evaluation_visualizations(
    target: Sequence[int] | np.ndarray,
    model_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    detailed: pd.DataFrame,
    threshold_comparison: pd.DataFrame,
    calibration_data: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save confusion, ROC, PR, threshold, metric, and calibration charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = _validate_target(target)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    confusion_path = output / "confusion_matrix_comparison.png"
    count = len(model_probabilities)
    figure, axes = plt.subplots(1, count, figsize=(5 * count, 4), squeeze=False)
    for axis, row in zip(axes[0], detailed.itertuples(index=False)):
        matrix = [
            [row.true_negative, row.false_positive],
            [row.false_negative, row.true_positive],
        ]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Paid", "Charged Off"],
            yticklabels=["Paid", "Charged Off"],
            ax=axis,
        )
        axis.set(
            title=f"{row.model}\nthreshold={row.threshold:.2f}",
            xlabel="Predicted",
            ylabel="Actual",
        )
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    paths["confusion_matrix_visualization"] = confusion_path

    roc_path = output / "roc_curves.png"
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name, probabilities in model_probabilities.items():
        from sklearn.metrics import RocCurveDisplay

        RocCurveDisplay.from_predictions(
            labels, probabilities, name=model_name, ax=axis
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(title="ROC curves — untouched holdout")
    figure.tight_layout()
    figure.savefig(roc_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    paths["roc_visualization"] = roc_path

    pr_path = output / "precision_recall_curves.png"
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name, probabilities in model_probabilities.items():
        from sklearn.metrics import PrecisionRecallDisplay

        PrecisionRecallDisplay.from_predictions(
            labels, probabilities, name=model_name, ax=axis
        )
    axis.axhline(float(labels.mean()), linestyle="--", color="gray", label="Prevalence")
    axis.set(title="Precision-recall curves — untouched holdout")
    figure.tight_layout()
    figure.savefig(pr_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    paths["precision_recall_visualization"] = pr_path

    threshold_path = output / "threshold_comparison.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name, group in threshold_comparison.groupby("model", sort=False):
        axis.plot(group["threshold"], group["recall"], marker="o", label=f"{model_name} recall")
        axis.plot(group["threshold"], group["precision"], marker=".", linestyle="--", label=f"{model_name} precision")
    axis.set(
        title="Threshold trade-offs — validation OOF predictions",
        xlabel="Probability threshold",
        ylabel="Metric",
        ylim=(0, 1),
    )
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(threshold_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    paths["threshold_visualization"] = threshold_path

    metric_path = output / "metric_comparison.png"
    figure, axis = plt.subplots(figsize=(10, 5))
    metric_frame = detailed.melt(
        id_vars="model",
        value_vars=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "specificity",
        ],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=metric_frame, x="metric", y="score", hue="model", ax=axis)
    axis.set(title="Evaluation metrics — untouched holdout", ylim=(0, 1), ylabel="Score")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(metric_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    paths["metric_comparison_visualization"] = metric_path

    calibration_path = output / "calibration_curves.png"
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for model_name, group in calibration_data.groupby("model", sort=False):
        raw = group.loc[group["probability_type"] == "raw"].dropna(
            subset=["mean_predicted_probability", "observed_default_rate"]
        )
        calibrated = group.loc[
            group["probability_type"] == "calibrated"
        ].dropna(subset=["mean_predicted_probability", "observed_default_rate"])
        if not raw.empty:
            axis.plot(
                raw["mean_predicted_probability"],
                raw["observed_default_rate"],
                marker="o",
                label=f"{model_name} raw",
            )
        if not calibrated.empty:
            axis.plot(
                calibrated["mean_predicted_probability"],
                calibrated["observed_default_rate"],
                marker=".",
                linestyle="--",
                label=f"{model_name} calibrated",
            )
    axis.set(
        title="Reliability — untouched holdout",
        xlabel="Mean predicted default probability",
        ylabel="Observed default frequency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(calibration_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    paths["calibration_visualization"] = calibration_path
    return paths


def _business_risk_lines(
    detailed: pd.DataFrame,
    false_negative_cost: float,
    false_positive_cost: float,
    thresholds: Mapping[str, float],
) -> list[str]:
    lines = [
        "Credit-risk business trade-off report",
        "=====================================",
        "Thresholds are selected from validation out-of-fold probabilities only.",
        "Final metrics below use the untouched holdout at each validation-selected threshold.",
        f"Assumed relative false-negative cost: {false_negative_cost:g} units per missed default.",
        f"Assumed relative false-positive cost: {false_positive_cost:g} units per false alarm.",
        "These are configurable relative cost weights, not currency values or observed losses.",
        "",
        "A false negative is a charged-off loan classified as paid and represents "
        "unidentified credit-loss exposure. A false positive is a fully paid loan "
        "classified as charged off and represents a potentially declined or "
        "unnecessarily reviewed borrower.",
        "",
    ]
    for row in detailed.itertuples(index=False):
        expected_units = (
            row.false_negative * false_negative_cost
            + row.false_positive * false_positive_cost
        )
        lines.extend(
            [
                f"{row.model} at threshold {thresholds[row.model]:.2f}:",
                f"  Default detection rate (recall): {row.recall:.2%}",
                f"  False alarm rate among paid (FPR): {row.false_positive_rate:.2%}",
                f"  Missed defaults (FN): {row.false_negative}",
                f"  False alarms (FP): {row.false_positive}",
                f"  Relative cost units: {expected_units:g}",
                f"  MCC reliability summary: {row.matthews_correlation_coefficient:.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "These results do not estimate actual financial losses. Real costs vary "
            "with exposure at default, loss-given-default, underwriting policy, "
            "review costs, and borrower outcomes; provide institution-specific "
            "cost data before operational use.",
        ]
    )
    return lines


def _validate_inputs(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _validate_target(target)
    scores = _validate_probabilities(probabilities, len(labels))
    if len(np.unique(labels)) != 2:
        raise ValueError("Evaluation requires both target classes.")
    return labels, scores


def _validate_target(target: Sequence[int] | np.ndarray) -> np.ndarray:
    labels = np.asarray(target)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Target must be a non-empty 1-D array.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Target must contain only binary values 0 and 1.")
    return labels.astype("int8")


def _validate_probabilities(
    probabilities: Sequence[float] | np.ndarray, expected_length: int
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or len(scores) != expected_length:
        raise ValueError("Probabilities must be a 1-D array matching target length.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return scores
