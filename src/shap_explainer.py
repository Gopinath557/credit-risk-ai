"""SHAP explanations for the fitted credit-risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.model_utils import predict_applicants


@dataclass
class ShapResult:
    """A normalized positive-class SHAP explanation and its output scale."""

    explanation: shap.Explanation
    contributions: pd.DataFrame
    default_probability: float
    predicted_default: int
    output_scale: str


@dataclass
class GlobalShapResult:
    """Aggregate global importance and sampled SHAP values."""

    importance: pd.DataFrame
    explanation: shap.Explanation
    output_scale: str
    sampled_rows: int


def transform_applicants(
    model: Any,
    applicants: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Apply the fitted preprocessing and one-hot steps in the saved pipeline."""
    if applicants.empty:
        raise ValueError("At least one applicant row is required for explanation.")
    if "target" in applicants.columns:
        raise ValueError("Applicant features must not include the target column.")

    pipeline_features = list(getattr(model, "feature_names_in_", []))
    if not pipeline_features:
        preprocessor = model.named_steps.get("preprocessor")
        pipeline_features = list(getattr(preprocessor, "feature_names_in_", []))
    if not pipeline_features:
        raise ValueError("The saved model does not expose its fitted input features.")
    missing = [column for column in pipeline_features if column not in applicants]
    extra = [column for column in applicants if column not in pipeline_features]
    if missing or extra:
        raise ValueError(
            "Applicant feature columns do not match the saved model; "
            f"missing={missing}, extra={extra}."
        )

    ordered = applicants.loc[:, pipeline_features]
    try:
        preprocessed = model.named_steps["preprocessor"].transform(ordered)
        transformed = model.named_steps["onehot"].transform(preprocessed)
        names = list(
            model.named_steps["onehot"].get_feature_names_out(pipeline_features)
        )
    except (AttributeError, KeyError, ValueError) as error:
        raise ValueError(f"Could not transform applicant features: {error}") from error
    values = np.asarray(transformed, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("The saved model's transformed feature schema is invalid.")
    return values, names, pipeline_features


def explain_individual(
    model: Any,
    reference_data: pd.DataFrame,
    applicant: pd.DataFrame,
    max_background_rows: int = 100,
    random_state: int = 42,
    output_dir: str | Path | None = None,
    max_display: int = 15,
) -> ShapResult:
    """Explain one applicant's positive-class prediction using the saved pipeline."""
    if len(applicant) != 1:
        raise ValueError("Individual SHAP explanation requires exactly one applicant.")
    explainer, background_features, feature_names, output_scale = _create_explainer(
        model, reference_data, max_background_rows, random_state
    )
    applicant_features, applicant_names, _ = transform_applicants(model, applicant)
    if applicant_names != feature_names:
        raise ValueError("Applicant and SHAP background feature order differs.")

    raw_explanation = explainer(applicant_features)
    normalized = _positive_class_explanation(
        raw_explanation,
        feature_names,
        applicant_features,
        output_scale,
    )
    predictions = predict_applicants(model, applicant)
    contributions = _local_contribution_table(
        model,
        applicant.iloc[0],
        feature_names,
        np.asarray(normalized.values)[0],
        output_scale,
    )
    result = ShapResult(
        explanation=normalized,
        contributions=contributions,
        default_probability=float(predictions.loc[0, "default_probability"]),
        predicted_default=int(predictions.loc[0, "predicted_default"]),
        output_scale=output_scale,
    )
    if output_dir is not None:
        _save_local_plots(result, output_dir, max_display)
    return result


def explain_global(
    model: Any,
    reference_data: pd.DataFrame,
    max_samples: int = 250,
    max_background_rows: int = 100,
    random_state: int = 42,
    output_dir: str | Path | None = None,
    max_display: int = 20,
) -> GlobalShapResult:
    """Compute global feature impact from a deterministic sample of real rows."""
    if max_samples < 1:
        raise ValueError("max_samples must be at least one.")
    if reference_data.empty:
        raise ValueError("Reference data is empty.")
    rng = np.random.default_rng(random_state)
    sample_count = min(max_samples, len(reference_data))
    if sample_count == len(reference_data):
        sample = reference_data.copy()
    else:
        selected = np.sort(rng.choice(len(reference_data), sample_count, replace=False))
        sample = reference_data.iloc[selected].copy()

    explainer, _, feature_names, output_scale = _create_explainer(
        model, reference_data, max_background_rows, random_state
    )
    transformed_sample, sample_names, _ = transform_applicants(model, sample)
    if sample_names != feature_names:
        raise ValueError("Global sample and SHAP background feature order differs.")
    raw_explanation = explainer(transformed_sample)
    normalized = _positive_class_explanation(
        raw_explanation,
        feature_names,
        transformed_sample,
        output_scale,
    )
    importance = _global_importance(model, normalized, feature_names, output_scale)
    result = GlobalShapResult(
        importance=importance,
        explanation=normalized,
        output_scale=output_scale,
        sampled_rows=sample_count,
    )
    if output_dir is not None:
        _save_global_plots(result, output_dir, max_display)
    return result


def _create_explainer(
    model: Any,
    reference_data: pd.DataFrame,
    max_background_rows: int,
    random_state: int,
) -> tuple[Any, np.ndarray, list[str], str]:
    if max_background_rows < 1:
        raise ValueError("max_background_rows must be at least one.")
    if reference_data.empty:
        raise ValueError("Reference data is empty.")
    pipeline_features = list(getattr(model, "feature_names_in_", []))
    if not pipeline_features:
        preprocessor = model.named_steps.get("preprocessor")
        pipeline_features = list(getattr(preprocessor, "feature_names_in_", []))
    if not pipeline_features:
        raise ValueError("The saved model does not expose its fitted input features.")
    missing = [column for column in pipeline_features if column not in reference_data]
    if missing:
        raise ValueError(
            f"Reference data is missing model features: {missing}."
        )
    reference = reference_data.loc[:, pipeline_features]
    background_count = min(max_background_rows, len(reference))
    if background_count == len(reference):
        background = reference.copy()
    else:
        background = reference.sample(
            n=background_count, random_state=random_state, replace=False
        )
    transformed_background, feature_names, _ = transform_applicants(model, background)
    classifier = model.named_steps.get("classifier")
    if classifier is None:
        raise ValueError("The saved model pipeline has no classifier step.")
    if isinstance(classifier, (RandomForestClassifier, DecisionTreeClassifier)):
        explainer = shap.TreeExplainer(
            classifier,
            data=transformed_background,
            feature_perturbation="interventional",
            model_output="probability",
        )
        output_scale = "default_probability"
    elif isinstance(classifier, LogisticRegression):
        explainer = shap.LinearExplainer(classifier, transformed_background)
        output_scale = "log_odds"
    else:
        raise ValueError(
            "SHAP currently supports the trained Logistic Regression, Decision "
            "Tree, and Random Forest classifiers."
        )
    return explainer, transformed_background, feature_names, output_scale


def _positive_class_explanation(
    explanation: shap.Explanation,
    feature_names: list[str],
    transformed_values: np.ndarray,
    output_scale: str,
) -> shap.Explanation:
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        if values.shape[-1] < 2:
            raise ValueError("SHAP output does not contain a default class.")
        values = values[..., 1]
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(
            f"Unexpected SHAP value shape {values.shape} for {len(feature_names)} features."
        )
    if base_values.ndim == 2:
        if base_values.shape[-1] < 2:
            raise ValueError("SHAP base values do not contain a default class.")
        base_values = base_values[:, 1]
    elif base_values.ndim == 0:
        base_values = np.full(values.shape[0], float(base_values))
    if base_values.shape != (values.shape[0],):
        raise ValueError(f"Unexpected SHAP base-value shape {base_values.shape}.")
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=transformed_values,
        feature_names=feature_names,
        output_names=f"Charged Off ({output_scale})",
    )


def _source_feature(encoded_name: str, raw_features: list[str]) -> str:
    for candidate in sorted(raw_features, key=len, reverse=True):
        if encoded_name == candidate or encoded_name.startswith(f"{candidate}_"):
            return candidate
    return encoded_name


def _local_contribution_table(
    model: Any,
    applicant: pd.Series,
    transformed_names: list[str],
    values: np.ndarray,
    output_scale: str,
) -> pd.DataFrame:
    raw_features = list(getattr(model, "feature_names_in_", []))
    if not raw_features:
        raw_features = list(model.named_steps["preprocessor"].feature_names_in_)
    rows = [
        {
            "feature": _source_feature(name, raw_features),
            "transformed_feature": name,
            "applicant_value": str(applicant[_source_feature(name, raw_features)]),
            "contribution": float(value),
            "output_scale": output_scale,
        }
        for name, value in zip(transformed_names, values)
    ]
    return (
        pd.DataFrame(rows)
        .groupby(["feature", "output_scale"], as_index=False, sort=False)
        .agg(
            contribution=("contribution", "sum"),
            transformed_features=("transformed_feature", lambda items: ", ".join(items)),
            applicant_value=("applicant_value", "first"),
        )
        .sort_values("contribution", key=lambda column: column.abs(), ascending=False)
        .reset_index(drop=True)
    )


def _global_importance(
    model: Any,
    explanation: shap.Explanation,
    transformed_names: list[str],
    output_scale: str,
) -> pd.DataFrame:
    raw_features = list(getattr(model, "feature_names_in_", []))
    if not raw_features:
        raw_features = list(model.named_steps["preprocessor"].feature_names_in_)
    rows = [
        {
            "feature": _source_feature(name, raw_features),
            "transformed_feature": name,
            "mean_absolute_contribution": float(np.mean(np.abs(explanation.values[:, index]))),
            "mean_signed_contribution": float(np.mean(explanation.values[:, index])),
            "output_scale": output_scale,
        }
        for index, name in enumerate(transformed_names)
    ]
    return (
        pd.DataFrame(rows)
        .groupby(["feature", "output_scale"], as_index=False, sort=False)
        .agg(
            mean_absolute_contribution=("mean_absolute_contribution", "sum"),
            mean_signed_contribution=("mean_signed_contribution", "sum"),
            transformed_features=("transformed_feature", lambda items: ", ".join(items)),
        )
        .sort_values("mean_absolute_contribution", ascending=False)
        .reset_index(drop=True)
    )


def _save_local_plots(
    result: ShapResult,
    output_dir: str | Path,
    max_display: int,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "waterfall": output / "shap_waterfall.png",
        "force": output / "shap_force.html",
    }
    shap.plots.waterfall(
        result.explanation[0],
        max_display=min(max_display, len(result.explanation.feature_names)),
        show=False,
    )
    figure = plt.gcf()
    figure.savefig(paths["waterfall"], dpi=140, bbox_inches="tight")
    plt.close(figure)
    try:
        force_plot = shap.plots.force(result.explanation[0])
        shap.save_html(str(paths["force"]), force_plot)
    except (AttributeError, TypeError, ValueError) as error:
        paths["force"].write_text(
            "Force visualization is unavailable for this SHAP version/model: "
            f"{error}",
            encoding="utf-8",
        )
    return paths


def _save_global_plots(
    result: GlobalShapResult,
    output_dir: str | Path,
    max_display: int,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output / "shap_summary.png",
        "bar": output / "shap_bar.png",
        "importance": output / "global_feature_importance.csv",
    }
    result.importance.to_csv(paths["importance"], index=False)
    shap.plots.beeswarm(
        result.explanation,
        max_display=min(max_display, len(result.explanation.feature_names)),
        show=False,
    )
    figure = plt.gcf()
    figure.savefig(paths["summary"], dpi=140, bbox_inches="tight")
    plt.close(figure)
    shap.plots.bar(
        result.explanation,
        max_display=min(max_display, len(result.explanation.feature_names)),
        show=False,
    )
    figure = plt.gcf()
    figure.savefig(paths["bar"], dpi=140, bbox_inches="tight")
    plt.close(figure)
    return paths
