"""Evidence-based narrative and report generation for model explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.lime_explainer import explain_applicant
from src.model_utils import predict_applicants
from src.shap_explainer import explain_global, explain_individual


def generate_individual_explanation(
    model_name: str,
    probability: float,
    contributions: pd.DataFrame,
    output_scale: str,
    predicted_default: int | None = None,
    max_features: int = 3,
) -> str:
    """Describe only observed local contributions and their documented scale."""
    if not 0 <= probability <= 1:
        raise ValueError("Default probability must be between zero and one.")
    if contributions.empty or not {"feature", "contribution"}.issubset(
        contributions.columns
    ):
        raise ValueError("Contributions must include feature and contribution columns.")
    if max_features < 1:
        raise ValueError("max_features must be at least one.")
    if predicted_default is None:
        predicted_default = int(probability >= 0.5)
    if predicted_default not in {0, 1}:
        raise ValueError("predicted_default must be 0 or 1.")

    effect_scale = {
        "default_probability": "default-probability contribution",
        "log_odds": "log-odds contribution",
        "local_probability_surrogate": "local surrogate contribution",
    }.get(output_scale, "model-score contribution")
    positive = contributions.loc[contributions["contribution"] > 0].head(max_features)
    negative = contributions.loc[contributions["contribution"] < 0].head(max_features)
    outcome = "higher" if predicted_default else "lower"
    sentences = [
        f"{model_name} assigns a {probability:.1%} predicted default probability "
        f"and classifies this applicant as {outcome} risk "
        f"(class {predicted_default}; the model's 0.5 classification threshold)."
    ]
    if not positive.empty:
        names = ", ".join(
            _humanize(str(feature)) for feature in positive["feature"].tolist()
        )
        sentences.append(
            f"Features with positive {effect_scale}s relative to the model baseline "
            f"include {names}."
        )
    else:
        sentences.append("No positive local feature contributions were identified.")
    if not negative.empty:
        names = ", ".join(
            _humanize(str(feature)) for feature in negative["feature"].tolist()
        )
        sentences.append(
            f"Features with negative {effect_scale}s relative to the model baseline "
            f"include {names}."
        )
    else:
        sentences.append("No negative local feature contributions were identified.")
    sentences.append(
        "These values describe how this fitted model uses the supplied features; "
        "they are associations, not causes or definitive reasons for a future default."
    )
    return " ".join(sentences)


def generate_explainability_report(
    model: Any,
    reference_data: pd.DataFrame,
    applicant: pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
    max_global_samples: int = 250,
    max_background_rows: int = 100,
) -> dict[str, Path]:
    """Write global SHAP, individual SHAP/LIME, charts, and narrative outputs."""
    if len(applicant) != 1:
        raise ValueError("Report generation requires exactly one applicant.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    global_result = explain_global(
        model,
        reference_data,
        max_samples=max_global_samples,
        max_background_rows=max_background_rows,
        output_dir=output,
    )
    shap_result = explain_individual(
        model,
        reference_data,
        applicant,
        max_background_rows=max_background_rows,
        output_dir=output,
    )
    lime_path = output / "lime_explanation.html"
    lime_contributions, lime_probability, lime_prediction = explain_applicant(
        model,
        reference_data,
        applicant,
        max_background_rows=max_background_rows,
        output_path=lime_path,
    )
    model_prediction = predict_applicants(model, applicant)
    if not abs(
        lime_probability - float(model_prediction.loc[0, "default_probability"])
    ) < 1e-12:
        raise ValueError("LIME and saved-pipeline probabilities are inconsistent.")

    shap_narrative = generate_individual_explanation(
        model_name,
        shap_result.default_probability,
        shap_result.contributions,
        shap_result.output_scale,
        shap_result.predicted_default,
    )
    lime_narrative = generate_individual_explanation(
        f"{model_name} LIME local surrogate",
        lime_probability,
        lime_contributions,
        "local_probability_surrogate",
        lime_prediction,
    )
    shap_contribution_path = output / "individual_shap_contributions.csv"
    lime_contribution_path = output / "lime_feature_contributions.csv"
    shap_result.contributions.to_csv(shap_contribution_path, index=False)
    lime_contributions.to_csv(lime_contribution_path, index=False)
    report_path = output / "explainability_report.txt"
    report_path.write_text(
        "\n".join(
            [
                "Credit risk explainability report",
                "==================================",
                f"Selected model: {model_name}",
                f"Global SHAP sample size: {global_result.sampled_rows}",
                f"SHAP output scale: {global_result.output_scale}",
                "Global feature importance is mean absolute SHAP contribution, "
                "aggregated across one-hot encoded categories when applicable.",
                "",
                "Global top features:",
                *[
                    f"  {row.feature}: {row.mean_absolute_contribution:.6g} "
                    f"mean absolute {row.output_scale} contribution"
                    for row in global_result.importance.head(10).itertuples(index=False)
                ],
                "",
                "Individual SHAP explanation:",
                shap_narrative,
                "",
                "Individual LIME explanation:",
                lime_narrative,
                "",
                "SHAP and LIME explain fitted model behavior for the selected "
                "inputs. They do not establish causal effects or guarantee future "
                "loan outcomes. SHAP for Logistic Regression is expressed in "
                "log-odds; tree-model SHAP is expressed in default-probability units. "
                "LIME weights describe a local surrogate and are not additive SHAP values.",
            ]
        ),
        encoding="utf-8",
    )

    importance_path = output / "global_feature_importance.csv"
    if not importance_path.is_file():
        raise OSError("Global SHAP importance output was not generated.")
    return {
        "global_feature_importance": importance_path,
        "shap_summary": output / "shap_summary.png",
        "shap_bar": output / "shap_bar.png",
        "shap_waterfall": output / "shap_waterfall.png",
        "shap_force": output / "shap_force.html",
        "individual_shap_contributions": shap_contribution_path,
        "lime_explanation": lime_path,
        "lime_feature_contributions": lime_contribution_path,
        "explainability_report": report_path,
    }


def _humanize(feature_name: str) -> str:
    return feature_name.replace("_", " ")
