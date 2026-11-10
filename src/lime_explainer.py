"""LIME local explanations using the trained pipeline's fitted transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.model_utils import predict_applicants
from src.shap_explainer import transform_applicants


def explain_applicant(
    model: Any,
    reference_data: pd.DataFrame,
    applicant: pd.DataFrame,
    num_features: int = 10,
    max_background_rows: int = 1000,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, float, int]:
    """Return local class-1 LIME contributions for exactly one applicant."""
    if len(applicant) != 1:
        raise ValueError("LIME explanation requires exactly one applicant.")
    if num_features < 1 or max_background_rows < 1:
        raise ValueError("LIME feature and background limits must be positive.")
    if reference_data.empty:
        raise ValueError("Reference data is empty.")
    pipeline_features = list(getattr(model, "feature_names_in_", []))
    if not pipeline_features:
        raise ValueError("The saved model does not expose its fitted input features.")
    missing = [column for column in pipeline_features if column not in reference_data]
    if missing:
        raise ValueError(f"Reference data is missing model features: {missing}.")
    reference = reference_data.loc[:, pipeline_features]
    if len(reference) > max_background_rows:
        reference = reference.sample(
            n=max_background_rows, random_state=random_state, replace=False
        )

    training_values, feature_names, _ = transform_applicants(model, reference)
    applicant_values, applicant_names, _ = transform_applicants(model, applicant)
    if feature_names != applicant_names:
        raise ValueError("Applicant and LIME background feature order differs.")
    classifier = model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "predict_proba"):
        raise ValueError("The saved model pipeline has no probability classifier.")
    explainer = LimeTabularExplainer(
        training_data=training_values,
        feature_names=feature_names,
        class_names=["Fully Paid", "Charged Off"],
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )
    lime_explanation = explainer.explain_instance(
        applicant_values[0],
        classifier.predict_proba,
        labels=(1,),
        num_features=min(num_features, len(feature_names)),
    )
    raw_features = list(getattr(model, "feature_names_in_", []))
    contribution_rows = []
    for transformed_index, contribution in lime_explanation.local_exp[1]:
        transformed_name = feature_names[transformed_index]
        source_feature = _source_feature(transformed_name, raw_features)
        contribution_rows.append(
            {
                "feature": source_feature,
                "condition": lime_explanation.domain_mapper.feature_names[
                    transformed_index
                ],
                "transformed_feature": transformed_name,
                "applicant_value": str(applicant.iloc[0][source_feature]),
                "contribution": float(contribution),
                "output_scale": "local_probability_surrogate",
            }
        )
    contributions = (
        pd.DataFrame(contribution_rows)
        .sort_values("contribution", key=lambda column: column.abs(), ascending=False)
        .reset_index(drop=True)
    )
    prediction = predict_applicants(model, applicant)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(lime_explanation.as_html(labels=(1,)), encoding="utf-8")
    return (
        contributions,
        float(prediction.loc[0, "default_probability"]),
        int(prediction.loc[0, "predicted_default"]),
    )


def _source_feature(encoded_name: str, raw_features: list[str]) -> str:
    for candidate in sorted(raw_features, key=len, reverse=True):
        if encoded_name == candidate or encoded_name.startswith(f"{candidate}_"):
            return candidate
    return encoded_name
