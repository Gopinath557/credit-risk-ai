from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.explanation_generator import (
    generate_explainability_report,
    generate_individual_explanation,
)
from src.lime_explainer import explain_applicant
from src.shap_explainer import explain_global, explain_individual
from src.model_training import _build_estimator_pipeline


@pytest.fixture
def fitted_pipelines() -> tuple[pd.DataFrame, object, object]:
    rng = np.random.default_rng(128)
    target = np.array([0] * 40 + [1] * 20, dtype="int64")
    rng.shuffle(target)
    data = pd.DataFrame(
        {
            "loan_amnt": rng.normal(12_000, 2_500, len(target)) + target * 2_000,
            "annual_inc": rng.normal(60_000, 12_000, len(target)),
            "int_rate": rng.normal(11, 3, len(target)) + target * 3,
        }
    )
    pipelines = []
    for estimator in (
        RandomForestClassifier(n_estimators=8, max_depth=3, random_state=21),
        LogisticRegression(max_iter=500, random_state=21),
    ):
        pipeline, _ = _build_estimator_pipeline(
            numeric_features=list(data.columns),
            categorical_features=[],
            classifier=estimator,
            training_target=pd.Series(target),
            random_state=21,
            requested_neighbors=3,
        )
        pipeline.fit(data, target)
        pipelines.append(pipeline)
    return data, pipelines[0], pipelines[1]


def test_pipeline_transform_preserves_order_and_rejects_missing_features(
    fitted_pipelines,
) -> None:
    data, tree_model, _ = fitted_pipelines
    result = explain_individual(tree_model, data, data.iloc[[0]])

    assert result.default_probability == pytest.approx(
        tree_model.predict_proba(data.iloc[[0]])[0, 1]
    )
    assert result.output_scale == "default_probability"
    assert set(result.contributions["feature"]) == set(data.columns)
    with pytest.raises(ValueError, match="missing"):
        explain_individual(
            tree_model,
            data,
            data.drop(columns=["int_rate"]).iloc[[0]],
        )


def test_tree_shap_probability_contributions_are_additive(fitted_pipelines) -> None:
    data, tree_model, _ = fitted_pipelines
    result = explain_individual(tree_model, data, data.iloc[[3]])
    explanation = result.explanation
    reconstructed_probability = float(
        explanation.base_values[0] + explanation.values[0].sum()
    )

    assert reconstructed_probability == pytest.approx(
        result.default_probability, abs=1e-5
    )
    assert (result.contributions["contribution"] > 0).any()
    assert (result.contributions["contribution"] < 0).any()


def test_logistic_shap_contributions_are_explicitly_log_odds(fitted_pipelines) -> None:
    data, _, logistic_model = fitted_pipelines
    result = explain_individual(logistic_model, data, data.iloc[[1]])
    transformed = (
        logistic_model.named_steps["onehot"].transform(
            logistic_model.named_steps["preprocessor"].transform(data.iloc[[1]])
        )
    )
    decision_score = float(
        logistic_model.named_steps["classifier"].decision_function(transformed)[0]
    )

    assert result.output_scale == "log_odds"
    assert result.explanation.base_values[0] + result.explanation.values[0].sum() == pytest.approx(
        decision_score, abs=1e-6
    )
    assert 0 <= result.default_probability <= 1


def test_global_shap_creates_importance_and_summary_visualizations(
    fitted_pipelines, tmp_path
) -> None:
    data, tree_model, _ = fitted_pipelines
    result = explain_global(
        tree_model,
        data,
        max_samples=20,
        max_background_rows=12,
        output_dir=tmp_path,
    )

    assert result.sampled_rows == 20
    assert result.importance["feature"].tolist()[0] in data.columns
    assert set(result.importance["output_scale"]) == {"default_probability"}
    assert (tmp_path / "global_feature_importance.csv").is_file()
    assert (tmp_path / "shap_summary.png").stat().st_size > 0
    assert (tmp_path / "shap_summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "shap_bar.png").stat().st_size > 0


def test_lime_uses_pipeline_prediction_and_writes_html(
    fitted_pipelines, tmp_path
) -> None:
    data, tree_model, _ = fitted_pipelines
    applicant = data.iloc[[2]]
    output_path = tmp_path / "lime.html"
    contributions, probability, prediction = explain_applicant(
        tree_model,
        data,
        applicant,
        num_features=3,
        max_background_rows=40,
        output_path=output_path,
    )

    assert probability == pytest.approx(tree_model.predict_proba(applicant)[0, 1])
    assert prediction == int(tree_model.predict(applicant)[0])
    assert not contributions.empty
    assert contributions["feature"].isin(data.columns).all()
    assert set(contributions["output_scale"]) == {"local_probability_surrogate"}
    assert output_path.is_file()
    assert "<html" in output_path.read_text(encoding="utf-8").lower()
    with pytest.raises(ValueError, match="exactly one"):
        explain_applicant(tree_model, data, data.iloc[:2])


def test_explanation_narrative_uses_actual_positive_and_negative_features() -> None:
    contributions = pd.DataFrame(
        {
            "feature": ["int_rate", "annual_inc", "loan_amnt"],
            "contribution": [0.12, -0.08, 0.02],
        }
    )
    narrative = generate_individual_explanation(
        "Random Forest",
        probability=0.73,
        contributions=contributions,
        output_scale="default_probability",
        predicted_default=1,
        max_features=2,
    )

    assert "73.0%" in narrative
    assert "int rate" in narrative
    assert "annual inc" in narrative
    assert "associations, not causes" in narrative


def test_explainability_report_writes_all_requested_outputs(
    fitted_pipelines, tmp_path
) -> None:
    data, tree_model, _ = fitted_pipelines
    outputs = generate_explainability_report(
        tree_model,
        data,
        data.iloc[[4]],
        "Random Forest",
        tmp_path,
        max_global_samples=20,
        max_background_rows=12,
    )

    expected = {
        "global_feature_importance",
        "shap_summary",
        "shap_bar",
        "shap_waterfall",
        "shap_force",
        "individual_shap_contributions",
        "lime_explanation",
        "lime_feature_contributions",
        "explainability_report",
    }
    assert expected.issubset(outputs)
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    png_signature = b"\x89PNG\r\n\x1a\n"
    for chart_name in ("shap_summary", "shap_bar", "shap_waterfall"):
        assert outputs[chart_name].read_bytes().startswith(png_signature)
    assert outputs["shap_force"].read_text(encoding="utf-8").lstrip().lower().startswith(
        "<html"
    )
    report = outputs["explainability_report"].read_text(encoding="utf-8")
    assert "associations, not causes" in report
    assert "LIME weights describe a local surrogate" in report


def test_unsupported_classifier_is_reported() -> None:
    class UnsupportedClassifier:
        pass

    model = UnsupportedClassifier()
    with pytest.raises((AttributeError, ValueError)):
        explain_individual(model, pd.DataFrame({"value": [1]}), pd.DataFrame({"value": [1]}))
