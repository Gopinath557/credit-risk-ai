import json

import numpy as np
import pandas as pd
import pytest

from src.model_selection import comparison_table, select_best_model
from src.model_training import train_models
from src.model_utils import (
    calculate_binary_metrics,
    load_model,
    predict_applicants,
    stratified_train_test_split,
    validate_modeling_data,
)
from src.smote_handler import class_distribution, create_sampling_plan


@pytest.fixture
def mixed_training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 90
    target = np.array([0] * 60 + [1] * 30)
    rng.shuffle(target)
    loan_amount = rng.normal(12_000, 3_000, rows) + target * 2_500
    income = rng.normal(65_000, 15_000, rows)
    grade = np.where(target == 1, rng.choice(["C", "D", "E"], rows), rng.choice(["A", "B", "C"], rows))
    return pd.DataFrame(
        {
            "loan_amnt": loan_amount,
            "annual_inc": income,
            "int_rate": np.where(target == 1, 16, 9) + rng.normal(0, 1, rows),
            "grade": grade,
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], rows),
            "employment_length_years": rng.choice([1, 3, 5, 10, np.nan], rows),
            "target": target,
        }
    )


def test_stratified_split_is_reproducible_and_preserves_holdout_distribution(
    mixed_training_frame: pd.DataFrame,
) -> None:
    features, target = validate_modeling_data(mixed_training_frame)
    first = stratified_train_test_split(features, target, random_state=123)
    second = stratified_train_test_split(features, target, random_state=123)

    x_train, x_test, y_train, y_test = first
    assert len(x_test) == 18
    assert len(x_train) == 72
    assert y_test.value_counts(normalize=True).to_dict() == target.value_counts(normalize=True).to_dict()
    pd.testing.assert_frame_equal(x_train, second[0])
    pd.testing.assert_series_equal(y_test, second[3])


def test_smote_plan_uses_smote_nc_and_handles_small_minority() -> None:
    mixed_plan = create_sampling_plan(
        np.array([0] * 20 + [1] * 4),
        numeric_count=2,
        categorical_count=3,
        random_state=11,
        requested_neighbors=5,
    )
    fallback_plan = create_sampling_plan(
        np.array([0] * 5 + [1]),
        numeric_count=1,
        categorical_count=1,
        random_state=11,
    )
    categorical_plan = create_sampling_plan(
        np.array([0] * 12 + [1] * 4),
        numeric_count=0,
        categorical_count=2,
        random_state=11,
    )

    assert mixed_plan.name == "SMOTENC (k_neighbors=3)"
    assert mixed_plan.categorical_indices == (2, 3, 4)
    assert "RandomOverSampler" in fallback_plan.name
    assert "SMOTEN" in categorical_plan.name
    assert mixed_plan.after_estimate == {0: 20, 1: 20}


def test_class_distribution_is_counted_correctly() -> None:
    distribution = class_distribution(np.array([0, 0, 0, 1]))

    assert distribution["applications"].tolist() == [3, 1]
    assert distribution["share"].tolist() == pytest.approx([0.75, 0.25])


def test_training_rejects_leakage_columns(mixed_training_frame: pd.DataFrame) -> None:
    leaky = mixed_training_frame.assign(total_pymnt=1234)

    with pytest.raises(ValueError, match="target leakage"):
        validate_modeling_data(leaky)


def test_model_selection_does_not_choose_accuracy_only() -> None:
    metrics = {
        "high accuracy low recall": {
            "accuracy": 0.99,
            "recall": 0.2,
            "precision": 0.9,
            "f1": 0.33,
            "roc_auc": 0.7,
            "average_precision": 0.3,
        },
        "risk recall model": {
            "accuracy": 0.8,
            "recall": 0.85,
            "precision": 0.5,
            "f1": 0.63,
            "roc_auc": 0.82,
            "average_precision": 0.68,
        },
    }

    selected, rationale = select_best_model(metrics)
    table = comparison_table(metrics)

    assert selected == "risk recall model"
    assert "recall" in rationale.lower()
    assert table.iloc[0]["model"] == selected


def test_metric_calculation_and_invalid_probability() -> None:
    metrics = calculate_binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        np.array([0.1, 0.6, 0.7, 0.9]),
    )

    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["average_precision"] >= 0
    with pytest.raises(ValueError, match="probabilities"):
        calculate_binary_metrics(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0.2, 1.1]),
        )


def test_all_models_train_and_artifacts_round_trip(
    mixed_training_frame: pd.DataFrame, tmp_path
) -> None:
    result = train_models(
        mixed_training_frame,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        random_state=7,
        cv_splits=2,
        requested_neighbors=3,
    )

    assert set(result.comparison["model"]) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }
    assert set(result.comparison.columns).issuperset(
        {"accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"}
    )
    assert result.train_size == 72
    assert result.test_size == 18
    assert result.class_distribution_before["applications"].sum() == result.train_size
    assert result.class_distribution_after["applications"].nunique() == 1
    assert all(path.is_file() for path in result.artifacts.values())
    assert all(path.is_file() for path in result.reports.values())
    validation_scores = pd.read_csv(result.validation_predictions)
    holdout_scores = pd.read_csv(result.holdout_predictions)
    expected_score_columns = {
        "target",
        "probability__Logistic Regression",
        "probability__Decision Tree",
        "probability__Random Forest",
    }
    assert set(validation_scores.columns) == expected_score_columns
    assert set(holdout_scores.columns) == expected_score_columns
    assert len(validation_scores) == result.train_size
    assert len(holdout_scores) == result.test_size
    assert validation_scores["target"].isin([0, 1]).all()
    assert holdout_scores["target"].isin([0, 1]).all()
    probability_columns = sorted(expected_score_columns - {"target"})
    assert validation_scores[probability_columns].apply(
        lambda column: column.between(0, 1).all()
    ).all()
    assert holdout_scores[probability_columns].apply(
        lambda column: column.between(0, 1).all()
    ).all()
    metadata = json.loads(result.artifacts["model_metadata"].read_text(encoding="utf-8"))
    assert metadata["model_name"] == result.best_model_name
    assert metadata["test_rows"] == 18
    loaded = load_model(result.artifacts["best_model"])
    applicant_data = mixed_training_frame.drop(columns="target").iloc[:3]
    predictions = predict_applicants(loaded, applicant_data)
    assert predictions["predicted_default"].isin([0, 1]).all()
    assert predictions["default_probability"].between(0, 1).all()

    repeat = train_models(
        mixed_training_frame,
        models_dir=tmp_path / "models_repeat",
        reports_dir=tmp_path / "reports_repeat",
        random_state=7,
        cv_splits=2,
        requested_neighbors=3,
    )
    pd.testing.assert_frame_equal(
        result.comparison,
        repeat.comparison,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_prediction_rejects_target_and_wrong_input_columns(
    mixed_training_frame: pd.DataFrame, tmp_path
) -> None:
    result = train_models(
        mixed_training_frame,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        cv_splits=2,
    )

    with pytest.raises(ValueError, match="target column"):
        predict_applicants(result.best_pipeline, mixed_training_frame.iloc[:2])
    with pytest.raises(ValueError, match="feature columns"):
        predict_applicants(
            result.best_pipeline,
            pd.DataFrame({"loan_amnt": [2000]}),
        )
