from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from streamlit.testing.v1 import AppTest

from src.model_training import _build_estimator_pipeline
from src.model_utils import save_model_artifacts

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_initial_app_renders_without_errors() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert any("Credit Risk Intelligence" in title.value for title in app.title)
    assert any("Phases 1–6 are implemented" in item.value for item in app.info)
    assert any("No dataset is present" in item.value for item in app.warning)


def test_eda_page_explains_when_processed_data_is_missing() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    app.sidebar.radio[0].set_value("Exploratory data analysis").run(timeout=10)

    assert not app.exception
    assert any("No cleaned dataset is available yet" in item.value for item in app.info)


def test_training_page_does_not_claim_results_without_processed_data() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    app.sidebar.radio[0].set_value("Model training").run(timeout=10)

    assert not app.exception
    assert any("No Phase 2 processed dataset is available" in item.value for item in app.info)


def test_evaluation_page_does_not_claim_metrics_without_training() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    app.sidebar.radio[0].set_value("Model evaluation").run(timeout=10)

    assert not app.exception
    assert any("No evaluation prediction bundles are available" in item.value for item in app.info)


def test_explainability_page_does_not_claim_results_without_trained_model() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    app.sidebar.radio[0].set_value("Explainable AI").run(timeout=10)

    assert not app.exception
    assert any("No trained model artifact is available" in item.value for item in app.info)


def test_explainability_page_generates_reports_from_saved_model(tmp_path) -> None:
    app_path = tmp_path / "app.py"
    shutil.copy2(APP_PATH, app_path)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    rng = np.random.default_rng(42)
    target = np.array([0] * 30 + [1] * 15)
    rng.shuffle(target)
    features = pd.DataFrame(
        {
            "loan_amnt": rng.normal(10_000, 2_000, len(target)) + target * 1_000,
            "int_rate": rng.normal(10, 2, len(target)) + target * 2,
        }
    )
    training_pipeline, _ = _build_estimator_pipeline(
        numeric_features=list(features.columns),
        categorical_features=[],
        classifier=RandomForestClassifier(
            n_estimators=8,
            max_depth=3,
            random_state=42,
        ),
        training_target=pd.Series(target),
        random_state=42,
        requested_neighbors=3,
    )
    training_pipeline.fit(features, target)
    save_model_artifacts(
        training_pipeline,
        list(features.columns),
        {"model_name": "Random Forest"},
        models_dir,
    )
    pd.concat(
        [features, pd.Series(target, name="target")],
        axis=1,
    ).to_csv(dataset_dir / "loan_data_processed.csv", index=False)

    app = AppTest.from_file(str(app_path)).run(timeout=10)
    app.sidebar.radio[0].set_value("Explainable AI").run(timeout=20)
    assert not app.exception
    explain_button = next(
        button
        for button in app.button
        if button.label == "Generate SHAP and LIME explanations"
    )
    explain_button.click().run(timeout=60)

    assert not app.exception
    assert any("Global feature importance" in item.value for item in app.subheader)
    assert (reports_dir / "explainability" / "global_feature_importance.csv").is_file()
    assert (reports_dir / "explainability" / "lime_explanation.html").is_file()
    assert (reports_dir / "explainability" / "shap_waterfall.png").is_file()


def test_evaluation_dashboard_renders_saved_validation_and_holdout_scores(
    tmp_path,
) -> None:
    app_path = tmp_path / "app.py"
    shutil.copy2(APP_PATH, app_path)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    validation = pd.DataFrame(
        {
            "target": [0] * 8 + [1] * 8,
            "probability__Logistic Regression": [
                0.05, 0.1, 0.2, 0.25, 0.4, 0.45, 0.55, 0.6,
                0.3, 0.5, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95,
            ],
        }
    )
    holdout = pd.DataFrame(
        {
            "target": [0] * 6 + [1] * 6,
            "probability__Logistic Regression": [
                0.1, 0.2, 0.3, 0.4, 0.6, 0.7,
                0.25, 0.55, 0.65, 0.75, 0.85, 0.95,
            ],
        }
    )
    validation.to_csv(reports_dir / "validation_oof_predictions.csv", index=False)
    holdout.to_csv(reports_dir / "holdout_predictions.csv", index=False)

    app = AppTest.from_file(str(app_path)).run(timeout=10)
    app.sidebar.radio[0].set_value("Model evaluation").run(timeout=20)

    assert not app.exception
    assert any("Final metrics — untouched holdout" in item.value for item in app.subheader)
    assert any("Probability calibration" in item.value for item in app.subheader)
    assert len(app.get("plotly_chart")) >= 5


def test_training_page_does_not_claim_placeholder_files_are_results(tmp_path) -> None:
    app_path = tmp_path / "app.py"
    shutil.copy2(APP_PATH, app_path)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (tmp_path / "reports").mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    pd.DataFrame(
        {
            "loan_amnt": [1000, 2000, 3000, 4000],
            "target": [0, 0, 1, 1],
        }
    ).to_csv(dataset_dir / "loan_data_processed.csv", index=False)
    (tmp_path / "reports" / "model_comparison.csv").write_text(
        "model,recall,precision\n", encoding="utf-8"
    )
    (models_dir / "model_metadata.json").write_text(
        '{"training_status":"not_run","model_name":null}', encoding="utf-8"
    )

    app = AppTest.from_file(str(app_path)).run(timeout=10)
    app.sidebar.radio[0].set_value("Model training").run(timeout=10)

    assert not app.exception
    assert any("No completed model training run" in item.value for item in app.info)


def test_eda_dashboard_renders_with_processed_data(tmp_path) -> None:
    app_path = tmp_path / "app.py"
    shutil.copy2(APP_PATH, app_path)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (tmp_path / "reports").mkdir()
    pd.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "loan_amnt": [1000, 2000, 3000, 4000],
            "annual_inc": [30000, 40000, 35000, 45000],
            "int_rate": [7.0, 8.0, 15.0, 14.0],
            "dti": [5.0, 10.0, 20.0, 25.0],
            "grade": ["A", "B", "C", "D"],
            "purpose": ["car", "home", "car", "debt"],
            "home_ownership": ["RENT", "OWN", "RENT", "MORTGAGE"],
        }
    ).to_csv(dataset_dir / "loan_data_processed.csv", index=False)

    app = AppTest.from_file(str(app_path)).run(timeout=20)
    app.sidebar.radio[0].set_value("Exploratory data analysis").run(timeout=20)

    assert not app.exception
    assert any("Filtered dataset overview" in item.value for item in app.subheader)
    assert any("Descriptive statistics" in item.value for item in app.subheader)
    app.segmented_control[0].set_value("Relationships").run(timeout=20)
    assert not app.exception
    assert len(app.get("plotly_chart")) > 0
