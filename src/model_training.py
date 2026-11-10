"""Leakage-aware training, cross-validation, comparison, and artifact creation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.model_selection import comparison_table, select_best_model
from src.model_utils import (
    calculate_binary_metrics,
    save_model_artifacts,
    stratified_train_test_split,
    validate_modeling_data,
)
from src.preprocessing import feature_columns
from src.smote_handler import (
    build_sampling_preprocessor,
    class_distribution,
    create_sampling_plan,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_MINIMUM_RECALL: float | None = None


@dataclass
class TrainingResult:
    """In-memory result for an actual completed model-training run."""

    best_model_name: str
    best_pipeline: ImbalancedPipeline
    comparison: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    best_parameters: dict[str, dict[str, Any]]
    cross_validation_scores: dict[str, float]
    feature_names: list[str]
    class_distribution_before: pd.DataFrame
    class_distribution_after: pd.DataFrame
    sampling_method: str
    train_size: int
    test_size: int
    reports: dict[str, Path]
    artifacts: dict[str, Path]
    selection_rationale: str
    validation_predictions: Path
    holdout_predictions: Path


def _post_resample_encoder(
    numeric_count: int, categorical_count: int
) -> ColumnTransformer:
    numeric_indices = list(range(numeric_count))
    categorical_indices = list(
        range(numeric_count, numeric_count + categorical_count)
    )
    transformers: list[tuple[str, object, list[int]]] = []
    if numeric_indices:
        transformers.append(("numeric", "passthrough", numeric_indices))
    if categorical_indices:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_indices,
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def _build_estimator_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    classifier: object,
    training_target: pd.Series,
    random_state: int,
    requested_neighbors: int,
) -> tuple[ImbalancedPipeline, str]:
    plan = create_sampling_plan(
        training_target,
        len(numeric_features),
        len(categorical_features),
        random_state,
        requested_neighbors,
    )
    steps = [
        (
            "preprocessor",
            build_sampling_preprocessor(numeric_features, categorical_features),
        ),
        ("sampler", plan.sampler),
        ("onehot", _post_resample_encoder(len(numeric_features), len(categorical_features))),
        ("classifier", classifier),
    ]
    return ImbalancedPipeline(steps), plan.name


def _candidate_grids(random_state: int) -> list[tuple[str, object, list[dict[str, Any]]]]:
    return [
        (
            "Logistic Regression",
            LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                random_state=random_state,
            ),
            [
                {"classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None]},
                {"classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": ["balanced"]},
            ],
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=random_state),
            [
                {
                    "classifier__max_depth": [None, 10],
                    "classifier__min_samples_split": [2, 10],
                    "classifier__min_samples_leaf": [1, 5],
                    "classifier__class_weight": [None, "balanced"],
                }
            ],
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=1,
            ),
            [
                {
                    "classifier__max_depth": [None],
                    "classifier__min_samples_split": [2],
                    "classifier__min_samples_leaf": [1],
                    "classifier__class_weight": [None],
                },
                {
                    "classifier__max_depth": [15],
                    "classifier__min_samples_split": [2],
                    "classifier__min_samples_leaf": [1],
                    "classifier__class_weight": ["balanced"],
                },
                {
                    "classifier__max_depth": [15],
                    "classifier__min_samples_split": [10],
                    "classifier__min_samples_leaf": [3],
                    "classifier__class_weight": [None],
                },
                {
                    "classifier__max_depth": [None],
                    "classifier__min_samples_split": [10],
                    "classifier__min_samples_leaf": [3],
                    "classifier__class_weight": ["balanced"],
                },
            ],
        ),
    ]


def train_models(
    dataframe: pd.DataFrame,
    models_dir: str | Path,
    reports_dir: str | Path,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    cv_splits: int = 3,
    requested_neighbors: int = 5,
    minimum_recall: float | None = DEFAULT_MINIMUM_RECALL,
) -> TrainingResult:
    """Train three tuned models, compare on untouched holdout, and save artifacts."""
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    features, target = validate_modeling_data(dataframe)
    numeric_features, categorical_features = feature_columns(dataframe)
    class_counts = target.value_counts()
    if int(class_counts.min()) < cv_splits:
        raise ValueError(
            f"At least {cv_splits} examples of each class are required for "
            "stratified cross-validation."
        )

    x_train, x_test, y_train, y_test = stratified_train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    before_distribution = class_distribution(y_train)
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    minimum_cv_minority = min(
        int(y_fold_train.value_counts().min())
        for train_indices, _ in cv.split(x_train, y_train)
        for y_fold_train in [y_train.iloc[train_indices]]
    )
    cv_neighbors = min(requested_neighbors, minimum_cv_minority - 1)
    if cv_neighbors < 1:
        raise ValueError(
            "Insufficient minority examples within cross-validation folds "
            "for SMOTE or SMOTENC."
        )
    scorer = "average_precision"
    fitted_models: dict[str, ImbalancedPipeline] = {}
    metrics: dict[str, dict[str, float]] = {}
    validation_metrics: dict[str, dict[str, float]] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    holdout_probabilities: dict[str, np.ndarray] = {}
    best_parameters: dict[str, dict[str, Any]] = {}
    cv_scores: dict[str, float] = {}
    sampling_method = ""

    for name, classifier, parameter_grid in _candidate_grids(random_state):
        pipeline, sampler_name = _build_estimator_pipeline(
            numeric_features,
            categorical_features,
            classifier,
            y_train,
            random_state,
            cv_neighbors,
        )
        search = GridSearchCV(
            pipeline,
            param_grid=parameter_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        estimator = search.best_estimator_
        prediction = estimator.predict(x_test)
        probability = estimator.predict_proba(x_test)[:, 1]
        metrics[name] = calculate_binary_metrics(
            y_test, prediction, probability
        )
        oof_class_probabilities = cross_val_predict(
            estimator,
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )
        positive_class_index = list(estimator.classes_).index(1)
        oof_probability = oof_class_probabilities[:, positive_class_index]
        validation_probabilities[name] = oof_probability
        holdout_probabilities[name] = probability
        validation_metrics[name] = calculate_binary_metrics(
            y_train,
            (oof_probability >= 0.5).astype("int8"),
            oof_probability,
        )
        fitted_models[name] = estimator
        best_parameters[name] = dict(search.best_params_)
        cv_scores[name] = float(search.best_score_)
        sampling_method = sampler_name

    best_name, selection_rationale = select_best_model(
        validation_metrics, minimum_recall=minimum_recall
    )
    best_pipeline = fitted_models[best_name]
    x_train_transformed = best_pipeline.named_steps["preprocessor"].transform(x_train)
    _, y_resampled = best_pipeline.named_steps["sampler"].fit_resample(
        x_train_transformed, y_train
    )
    after_distribution = class_distribution(y_resampled)
    feature_names = list(
        best_pipeline.named_steps["onehot"].get_feature_names_out(
            numeric_features + categorical_features
        )
    )
    report_paths = _write_training_reports(
        reports_dir,
        metrics,
        cv_scores,
        best_parameters,
        best_name,
        selection_rationale,
        before_distribution,
        after_distribution,
        sampling_method,
        len(x_train),
        len(x_test),
        random_state,
        cv_splits,
        minimum_recall,
    )
    importance_paths = _save_training_visualizations(
        reports_dir,
        metrics,
        best_name,
        best_pipeline,
        feature_names,
        before_distribution,
        after_distribution,
        y_test,
        best_pipeline.predict(x_test),
        best_pipeline.predict_proba(x_test)[:, 1],
    )
    report_paths.update(importance_paths)

    metadata = {
        "model_name": best_name,
        "target": {"column": "target", "0": "Fully Paid", "1": "Charged Off"},
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "test_size": test_size,
        "cv_splits": cv_splits,
        "tuning_metric": scorer,
        "selection_rationale": selection_rationale,
        "minimum_recall_requirement": minimum_recall,
        "sampling_method": sampling_method,
        "class_distribution_before_smote": _distribution_mapping(before_distribution),
        "class_distribution_after_smote": _distribution_mapping(after_distribution),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "raw_feature_names": features.columns.tolist(),
        "transformed_feature_names": feature_names,
        "holdout_metrics": metrics[best_name],
        "all_model_holdout_metrics": metrics,
        "all_model_validation_oof_metrics_at_0_5": validation_metrics,
        "cross_validation_average_precision": cv_scores,
        "best_parameters": best_parameters,
        "probability_threshold": 0.5,
        "limitations": [
            "Holdout metrics are from a random stratified split, not temporal validation.",
            "The historical dataset may contain vintage, selection, and reporting bias.",
            "Observed loan outcomes do not establish causal borrower risk effects.",
        ],
    }
    artifact_paths = save_model_artifacts(
        best_pipeline, feature_names, metadata, models_dir
    )
    validation_prediction_path, holdout_prediction_path = _save_prediction_bundles(
        reports_dir,
        y_train,
        validation_probabilities,
        y_test,
        holdout_probabilities,
    )
    report_paths["validation_oof_predictions"] = validation_prediction_path
    report_paths["holdout_predictions"] = holdout_prediction_path
    return TrainingResult(
        best_model_name=best_name,
        best_pipeline=best_pipeline,
        comparison=comparison_table(metrics),
        metrics=metrics,
        best_parameters=best_parameters,
        cross_validation_scores=cv_scores,
        feature_names=feature_names,
        class_distribution_before=before_distribution,
        class_distribution_after=after_distribution,
        sampling_method=sampling_method,
        train_size=len(x_train),
        test_size=len(x_test),
        reports=report_paths,
        artifacts=artifact_paths,
        selection_rationale=selection_rationale,
        validation_predictions=validation_prediction_path,
        holdout_predictions=holdout_prediction_path,
    )


def _save_prediction_bundles(
    reports_dir: str | Path,
    validation_target: pd.Series,
    validation_probabilities: dict[str, np.ndarray],
    holdout_target: pd.Series,
    holdout_probabilities: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    """Persist only labels and scores, never row-level applicant features."""
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    validation = pd.DataFrame({"target": validation_target.to_numpy()})
    for model_name, scores in validation_probabilities.items():
        validation[f"probability__{model_name}"] = scores
    holdout = pd.DataFrame({"target": holdout_target.to_numpy()})
    for model_name, scores in holdout_probabilities.items():
        holdout[f"probability__{model_name}"] = scores
    validation_path = directory / "validation_oof_predictions.csv"
    holdout_path = directory / "holdout_predictions.csv"
    validation.to_csv(validation_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    return validation_path, holdout_path


def _distribution_mapping(dataframe: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        str(row.loan_status): {
            "target": int(row.target),
            "applications": int(row.applications),
        }
        for row in dataframe.itertuples(index=False)
    }


def _write_training_reports(
    reports_dir: str | Path,
    metrics: dict[str, dict[str, float]],
    cv_scores: dict[str, float],
    best_parameters: dict[str, dict[str, Any]],
    best_name: str,
    selection_rationale: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    sampling_method: str,
    train_size: int,
    test_size: int,
    random_state: int,
    cv_splits: int,
    minimum_recall: float | None,
) -> dict[str, Path]:
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    comparison = comparison_table(metrics)
    comparison["cross_validation_average_precision"] = comparison["model"].map(cv_scores)
    comparison["best_parameters"] = comparison["model"].map(
        lambda name: json.dumps(best_parameters[name], sort_keys=True)
    )
    comparison_path = directory / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    distribution_path = directory / "class_distribution.csv"
    pd.concat(
        [
            before.assign(stage="before_resampling"),
            after.assign(stage="after_training_resampling"),
        ],
        ignore_index=True,
    )[["stage", "target", "loan_status", "applications", "share"]].to_csv(
        distribution_path, index=False
    )
    report_path = directory / "training_report.txt"
    report_path.write_text(
        "\n".join(
            [
                "Credit risk model training report",
                "=================================",
                "Dataset: Phase 2 cleaned input provided to training.",
                f"Train rows: {train_size:,}",
                f"Untouched stratified test rows: {test_size:,}",
                f"Random seed: {random_state}",
                f"Stratified CV folds: {cv_splits}",
                "Tuning metric: average precision (PR-AUC).",
                f"Training sampler: {sampling_method}",
                "Resampling is inside the imbalanced-learn pipeline, so it is fit "
                "only on each CV training fold and the final training partition. "
                "The holdout test set is never resampled.",
                "",
                f"Selected model: {best_name}",
                f"Selection rationale: {selection_rationale}",
                (
                    f"Minimum recall constraint: {minimum_recall:.3f}"
                    if minimum_recall is not None
                    else "Minimum recall constraint: none configured"
                ),
                "",
                "Untouched test-set metrics (probability threshold 0.5):",
                comparison.to_string(index=False),
                "",
                "Limitations: random split is not temporal validation; class "
                "balancing changes the training prior; threshold choice requires "
                "business validation; historical data may have vintage and "
                "selection bias. Metrics are not guarantees of future performance.",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "model_comparison": comparison_path,
        "class_distribution": distribution_path,
        "training_report": report_path,
    }


def _save_training_visualizations(
    reports_dir: str | Path,
    metrics: dict[str, dict[str, float]],
    best_name: str,
    pipeline: ImbalancedPipeline,
    feature_names: list[str],
    before: pd.DataFrame,
    after: pd.DataFrame,
    target: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    directory = Path(reports_dir) / "visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    class_path = directory / "class_distribution_smote.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, frame, title in zip(
        axes,
        (before, after),
        ("Training class distribution before resampling", "Training class distribution after resampling"),
    ):
        axis.bar(
            frame["loan_status"],
            frame["applications"],
            color=["#11a8c4", "#e36f58"],
        )
        axis.set(title=title, ylabel="Applications")
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(class_path, dpi=140, bbox_inches="tight")
    plt.close(figure)

    comparison = comparison_table(metrics)
    performance_path = directory / "model_comparison.png"
    figure, axis = plt.subplots(figsize=(10, 5))
    metric_plot = comparison.melt(
        id_vars="model",
        value_vars=["recall", "precision", "f1", "roc_auc", "average_precision"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(
        data=metric_plot,
        x="metric",
        y="score",
        hue="model",
        ax=axis,
    )
    axis.set(title="Untouched holdout model comparison", ylim=(0, 1), ylabel="Score")
    figure.tight_layout()
    figure.savefig(performance_path, dpi=140, bbox_inches="tight")
    plt.close(figure)

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_)
        importance_title = f"{best_name} feature importance"
    elif hasattr(classifier, "coef_"):
        importance = np.abs(np.asarray(classifier.coef_)[0])
        importance_title = f"{best_name} absolute coefficient importance"
    else:
        importance = np.array([])
        importance_title = ""
    importance_path = directory / "feature_importance.png"
    if len(importance):
        top_indices = np.argsort(importance)[-20:]
        figure, axis = plt.subplots(figsize=(9, max(4, len(top_indices) * 0.3)))
        axis.barh(
            np.asarray(feature_names)[top_indices],
            importance[top_indices],
            color="#11a8c4",
        )
        axis.set(title=importance_title, xlabel="Importance", ylabel="Feature")
        figure.tight_layout()
        figure.savefig(importance_path, dpi=140, bbox_inches="tight")
        plt.close(figure)
    else:
        importance_path.write_text(
            "Selected estimator does not expose feature importance or coefficients.\n",
            encoding="utf-8",
        )

    curve_path = directory / "training_performance_curves.png"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    RocCurveDisplay.from_predictions(target, probability, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(target, probability, ax=axes[1])
    axes[0].set_title("ROC curve — untouched holdout")
    axes[1].set_title("Precision-recall curve — untouched holdout")
    figure.tight_layout()
    figure.savefig(curve_path, dpi=140, bbox_inches="tight")
    plt.close(figure)

    paths = {
        "class_distribution_chart": class_path,
        "model_comparison_chart": performance_path,
        "feature_importance_chart": importance_path,
        "performance_curves": curve_path,
    }
    return paths
