"""Modeling validation, holdout splits, metrics, and artifact persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "target"
FORBIDDEN_FEATURES = {
    "loan_status",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "out_prncp",
    "out_prncp_inv",
    "funded_amnt",
    "funded_amnt_inv",
    "installment",
    "settlement_status",
    "settlement_amount",
    "settlement_date",
}


def validate_modeling_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate target, rows, feature types, and known target-leakage columns."""
    if dataframe.empty:
        raise ValueError("The processed dataset contains no rows.")
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError("The processed dataset must contain a 'target' column.")
    duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate feature names found: {duplicate_columns}")

    forbidden = [
        column
        for column in dataframe.columns
        if _is_forbidden_feature(column)
    ]
    if forbidden:
        raise ValueError(
            "Potential target leakage or identifiers remain in the processed "
            f"dataset: {forbidden}. Reprocess it using Phase 2 cleaning."
        )
    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Target values must be non-missing binary values 0 or 1.")
    features = dataframe.drop(columns=[TARGET_COLUMN]).copy()
    if features.empty:
        raise ValueError("No feature columns remain after excluding the target.")
    if len(target.value_counts()) != 2:
        raise ValueError("Training requires both target classes 0 and 1.")
    if features.select_dtypes(include=["datetime", "datetimetz"]).shape[1]:
        raise ValueError(
            "Datetime columns must be engineered or excluded before model training."
        )
    target = target.astype("int8").reset_index(drop=True)
    features = features.reset_index(drop=True)
    return features, target


def _is_forbidden_feature(column: Any) -> bool:
    normalized = str(column).strip().casefold()
    return (
        normalized in FORBIDDEN_FEATURES
        or normalized in {"id", "member_id", "url"}
        or normalized.startswith(
            (
                "total_pymnt",
                "total_rec",
                "last_pymnt",
                "last_fico",
                "hardship_",
                "settlement_",
            )
        )
    )


def stratified_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a reproducible stratified holdout split with test data untouched."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if len(features) != len(target):
        raise ValueError("Feature and target row counts do not match.")
    if target.value_counts().min() < 2:
        raise ValueError("Each class needs at least two rows for stratified splitting.")
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
    except ValueError as error:
        raise ValueError(f"Could not create a stratified holdout split: {error}") from error
    return (
        x_train.reset_index(drop=True),
        x_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def calculate_binary_metrics(
    target: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate threshold-0.5 classification and probability ranking metrics."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = np.asarray(target)
    predictions = np.asarray(prediction)
    probabilities = np.asarray(probability)
    if labels.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("Target and prediction arrays must be matching 1-D arrays.")
    if probabilities.ndim == 2:
        if probabilities.shape[1] != 2:
            raise ValueError("Probability output must have exactly two class columns.")
        probabilities = probabilities[:, 1]
    if probabilities.shape != labels.shape:
        raise ValueError("Target and positive-class probability shapes do not match.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Metrics require binary target values 0 and 1.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Metrics require both classes in the holdout target.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Predicted probabilities must be finite values from 0 to 1.")

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def save_model_artifacts(
    pipeline: Any,
    feature_names: list[str],
    metadata: dict[str, Any],
    models_dir: str | Path,
) -> dict[str, Path]:
    """Persist the complete fitted model pipeline and JSON model documentation."""
    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "best_model": output_dir / "best_model.joblib",
        "preprocessing_pipeline": output_dir / "preprocessing_pipeline.joblib",
        "model_metadata": output_dir / "model_metadata.json",
        "feature_names": output_dir / "feature_names.json",
    }
    joblib.dump(pipeline, paths["best_model"], compress=3)
    joblib.dump(
        {
            "preprocessor": pipeline.named_steps["preprocessor"],
            "post_resample_encoder": pipeline.named_steps["onehot"],
        },
        paths["preprocessing_pipeline"],
        compress=3,
    )
    paths["feature_names"].write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    paths["model_metadata"].write_text(
        json.dumps(metadata, indent=2, default=_json_value),
        encoding="utf-8",
    )
    return paths


def load_model(model_path: str | Path) -> Any:
    """Load a persisted full preprocessing/sampling/classification pipeline."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    model = joblib.load(path)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("Model artifact does not expose predict and predict_proba.")
    return model


def predict_applicants(
    model: Any,
    applicants: pd.DataFrame,
) -> pd.DataFrame:
    """Return binary prediction and class-1 probability using the saved pipeline."""
    if applicants.empty:
        raise ValueError("At least one applicant row is required.")
    if TARGET_COLUMN in applicants.columns:
        raise ValueError("Applicant input must not contain the target column.")
    expected = list(getattr(model, "feature_names_in_", []))
    if expected:
        missing = [column for column in expected if column not in applicants.columns]
        extra = [column for column in applicants.columns if column not in expected]
        if missing or extra:
            raise ValueError(
                f"Applicant feature columns do not match the model; "
                f"missing={missing}, extra={extra}."
            )
        applicants = applicants[expected]
    predictions = model.predict(applicants)
    probabilities = model.predict_proba(applicants)[:, 1]
    return pd.DataFrame(
        {
            "predicted_default": np.asarray(predictions, dtype="int8"),
            "default_probability": np.asarray(probabilities, dtype=float),
        }
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        converted = float(value)
        return converted if np.isfinite(converted) else None
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def artifact_safe_name(value: str) -> str:
    """Normalize model names for filenames and report headings."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
