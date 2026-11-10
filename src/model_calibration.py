"""Validation-fitted probability calibration and holdout reliability analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def fit_probability_calibrator(
    validation_target: Sequence[int] | np.ndarray,
    validation_probabilities: Sequence[float] | np.ndarray,
    method: str = "sigmoid",
) -> LogisticRegression:
    """Fit a Platt sigmoid to validation out-of-fold probabilities only."""
    labels, scores = _validate_inputs(validation_target, validation_probabilities)
    if method != "sigmoid":
        raise ValueError("Only sigmoid (Platt) calibration is currently supported.")
    logits = _logit(scores).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    calibrator.fit(logits, labels)
    return calibrator


def apply_probability_calibrator(
    calibrator: LogisticRegression,
    probabilities: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply a validation-fitted sigmoid calibrator to new probabilities."""
    scores = _validate_probabilities(probabilities)
    return calibrator.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]


def calibration_report(
    validation_target: Sequence[int] | np.ndarray,
    validation_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    holdout_target: Sequence[int] | np.ndarray,
    holdout_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    method: str = "sigmoid",
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit per-model calibration on OOF data and report reliability on holdout."""
    validation_labels = _validate_target(validation_target)
    holdout_labels = _validate_target(holdout_target)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    rows = []
    calibrated: dict[str, np.ndarray] = {}
    for model_name, validation_scores in validation_probabilities.items():
        if model_name not in holdout_probabilities:
            raise ValueError(f"Missing holdout probabilities for {model_name!r}.")
        valid_scores = _validate_probabilities(
            validation_scores, len(validation_labels)
        )
        test_scores = _validate_probabilities(
            holdout_probabilities[model_name], len(holdout_labels)
        )
        if len(np.unique(validation_labels)) != 2:
            raise ValueError("Calibration validation labels require both classes.")
        if len(np.unique(holdout_labels)) != 2:
            raise ValueError("Calibration holdout labels require both classes.")
        calibrator = fit_probability_calibrator(
            validation_labels, valid_scores, method=method
        )
        calibrated_scores = apply_probability_calibrator(calibrator, test_scores)
        calibrated[model_name] = calibrated_scores

        for probability_type, scores in (
            ("raw", test_scores),
            ("calibrated", calibrated_scores),
        ):
            observed, predicted = calibration_curve(
                holdout_labels,
                scores,
                n_bins=n_bins,
                strategy="quantile",
            )
            for bin_number, (observed_rate, mean_score) in enumerate(
                zip(observed, predicted), start=1
            ):
                rows.append(
                    {
                        "model": model_name,
                        "probability_type": probability_type,
                        "bin": bin_number,
                        "mean_predicted_probability": float(mean_score),
                        "observed_default_rate": float(observed_rate),
                        "holdout_applications": len(holdout_labels),
                        "brier_score": float(
                            brier_score_loss(
                                holdout_labels,
                                scores,
                            )
                        ),
                        "calibration_method": (
                            "none" if probability_type == "raw" else method
                        ),
                    }
                )
    if not rows:
        raise ValueError("No validation/holdout model probabilities were provided.")
    return pd.DataFrame(rows), calibrated


def _validate_inputs(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _validate_target(target)
    scores = _validate_probabilities(probabilities, len(labels))
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration requires both target classes.")
    return labels, scores


def _validate_target(target: Sequence[int] | np.ndarray) -> np.ndarray:
    labels = np.asarray(target)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Target must be a non-empty 1-D array.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Target must contain only binary values 0 and 1.")
    return labels.astype("int8")


def _validate_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    expected_length: int | None = None,
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or (expected_length is not None and len(scores) != expected_length):
        raise ValueError("Probabilities must be a 1-D array matching target length.")
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Probabilities must be non-empty and finite.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be in [0, 1].")
    return scores


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))
