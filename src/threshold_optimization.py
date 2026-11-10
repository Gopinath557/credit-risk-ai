"""Validation-only probability threshold comparisons and cost minimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def compare_thresholds(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Compare threshold metrics and relative costs on validation predictions."""
    from src.model_evaluation import binary_classification_metrics

    y_true, y_probability = _validate_inputs(target, probabilities)
    _validate_costs(false_negative_cost, false_positive_cost)
    validated_thresholds = _validate_thresholds(thresholds)
    rows = []
    for threshold in validated_thresholds:
        metrics = binary_classification_metrics(
            y_true, y_probability, threshold=threshold
        )
        total_cost = (
            metrics["false_negative"] * false_negative_cost
            + metrics["false_positive"] * false_positive_cost
        )
        rows.append(
            {
                "threshold": threshold,
                **metrics,
                "false_negative_cost": false_negative_cost,
                "false_positive_cost": false_positive_cost,
                "total_relative_cost": float(total_cost),
                "relative_cost_per_application": float(total_cost / len(y_true)),
                "threshold_source": "validation_oof",
            }
        )
    return pd.DataFrame(rows)


def compare_model_thresholds(
    targets: Sequence[int] | np.ndarray,
    validation_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Calculate validation OOF threshold curves for each named model."""
    labels = np.asarray(targets)
    rows = []
    for model_name, probabilities in validation_probabilities.items():
        comparison = compare_thresholds(
            labels,
            probabilities,
            false_negative_cost,
            false_positive_cost,
            thresholds,
        )
        comparison.insert(0, "model", model_name)
        rows.append(comparison)
    if not rows:
        raise ValueError("No validation model probabilities were provided.")
    return pd.concat(rows, ignore_index=True)


def select_cost_threshold(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> tuple[float, pd.DataFrame]:
    """Choose the lowest-cost threshold using validation data only."""
    comparison = compare_thresholds(
        target,
        probabilities,
        false_negative_cost,
        false_positive_cost,
        thresholds,
    )
    selected = comparison.sort_values(
        [
            "total_relative_cost",
            "false_negative",
            "false_positive",
            "threshold",
        ],
        ascending=True,
        kind="stable",
    ).iloc[0]
    return float(selected["threshold"]), comparison


def select_cost_thresholds(
    target: Sequence[int] | np.ndarray,
    validation_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Select a separate validation-cost threshold for every model."""
    table = compare_model_thresholds(
        target,
        validation_probabilities,
        false_negative_cost,
        false_positive_cost,
        thresholds,
    )
    selected: dict[str, float] = {}
    for model_name, group in table.groupby("model", sort=False):
        choice = group.sort_values(
            [
                "total_relative_cost",
                "false_negative",
                "false_positive",
                "threshold",
            ],
            ascending=True,
            kind="stable",
        ).iloc[0]
        selected[str(model_name)] = float(choice["threshold"])
    return selected, table


def _validate_inputs(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(target)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Target and probability inputs must be matching 1-D arrays.")
    if len(labels) == 0:
        raise ValueError("Evaluation arrays cannot be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Target must contain only binary values 0 and 1.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Evaluation requires both target classes.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return labels.astype("int8"), scores


def _validate_costs(false_negative_cost: float, false_positive_cost: float) -> None:
    if (
        not np.isfinite(false_negative_cost)
        or not np.isfinite(false_positive_cost)
        or false_negative_cost < 0
        or false_positive_cost < 0
        or false_negative_cost + false_positive_cost == 0
    ):
        raise ValueError("Costs must be finite, non-negative, and not both zero.")


def _validate_thresholds(thresholds: Sequence[float]) -> list[float]:
    values = [float(value) for value in thresholds]
    if not values:
        raise ValueError("At least one threshold is required.")
    if any(not np.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("Thresholds must be finite values in [0, 1].")
    if len(set(values)) != len(values):
        raise ValueError("Thresholds must not contain duplicates.")
    return values
