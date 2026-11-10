"""Configurable relative-cost summaries for classification errors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.model_evaluation import binary_classification_metrics


def calculate_business_cost(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, float | int]:
    """Calculate relative error cost; costs are arbitrary units, not dollars."""
    if (
        not np.isfinite(false_negative_cost)
        or not np.isfinite(false_positive_cost)
        or false_negative_cost < 0
        or false_positive_cost < 0
        or false_negative_cost + false_positive_cost == 0
    ):
        raise ValueError("Costs must be finite, non-negative, and not both zero.")
    metrics = binary_classification_metrics(target, probabilities, threshold)
    false_negative_total = metrics["false_negative"] * false_negative_cost
    false_positive_total = metrics["false_positive"] * false_positive_cost
    return {
        **metrics,
        "false_negative_cost_per_case": float(false_negative_cost),
        "false_positive_cost_per_case": float(false_positive_cost),
        "false_negative_relative_cost": float(false_negative_total),
        "false_positive_relative_cost": float(false_positive_total),
        "total_relative_cost": float(false_negative_total + false_positive_total),
        "relative_cost_per_application": float(
            (false_negative_total + false_positive_total) / len(target)
        ),
    }


def business_risk_table(
    target: Sequence[int] | np.ndarray,
    model_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    thresholds: Mapping[str, float],
    false_negative_cost: float,
    false_positive_cost: float,
) -> pd.DataFrame:
    """Compare relative error costs at validation-selected model thresholds."""
    rows = []
    for model_name, probabilities in model_probabilities.items():
        if model_name not in thresholds:
            raise ValueError(f"No validation-selected threshold for {model_name!r}.")
        rows.append(
            {
                "model": model_name,
                **calculate_business_cost(
                    target,
                    probabilities,
                    thresholds[model_name],
                    false_negative_cost,
                    false_positive_cost,
                ),
            }
        )
    if not rows:
        raise ValueError("No model probabilities were provided.")
    return pd.DataFrame(rows)
