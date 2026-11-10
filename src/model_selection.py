"""Business-oriented selection among tuned candidate classifiers."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


METRIC_COLUMNS = [
    "recall",
    "precision",
    "f1",
    "average_precision",
    "roc_auc",
    "accuracy",
]


def comparison_table(
    model_metrics: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Build a model comparison table ordered by risk-oriented metrics."""
    rows = []
    for model_name, metrics in model_metrics.items():
        rows.append({"model": model_name, **{key: metrics.get(key, float("nan")) for key in METRIC_COLUMNS}})
    if not rows:
        return pd.DataFrame(columns=["model", *METRIC_COLUMNS])
    return pd.DataFrame(rows).sort_values(
        ["recall", "average_precision", "f1", "precision", "roc_auc", "accuracy"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def select_best_model(
    metrics: Mapping[str, Mapping[str, float]],
    minimum_recall: float | None = None,
) -> tuple[str, str]:
    """Select by recall eligibility then PR-AUC, F1, precision, and ROC-AUC."""
    table = comparison_table(metrics)
    if table.empty:
        raise ValueError("No trained model metrics are available for selection.")
    eligible = table
    rationale = "Ordered by recall, average precision, F1, precision, then ROC-AUC."
    if minimum_recall is not None:
        if not 0 <= minimum_recall <= 1:
            raise ValueError("minimum_recall must be in [0, 1].")
        eligible = table.loc[table["recall"] >= minimum_recall]
        if eligible.empty:
            rationale = (
                f"No candidate reached the requested recall of {minimum_recall:.2f}; "
                "selected the model with the highest recall, then average precision."
            )
            eligible = table
        else:
            rationale = (
                f"Among candidates meeting recall >= {minimum_recall:.2f}, ordered by "
                "average precision, F1, precision, then ROC-AUC."
            )
            eligible = eligible.sort_values(
                ["average_precision", "f1", "precision", "roc_auc", "recall"],
                ascending=False,
                kind="stable",
            )
    return str(eligible.iloc[0]["model"]), rationale
