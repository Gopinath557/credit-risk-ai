"""Reusable feature preprocessing for LendingClub model pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Build a train-fittable transformer for numeric and categorical columns."""
    numeric = list(numeric_features)
    categorical = list(categorical_features)
    if not numeric and not categorical:
        raise ValueError("At least one numeric or categorical feature is required.")
    if len(set(numeric + categorical)) != len(numeric) + len(categorical):
        raise ValueError("Feature names must be unique and belong to one feature group.")

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__missing__"),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_columns(
    dataframe: pd.DataFrame, target_column: str = "target"
) -> tuple[list[str], list[str]]:
    """Return modeling feature names grouped by numeric and categorical dtype."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} is missing.")
    if dataframe[target_column].isna().any():
        raise ValueError("Target values cannot be missing.")
    if not dataframe[target_column].isin([0, 1]).all():
        raise ValueError("Target values must be binary: 0 or 1.")

    features = dataframe.drop(columns=[target_column])
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [
        column for column in features.columns if column not in numeric
    ]
    if not numeric and not categorical:
        raise ValueError("No modeling features remain after excluding the target.")
    return numeric, categorical
