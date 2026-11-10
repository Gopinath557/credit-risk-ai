"""Training-fold-only resampling for mixed LendingClub feature types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SMOTEN
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@dataclass(frozen=True)
class SamplingPlan:
    """Sampler choice and validated class counts for a training partition."""

    sampler: object
    name: str
    before: dict[int, int]
    after_estimate: dict[int, int]
    categorical_indices: tuple[int, ...]


def build_sampling_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Impute/scale numbers and impute/ordinal-encode categories for SMOTENC."""
    if not numeric_features and not categorical_features:
        raise ValueError("At least one feature is required for resampling.")
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__missing__",
                            ),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def create_sampling_plan(
    target: pd.Series | np.ndarray,
    numeric_count: int,
    categorical_count: int,
    random_state: int = 42,
    requested_neighbors: int = 5,
) -> SamplingPlan:
    """Choose SMOTE/SMOTENC/SMOTEN, falling back to random oversampling as needed."""
    labels = pd.Series(np.asarray(target).reshape(-1))
    if labels.empty or labels.isna().any():
        raise ValueError("Training labels must be non-empty and non-missing.")
    counts_series = labels.value_counts().sort_index()
    if len(counts_series) != 2 or set(counts_series.index) != {0, 1}:
        raise ValueError("Training data must contain both binary classes 0 and 1.")
    if numeric_count < 0 or categorical_count < 0:
        raise ValueError("Feature counts cannot be negative.")
    if requested_neighbors < 1:
        raise ValueError("requested_neighbors must be at least 1.")
    if numeric_count + categorical_count < 1:
        raise ValueError("At least one feature is required.")
    before = {int(label): int(count) for label, count in counts_series.items()}
    minority_count = min(before.values())
    majority_count = max(before.values())
    categorical_indices = tuple(
        range(numeric_count, numeric_count + categorical_count)
    )

    if minority_count < 2:
        sampler: object = RandomOverSampler(random_state=random_state)
        name = "RandomOverSampler (SMOTE requires at least two minority samples)"
    else:
        neighbors = min(requested_neighbors, minority_count - 1)
        if numeric_count and categorical_count:
            sampler = SMOTENC(
                categorical_features=list(categorical_indices),
                random_state=random_state,
                k_neighbors=neighbors,
            )
            name = f"SMOTENC (k_neighbors={neighbors})"
        elif categorical_count:
            sampler = SMOTEN(random_state=random_state, k_neighbors=neighbors)
            name = f"SMOTEN (k_neighbors={neighbors})"
        elif numeric_count:
            sampler = SMOTE(random_state=random_state, k_neighbors=neighbors)
            name = f"SMOTE (k_neighbors={neighbors})"
        else:
            sampler = RandomOverSampler(random_state=random_state)
            name = "RandomOverSampler (no usable feature type)"

    after_estimate = dict(before)
    after_estimate[0] = majority_count
    after_estimate[1] = majority_count
    return SamplingPlan(
        sampler=sampler,
        name=name,
        before=before,
        after_estimate=after_estimate,
        categorical_indices=categorical_indices,
    )


def class_distribution(
    target: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Summarize class counts before or after resampling."""
    labels = pd.Series(np.asarray(target).reshape(-1), name="target")
    counts = labels.value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "target": [0, 1],
            "loan_status": ["Fully Paid", "Charged Off"],
            "applications": counts.to_numpy(dtype=int),
            "share": (
                counts.to_numpy(dtype=float) / total if total else [0.0, 0.0]
            ),
        }
    )
