"""Streamlit entry point for the credit risk modeling project."""

import json
import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import (
    DatasetLoadError,
    inspect_dataset,
    load_lendingclub_csv,
    write_dataset_reports,
)
from src.data_validation import validate_and_clean_dataset
from src.eda import create_eda_figures, generate_eda_reports, save_dataset_charts
from src.statistical_analysis import (
    analyze_target_distribution,
    create_insights,
    describe_numeric_features,
    default_rate_analysis,
    filter_eda_data,
)
from src.model_training import train_models
from src.business_risk_analysis import business_risk_table
from src.model_calibration import calibration_report
from src.model_evaluation import (
    binary_classification_metrics,
    confusion_matrix_figure,
    evaluate_models,
    generate_evaluation_reports,
    precision_recall_figure,
    roc_figure,
    threshold_figure,
)
from src.threshold_optimization import (
    select_cost_thresholds,
    compare_thresholds,
)
from src.explanation_generator import generate_explainability_report
from src.model_utils import load_model


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
REPORTS_DIR = ROOT / "reports"
EXPLAINABILITY_DIR = REPORTS_DIR / "explainability"
MODELS_DIR = ROOT / "models"


@st.cache_data(show_spinner=False)
def load_processed_data(path: str, modified_ns: int) -> pd.DataFrame:
    """Cache cleaned CSV reads and refresh when the file changes."""
    del modified_ns
    return pd.read_csv(path, low_memory=False)


def render_evaluation_table(dataframe: pd.DataFrame) -> None:
    """Render a read-only evaluation table without converting labels or scores."""
    st.markdown(
        dataframe.to_html(index=False, border=0, classes="credit-risk-table"),
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_saved_model(path: str, modified_ns: int):
    """Cache the fitted model pipeline until its artifact changes."""
    del modified_ns
    return load_model(path)


def build_manual_applicant_form(reference_data: pd.DataFrame) -> pd.DataFrame | None:
    """Collect a single applicant using feature types and categories in the data."""
    values: dict[str, object] = {}
    with st.form("xai_manual_applicant"):
        columns = st.columns(2)
        for index, feature in enumerate(reference_data.columns):
            column = columns[index % len(columns)]
            series = reference_data[feature]
            with column:
                if pd.api.types.is_numeric_dtype(series):
                    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
                    default = float(numeric_values.median()) if not numeric_values.empty else 0.0
                    values[feature] = st.number_input(
                        feature.replace("_", " "),
                        value=default,
                        key=f"xai_manual_{feature}",
                    )
                else:
                    options = (
                        series.dropna()
                        .astype(str)
                        .value_counts()
                        .index[:100]
                        .tolist()
                    )
                    if options and series.nunique(dropna=True) <= 100:
                        values[feature] = st.selectbox(
                            feature.replace("_", " "),
                            options,
                            key=f"xai_manual_{feature}",
                        )
                    else:
                        mode = series.dropna().astype(str).mode()
                        values[feature] = st.text_input(
                            feature.replace("_", " "),
                            value=str(mode.iloc[0]) if not mode.empty else "",
                            key=f"xai_manual_{feature}",
                        )
        submitted = st.form_submit_button(
            "Use applicant details",
            type="primary",
            width="stretch",
        )
    return pd.DataFrame([values], columns=reference_data.columns) if submitted else None


st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="📊",
    layout="wide",
)

pages = [
    "Dataset preparation",
    "Exploratory data analysis",
    "Model training",
    "Model evaluation",
    "Explainable AI",
]
page_icons = {
    "Dataset preparation": "database",
    "Exploratory data analysis": "monitoring",
    "Model training": "model_training",
    "Model evaluation": "fact_check",
    "Explainable AI": "psychology",
}

with st.sidebar:
    st.title("Credit risk", icon=":material/account_balance:")
    st.caption("LENDING ANALYTICS")
    selected_page = st.radio(
        "Workspace",
        pages,
        format_func=lambda page: f":material/{page_icons[page]}:  {page}",
    )
    st.divider()
    model_available = (MODELS_DIR / "best_model.joblib").is_file()
    st.badge(
        "Model ready" if model_available else "Model not trained",
        icon=":material/check_circle:" if model_available else ":material/info:",
        color="green" if model_available else "orange",
    )
    st.caption("LendingClub credit-risk workbench")

st.title("Credit Risk Intelligence", icon=":material/account_balance:")
st.caption("Portfolio analysis, default-risk modeling, and explainable decisions")

if selected_page == "Model training":
    st.header("Model training")
    st.caption(
        "Tune and select classifiers using stratified training-partition "
        "cross-validation and out-of-fold scores. Holdout metrics are reported "
        "separately; the test partition is never resampled or used for threshold selection."
    )
    processed_path = DATASET_DIR / "loan_data_processed.csv"
    if not processed_path.is_file():
        st.info(
            "No Phase 2 processed dataset is available. Process a genuine "
            "LendingClub dataset on the Dataset preparation page first. Training "
            "metrics and model artifacts will not be generated without real data."
        )
        st.stop()
    try:
        training_data = load_processed_data(
            str(processed_path), processed_path.stat().st_mtime_ns
        )
        from src.model_utils import validate_modeling_data

        model_features, model_target = validate_modeling_data(training_data)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Cannot prepare this dataset for training: {error}")
        st.stop()

    st.subheader("Training data")
    class_counts = model_target.value_counts().reindex([0, 1], fill_value=0)
    data_metrics = st.columns(4)
    data_metrics[0].metric("Eligible applications", f"{len(training_data):,}")
    data_metrics[1].metric("Application-time features", f"{len(model_features.columns):,}")
    data_metrics[2].metric("Fully paid", f"{class_counts.loc[0]:,}")
    data_metrics[3].metric("Charged off", f"{class_counts.loc[1]:,}")
    st.caption(
        "Rows/columns were checked for known status, payment, outcome and identifier "
        "leakage before training. Stratified split: 80% training / 20% holdout; "
        "the holdout keeps its original class distribution and is reserved for reporting."
    )

    with st.form("model_training_options"):
        cv_splits = st.selectbox("Stratified CV folds", [3, 5], index=0)
        recall_requirement = st.selectbox(
            "Minimum validation recall requirement (optional)",
            ["No minimum", "0.60", "0.70", "0.80"],
            index=0,
            help="A selection constraint, not a probability threshold.",
        )
        train_submitted = st.form_submit_button(
            "Train and compare all models",
            type="primary",
            width="stretch",
        )
    if train_submitted:
        minimum_recall = (
            None if recall_requirement == "No minimum" else float(recall_requirement)
        )
        try:
            with st.spinner(
                "Running stratified cross-validation and training all three models..."
            ):
                result = train_models(
                    training_data,
                    models_dir=ROOT / "models",
                    reports_dir=REPORTS_DIR,
                    cv_splits=int(cv_splits),
                    minimum_recall=minimum_recall,
                )
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f"Model training failed: {error}")
            st.stop()
        st.session_state["phase4_best_model"] = result.best_model_name
        st.session_state["phase4_metrics"] = result.comparison.to_dict(orient="records")
        st.session_state["phase4_sampling_method"] = result.sampling_method
        st.session_state["phase4_selection_rationale"] = result.selection_rationale
        st.session_state["phase4_artifacts"] = {
            key: str(path) for key, path in result.artifacts.items()
        }
        st.session_state["phase4_reports"] = {
            key: str(path) for key, path in result.reports.items()
        }
        st.success(f"Training complete. Selected model: {result.best_model_name}.")

    comparison_path = REPORTS_DIR / "model_comparison.csv"
    distribution_path = REPORTS_DIR / "class_distribution.csv"
    metadata_path = ROOT / "models" / "model_metadata.json"
    if comparison_path.is_file() and metadata_path.is_file():
        try:
            comparison = pd.read_csv(comparison_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_name = str(metadata.get("model_name", "Unknown"))
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Could not load saved training results: {error}")
            st.stop()
        if (
            not isinstance(metadata, dict)
            or not metadata.get("model_name")
            or comparison.empty
        ):
            st.info(
                "No completed model training run is available yet. Start training "
                "to generate real holdout metrics, reports, visualizations, and artifacts."
            )
            st.stop()

        st.subheader("Model comparison — untouched holdout")
        st.dataframe(comparison, hide_index=True)
        chart_metrics = [
            metric
            for metric in ("recall", "precision", "f1", "roc_auc", "average_precision")
            if metric in comparison.columns
        ]
        if chart_metrics:
            chart_data = comparison.melt(
                id_vars="model",
                value_vars=chart_metrics,
                var_name="Metric",
                value_name="Score",
            )
            st.plotly_chart(
                px.bar(
                    chart_data,
                    x="Metric",
                    y="Score",
                    color="model",
                    barmode="group",
                    title="Risk-focused metrics on original-distribution holdout",
                    hover_data={"Score": ":.3f"},
                ),
                width="stretch",
                alt="Holdout recall, precision, F1, ROC-AUC, and average precision by model",
            )
        st.success(f"Selected model: {model_name}")
        st.write(str(metadata.get("selection_rationale", "")))
        st.write(
            f"Training sampler: {metadata.get('sampling_method', 'Unknown')}. "
            "It is applied only inside each training fold and final training data."
        )
        if distribution_path.is_file():
            distribution = pd.read_csv(distribution_path)
            st.subheader("Training class distribution before and after resampling")
            st.plotly_chart(
                px.bar(
                    distribution,
                    x="loan_status",
                    y="applications",
                    color="stage",
                    barmode="group",
                    title="Training partition only; holdout data remains unchanged",
                    hover_data={"share": ":.2%"},
                    labels={
                        "loan_status": "Loan outcome",
                        "applications": "Applications",
                        "stage": "Training stage",
                    },
                ),
                width="stretch",
                alt="Training class counts before and after resampling",
            )

        importance_path = REPORTS_DIR / "visualizations" / "feature_importance.png"
        curves_path = REPORTS_DIR / "visualizations" / "training_performance_curves.png"
        if importance_path.is_file() and importance_path.suffix == ".png":
            st.subheader("Selected model feature importance")
            if importance_path.stat().st_size:
                try:
                    st.image(
                        str(importance_path),
                        alt="Feature importance for selected model",
                    )
                except (OSError, ValueError):
                    st.caption("Feature-importance chart is not available.")
        if curves_path.is_file():
            st.subheader("Untouched holdout curves")
            st.image(str(curves_path), alt="ROC and precision-recall holdout curves")

        if comparison_path.is_file():
            st.download_button(
                "Download model comparison",
                data=comparison_path.read_bytes(),
                file_name="model_comparison.csv",
                mime="text/csv",
            )
        training_report_path = REPORTS_DIR / "training_report.txt"
        if training_report_path.is_file():
            st.download_button(
                "Download training report",
                data=training_report_path.read_bytes(),
                file_name="training_report.txt",
                mime="text/plain",
            )
    else:
        st.info(
            "No completed model training run is available yet. Start training to "
            "generate real holdout metrics, reports, visualizations, and artifacts."
        )
    st.stop()

if selected_page == "Model evaluation":
    st.header("Model evaluation and credit risk analysis")
    st.caption(
        "Thresholds are selected from training-partition out-of-fold "
        "probabilities. Final metrics and curves use the untouched holdout."
    )
    validation_path = REPORTS_DIR / "validation_oof_predictions.csv"
    holdout_path = REPORTS_DIR / "holdout_predictions.csv"
    if not validation_path.is_file() or not holdout_path.is_file():
        st.info(
            "No evaluation prediction bundles are available. Run Model training "
            "on a genuine Phase 2 LendingClub dataset first. No evaluation "
            "metrics or charts are fabricated."
        )
        st.stop()
    try:
        validation_predictions = pd.read_csv(validation_path)
        holdout_predictions = pd.read_csv(holdout_path)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not load saved evaluation predictions: {error}")
        st.stop()
    validation_model_columns = [
        column
        for column in validation_predictions.columns
        if column.startswith("probability__")
    ]
    holdout_model_columns = [
        column
        for column in holdout_predictions.columns
        if column.startswith("probability__")
    ]
    model_names = [
        column.removeprefix("probability__")
        for column in validation_model_columns
        if column in holdout_model_columns
    ]
    if (
        "target" not in validation_predictions.columns
        or "target" not in holdout_predictions.columns
        or not model_names
    ):
        st.error("Saved prediction bundles have an invalid or incomplete schema.")
        st.stop()

    validation_probabilities = {
        model_name: validation_predictions[f"probability__{model_name}"].to_numpy()
        for model_name in model_names
    }
    holdout_probabilities = {
        model_name: holdout_predictions[f"probability__{model_name}"].to_numpy()
        for model_name in model_names
    }
    validation_target = validation_predictions["target"].to_numpy()
    holdout_target = holdout_predictions["target"].to_numpy()
    model_name = st.selectbox("Model", model_names)
    cost_columns = st.columns(2)
    false_negative_cost = cost_columns[0].number_input(
        "Relative cost of a missed default (FN)",
        min_value=0.0,
        value=5.0,
        step=1.0,
        help="Relative, non-currency units. Used to select a threshold on OOF validation data.",
    )
    false_positive_cost = cost_columns[1].number_input(
        "Relative cost of a false alarm (FP)",
        min_value=0.0,
        value=1.0,
        step=1.0,
        help="Relative, non-currency units. Used to select a threshold on OOF validation data.",
    )
    if false_negative_cost + false_positive_cost == 0:
        st.error("At least one relative cost must be greater than zero.")
        st.stop()

    try:
        selected_thresholds, all_validation_thresholds = select_cost_thresholds(
            validation_target,
            validation_probabilities,
            false_negative_cost,
            false_positive_cost,
        )
        final_metrics = evaluate_models(
            holdout_target,
            holdout_probabilities,
            selected_thresholds,
        )
        validation_thresholds = compare_thresholds(
            validation_target,
            validation_probabilities[model_name],
            false_negative_cost,
            false_positive_cost,
        )
        calibration_data, _ = calibration_report(
            validation_target,
            validation_probabilities,
            holdout_target,
            holdout_probabilities,
            method="sigmoid",
        )
        holdout_risk = business_risk_table(
            holdout_target,
            holdout_probabilities,
            selected_thresholds,
            false_negative_cost,
            false_positive_cost,
        )
        roc_chart = roc_figure(holdout_target, holdout_probabilities)
        pr_chart = precision_recall_figure(holdout_target, holdout_probabilities)
    except (ValueError, KeyError) as error:
        st.error(f"Evaluation failed: {error}")
        st.stop()

    roc_scores = {
        row["model"]: row["roc_auc"]
        for _, row in final_metrics.iterrows()
    }
    strongest_roc_model = max(roc_scores, key=roc_scores.get)
    st.metric(
        "Strongest holdout ROC-AUC",
        strongest_roc_model,
        f"{roc_scores[strongest_roc_model]:.3f}",
    )

    st.subheader("Threshold selection (validation OOF only)")
    model_threshold = selected_thresholds[model_name]
    st.write(
        f"Cost-minimizing threshold for **{model_name}**: **{model_threshold:.2f}** "
        "selected from out-of-fold validation probabilities—not the test set."
    )
    threshold_slider = st.slider(
        "Explore validation threshold trade-off",
        min_value=0.30,
        max_value=0.70,
        value=0.50,
        step=0.01,
        help="This slider updates validation-only metrics. It does not tune or change holdout metrics.",
    )
    selected_validation_metrics = binary_classification_metrics(
        validation_target,
        validation_probabilities[model_name],
        threshold_slider,
    )
    validation_metric_columns = st.columns(4)
    validation_metric_columns[0].metric(
        "Validation recall", f"{selected_validation_metrics['recall']:.2%}"
    )
    validation_metric_columns[1].metric(
        "Validation precision", f"{selected_validation_metrics['precision']:.2%}"
    )
    validation_metric_columns[2].metric(
        "Validation F1", f"{selected_validation_metrics['f1']:.3f}"
    )
    validation_metric_columns[3].metric(
        "Validation relative cost",
        f"{selected_validation_metrics['false_negative'] * false_negative_cost + selected_validation_metrics['false_positive'] * false_positive_cost:,.1f}",
    )
    threshold_chart = threshold_figure(validation_thresholds)
    threshold_chart.add_vline(
        x=threshold_slider,
        line_dash="dash",
        annotation_text="Exploration threshold",
    )
    threshold_chart.add_vline(
        x=model_threshold,
        line_dash="dot",
        annotation_text="Cost-selected",
    )
    st.plotly_chart(
        threshold_chart,
        width="stretch",
        alt="Validation precision, recall and F1 trade-offs by threshold",
    )
    render_evaluation_table(all_validation_thresholds.round(4))
    st.caption(
        "Precision-recall analysis is particularly informative for imbalanced "
        "default outcomes: ROC curves can appear favorable while precision remains "
        "low because most applications are non-defaults. The PR baseline is the "
        "observed default prevalence."
    )

    st.subheader("Final metrics — untouched holdout")
    render_evaluation_table(final_metrics.round(4))
    selected_row = final_metrics.loc[final_metrics["model"] == model_name].iloc[0]
    st.plotly_chart(
        confusion_matrix_figure(
            holdout_target,
            holdout_probabilities[model_name],
            selected_thresholds[model_name],
            model_name,
        ),
        width="stretch",
        alt="Holdout confusion matrix with counts of each outcome and prediction",
    )
    st.write(
        f"At the validation-selected threshold of {selected_thresholds[model_name]:.2f}: "
        f"TP {int(selected_row['true_positive'])}, TN {int(selected_row['true_negative'])}, "
        f"FP {int(selected_row['false_positive'])}, FN {int(selected_row['false_negative'])}."
    )
    st.caption(
        "False negatives are charged-off loans not flagged, creating unidentified "
        "credit-loss exposure. False positives are fully paid loans flagged as "
        "defaults and may represent unnecessary declines or reviews. Counts are "
        "not monetary losses."
    )

    st.subheader("ROC curve comparison")
    st.plotly_chart(
        roc_chart,
        width="stretch",
        alt="ROC curves for all trained models on the untouched holdout",
    )
    st.subheader("Precision-recall curve comparison")
    st.plotly_chart(
        pr_chart,
        width="stretch",
        alt="Precision-recall curves and default prevalence baseline for all models",
    )

    st.subheader("Relative business risk")
    render_evaluation_table(holdout_risk.round(4))
    st.caption(
        "Relative cost units use the configured FN/FP weights. They are assumptions, "
        "not currency or observed borrower losses. Add institution-specific exposure, "
        "loss-given-default, and review costs before using this operationally."
    )

    st.subheader("Probability calibration")
    calibration_summary = (
        calibration_data.groupby(["model", "probability_type"], as_index=False)
        .agg(brier_score=("brier_score", "first"))
    )
    render_evaluation_table(calibration_summary.round(4))
    calibration_chart = px.line(
        calibration_data,
        x="mean_predicted_probability",
        y="observed_default_rate",
        color="model",
        line_dash="probability_type",
        markers=True,
        title="Reliability on untouched holdout",
        labels={
            "mean_predicted_probability": "Mean predicted default probability",
            "observed_default_rate": "Observed default frequency",
            "probability_type": "Probability",
        },
        hover_data=["bin", "brier_score"],
    )
    calibration_chart.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={"dash": "dash", "color": "gray"},
    )
    st.plotly_chart(
        calibration_chart,
        width="stretch",
        alt="Raw and sigmoid-calibrated probability reliability against observed holdout default rates",
    )
    st.caption(
        "Sigmoid calibrators are fit only on training-partition OOF scores, then "
        "assessed on the holdout. Brier score summarizes probability error; lower "
        "is better. Calibration does not guarantee future calibration."
    )

    if st.button("Generate evaluation reports and charts", type="primary"):
        try:
            report_paths = generate_evaluation_reports(
                validation_target,
                validation_probabilities,
                holdout_target,
                holdout_probabilities,
                REPORTS_DIR,
                false_negative_cost,
                false_positive_cost,
            )
        except (ValueError, OSError) as error:
            st.error(f"Could not generate evaluation outputs: {error}")
        else:
            st.success(
                f"Generated {len(report_paths)} evaluation reports and visualizations."
            )
    report_file_names = (
        "detailed_evaluation.csv",
        "threshold_comparison.csv",
        "calibration_report.csv",
        "business_risk_report.txt",
    )
    report_columns = st.columns(4)
    for column, file_name in zip(report_columns, report_file_names):
        path = REPORTS_DIR / file_name
        if path.is_file():
            column.download_button(
                f"Download {file_name.replace('_', ' ').replace('.csv', '').replace('.txt', '')}",
                data=path.read_bytes(),
                file_name=file_name,
                mime="text/plain" if file_name.endswith(".txt") else "text/csv",
            )
    st.stop()

if selected_page == "Explainable AI":
    st.header("Explainable AI")
    st.caption(
        "SHAP and LIME describe behavior of the selected fitted model for the "
        "provided inputs. Their contributions are associations, not causes or "
        "guarantees of future repayment outcomes."
    )
    model_path = MODELS_DIR / "best_model.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"
    processed_path = DATASET_DIR / "loan_data_processed.csv"
    if not model_path.is_file() or not metadata_path.is_file():
        st.info(
            "No trained model artifact is available. Process a genuine LendingClub "
            "dataset and complete model training before generating SHAP or LIME explanations."
        )
        st.stop()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or not metadata.get("model_name"):
            st.info(
                "Model metadata indicates training has not been run. No explanations "
                "or feature importance have been fabricated."
            )
            st.stop()
        model = load_saved_model(str(model_path), model_path.stat().st_mtime_ns)
        processed = load_processed_data(
            str(processed_path), processed_path.stat().st_mtime_ns
        )
    except FileNotFoundError:
        st.info(
            "The processed dataset is missing. Restore the dataset used for model "
            "training before generating applicant explanations."
        )
        st.stop()
    except (OSError, ValueError, KeyError, ImportError) as error:
        st.error(f"Could not load the saved model or dataset: {error}")
        st.stop()
    if "target" not in processed.columns:
        st.error("The processed dataset does not contain its required target column.")
        st.stop()
    reference_data = processed.drop(columns=["target"])
    model_features = list(getattr(model, "feature_names_in_", []))
    missing_features = [name for name in model_features if name not in reference_data]
    if not model_features or missing_features:
        st.error(
            "The saved model and processed dataset schemas do not match. "
            f"Missing model features: {missing_features}."
        )
        st.stop()
    reference_data = reference_data.loc[:, model_features]
    st.write(f"Selected trained model: **{metadata['model_name']}**")

    input_source = st.radio(
        "Applicant input",
        ["Select a processed application", "Enter applicant details"],
        horizontal=True,
    )
    applicant: pd.DataFrame | None = None
    if input_source == "Select a processed application":
        selected_row = st.selectbox(
            "Application record",
            range(len(reference_data)),
            format_func=lambda row: f"Processed record {row + 1:,}",
        )
        applicant = reference_data.iloc[[selected_row]].copy()
        st.caption(
            "The row is selected from the processed dataset. Its known outcome is "
            "not shown to the model or included in the explanation."
        )
    else:
        applicant = build_manual_applicant_form(reference_data)

    if applicant is not None:
        st.subheader("Individual applicant prediction")
        try:
            from src.model_utils import predict_applicants

            prediction = predict_applicants(model, applicant).iloc[0]
        except ValueError as error:
            st.error(f"Applicant inputs do not match the model: {error}")
            st.stop()
        predicted_label = (
            "High risk class" if int(prediction["predicted_default"]) == 1
            else "Lower risk class"
        )
        with st.container(horizontal=True):
            st.metric("Predicted default probability", f"{prediction['default_probability']:.1%}", border=True)
            st.metric("Model classification", predicted_label, border=True)
        st.caption(
            "The classification shown uses the saved model's default prediction "
            "rule. Probability is not a guaranteed outcome."
        )
        explain_clicked = st.button(
            "Generate SHAP and LIME explanations",
            type="primary",
            icon=":material/insights:",
        )
        applicant_fingerprint = hashlib.sha256(
            applicant.to_json(orient="split", date_format="iso").encode("utf-8")
        ).hexdigest()
        model_fingerprint = f"{model_path.stat().st_mtime_ns}:{metadata['model_name']}"
        if explain_clicked:
            try:
                with st.spinner(
                    "Computing global SHAP, individual SHAP, and local LIME explanations..."
                ):
                    explanation_paths = generate_explainability_report(
                        model,
                        reference_data,
                        applicant,
                        str(metadata["model_name"]),
                        EXPLAINABILITY_DIR,
                    )
            except (ValueError, OSError, ImportError, KeyError) as error:
                st.error(f"Explainability generation failed: {error}")
                st.stop()
            st.session_state["phase6_context"] = {
                "applicant_fingerprint": applicant_fingerprint,
                "model_fingerprint": model_fingerprint,
                "paths": {name: str(path) for name, path in explanation_paths.items()},
            }
            st.success("SHAP and LIME reports and visualizations were generated.")

        context = st.session_state.get("phase6_context", {})
        current_context = (
            context.get("applicant_fingerprint") == applicant_fingerprint
            and context.get("model_fingerprint") == model_fingerprint
        )
        if current_context:
            paths = {
                name: Path(path) for name, path in context.get("paths", {}).items()
            }
            report_path = paths.get("explainability_report")
            shap_contributions_path = paths.get("individual_shap_contributions")
            lime_contributions_path = paths.get("lime_feature_contributions")
            importance_path = paths.get("global_feature_importance")

            st.subheader("Global feature importance")
            if importance_path and importance_path.is_file():
                importance = pd.read_csv(importance_path)
                render_evaluation_table(importance.head(20).round(5))
            summary_path = paths.get("shap_summary")
            bar_path = paths.get("shap_bar")
            if summary_path and summary_path.is_file():
                st.image(
                    str(summary_path),
                    alt="SHAP global summary showing feature value and contribution direction",
                )
            if bar_path and bar_path.is_file():
                st.image(
                    str(bar_path),
                    alt="SHAP global mean absolute feature contribution bar chart",
                )

            st.subheader("Individual SHAP explanation")
            if shap_contributions_path and shap_contributions_path.is_file():
                shap_contributions = pd.read_csv(shap_contributions_path)
                st.plotly_chart(
                    px.bar(
                        shap_contributions.sort_values(
                            "contribution", key=lambda values: values.abs()
                        ),
                        x="contribution",
                        y="feature",
                        color="contribution",
                        color_continuous_scale=["#168a9c", "#e36f58"],
                        title="Applicant SHAP contributions",
                        labels={
                            "contribution": "Contribution on model output scale",
                            "feature": "Feature",
                        },
                        hover_data=["applicant_value", "output_scale"],
                    ),
                    width="stretch",
                    alt="Positive and negative SHAP feature contributions for selected applicant",
                )
            waterfall_path = paths.get("shap_waterfall")
            if waterfall_path and waterfall_path.is_file():
                st.image(
                    str(waterfall_path),
                    alt="SHAP waterfall showing feature contributions for one applicant",
                )
            force_path = paths.get("shap_force")
            if force_path and force_path.is_file():
                st.download_button(
                    "Download SHAP force plot",
                    data=force_path.read_bytes(),
                    file_name=force_path.name,
                    mime="text/html",
                )

            st.subheader("LIME local explanation")
            if lime_contributions_path and lime_contributions_path.is_file():
                lime_contributions = pd.read_csv(lime_contributions_path)
                st.plotly_chart(
                    px.bar(
                        lime_contributions.sort_values(
                            "contribution", key=lambda values: values.abs()
                        ),
                        x="contribution",
                        y="condition",
                        color="contribution",
                        color_continuous_scale=["#168a9c", "#e36f58"],
                        title="LIME local surrogate contributions",
                        labels={
                            "contribution": "Local surrogate weight",
                            "condition": "Feature condition",
                        },
                        hover_data=["feature", "applicant_value"],
                    ),
                    width="stretch",
                    alt="LIME positive and negative local surrogate weights",
                )
            lime_path = paths.get("lime_explanation")
            if lime_path and lime_path.is_file():
                st.download_button(
                    "Download LIME explanation",
                    data=lime_path.read_bytes(),
                    file_name=lime_path.name,
                    mime="text/html",
                )

            st.subheader("Risk factor analysis")
            st.caption(
                "Positive values increase the model output relative to its baseline; "
                "negative values decrease it. Logistic Regression SHAP values are "
                "log-odds, tree-model SHAP values are probabilities, and LIME weights "
                "are local-surrogate terms. These scales are not interchangeable."
            )
            if report_path and report_path.is_file():
                report_text = report_path.read_text(encoding="utf-8")
                st.write(report_text)
                st.download_button(
                    "Download explainability report",
                    data=report_text,
                    file_name=report_path.name,
                    mime="text/plain",
                )
            for path_name in (
                "global_feature_importance",
                "individual_shap_contributions",
                "lime_feature_contributions",
            ):
                path = paths.get(path_name)
                if path and path.is_file():
                    st.download_button(
                        f"Download {path.name}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="text/csv",
                        key=f"xai_download_{path_name}",
                    )
    st.stop()

if selected_page == "Exploratory data analysis":
    st.header("Exploratory data analysis")
    st.caption(
        "Descriptive analysis of the cleaned historical dataset. Observed "
        "relationships are not causal effects or model predictions."
    )
    processed_path = DATASET_DIR / "loan_data_processed.csv"
    if not processed_path.is_file():
        st.info(
            "No cleaned dataset is available yet. Process a genuine LendingClub "
            "CSV on the Dataset preparation page first. No findings or charts "
            "are generated from fabricated records."
        )
        st.stop()

    try:
        processed_data = load_processed_data(
            str(processed_path), processed_path.stat().st_mtime_ns
        )
        target_choices = analyze_target_distribution(processed_data)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not load the cleaned dataset for analysis: {error}")
        st.stop()

    available_statuses = [
        status for status, count in target_choices.items() if count > 0
    ]
    selected_statuses = st.sidebar.multiselect(
        "Loan outcomes",
        options=available_statuses,
        default=available_statuses,
    )
    outcome_codes = {
        code
        for code, label in ((0, "Fully Paid"), (1, "Charged Off"))
        if label in selected_statuses
    }

    category_columns = [
        column
        for column in ("grade", "purpose", "home_ownership", "term", "verification_status")
        if column in processed_data.columns
    ]
    category_filter = st.sidebar.selectbox(
        "Filter category",
        options=["No category filter", *category_columns],
    )
    selected_category_values: set[str] | None = None
    if category_filter != "No category filter":
        category_values = (
            processed_data[category_filter]
            .astype("string")
            .fillna("<missing>")
        )
        selected_values = st.sidebar.multiselect(
            f"{category_filter.replace('_', ' ').title()}",
            options=sorted(category_values.unique().tolist()),
            default=sorted(category_values.unique().tolist()),
        )
        selected_category_values = set(selected_values)

    numeric_filter_columns = [
        column
        for column in (
            "loan_amnt",
            "annual_inc",
            "int_rate",
            "dti",
            "revol_util",
            "employment_length_years",
        )
        if column in processed_data.columns
        and pd.to_numeric(processed_data[column], errors="coerce").notna().any()
    ]
    numeric_filter = st.sidebar.selectbox(
        "Numeric range filter",
        options=["No numeric filter", *numeric_filter_columns],
    )
    selected_numeric_range: tuple[float, float] | None = None
    if numeric_filter != "No numeric filter":
        numeric_values = pd.to_numeric(
            processed_data[numeric_filter], errors="coerce"
        )
        low, high = float(numeric_values.min()), float(numeric_values.max())
        if low < high:
            selected_numeric_range = st.sidebar.slider(
                f"{numeric_filter.replace('_', ' ').title()} range",
                min_value=low,
                max_value=high,
                value=(low, high),
            )
        else:
            st.sidebar.caption("This feature has only one observed value.")

    try:
        filtered_data = filter_eda_data(
            processed_data,
            outcomes=outcome_codes,
            category_column=(
                category_filter if category_filter != "No category filter" else None
            ),
            categories=selected_category_values,
            numeric_column=(
                numeric_filter if numeric_filter != "No numeric filter" else None
            ),
            numeric_range=selected_numeric_range,
        )
        figures = create_eda_figures(filtered_data)
    except ValueError as error:
        st.error(f"Could not create EDA charts for the current filters: {error}")
        st.stop()

    st.subheader("Filtered dataset overview")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Applications", f"{len(filtered_data):,}")
    if filtered_data.empty:
        metric_columns[1].metric("Observed default rate", "—")
        metric_columns[2].metric("Median loan amount", "—")
        st.warning("The current filters select no applications.")
        st.stop()
    metric_columns[1].metric(
        "Observed default rate", f"{filtered_data['target'].mean():.2%}"
    )
    if "loan_amnt" in filtered_data.columns:
        median_loan = pd.to_numeric(
            filtered_data["loan_amnt"], errors="coerce"
        ).median()
        metric_columns[2].metric(
            "Median loan amount",
            f"{median_loan:,.0f}" if pd.notna(median_loan) else "—",
        )
    else:
        metric_columns[2].metric("Median loan amount", "Not available")

    chart_sections = {
        "Overview": ["target_distribution"],
        "Distributions": [
            name
            for name in figures
            if name.startswith(("histogram_", "count_"))
        ],
        "Default rates": [
            name
            for name in figures
            if name.startswith("default_rate_")
        ],
        "Relationships": [
            name
            for name in (
                "correlation_heatmap",
                "income_vs_loan_amount",
                "interest_rate_by_grade",
                "pairwise_relationships",
                "default_rate_by_dti",
            )
            if name in figures
        ],
    }
    chart_section = st.segmented_control(
        "Analysis view",
        options=list(chart_sections),
        default="Overview",
    )
    available_charts = [
        name for name in chart_sections[chart_section] if name in figures
    ]
    if not available_charts:
        st.info("No applicable features for this analysis view are present.")
    else:
        for start in range(0, len(available_charts), 2):
            chart_columns = st.columns(2)
            for chart_column, chart_name in zip(
                chart_columns, available_charts[start : start + 2]
            ):
                with chart_column:
                    st.plotly_chart(
                        figures[chart_name],
                        width="stretch",
                        key=f"eda_{chart_name}",
                        alt=chart_name.replace("_", " "),
                    )

    st.subheader("Descriptive statistics")
    st.text(
        describe_numeric_features(filtered_data)
        .round(3)
        .to_string(index=False)
    )
    st.subheader("Default-rate analysis")
    rate_table = default_rate_analysis(filtered_data)
    st.text(
        rate_table.round({"default_rate": 4}).to_string(index=False)
    )
    st.download_button(
        "Download filtered default-rate analysis",
        data=rate_table.to_csv(index=False).encode("utf-8"),
        file_name="default_rate_analysis.csv",
        mime="text/csv",
    )
    with st.expander("Descriptive risk insights"):
        st.text(create_insights(filtered_data))

    st.subheader("Generate downloadable reports")
    st.caption(
        "Reports and aggregate PNG charts are saved for the current filter "
        "selection. PNGs contain aggregated counts/rates rather than applicant rows."
    )
    if st.button("Generate or refresh EDA reports", type="primary"):
        try:
            report_paths, visualization_paths = generate_eda_reports(
                filtered_data, REPORTS_DIR
            )
        except (OSError, ValueError) as error:
            st.error(f"Could not generate EDA reports: {error}")
        else:
            st.success(
                f"Generated {len(report_paths)} reports and "
                f"{len(visualization_paths)} aggregate visualizations."
            )
    report_file_names = {
        "descriptive_statistics": "descriptive_statistics.csv",
        "correlation_matrix": "correlation_matrix.csv",
        "default_rate_analysis": "default_rate_analysis.csv",
        "eda_insights": "eda_insights.txt",
    }
    download_columns = st.columns(4)
    for column, (label, file_name) in zip(
        download_columns, report_file_names.items()
    ):
        report_path = REPORTS_DIR / file_name
        if report_path.is_file():
            column.download_button(
                f"Download {label.replace('_', ' ')}",
                data=report_path.read_bytes(),
                file_name=file_name,
                mime="text/plain" if file_name.endswith(".txt") else "text/csv",
            )
    visualization_dir = REPORTS_DIR / "visualizations"
    if visualization_dir.is_dir():
        visualization_files = sorted(visualization_dir.glob("*.png"))
        if visualization_files:
            st.caption(
                "Generated visualizations: "
                + ", ".join(path.name for path in visualization_files)
            )
    st.stop()

st.info(
    "Phases 1–6 are implemented. A genuine LendingClub dataset is required to "
    "prepare data, train a model, and produce real evaluation or explanation results."
)

data_available = (DATASET_DIR / "loan_data_processed.csv").is_file()
model_available = (MODELS_DIR / "best_model.joblib").is_file()
with st.container(horizontal=True):
    st.metric("Project phase", "Phase 6", border=True)
    st.metric(
        "Processed dataset",
        "Ready" if data_available else "Not added",
        border=True,
    )
    st.metric(
        "Trained model",
        "Ready" if model_available else "Not trained",
        border=True,
    )

st.subheader("Start with LendingClub data")
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Prepare a dataset")
        st.write(
            "Upload a genuine historical-loan CSV or place it at "
            "`dataset/loan_data.csv`. The original source is never overwritten."
        )
        st.caption(
            "Only eligible Fully Paid and Charged Off outcomes are retained; "
            "post-origination fields are excluded."
        )
with right:
    with st.container(border=True):
        st.subheader("Explore the workflow")
        st.write(
            "Use the sidebar to inspect data, explore observed risk patterns, "
            "train and evaluate models, and explain predictions."
        )
        st.caption(
            "No sample records or real model results are included. Results appear "
            "only after processing and training on genuine data."
        )

st.divider()
st.header("Dataset inspection and preparation")
st.markdown(
    "Upload a LendingClub CSV. Only **Fully Paid** and **Charged Off** records "
    "are retained; unresolved outcomes, identifiers, and post-origination fields "
    "are excluded. Missing feature values are imputed later, after the training "
    "split, to avoid leakage."
)
uploaded_file = st.file_uploader(
    "Choose a LendingClub CSV",
    type=["csv"],
    help="The uploaded original stays in memory and is not saved or overwritten.",
)
local_dataset = DATASET_DIR / "loan_data.csv"
source_label = uploaded_file.name if uploaded_file is not None else str(local_dataset)
if uploaded_file is None and not local_dataset.is_file():
    st.warning(
        "No dataset is present. See README.md for Kaggle download instructions; "
        "no sample records are bundled."
    )
if st.button(
    "Inspect and prepare dataset",
    type="primary",
    disabled=uploaded_file is None and not local_dataset.is_file(),
):
    for key in (
        "phase2_processed_path",
        "phase2_target_counts",
        "phase2_raw_inspection",
        "phase2_report_paths",
        "phase2_chart_paths",
    ):
        st.session_state.pop(key, None)
    source = uploaded_file if uploaded_file is not None else local_dataset
    with st.spinner("Loading, validating, and cleaning the dataset..."):
        try:
            raw_dataframe = load_lendingclub_csv(source)
            raw_inspection = inspect_dataset(raw_dataframe)
            cleaned_dataframe, cleaning_report = validate_and_clean_dataset(
                raw_dataframe
            )
            processed_path = DATASET_DIR / "loan_data_processed.csv"
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            cleaned_dataframe.to_csv(processed_path, index=False)
            report_paths = write_dataset_reports(
                raw_dataframe, cleaned_dataframe, cleaning_report, REPORTS_DIR
            )
            chart_paths = save_dataset_charts(cleaned_dataframe, REPORTS_DIR)
        except (DatasetLoadError, ValueError, OSError) as error:
            st.error(f"Dataset processing failed: {error}")
        else:
            st.session_state["phase2_processed_path"] = str(processed_path)
            st.session_state["phase2_target_counts"] = (
                cleaned_dataframe["target"].value_counts().to_dict()
            )
            st.session_state["phase2_raw_inspection"] = raw_inspection
            st.session_state["phase2_report_paths"] = {
                name: str(path) for name, path in report_paths.items()
            }
            st.session_state["phase2_chart_paths"] = [
                str(path) for path in chart_paths
            ]
            st.success(
                f"Processed {len(cleaned_dataframe):,} eligible records from "
                f"{source_label}. The raw source was preserved."
            )

if "phase2_processed_path" in st.session_state:
    processed_path = Path(st.session_state["phase2_processed_path"])
    if processed_path.is_file():
        target_counts = st.session_state["phase2_target_counts"]
        raw_inspection = st.session_state["phase2_raw_inspection"]
        st.subheader("Processed dataset")
        metric_columns = st.columns(5)
        metric_columns[0].metric(
            "Source rows", f"{raw_inspection['rows']:,}"
        )
        metric_columns[1].metric(
            "Source columns", f"{raw_inspection['columns']:,}"
        )
        metric_columns[2].metric("Eligible loans", f"{sum(target_counts.values()):,}")
        metric_columns[3].metric("Fully paid", f"{target_counts.get(0, 0):,}")
        metric_columns[4].metric("Charged off", f"{target_counts.get(1, 0):,}")
        with st.expander("Source dataset schema and inspection"):
            schema_rows = [
                {
                    "column": column,
                    "dtype": raw_inspection["dtypes"][column],
                    "missing_values": raw_inspection["missing_values"][column],
                    "unique_values": raw_inspection["unique_values"][column],
                }
                for column in raw_inspection["column_names"]
            ]
            st.write(
                f"Exact duplicate rows: {raw_inspection['duplicate_rows']:,}"
            )
            st.dataframe(schema_rows, hide_index=True)
            st.write("Raw loan status counts", raw_inspection["loan_status_counts"])
        st.caption(
            "Missing values in feature columns are intentionally kept in the "
            "processed CSV and will be imputed by a training-fitted pipeline."
        )
        st.download_button(
            "Download processed CSV",
            data=processed_path.read_bytes(),
            file_name="loan_data_processed.csv",
            mime="text/csv",
        )
        try:
            processed_preview = load_lendingclub_csv(processed_path)
        except DatasetLoadError as error:
            st.error(f"Could not reload the processed dataset: {error}")
        else:
            st.subheader("Preview")
            st.dataframe(processed_preview.head(100), hide_index=True)
            target_chart = (
                processed_preview["target"]
                .value_counts()
                .rename_axis("target")
                .reset_index(name="applications")
            )
            target_chart["loan_status"] = target_chart["target"].map(
                {0: "Fully Paid", 1: "Charged Off"}
            )
            st.subheader("Target distribution")
            import plotly.express as px

            st.plotly_chart(
                px.bar(
                    target_chart,
                    x="loan_status",
                    y="applications",
                    color="loan_status",
                    color_discrete_map={
                        "Fully Paid": "#11a8c4",
                        "Charged Off": "#e36f58",
                    },
                    labels={
                        "loan_status": "Loan status",
                        "applications": "Applications",
                    },
                ),
                width="stretch",
                alt="Counts of fully paid and charged off applications",
            )
            st.subheader("Missing values")
            missing = (
                processed_preview.isna()
                .sum()
                .loc[lambda values: values > 0]
                .sort_values(ascending=False)
                .head(30)
            )
            if not missing.empty:
                st.bar_chart(missing, horizontal=True)
            else:
                st.success("The processed dataset has no missing feature values.")
        st.caption(
            "Saved reports: "
            + ", ".join(
                Path(path).name
                for path in st.session_state.get("phase2_report_paths", {}).values()
            )
        )
