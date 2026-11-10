# Credit Risk Modeling (credit-risk-ai)

An AI-powered credit risk modeling system for estimating loan default risk,
with a Streamlit dashboard and explainability features.

> **Development status:** Phase 6 (dataset preparation, EDA, model training,
> and evaluation workflow). No LendingClub loan records are bundled. No
> production model, real-data performance results, or trained-model
> explanations are available until a genuine dataset is processed and training
> is run.

## Project structure

```text
Credit-Risk-Modeling/
├── dataset/                 # User-provided LendingClub CSV data (not committed)
├── models/                  # Generated model artifacts (not committed)
├── notebooks/               # Exploratory notebooks
├── reports/                 # Generated evaluation and explainability reports
├── src/
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── statistical_analysis.py
│   ├── smote_handler.py
│   ├── model_utils.py
│   ├── model_selection.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── threshold_optimization.py
│   ├── business_risk_analysis.py
│   ├── model_calibration.py
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│   ├── explanation_generator.py
│   ├── train.py
│   ├── evaluate.py
│   └── prediction.py
├── tests/
│   ├── test_app.py
│   ├── test_data_processing.py
│   ├── test_eda.py
│   ├── test_model_training.py
│   ├── test_model_evaluation.py
│   └── test_explainability.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.11–3.14 and pip.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run commands with the environment interpreter
directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard uses a navy-and-cyan fintech theme with light and dark appearance
variants, a branded sidebar workspace navigator, compact project-status metrics,
and bordered content cards. Theme settings live in
`.streamlit\config.toml` and apply when the Streamlit app is restarted or rerun.

Upload a CSV in the app, or place a LendingClub CSV at
`dataset\loan_data.csv` and select **Inspect and prepare dataset**. The app
creates `dataset\loan_data_processed.csv` and reports under `reports\`.
Uploaded raw files are processed in memory and are not saved or overwritten.
The source file at `dataset\loan_data.csv` is left untouched. Processed loan
data and Phase 2 reports are git-ignored; the requested aggregate Phase 3 EDA
reports are explicitly allowed by `.gitignore`.

## Dataset acquisition

No sample records or fabricated data are included. Kaggle's public catalog
currently lists [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
(`wordsforthewise/lending-club`), describing historical accepted and rejected
loan data; its catalog metadata reports CC0. The archive is large (over
600 MB), and its dataset file inventory/download endpoint was not available in
this environment, so no raw CSV could be acquired or verified here. Check the
dataset page and its current license/terms before downloading or using it.

To download using Kaggle's CLI (install/configure credentials separately and
confirm access on Kaggle first):

```powershell
python -m pip install kaggle
kaggle datasets download -d wordsforthewise/lending-club --unzip -p dataset\source
```

The archive can be large and may contain accepted and rejected applications,
several quarterly files, or different schema versions. Choose one accepted-loan
CSV with `loan_status` and `loan_amnt` (for example an `accepted_...csv` or
`LoanStats_...csv`) and copy it to `dataset\loan_data.csv`; rejected-application
files do not have the required repayment outcome. Do not concatenate schema
vintages without first reconciling their columns. Alternatively, upload a
compatible file directly in the app. Expected target column: `loan_status`,
with `Fully Paid` and
`Charged Off` values. Rows with other, missing, or unresolved statuses,
including loans marked as not meeting credit policy, are excluded.

LendingClub public historical data is a retrospective snapshot, not a complete
record of information available to every lender at decision time. Schema and
status conventions vary over time; records may contain missing or revised
values. The cleaning step excludes known payment/outcome fields, identifiers,
and text fields. Missing feature values remain in the processed CSV and should
be imputed only after the training/test split using the provided scikit-learn
preprocessor. Extreme numeric values are reported for review, not blindly
removed.

## Phase 2 outputs

- `dataset\loan_data_processed.csv`: de-duplicated eligible loans, binary
  target (`0 = Fully Paid`, `1 = Charged Off`), application-time features, and
  engineered features. Missing feature values remain for a training-fitted
  imputer.
- `reports\dataset_summary.txt`, `reports\missing_values.csv`,
  `reports\data_cleaning_report.txt`: raw inspection and cleaning results.
- `reports\target_distribution.csv`, `reports\feature_descriptions.csv`, and
  diagnostic PNG charts: target, feature, and missingness summaries.
- `src\preprocessing.py`: reusable `ColumnTransformer` with train-fitted
  numeric imputation/scaling and categorical imputation/one-hot encoding.

The three requested text/CSV reports in `reports\` start as explicit
"not generated" notices because no source dataset is bundled. Processing a
real dataset replaces them with data-derived results.

## Phase 3 exploratory analysis

After processing a dataset, select **Exploratory data analysis** in the
dashboard. The page offers outcome, category, and numeric-range filters,
interactive Plotly charts, descriptive numeric statistics, and observed
default-rate summaries for available features only. Numeric variables are
quantile-binned for descriptive rate comparisons; category names are never
assumed to exist. Statistics and reports reflect the current filter selection.

**Observed default rates are descriptive, not predicted probabilities.**
Correlations and segment differences do not establish causal effects and can
reflect selection, historical policy, vintage, or data quality. Small groups
are shown with their application counts because their rates can be unstable.

Generate or refresh reports from the EDA page to write
`reports\descriptive_statistics.csv`, `reports\correlation_matrix.csv`,
`reports\eda_insights.txt`, and `reports\default_rate_analysis.csv`.
Aggregated PNG charts are written to `reports\visualizations\`. The app's
interactive charts may sample at most 10,000 rows for scatterplots and 2,000
rows for pairwise plots (fixed random seed); summaries and exported aggregate
charts use all filtered rows. No applicant-level values are embedded in saved
visualization images. With no processed data, report placeholders say that no
results are available rather than fabricating statistics or charts.

## Phase 4 model training

After processing a real dataset, select **Model training** in the dashboard and
start the training run. The workflow rejects known outcome leakage fields,
uses a reproducible 80/20 stratified split, and tunes logistic regression,
decision tree, and random forest using stratified cross-validation scored by
average precision. Numeric features are imputed/scaled and categorical features
are imputed/ordinal encoded before training-fold-only SMOTE, SMOTENC, or SMOTEN;
the resulting categories are one-hot encoded for classification. When the
minority training class has fewer than two observations, random oversampling
is used and explicitly reported. The untouched holdout is never resampled.

Holdout comparisons include accuracy, precision, recall, F1, ROC-AUC, and
average precision at a 0.5 decision threshold. Model selection prioritizes
recall, average precision, F1, precision, and ROC-AUC—not accuracy—using
training-partition out-of-fold results. The untouched holdout is reserved for
final performance reporting. An optional minimum recall constraint is
configurable in the training page. This is a descriptive model selection
criterion, not a fully specified lender loss function; thresholds and costs
require separate business validation.

On a successful real-data run, the app writes `models\best_model.joblib`,
`models\preprocessing_pipeline.joblib`, `models\model_metadata.json`, and
`models\feature_names.json`; it also writes comparison, class distribution,
and training reports plus charts under `reports\`. Trained binary artifacts
remain git-ignored. Metadata and report files in the repository initially
state that training has not been run because no source data is present.

The evaluation page compares accuracy, precision, recall, F1, ROC-AUC,
average precision, specificity, false-positive and false-negative rates, and
Matthews correlation. It also provides confusion matrices, ROC and
precision-recall curves, and calibration reliability analysis.

The evaluation run uses the training partition's out-of-fold (OOF) probability
scores to compare thresholds 0.30–0.70 and select one per model by configurable
false-negative and false-positive relative costs (defaults 5:1). A dashboard
slider explores threshold trade-offs on those same validation scores only.
Final metrics, confusion matrices, curves, relative-cost summaries, and raw
versus sigmoid-calibrated probability reliability are calculated on the
untouched holdout. Sigmoid calibrators are fitted on OOF scores, never the
holdout. Cost weights are assumed relative units, not dollars or estimated
losses; operational use requires institution-specific exposure, loss-given-
default, review-cost, and policy inputs.

Because hyperparameters are selected through cross-validation on the training
partition before OOF probabilities are generated for the selected estimators,
the OOF threshold scores may be somewhat optimistic from model-selection reuse.
The holdout remains untouched for final reporting, but temporal validation and
external validation are still needed before deployment. The holdout is a
random split, not temporal validation; LendingClub historical data can have
vintage, selection, and reporting bias, and SMOTE changes the training class
prior.

Successful evaluation writes aggregate metrics and threshold, calibration,
business-risk, ROC-coordinate, precision-recall-coordinate reports under
`reports\`, plus confusion, ROC, PR, threshold, metric, and calibration charts
under `reports\visualizations\`. Applicant feature rows are not stored in
evaluation bundles; OOF/holdout bundles contain labels and probabilities only
and remain git-ignored. Evaluation placeholders explicitly indicate that no
results are available before a genuine training run.

## Phase 6 explainable AI

After a genuine training run, choose **Explainable AI** to explain the selected
saved model and an individual applicant. Input can be a processed application
or manually entered application-time features. The app uses the model's saved
preprocessor, one-hot encoder, and classifier in their fitted order; it does
not train or substitute a new model for explanations.

Global and individual SHAP explanations support the trained Logistic
Regression, Decision Tree, and Random Forest models. Tree SHAP values are in
default-probability units; Logistic Regression SHAP values are in log-odds.
Global importance aggregates one-hot contributions to the original input
feature. LIME uses the same fitted transforms and classifier probabilities;
its values describe a local surrogate and are not additive SHAP values.
Individual reports include positive and negative contributions, a SHAP
waterfall, a force plot when supported, and a LIME HTML explanation.

Global SHAP is computed on a deterministic, bounded sample of feature rows
from the processed dataset (without target labels); explainer background
samples are also bounded. Individual applicant values are included in local
contribution files and the LIME HTML, so generated artifacts under
`reports\explainability\` are git-ignored and should be handled as sensitive
local data. Clear them when no longer needed and do not share them without
authorization. Explanations reflect model associations, not causal reasons or
guarantees of future default. In the absence of a genuine dataset and fitted
artifact, the report files remain explicit "not generated" notices; test
fixtures validate code only and are not presented as credit-risk findings.

## Tests

```powershell
python -m pytest
```

Do not upload or place personally identifiable applicant information in this
project.
#   c r e d i t - r i s k  
 