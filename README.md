# Explainable Customer Retention Decision-Support System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://ibm-telco-churn-prediction.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-Random%20Forest-2E8B57)](#modeling-approach)

## Live demo

Open the application here: **[Explainable Customer Retention Decision-Support System — Streamlit Demo](https://ibm-telco-churn-prediction.streamlit.app/)**.

The demo may be asleep after a period of inactivity. If Streamlit displays a wake-up message, select **Yes, get this app back up!** and wait for the application to load. The public demo uses the deployed application environment; the optional local Ollama summary is intended for local execution unless Ollama is hosted separately.

## Project overview

**Explainable Customer Retention Decision-Support System** is an interactive, explainable-AI application for estimating customer churn risk and supporting transparent retention investigations.

Customer churn creates recurring-revenue risk and increases customer-acquisition pressure. This project analyzes historical telecom customer behavior and provides an interactive application for estimating churn risk, understanding the model’s individual prediction factors, and prioritizing transparent retention investigations.

The project combines the original data-science workflow with a more maintainable explainable-AI application:

1. Exploratory analysis of customer, service, contract, and billing attributes.
2. Data cleaning and reproducible preprocessing.
3. Class-weighted Random Forest churn prediction.
4. Input validation and provisional risk bands.
5. Individual SHAP explanations rather than only global feature importance.
6. Rule-based recommendations with explicit reasons.
7. Numeric customer segmentation using K-means clustering.
8. Model-performance and threshold-analysis support.
9. Data-quality and monitoring utilities.
10. Optional structured business summaries using local Ollama or OpenAI fallback.

> **Responsible-use statement:** This is a decision-support prototype. The model identifies historical patterns associated with churn; it does not prove causation, customer intent, or the effectiveness of any retention intervention. Consequential decisions require human review.

## What the application provides

| Application view | Purpose |
|---|---|
| **Prediction** | Accepts a customer profile and displays predicted churn probability and a provisional Low/Medium/High risk band. |
| **Explanation** | Displays customer-level SHAP contributions, including factors pushing the prediction toward higher or lower churn risk. |
| **Recommended Actions** | Presents transparent investigation suggestions and the reason behind each suggestion. |
| **Segments** | Profiles K-means customer groupings using numeric tenure, charge, and senior-citizen features. |
| **Model Performance** | Displays saved evaluation results when the corresponding evaluation artifact is present. |
| **AI Summary** | Produces a structured business summary from verified model outputs using local Ollama first, with optional OpenAI fallback. |
| **Monitoring** | Reports reference-data quality and provides utilities for numeric drift and later-period performance checks. |

## Architecture

```text
Customer profile
      |
Input validation
      |
Saved scikit-learn pipeline
      |
Churn probability + provisional risk band
      |
+----------------------+-------------------------+------------------+
| Individual SHAP      | Rule-based investigation| K-means customer |
| explanation          | recommendations        | segmentation     |
+----------------------+-------------------------+------------------+
      |
Verified structured facts
      |
Optional Ollama/OpenAI business summary
      |
Streamlit interface
```

The saved prediction pipeline remains the source of truth. Explainability, recommendations, segmentation, monitoring, and language-model summaries are supporting layers around the validated model rather than replacements for it.

## Dataset

The project uses the IBM Telco Customer Churn dataset. It contains demographic, account, service, contract, billing, and churn information for telecom customers.

The main dataset file is:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The target column is `Churn`, represented in the raw data as `Yes` or `No` and converted to `1` or `0` for modeling. The `customerID` column is excluded from the feature set.

Review the dataset’s license and attribution requirements before redistributing the data or using it outside a portfolio or educational context.

## Data preparation and exploratory analysis

The original notebook documents the exploratory workflow. The reusable data loader performs the following steps:

- Removes duplicate rows.
- Converts `TotalCharges` from text to numeric values.
- Replaces blank or invalid `TotalCharges` values with zero.
- Removes `customerID` from the modeling features.
- Maps `Churn` to a binary target.
- Separates numeric and categorical features.
- Uses a scikit-learn `ColumnTransformer` so preprocessing remains connected to the saved model.
- Standardizes numeric features.
- One-hot encodes categorical features with unknown-category handling.

The analysis focuses on patterns involving tenure, monthly charges, total charges, contract type, services, payment methods, and customer demographics. These patterns are predictive associations in the available historical data and should not be interpreted as causal conclusions.

## Modeling approach

The current application model is a saved scikit-learn pipeline located at:

```text
models/churn_model.pkl
```

The pipeline contains a preprocessing stage and a class-weighted Random Forest classifier with the following configuration:

| Component | Configuration |
|---|---|
| Numeric preprocessing | `StandardScaler` |
| Categorical preprocessing | `OneHotEncoder(handle_unknown="ignore")` |
| Classifier | `RandomForestClassifier` |
| Number of trees | 300 |
| Maximum depth | 10 |
| Class weighting | `balanced` |
| Random state | 42 |

The project also includes evaluation utilities for comparing a majority-class baseline, logistic regression, Random Forest, and gradient boosting with stratified cross-validation. The evaluation workflow reports ROC-AUC, PR-AUC, recall, F1, and variability across folds when it is run and its outputs are saved.

## Prediction validation and risk bands

The application validates required fields, categorical values, numeric values, charges, and tenure before calling the model. Invalid data produces a clear error instead of silently generating a prediction.

The current prototype maps predicted probabilities to provisional risk bands:

| Churn probability | Risk band |
|---:|---|
| `< 0.30` | Low |
| `0.30–<0.60` | Medium |
| `≥ 0.60` | High |

These thresholds are not universal business rules. A production implementation should select them using intervention capacity, expected retention value, false-positive cost, false-negative cost, and probability calibration.

## Explainable AI with SHAP

The explanation layer transforms each customer through the pipeline preprocessor and applies `shap.TreeExplainer` to the Random Forest classifier. It returns the strongest individual feature contributions for that customer.

A positive contribution means that the transformed feature representation pushed that prediction toward higher modeled churn risk. A negative contribution pushed it toward lower modeled churn risk. These are model contributions, not proof that the feature caused the customer’s behavior.

The application deliberately avoids presenting global Random Forest feature importance as the explanation for an individual customer.

## Transparent retention recommendations

Recommendations are generated by explicit rules rather than allowing a language model to invent customer actions. Current investigation triggers include:

- Prioritizing customers whose predicted churn probability is at least 0.60 for retention review.
- Reviewing annual-contract or loyalty-plan eligibility for month-to-month customers.
- Reviewing onboarding and early-tenure support for customers with tenure of 12 months or less.
- Reviewing plan fit, pricing, and service value when monthly charges are at least 80.

These are investigation suggestions. They are not guaranteed retention solutions, automatic discounts, or automatic customer-treatment decisions.

## Customer segmentation

The segmentation module applies K-means clustering to standardized numeric features:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `SeniorCitizen`

Candidate cluster counts from 2 through 6 are evaluated using silhouette scores, and the highest-scoring candidate is selected. Segment profiles include customer count, churn rate when labels are available, median tenure, and median monthly charges.

The clusters are analytical groupings, not validated natural customer types. Their stability and usefulness should be tested before they are used in business processes.

## Monitoring prototype

The monitoring module provides utilities for:

- Missing-value rates and unique-value counts by feature.
- Numeric mean, median, standard-deviation, and mean-delta comparisons between reference and later datasets.
- Later-period ROC-AUC, PR-AUC, precision, and recall when labels become available.

The repository does not currently receive a live production data stream. Therefore, this is a monitoring prototype and design, not a claim of live production monitoring.

## Optional AI-generated business summary

The preferred local setup uses [Ollama](https://ollama.com/) so the business summary can be generated without an OpenAI API key. The application sends only a small structured object containing verified risk, probability, explanation, recommendation, and limitation fields.

Install Ollama and download the configured local model:

```powershell
ollama pull qwen3.5:4b
```

The application expects Ollama at:

```text
http://localhost:11434
```

Optional configuration variables are:

```powershell
$env:OLLAMA_URL="http://localhost:11434"
$env:OLLAMA_MODEL="qwen3.5:4b"
```

The application uses Ollama first. If Ollama is unavailable and `OPENAI_API_KEY` is configured, it uses OpenAI as a fallback. On Streamlit Cloud, add the key under **App settings → Secrets** rather than committing it to GitHub:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4o-mini"
```

`OPENAI_MODEL` is optional and can be changed to a model available to your API account. The app reads both environment variables and Streamlit Secrets, so the same code works locally and in the deployed app. `OPENAI_BASE_URL` is also supported for OpenAI-compatible providers.

> A local Ollama process works when the application runs on your own computer. A public Streamlit server cannot automatically access Ollama running on your private computer’s `localhost`.

If neither provider is configured, the AI Summary tab remains usable: it generates a deterministic summary from the verified churn probability, model-associated risk factors, and existing recommendations. This fallback does not pretend to be an LLM response, and its caveat is shown in the UI. Therefore, missing Ollama or OpenAI configuration no longer leaves the feature disabled or displays a dead-end setup error.

## Project structure

```text
Explainable-Customer-Retention-Decision-Support-System/
├── app.py
├── README.md
├── requirements.txt
├── Customer_Churn_Project.ipynb
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── churn_model.pkl
├── src/
│   ├── ai_summary.py
│   ├── explain.py
│   ├── metrics.py
│   ├── monitoring.py
│   ├── predict.py
│   ├── recommendations.py
│   └── segmentation.py
├── test_predict.py
├── test_explain_and_recommendations.py
└── test_segmentation_monitoring.py
```

## Installation and local execution

### Windows PowerShell

Run these commands from the folder containing `app.py` and `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

If OneDrive causes virtual-environment file-locking problems, create the environment outside the synchronized project folder:

```powershell
python -m venv "$HOME\.venvs\customer-churn"
& "$HOME\.venvs\customer-churn\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pytest -q
streamlit run app.py
```

## Testing

The test suite covers:

- Probability bounds and risk-band output.
- Missing required fields.
- Negative charges.
- Unknown categorical values.
- Impossible tenure values.
- SHAP explanation structure and feature names.
- Recommendation reasons and priority behavior.
- Segment fitting and profiling.
- Monitoring metric output.

Run the tests with:

```bash
python -m pytest -q
```

## Deployment and live-demo status

The project is suitable for portfolio and demonstration deployment. The public Streamlit URL is included at the top of this README, but free Streamlit deployments may sleep after inactivity and may need to be awakened.

Before describing the project as production-ready, complete the following:

1. Pin and reproduce the complete model-training environment.
2. Verify the serialized model under the exact deployment dependency versions.
3. Save and commit reproducible evaluation artifacts if the Model Performance tab is expected to display them.
4. Select a decision threshold using actual retention capacity and intervention-cost assumptions.
5. Connect the monitoring prototype to later-period data and labels.
6. Review privacy, fairness, access control, and human-approval requirements.
7. Use a hosted language-model provider or securely hosted Ollama service if public users must access the AI Summary feature.

## Responsible use and limitations

Historical customer data may not represent current products, prices, service quality, or customer populations. The model may reproduce patterns and biases present in the historical data.

The application should not be used to deny service, penalize customers, or make fully automated customer-treatment decisions. SHAP values describe model behavior rather than causation. Rule-based recommendations are investigation prompts. K-means segments are exploratory groupings. Human review is required for consequential decisions.

## References

[1]: [Streamlit documentation](https://docs.streamlit.io/)
[2]: [scikit-learn documentation](https://scikit-learn.org/stable/)
[3]: [SHAP documentation](https://shap.readthedocs.io/)
[4]: [Ollama documentation](https://docs.ollama.com/)
[5]: [IBM Telco Customer Churn dataset information](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
