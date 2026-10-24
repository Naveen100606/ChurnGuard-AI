# ChurnGuard AI — Customer Churn Prediction & Retention Analytics

An end-to-end machine learning project that analyzes historical retail transactions, predicts customer churn risk, and explores customer-retention campaign scenarios.

## Project Overview

Customer churn can reduce revenue and long-term customer value. ChurnGuard AI uses historical purchasing behavior to identify customers who may stop purchasing, helping businesses prioritize customers for retention analysis.

The project combines SQL-based data analysis, Python feature engineering, machine learning, model interpretation, and an interactive Streamlit dashboard.

## Objectives

- Analyze historical customer purchasing behavior using SQL and Python.
- Build customer-level behavioral features from transaction data.
- Define churn using a future 90-day purchase inactivity window.
- Train and evaluate a Random Forest classifier using chronological validation.
- Rank customers by model-estimated churn risk.
- Explore hypothetical retention campaign costs and value.

## Dataset

**Dataset:** Online Retail II transaction data  
**Period:** December 2009 – December 2011

The dataset contains historical retail transactions, including invoice details, product quantities, prices, dates, and customer identifiers.

Data preparation excludes transactions that do not qualify as purchases, including cancellations, non-positive quantities or prices, and records without customer identifiers.

The data represents historical UK online retail transactions, not modern Indian retail data.

## Tech Stack

- **SQL:** MySQL
- **Programming:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn, Random Forest
- **Model interpretation:** SHAP
- **Visualization and dashboard:** Streamlit, Matplotlib
- **Model persistence:** Joblib
- **Development:** Jupyter Notebook, Git, GitHub

## Methodology

### 1. Data Cleaning and SQL Analysis

- Audited missing values, duplicate transactions, cancellations, and invalid quantities and prices.
- Created an invoice-level dataset using MySQL.
- Analyzed revenue trends, order volumes, and customer purchasing behavior.
- Built customer-level features for churn prediction.

### 2. Feature Engineering

The model uses four behavioral features calculated from a historical 90-day observation window:

| Feature | Description |
|---|---|
| `orders_90d` | Number of orders |
| `spend_90d` | Total customer spending |
| `quantity_90d` | Total quantity purchased |
| `recency_days` | Days since the most recent purchase at the cutoff |

### 3. Churn Definition

A customer is labelled as churned when they make no qualifying purchase during the following 90-day label window.

- `1` — No qualifying purchase during the label window.
- `0` — At least one qualifying purchase during the label window.

This is an operational definition for the project and does not prove that a customer has permanently left a business.

### 4. Machine Learning

A Random Forest classifier was trained on historical customer snapshots.

Chronological validation was used to select a classification threshold on a validation period and evaluate the final model on a later historical test snapshot.

The selected threshold was **0.50**.

### 5. Model Evaluation

Final chronological test results at threshold 0.50:

| Metric | Result |
|---|---:|
| Accuracy | 59.1% |
| Precision | 42.4% |
| Recall | 79.1% |
| F1-score | 0.552 |
| ROC-AUC | 0.704 |

The model identified 485 of 613 actual churners in the test snapshot, while incorrectly flagging 658 customers who did not churn.

The model prioritizes detecting churners, at the cost of generating false positives. Accuracy alone is therefore insufficient to assess its usefulness.

### 6. Interactive Dashboard

The Streamlit dashboard provides:

- Customer churn-risk scores.
- Adjustable risk threshold.
- Customer risk ranking.
- Risk distribution visualization.
- CSV download of flagged customers.
- Hypothetical campaign cost and net-value calculations.

Campaign outcomes depend on user-entered assumptions. The dashboard does not demonstrate that a real retention offer would prevent churn.

## Project Structure

```text
churnguard-ai/
├── data/
│   └── processed/
│       └── churn_training_data.csv
├── models/
│   └── churn_random_forest.joblib
├── notebooks/
│   └── 03_chronological_validation.ipynb
├── app.py/
│   └── app.py
├── README.md
└── requirements.txt
```

Adjust the notebook path above if your actual folder structure differs.

## Run the Dashboard Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd churnguard-ai
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Launch Streamlit

If the application remains inside the nested `app.py` folder:

```bash
python -m streamlit run app.py/app.py
```

If you move the application to the project root, use:

```bash
python -m streamlit run app.py
```

Ensure the processed dataset and saved model are available at the paths expected by the application.

## Limitations and Future Improvements

- Evaluate the model across additional chronological test periods.
- Test whether repeated customer appearances across snapshots affect evaluation.
- Improve precision and investigate false-positive costs.
- Evaluate probability calibration and threshold selection against business costs.
- Compare Random Forest against Logistic Regression and XGBoost using a consistent validation strategy.
- Test retention campaigns through controlled experiments before estimating actual causal impact.
- Add model monitoring and a reproducible data-processing pipeline.

## Conclusion

ChurnGuard AI demonstrates an end-to-end workflow for customer churn analytics, from transaction cleaning and SQL analysis to machine learning evaluation and an interactive retention dashboard.

The project emphasizes chronological evaluation and interpretable business trade-offs rather than relying on accuracy alone.
