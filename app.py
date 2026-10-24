import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION AND PROJECT PATHS
# ==================================================
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📊",
    layout="wide"
)

# Find the project root whether this file is stored directly in the
# project or inside a subfolder (such as the current `app.py` folder).
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR

for candidate_dir in (CURRENT_DIR, *CURRENT_DIR.parents):
    has_data_dir = (candidate_dir / "data").is_dir()
    has_models_dir = (candidate_dir / "models").is_dir()

    if has_data_dir and has_models_dir:
        BASE_DIR = candidate_dir
        break

DATA_PATH = BASE_DIR / "data" / "processed" / "churn_training_data.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_random_forest.joblib"

st.title("ChurnGuard AI")
st.subheader("Customer Retention & Churn Risk Dashboard")

st.write(
    "Identify customers at risk of churn using machine learning "
    "and explore hypothetical customer-retention campaign scenarios."
)

st.info(
    "This dashboard uses a trained Random Forest model. "
    "Risk scores are model estimates, and campaign ROI is an "
    "illustrative scenario rather than a guaranteed business outcome."
)



# ==================================================
# 2. LOAD CUSTOMER DATA
# ==================================================


# ==================================================
# 2. LOAD AND VALIDATE CUSTOMER DATA
# ==================================================

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Check whether the dataset exists
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

# Load the dataset
try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

# Validate the loaded dataset
if df.empty:
    st.error("The dataset is empty.")
    st.stop()

st.success(f"Loaded {len(df):,} customer records.")

required_columns = {
    "customer_id",
    "orders_90d",
    "spend_90d",
    "quantity_90d",
    "recency_days",
    "churn_label"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(f"Missing columns: {sorted(missing_columns)}")
    st.stop()

if df["customer_id"].isna().any():
    st.error("Missing customer IDs found.")
    st.stop()

if df["customer_id"].duplicated().any():
    st.error("Duplicate customer IDs found.")
    st.stop()

feature_columns = [
    "orders_90d",
    "spend_90d",
    "quantity_90d",
    "recency_days"
]

for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[feature_columns].isna().any().any():
    st.error("Missing or invalid model feature values found.")
    st.stop()

if not np.isfinite(df[feature_columns].to_numpy(dtype=float)).all():
    st.error("Infinite or invalid model feature values found.")
    st.stop()

if not df["churn_label"].isin([0, 1]).all():
    st.error("churn_label must contain only 0 and 1.")
    st.stop()

df["churn_label"] = df["churn_label"].astype(int)


# ==================================================
# 3. LOAD THE SAVED CHRONOLOGICAL RANDOM FOREST
# ==================================================
@st.cache_resource
def load_model(path, modified_time):
    package = joblib.load(path)

    if not isinstance(package, dict) or "model" not in package:
        raise ValueError(
            "Unexpected model package. Save a dictionary containing "
            "'model' and 'features' from the training notebook."
        )

    model = package["model"]
    features = package.get("features")
    threshold = float(package.get("threshold", 0.50))

    if not features:
        raise ValueError("The saved model package has no feature list.")

    if not 0 <= threshold <= 1:
        raise ValueError("The saved threshold must be between 0 and 1.")

    return model, features, threshold


if not MODEL_PATH.exists():
    st.error(
        f"Trained model not found:\n\n{MODEL_PATH}\n\n"
        "Run the model-saving cell in your notebook and verify "
        "that the file is saved in the project's models folder."
    )
    st.stop()

try:
    rf_model, model_features, saved_threshold = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

missing_features = set(model_features) - set(df.columns)

if missing_features:
    st.error(
        f"The dataset is missing model features: {sorted(missing_features)}"
    )
    st.stop()

if not hasattr(rf_model, "predict_proba"):
    st.error("The saved model does not support probability predictions.")
    st.stop()

# ==================================================
# 4. GENERATE MODEL PREDICTIONS
# ==================================================
X_customers = df[model_features].copy()

try:
    probabilities = rf_model.predict_proba(X_customers)

    # Find the probability column corresponding to churn label 1.
    classes = list(rf_model.classes_)

    if 1 not in classes:
        st.error("The trained model has no class labelled 1 (churn).")
        st.stop()

    churn_class_index = classes.index(1)
    df["churn_probability"] = probabilities[:, churn_class_index]

except Exception as exc:
    st.error(f"Could not generate churn predictions: {exc}")
    st.stop()

if (
    not np.isfinite(df["churn_probability"]).all()
    or not df["churn_probability"].between(0, 1).all()
):
    st.error("The model generated invalid probability values.")
    st.stop()

# ==================================================
# 5. SIDEBAR: CAMPAIGN ASSUMPTIONS
# ==================================================
st.sidebar.header("Campaign assumptions")

st.sidebar.caption(
    f"Saved model threshold: {saved_threshold:.2f}"
)

threshold = st.sidebar.slider(
    "Churn risk threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(np.clip(saved_threshold, 0.05, 0.95)),
    step=0.05,
    help=(
        "Customers with estimated churn probability at or above "
        "this threshold will be flagged."
    )
)

offer_cost = st.sidebar.number_input(
    "Offer cost per contacted customer (₹)",
    min_value=0.0,
    value=100.0,
    step=50.0
)

success_rate = st.sidebar.slider(
    "Assumed offer success rate",
    min_value=0,
    max_value=100,
    value=20,
    step=5
) / 100

customer_value = st.sidebar.number_input(
    "Estimated value per saved customer (₹)",
    min_value=0.0,
    value=2000.0,
    step=500.0
)

# ==================================================
# 6. FLAG AND RANK CUSTOMERS
# ==================================================
df["is_flagged"] = df["churn_probability"] >= threshold

selected = (
    df.loc[df["is_flagged"]]
    .copy()
    .sort_values("churn_probability", ascending=False)
)

contacted = len(selected)
campaign_cost = contacted * offer_cost

# ==================================================
# 7. ILLUSTRATIVE CAMPAIGN SCENARIO
# ==================================================
# Historical labels are used here only to estimate how many
# historically labelled churners were flagged.
# This does NOT establish that an offer would save those customers.

actual_churners_selected = int(selected["churn_label"].sum())

expected_saves = actual_churners_selected * success_rate
expected_value = expected_saves * customer_value
estimated_net_value = expected_value - campaign_cost

# ==================================================
# 8. DASHBOARD KPI CARDS
# ==================================================
st.header("Dashboard overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers analysed",
    f"{len(df):,}"
)

col2.metric(
    "Customers flagged",
    f"{contacted:,}"
)

col3.metric(
    "Estimated campaign cost",
    f"₹{campaign_cost:,.0f}"
)

col4.metric(
    "Illustrative net value",
    f"₹{estimated_net_value:,.0f}"
)

st.caption(
    "Net value is a hypothetical scenario based on historical labels "
    "and user-entered assumptions. It is not measured campaign ROI."
)

st.divider()

# ==================================================
# 9. HISTORICAL OUTCOME OVERVIEW
# ==================================================
st.header("Historical customer outcomes")

historical_churn_rate = df["churn_label"].mean()

if contacted > 0:
    flagged_historical_churn_rate = selected["churn_label"].mean()
else:
    flagged_historical_churn_rate = np.nan

col1, col2, col3 = st.columns(3)

col1.metric(
    "Historical churn rate",
    f"{historical_churn_rate:.1%}"
)

col2.metric(
    "Assumed offer success rate",
    f"{success_rate:.0%}"
)

col3.metric(
    "Flagged customers' historical churn rate",
    (
        f"{flagged_historical_churn_rate:.1%}"
        if pd.notna(flagged_historical_churn_rate)
        else "N/A"
    )
)

st.caption(
    "These outcomes describe the historical labelled dataset. "
    "They should not be interpreted as performance on new customers."
)

# ==================================================
# 10. CHURN RISK DISTRIBUTION
# ==================================================
st.header("Churn risk distribution")

risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.000001]

risk_labels = [
    "Low (0–20%)",
    "Moderate (20–40%)",
    "High (40–60%)",
    "Very high (60–80%)",
    "Extreme (80–100%)"
]

df["risk_category"] = pd.cut(
    df["churn_probability"],
    bins=risk_bins,
    labels=risk_labels,
    include_lowest=True,
    right=True
)

risk_counts = (
    df["risk_category"]
    .value_counts(sort=False)
    .reindex(risk_labels, fill_value=0)
)

chart_data = pd.DataFrame({
    "Risk category": risk_counts.index,
    "Number of customers": risk_counts.values
}).set_index("Risk category")

st.bar_chart(chart_data)

st.caption(
    "Risk categories group customers by model-estimated score. "
    "The scores have not necessarily been calibrated to real-world "
    "churn probabilities."
)

# ==================================================
# 11. CUSTOMER RISK RANKING
# ==================================================
st.header("Customer risk ranking")

customer_columns = [
    "customer_id",
    "orders_90d",
    "spend_90d",
    "quantity_90d",
    "recency_days",
    "churn_probability"
]

if contacted > 0:
    display_df = selected[customer_columns].head(100).copy()

    display_df["churn_probability_pct"] = (
        display_df.pop("churn_probability") * 100
    ).round(1)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info(
        "No customers meet the current threshold. "
        "Try lowering the threshold in the sidebar."
    )

# ==================================================
# 12. DOWNLOAD FLAGGED CUSTOMERS
# ==================================================
download_df = selected[customer_columns].copy()

download_df["churn_probability_pct"] = (
    download_df.pop("churn_probability") * 100
).round(1)

st.download_button(
    label="Download flagged customers (CSV)",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="churnguard_flagged_customers.csv",
    mime="text/csv",
    disabled=download_df.empty
)

# ==================================================
# 13. MODEL VALIDATION INFORMATION
# ==================================================
with st.expander("Model validation and limitations"):
    st.markdown('''
        
        **Model**

        This dashboard uses the saved chronological Random Forest model.

        **Features**

        The model uses orders, spending, quantity, and recency over the
        historical feature window.

        **Threshold**

        Customers are flagged when their model-estimated churn score
        meets or exceeds the selected threshold. The default comes from
        the saved model package.

        **Historical labels**

        The churn label indicates whether a customer made a qualifying
        purchase during the future 90-day label window used to construct
        the historical training dataset.

        **Chronological evaluation**

        The reported test metrics were calculated on a later historical
        snapshot. The dashboard dataset is not an independent test set,
        so its historical outcomes must not be presented as test metrics.

        **Probability interpretation**

        Model scores are not necessarily calibrated probabilities of
        real-world churn.

        **Campaign scenario**

        The assumed offer success rate is user-entered. The calculation
        does not establish that an offer prevents churn. A controlled
        experiment would be needed to estimate causal campaign impact.

        **Deployment**

        Before contacting real customers, validate the pipeline on newer
        data, check data privacy requirements, and monitor model drift.
        '''
    )

st.caption(
    "ChurnGuard AI | Random Forest | Chronological validation | "
    "Illustrative retention scenarios"
)
