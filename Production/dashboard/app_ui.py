import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError

try:
    from langchain_huggingface import HuggingFaceEndpoint
except Exception:
    HuggingFaceEndpoint = None

# -----------------------------------------------------------------------------
# App Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Supermarket Intelligence Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    .metric-card {
        background: #f7f9fc;
        border: 1px solid #e8edf3;
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Configuration for file paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "data" / "cleaned"
CUSTOMER_METRICS_DIR = BASE_DIR / "app" / "data" / "customer_day_metrics"
ABUSE_JSON_DIR = BASE_DIR / "app" / "data" / "abuse_detection_json"
MODELS_DIR = BASE_DIR / "Models"
FRAUD_MODELS_DIR = MODELS_DIR / "fraud_ml_models"
SALES_MODELS_DIR = MODELS_DIR / "sales_ml_models"

PRIMARY_DATA_PATH = DATA_DIR / "sql_supermarket.parquet"
CUSTOMER_METRICS_PATH = CUSTOMER_METRICS_DIR / "customer_day_metrics.parquet"
ABUSE_JSON_PATH = ABUSE_JSON_DIR / "abuse_analysis_results_2.json"
X_FEATURES_SALES = DATA_DIR / "X_features.parquet"
X_FEATURES_FRAUD = DATA_DIR / "X_features_fraud.parquet"
SCALER_PATH = SALES_MODELS_DIR / "scaler" / "scaler.joblib"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def read_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_model_name_from_path(path: Path) -> str:
    return path.stem.replace("_model", "").replace(".joblib", "")


def build_model_input_from_row(
    feature_row: pd.Series,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = feature_df.columns.tolist()
    input_df = pd.DataFrame([feature_row]).reindex(columns=feature_columns)
    return input_df


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not df.empty and col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_primary_data() -> pd.DataFrame:
    df = read_parquet(PRIMARY_DATA_PATH)
    if not df.empty:
        df = ensure_datetime(df, "order_date")
        df = ensure_datetime(df, "ship_date")
    return df


@st.cache_data(show_spinner=False)
def load_feature_sales_data() -> pd.DataFrame:
    return read_features(X_FEATURES_SALES)


@st.cache_data(show_spinner=False)
def load_feature_fraud_data() -> pd.DataFrame:
    return read_features(X_FEATURES_FRAUD)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


scaler = load_scaler()


@st.cache_data(show_spinner=False)
def load_customer_day_metrics() -> pd.DataFrame:
    df = read_parquet(CUSTOMER_METRICS_PATH)
    if not df.empty:
        df = ensure_datetime(df, "order_date")
    return df


@st.cache_data(show_spinner=False)
def load_abuse_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if not df.empty and "current_order_date" in df.columns:
        df["current_order_date"] = pd.to_datetime(
            df["current_order_date"], errors="coerce"
        )
    return df


@st.cache_resource(show_spinner=False)
def load_sales_models() -> Dict[str, Any]:
    model_files = [
        SALES_MODELS_DIR / "CatboostRegressor_model.joblib",
        SALES_MODELS_DIR / "DecisionTreeRegressor_model.joblib",
        SALES_MODELS_DIR / "RandomForestRegressor_model.joblib",
        SALES_MODELS_DIR / "XGBRegressor_model.joblib",
    ]
    models = {}
    for path in model_files:
        model = load_model(path)
        if model is not None:
            models[get_model_name_from_path(path)] = model
    return models


@st.cache_resource(show_spinner=False)
def load_fraud_models() -> Dict[str, Any]:
    model_files = [
        FRAUD_MODELS_DIR / "GradientBoostingClassifier.pkl",
        FRAUD_MODELS_DIR / "RandomForestClassifier.pkl",
        FRAUD_MODELS_DIR / "XGBClassifier.pkl",
    ]
    models = {}
    for path in model_files:
        model = load_model(path)
        if model is not None:
            models[get_model_name_from_path(path)] = model
    return models


@st.cache_resource(show_spinner=False)
def load_llm():
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key or HuggingFaceEndpoint is None:
        return None
    try:
        return HuggingFaceEndpoint(
            repo_id=os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
            task="text-generation",
            max_new_tokens=512,
            temperature=0.2,
            huggingfacehub_api_token=api_key,
            timeout=30,
        )
    except Exception:
        return None


def weighted_abuse_score(metrics_row: pd.Series) -> float:
    daily_orders = float(metrics_row.get("daily_orders", 0))
    weekly_orders = float(metrics_row.get("weekly_orders", 0))
    monthly_orders = float(metrics_row.get("monthly_orders", 0))
    spike_ratio = float(metrics_row.get("spike_ratio", 0))
    velocity_flag = float(metrics_row.get("velocity_alert_flag", 0))

    daily_component = min(daily_orders / 10.0, 1.0)
    weekly_component = min(weekly_orders / 30.0, 1.0)
    monthly_component = min(monthly_orders / 90.0, 1.0)
    spike_component = min(spike_ratio / 3.0, 1.0)

    score = (
        0.30 * daily_component
        + 0.20 * weekly_component
        + 0.20 * monthly_component
        + 0.20 * spike_component
        + 0.10 * velocity_flag
    )
    return float(np.clip(score, 0, 1))


def risk_level_from_score(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
    if score >= 0.6:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


def build_velocity_context(metrics_row: Optional[pd.Series]) -> str:
    if metrics_row is None or metrics_row.empty:
        return "No customer-day metric match was found."
    return (
        f"Daily Velocity Alert Metrics:\n"
        f"- Customer Target: {metrics_row.get('customer_name', 'Unknown')}\n"
        f"- Evaluation Date: {pd.to_datetime(metrics_row.get('order_date')).date() if pd.notna(metrics_row.get('order_date')) else 'Unknown'}\n"
        f"- Daily Orders: {int(metrics_row.get('daily_orders', 0))}\n"
        f"- Weekly Orders: {float(metrics_row.get('weekly_orders', 0)):.2f}\n"
        f"- Monthly Orders: {float(metrics_row.get('monthly_orders', 0)):.2f}\n"
        f"- Spike Ratio: {float(metrics_row.get('spike_ratio', 0)):.2f}\n"
        f"- Velocity Alert Flag: {int(metrics_row.get('velocity_alert_flag', 0))}"
    )


def get_customer_day_match(
    customer_day_metrics: pd.DataFrame,
    customer_name: str,
    order_date: pd.Timestamp,
) -> Optional[pd.Series]:
    if customer_day_metrics.empty:
        return None

    target_date = pd.to_datetime(order_date).normalize()
    subset = customer_day_metrics.copy()
    subset["order_date"] = pd.to_datetime(subset["order_date"]).dt.normalize()

    match = subset[
        (subset["customer_name"] == customer_name)
        & (subset["order_date"] == target_date)
    ]

    if not match.empty:
        return match.iloc[0]

    customer_subset = subset[subset["customer_name"] == customer_name].copy()
    if customer_subset.empty:
        return None

    customer_subset["date_distance"] = (
        customer_subset["order_date"] - target_date
    ).abs()
    return customer_subset.sort_values("date_distance").iloc[0]


def llm_abuse_summary(llm, context_text: str, score: float, risk_level: str) -> str:
    if llm is None:
        return (
            f"LLM unavailable. Risk={risk_level}, score={score:.2f}. "
            f"Use the metrics context below for manual review.\n\n{context_text}"
        )

    prompt = f"""
You are a security analyst for a supermarket transaction platform.

Analyze the following daily velocity alert metrics and produce:
1. Abuse pattern summary
2. Likely cause
3. Immediate action
4. Long-term recommendation

Context:
{context_text}

Risk level: {risk_level}
Abuse score: {score:.2f}

Return concise, production-ready analysis.
"""
    try:
        return llm.invoke(prompt)
    except Exception as e:
        return f"LLM error: {e}\n\n{context_text}"


def model_prediction_summary(model, input_df):
    try:
        if model is None:
            return None, None, "Model file not found or loaded."

        # Handle fitment checking dynamically based on the engine type
        from sklearn.utils.validation import check_is_fitted

        try:
            if hasattr(model, "is_fitted"):
                if not model.is_fitted():
                    return None, None, "Model is not fitted."
            else:
                check_is_fitted(model)

        except NotFittedError:
            return (
                None,
                None,
                "NotFittedError: This model pickle exists but was exported without being fitted.",
            )

        prediction = model.predict(input_df)
        confidence = None

        # Robust identification
        model_class_name = type(model).__name__.lower()
        estimator_type = getattr(model, "_estimator_type", None)

        # Use soft duck-typing verification to support XGBoost and CatBoost objects cleanly
        is_classifier = (
            "classifier" in model_class_name
            or estimator_type == "classifier"
            or hasattr(model, "predict_proba")
        )

        is_regressor = (
            "regressor" in model_class_name
            or estimator_type == "regressor"
            or hasattr(model, "predict")
        )

        if is_classifier:
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(input_df)))
            status = "ok"

        elif is_regressor:
            status = "ok"
        else:
            status = "Unknown"

        return prediction[0], confidence, status
    except Exception as e:
        return None, None, f"Runtime Inference Error: {str(e)}"


# -----------------------------------------------------------------------------
# Load Data and Models
# -----------------------------------------------------------------------------
primary_df = load_primary_data()
features_sales_df = load_feature_sales_data()
features_fraud_df = load_feature_fraud_data()
customer_day_metrics = load_customer_day_metrics()
abuse_df = load_abuse_json(ABUSE_JSON_PATH)
sales_models = load_sales_models()
fraud_models = load_fraud_models()
llm = load_llm()

# -----------------------------------------------------------------------------
# Session features Validation
# -----------------------------------------------------------------------------
if len(primary_df) != len(features_sales_df):
    st.error(
        f"""
        Primary dataset has {len(primary_df)} rows.
        Sales feature dataset has {len(features_sales_df)} rows.
        Both datasets must have identical row ordering.
        """
    )
    st.stop()

if len(primary_df) != len(features_fraud_df):
    st.error(
        f"""
        Primary dataset has {len(primary_df)} rows.
        Fraud feature dataset has {len(features_fraud_df)} rows.
        Both datasets must have identical row ordering.
        """
    )
    st.stop()

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = None
if "selected_date" not in st.session_state:
    st.session_state.selected_date = None
if "sales_model_name" not in st.session_state:
    st.session_state.sales_model_name = next(iter(sales_models.keys()), None)
if "fraud_model_name" not in st.session_state:
    st.session_state.fraud_model_name = next(iter(fraud_models.keys()), None)

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")

    st.subheader("Data Sources")
    st.write(f"Primary dataset: {PRIMARY_DATA_PATH.name}")
    st.write(f"Velocity metrics: {CUSTOMER_METRICS_PATH.name}")
    st.write(f"Abuse Data JSON: {ABUSE_JSON_PATH.name}")

    entity = st.radio(
        "Open module",
        ["Sales Prediction", "Fraud Prediction", "Abuse Detection", "Security Agent"],
        index=0,
    )

    if not primary_df.empty:
        customer_options = sorted(
            primary_df["customer_name"].dropna().astype(str).unique().tolist()
        )
        st.session_state.selected_customer = st.selectbox(
            "Customer",
            options=customer_options,
            index=(
                0
                if st.session_state.selected_customer not in customer_options
                else customer_options.index(st.session_state.selected_customer)
            ),
        )

        date_subset = primary_df[
            primary_df["customer_name"] == st.session_state.selected_customer
        ]
        date_options = sorted(
            date_subset["order_date"].dropna().dt.date.unique().tolist()
        )
        if date_options:
            default_date = (
                st.session_state.selected_date
                if st.session_state.selected_date in date_options
                else date_options[0]
            )
            st.session_state.selected_date = st.selectbox(
                "Order Date",
                options=date_options,
                index=date_options.index(default_date),
            )
        else:
            st.session_state.selected_date = None

    st.subheader("Available Models")
    if sales_models:
        st.session_state.sales_model_name = st.selectbox(
            "Sales model",
            options=list(sales_models.keys()),
            index=(
                list(sales_models.keys()).index(st.session_state.sales_model_name)
                if st.session_state.sales_model_name in sales_models
                else 0
            ),
        )
    else:
        st.info("No sales models found in Models/")

    if fraud_models:
        st.session_state.fraud_model_name = st.selectbox(
            "Fraud model",
            options=list(fraud_models.keys()),
            index=(
                list(fraud_models.keys()).index(st.session_state.fraud_model_name)
                if st.session_state.fraud_model_name in fraud_models
                else 0
            ),
        )
    else:
        st.info("No fraud models found in Models/fraud_ml_models/")

    st.subheader("LLM")
    if llm is not None:
        st.success("HuggingFace LLM ready")
    else:
        st.warning("LLM unavailable, fallback analysis only")

# -----------------------------------------------------------------------------
# Page Header
# -----------------------------------------------------------------------------
st.title("🛒 Supermarket AI Agent system helper")
st.caption(
    "Integrated sales prediction, fraud prediction, abuse detection, and security analysis"
)

if primary_df.empty:
    st.error("Primary dataset could not be loaded.")
    st.stop()

if customer_day_metrics.empty:
    st.warning(
        "customer_day_metrics.parquet could not be loaded. Abuse detection will use fallback signals."
    )

# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Sales Prediction",
        "🕵️ Fraud Prediction",
        "🚨 Abuse Detection",
        "🛡️ Security Agent",
        "📂 Saved JSON Results",
    ]
)

selected_row = primary_df[
    (primary_df["customer_name"] == st.session_state.selected_customer)
    & (primary_df["order_date"].dt.date == st.session_state.selected_date)
]

if selected_row.empty and st.session_state.selected_customer is not None:
    selected_row = (
        primary_df[primary_df["customer_name"] == st.session_state.selected_customer]
        .sort_values("order_date")
        .tail(1)
    )

if not selected_row.empty:
    selected_index = selected_row.index[0]
else:
    selected_index = primary_df.index[0]

selected_record = primary_df.loc[selected_index]
feature_sales_record = features_sales_df.loc[selected_index]
feature_fraud_record = features_fraud_df.loc[selected_index]

# -----------------------------------------------------------------------------
# Sales Prediction Tab
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("📈 Sales Optimization Pipeline")

    # Toggle between validating past records and forecasting upcoming demand
    sales_mode = st.radio(
        "Select Pipeline Mode",
        [
            "Phase 1: Historical Alignment (Minimize Error)",
            "Phase 2: Future Demand Forecasting (What-If Simulation)",
        ],
        horizontal=True,
    )

    if st.session_state.sales_model_name is None:
        st.info("No sales model available.")
    else:
        sales_model = sales_models.get(st.session_state.sales_model_name)
        sales_input_df = build_model_input_from_row(
            feature_sales_record, features_sales_df
        )

        if sales_mode == "Phase 1: Historical Alignment (Minimize Error)":
            st.markdown("### 🎯 Model Error Verification")
            st.caption(
                "Validating the production model by comparing predictions directly against true historical targets."
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Selected record**")
                st.dataframe(
                    pd.DataFrame([selected_record]).T.rename(columns={0: "value"}),
                    use_container_width=True,
                )

            with col2:
                st.markdown("**Model input preview**")
                st.dataframe(sales_input_df, use_container_width=True)

            if st.button("Execute Verification", type="primary"):
                # Apply standard scaling transformation to model features
                sales_input_scaled = pd.DataFrame(
                    scaler.transform(sales_input_df), columns=sales_input_df.columns
                )
                prediction, _, status = model_prediction_summary(
                    sales_model, sales_input_scaled
                )

                if status == "ok":
                    # Inverse log transform to revert target back to original currency scale
                    prediction_actual = np.expm1(prediction)
                    actual_sales = float(selected_record.get("sales", 0))
                    absolute_variance = abs(actual_sales - prediction_actual)
                    error_percentage = (
                        (absolute_variance / actual_sales * 100)
                        if actual_sales > 0
                        else 0
                    )

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Actual Store Sales", f"Rp.{actual_sales:,.2f}")
                    with c2:
                        st.metric(
                            "Model Predicted Sales", f"Rp.{prediction_actual:,.2f}"
                        )
                    with c3:
                        st.metric(
                            "Row-by-Row Variance Error",
                            f"{error_percentage:.2f}%",
                            delta=f"Rp.{absolute_variance:,.2f}",
                            delta_color="inverse",
                        )

                else:
                    st.error(f"Sales prediction failed: {status}")

        else:
            st.markdown("### 📊 Future Demand Forecasting")
            st.caption(
                "Simulate upcoming marketplace condition to forecast volume demand and revenue output."
            )

            st.markdown("#### Adjust Future Parameter Multipliers")
            sim_col1, sim_col2, sim_col3 = st.columns(3)

            # Allow interactive overrides of feature variables to simulate future events
            with sim_col1:
                quantity_mult = st.slider(
                    "Future Transaction Volume Scale", 0.5, 5.0, 1.0, step=0.1
                )
            with sim_col2:
                discount_mult = st.slider(
                    "Future Promotion/Discount Aggression", 0.0, 2.0, 1.0, step=0.1
                )
            with sim_col3:
                operational_shift = st.number_input(
                    "Bulk Processing Correction Adjustment", value=0.0
                )

            # Construct dynamic simulated features
            simulated_input_df = sales_input_df.copy()

            # Safely adjust feature targets
            for col in simulated_input_df.columns:
                if "quantity" in col.lower():
                    simulated_input_df[col] = simulated_input_df[col] * quantity_mult
                if "discount" in col.lower():
                    simulated_input_df[col] = simulated_input_df[col] * discount_mult

            st.markdown("**Engineered Simulation Vector Preview**")
            st.dataframe(simulated_input_df, use_container_width=True)

            if st.button("Generate Future Projections", type="primary"):
                # Scale the simulated feature block
                simulated_input_scaled = pd.DataFrame(
                    scaler.transform(simulated_input_df),
                    columns=simulated_input_df.columns,
                )
                sim_prediction, _, sim_status = model_prediction_summary(
                    sales_model, simulated_input_scaled
                )

                if sim_status == "ok":
                    forecasted_revenue = np.expm1(sim_prediction) + operational_shift

                    st.success(
                        "Future inference engine generated demand projections successfully!"
                    )
                    st.metric(
                        label="Forecasted Projected Revenue Output",
                        value=f"Rp.{forecasted_revenue:,.2f}",
                        delta=f"Based on adjusted {st.session_state.sales_model_name} architecture",
                    )
                else:
                    st.error(f"Simulation Prediction Blocked: {sim_status}")

# -----------------------------------------------------------------------------
# Fraud Prediction Tab
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Fraud Prediction")
    st.write("Use the classifier family to score transaction fraud risk.")

    if st.session_state.fraud_model_name is None:
        st.info("No fraud model available.")
    else:
        fraud_model = fraud_models.get(st.session_state.fraud_model_name)
        fraud_input_df = build_model_input_from_row(
            feature_fraud_record, features_fraud_df
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Selected transaction**")
            st.dataframe(
                pd.DataFrame([selected_record]).T.rename(columns={0: "value"}),
                use_container_width=True,
            )

        with col2:
            st.markdown("**Fraud model input preview**")
            st.dataframe(fraud_input_df, use_container_width=True)

        with st.expander("Debug Inference"):
            st.write("Input shape:", fraud_input_df.shape)
            st.write("Columns:")
            st.write(fraud_input_df.columns.tolist())
            st.write("Values:")

        if st.button("Run Fraud Prediction", type="primary"):
            pred, confidence, status = model_prediction_summary(
                fraud_model, fraud_input_df
            )

            if status == "ok":
                fraud_probability = None
                if hasattr(fraud_model, "predict_proba"):
                    try:
                        proba = fraud_model.predict_proba(fraud_input_df)
                        if proba is not None and len(proba) and proba.shape[1] > 1:
                            fraud_probability = float(proba[0][1])
                    except Exception:
                        fraud_probability = None

                c1, c2, c3 = st.columns(3)
                with c1:
                    # Clear representation of target classes
                    st.metric(
                        "Predicted Result",
                        (
                            "Fraud Alert 🚨"
                            if int(pred) == 1
                            else "Legitimate Transaction ✅"
                        ),
                    )
                with c2:
                    st.metric(
                        "Confidence Score",
                        f"{confidence:.2%}" if confidence is not None else "N/A",
                    )
                with c3:
                    st.metric(
                        "Fraud Probability",
                        (
                            f"{fraud_probability:.2%}"
                            if fraud_probability is not None
                            else "N/A"
                        ),
                    )
            else:
                # BUG FIX: If an unfitted model is detected, the error is handled gracefully inside a container instead of a full app exception
                st.error(f"Prediction Interrupted: {status}")

# -----------------------------------------------------------------------------
# Abuse Detection Tab
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Abuse Detection")
    st.write(
        "Uses customer-day metrics from the parquet file and rule-based velocity scoring."
    )

    metric_match = None
    if (
        st.session_state.selected_customer
        and st.session_state.selected_date is not None
    ):
        metric_match = get_customer_day_match(
            customer_day_metrics,
            st.session_state.selected_customer,
            pd.to_datetime(st.session_state.selected_date),
        )

    if metric_match is not None:
        context_text = build_velocity_context(metric_match)
        abuse_score = weighted_abuse_score(metric_match)
        risk_level = risk_level_from_score(abuse_score)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Daily Orders", int(metric_match.get("daily_orders", 0)))
        with c2:
            st.metric(
                "Weekly Orders", f"{float(metric_match.get('weekly_orders', 0)):.2f}"
            )
        with c3:
            st.metric(
                "Monthly Orders", f"{float(metric_match.get('monthly_orders', 0)):.2f}"
            )
        with c4:
            st.metric("Spike Ratio", f"{float(metric_match.get('spike_ratio', 0)):.2f}")

        st.markdown("**Velocity context**")
        st.code(context_text, language="text")

        st.markdown("**Rule-based risk score**")
        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.metric("Abuse score", f"{abuse_score:.2f}")
        with score_col2:
            st.metric("Risk level", risk_level)

        if int(metric_match.get("velocity_alert_flag", 0)) == 1:
            st.error("Velocity alert flag is active")
        else:
            st.success("Velocity alert flag is not active")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Daily", "Weekly", "Monthly"],
                y=[
                    float(metric_match.get("daily_orders", 0)),
                    float(metric_match.get("weekly_orders", 0)),
                    float(metric_match.get("monthly_orders", 0)),
                ],
                marker_color=["#EF476F", "#118AB2", "#06D6A0"],
            )
        )
        fig.update_layout(
            title="Customer Velocity Metrics",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No exact customer-day metric match found.")

# -----------------------------------------------------------------------------
# Security Agent Tab
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🛡️ Enterprise Threat Intelligence Agent")
    st.caption("Interactive chat framework connected to fine-tuned LLM execution pipelines.")

    if metric_match is None:
        st.warning("No metrics row available for LLM analysis.")
    else:
        context_text = build_velocity_context(metric_match)
        abuse_score = weighted_abuse_score(metric_match)
        risk_level = risk_level_from_score(abuse_score)

        # Create a unique session tracking key for customer context
        session_id = f"session_{st.session_state.selected_customer}_{st.session_state.selected_date}"

        # Initialize local frontend chat tracking container
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = {}

        if session_id not in st.session_state.chat_messages:
            st.session_state.chat_messages[session_id] = [
                {"role": "assistant", 
                 "content": f"System metrics loaded for *{st.session_state.selected_customer}*. How can I help you investigate this profile?"}
            ]

        # Layout Split: Left for technical logs, Right for conversational LLM output
        ui_col_left, ui_col_right = st.columns([1, 1.2])
        
        with ui_col_left:
            st.markdown("#### Real-Time Metric Context Feed")
            st.code(context_text, language="text")
            st.metric("Computed Anomaly Risk Level", risk_level, delta=f"Score: {abuse_score:.2f}")

        with ui_col_right:
            st.markdown("#### Interactive Investigation Window")
            
            # Container box to cleanly anchor chat streaming layout heights
            chat_container = st.container(height=450)
            
            with chat_container:
                # Render existing conversational logs stored in the session state
                for msg in st.session_state.chat_messages[session_id]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            # Listen for new prompt input vectors from the security engineer
            if user_prompt := st.chat_input("Ask for pattern evaluations or immediate lockdown paths..."):
                
                # Instantly append user message to UI state layout
                st.session_state.chat_messages[session_id].append({"role": "user", "content": user_prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.write(user_prompt)

                # Send request payload out to FastAPI network endpoint 
                import requests
                
                backend_url = "http://localhost:8000/api/v1/agent/chat" # Modify route port if inside Docker network mesh
                payload = {
                    "session_id": session_id,
                    "query": user_prompt,
                    "context": context_text
                }

            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing log traces..."):
                        try:
                            api_response = requests.post(backend_url, json=payload, timeout=45)
                            
                            if api_response.status_code == 200:
                                agent_reply = api_response.json().get("response")
                                st.write(agent_reply)
                                st.session_state.chat_messages[session_id].append({"role": "assistant", "content": agent_reply}) # Save assistant reply to memory state array
                            else:
                                
                                st.error(f"Agent API error: {api_response.status_code} - {api_response.text}")
                        except Exception as connection_err:
                                st.error(f"Failed to communicate with FastAPI Agent Engine: {connection_err}")

# -----------------------------------------------------------------------------
# Saved JSON Results Tab
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Saved Abuse Detection JSON")
    st.write("Review the saved production output from the abuse detection notebook.")

    if abuse_df.empty:
        st.warning("No saved abuse JSON data found.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Records", len(abuse_df))
        with c2:
            if "risk_level" in abuse_df.columns:
                high_critical = int(
                    abuse_df["risk_level"].isin(["HIGH", "CRITICAL"]).sum()
                )
            else:
                high_critical = 0
            st.metric("High / Critical", high_critical)
        with c3:
            if "risk_level" in abuse_df.columns:
                low_risk = int((abuse_df["risk_level"] == "LOW").sum())
            else:
                low_risk = 0
            st.metric("Low Risk", low_risk)

        display_cols = [
            col
            for col in [
                "customer_name",
                "current_order_date",
                "abuse_score",
                "risk_level",
                "patterns",
            ]
            if col in abuse_df.columns
        ]

        if display_cols:
            st.dataframe(abuse_df[display_cols], use_container_width=True)

        if "customer_name" in abuse_df.columns:
            selected_customer = st.selectbox(
                "Select customer",
                abuse_df["customer_name"].dropna().astype(str).unique().tolist(),
                key="json_customer_select",
            )

            customer_rows = abuse_df[
                abuse_df["customer_name"].astype(str) == str(selected_customer)
            ]
            if not customer_rows.empty:
                customer_row = customer_rows.iloc[0]

                st.markdown("### Historical Context")
                st.code(
                    str(customer_row.get("historical_context", "")), language="text"
                )

                st.markdown("### LLM Analysis")
                llm_text = customer_row.get("llm_analysis", "")
                if hasattr(llm_text, "content"):
                    llm_text = llm_text.content

                for line in str(llm_text).splitlines():
                    line = line.strip()
                    if line:
                        st.write(line)

                st.markdown("### Raw JSON Row")
                st.json(customer_row.to_dict())

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:gray; padding: 1rem;">
        <small>Supermarket Intelligence Platform | Updated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small>
    </div>
    """,
    unsafe_allow_html=True,
)
