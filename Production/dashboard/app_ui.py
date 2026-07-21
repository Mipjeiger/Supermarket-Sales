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
    return path.stem.replace("_model", "").replace(".joblib", "").replace(".pkl", "")


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
    if not SCALER_PATH.exists():
        return None
    try:
        return joblib.load(SCALER_PATH)
    except Exception:
        return None


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
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if not df.empty and "current_order_date" in df.columns:
            df["current_order_date"] = pd.to_datetime(
                df["current_order_date"], errors="coerce"
            )
        return df
    except Exception:
        return pd.DataFrame()


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


def llm_abuse_summary(llm_model, context_text: str, score: float, risk_level: str) -> str:
    if llm_model is None:
        return (
            f"⚠️ LLM unavailable. Risk={risk_level}, score={score:.2f}.\n\n"
            f"**Automated Summary:** High activity detected on customer profile. Recommend manual audit.\n\n"
            f"**Context:**\n{context_text}"
        )

    prompt = f"""
You are an enterprise security agent analyzing supermarket transaction velocity.

Context Metrics:
{context_text}

Risk Level: {risk_level}
Abuse Score: {score:.2f}

Provide a concise, 4-bullet investigation summary:
1. Pattern Summary
2. Likely Root Cause
3. Immediate Action Required
4. Long-term Prevention
"""
    try:
        return llm_model.invoke(prompt)
    except Exception as e:
        return f"LLM Inference Error: {e}\n\nFallback Context:\n{context_text}"


def model_prediction_summary(model, input_df):
    try:
        if model is None:
            return None, None, "Model file not found or loaded."

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
                "NotFittedError: Model pickle exists but was exported without fitting.",
            )

        prediction = model.predict(input_df)
        confidence = None

        model_class_name = type(model).__name__.lower()
        estimator_type = getattr(model, "_estimator_type", None)

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

        if is_classifier and hasattr(model, "predict_proba"):
            try:
                confidence = float(np.max(model.predict_proba(input_df)))
            except Exception:
                confidence = None

        if is_regressor and hasattr(model, "predict"):
            try:
                confidence = float(np.std(model.predict(input_df)))
            except Exception:
                confidence = None

        return prediction[0], confidence, "ok"
    except Exception as e:
        return None, None, f"Runtime Error: {str(e)}"


# -----------------------------------------------------------------------------
# Data & Model Initialization
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
# Feature Set Alignment Validation
# -----------------------------------------------------------------------------
if not primary_df.empty and not features_sales_df.empty:
    if len(primary_df) != len(features_sales_df):
        st.error(
            f"Row count mismatch! Primary: {len(primary_df)}, Sales Features: {len(features_sales_df)}."
        )
        st.stop()

if not primary_df.empty and not features_fraud_df.empty:
    if len(primary_df) != len(features_fraud_df):
        st.error(
            f"Row count mismatch! Primary: {len(primary_df)}, Fraud Features: {len(features_fraud_df)}."
        )
        st.stop()

# -----------------------------------------------------------------------------
# Session State Setup
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
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")

    st.subheader("Data Sources")
    st.caption(f"📍 Primary: `{PRIMARY_DATA_PATH.name}`")
    st.caption(f"📍 Metrics: `{CUSTOMER_METRICS_PATH.name}`")
    st.caption(f"📍 Abuse JSON: `{ABUSE_JSON_PATH.name}`")

    if not primary_df.empty:
        customer_options = sorted(
            primary_df["customer_name"].dropna().astype(str).unique().tolist()
        )
        st.session_state.selected_customer = st.selectbox(
            "Target Customer",
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

    st.subheader("Model Selections")
    if sales_models:
        st.session_state.sales_model_name = st.selectbox(
            "Sales Model",
            options=list(sales_models.keys()),
        )
    else:
        st.info("No sales models found in `Models/sales_ml_models`")

    if fraud_models:
        st.session_state.fraud_model_name = st.selectbox(
            "Fraud Model",
            options=list(fraud_models.keys()),
        )
    else:
        st.info("No fraud models found in `Models/fraud_ml_models`")

    st.subheader("LLM Engine")
    if llm is not None:
        st.success("HuggingFace LLM Ready 🚀")
    else:
        st.warning("LLM Offline (Fallback Mode)")

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("🛒 Supermarket AI Agent System")
st.caption(
    "Unified interface for Sales Forecasting, Fraud Detection, Abuse Analytics & Threat Intelligence"
)

if primary_df.empty:
    st.error("Primary dataset could not be loaded. Please verify file paths.")
    st.stop()

# -----------------------------------------------------------------------------
# Selected Record Context
# -----------------------------------------------------------------------------
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

selected_index = selected_row.index[0] if not selected_row.empty else primary_df.index[0]

selected_record = primary_df.loc[selected_index]
feature_sales_record = features_sales_df.loc[selected_index] if not features_sales_df.empty else pd.Series()
feature_fraud_record = features_fraud_df.loc[selected_index] if not features_fraud_df.empty else pd.Series()

# -----------------------------------------------------------------------------
# Tabs Interface
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

# -----------------------------------------------------------------------------
# TAB 1: Sales Prediction
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("📈 Sales Optimization Pipeline")

    sales_mode = st.radio(
        "Select Pipeline Mode",
        [
            "Phase 1: Historical Alignment (Minimize Error)",
            "Phase 2: Future Demand Forecasting (What-If Simulation)",
        ],
        horizontal=True,
    )

    if not st.session_state.sales_model_name or st.session_state.sales_model_name not in sales_models:
        st.info("No valid sales model selected.")
    else:
        sales_model = sales_models.get(st.session_state.sales_model_name)
        sales_input_df = build_model_input_from_row(feature_sales_record, features_sales_df)

        if sales_mode == "Phase 1: Historical Alignment (Minimize Error)":
            st.markdown("### 🎯 Model Error Verification")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Selected Record**")
                st.dataframe(pd.DataFrame([selected_record]).T.rename(columns={0: "value"}), use_container_width=True)
            with col2:
                st.markdown("**Model Feature Vector**")
                st.dataframe(sales_input_df, use_container_width=True)

            if st.button("Execute Verification", type="primary"):
                try:
                    sales_input_scaled = (
                        pd.DataFrame(scaler.transform(sales_input_df), columns=sales_input_df.columns)
                        if scaler is not None else sales_input_df
                    )
                    prediction, _, status = model_prediction_summary(sales_model, sales_input_scaled)

                    if status == "ok":
                        prediction_actual = np.expm1(prediction)
                        actual_sales = float(selected_record.get("sales", 0))
                        absolute_variance = abs(actual_sales - prediction_actual)
                        error_percentage = (absolute_variance / actual_sales * 100) if actual_sales > 0 else 0

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Actual Store Sales", f"Rp {actual_sales:,.2f}")
                        c2.metric("Model Predicted Sales", f"Rp {prediction_actual:,.2f}")
                        c3.metric(
                            "Row Variance Error",
                            f"{error_percentage:.2f}%",
                            delta=f"Rp {absolute_variance:,.2f}",
                            delta_color="inverse",
                        )
                    else:
                        st.error(f"Prediction Error: {status}")
                except Exception as e:
                    st.error(f"Inference pipeline execution error: {str(e)}")

        else:
            st.markdown("### 📊 Future Demand Forecasting")
            st.markdown("#### Adjust Future Parameter Multipliers")
            sim_col1, sim_col2, sim_col3 = st.columns(3)

            with sim_col1:
                quantity_mult = st.slider("Transaction Volume Scale", 0.5, 5.0, 1.0, step=0.1)
            with sim_col2:
                discount_mult = st.slider("Promotion/Discount Aggression", 0.0, 2.0, 1.0, step=0.1)
            with sim_col3:
                operational_shift = st.number_input("Bulk Processing Correction Adjustment", value=0.0)

            simulated_input_df = sales_input_df.copy()
            for col in simulated_input_df.columns:
                if "quantity" in col.lower():
                    simulated_input_df[col] *= quantity_mult
                if "discount" in col.lower():
                    simulated_input_df[col] *= discount_mult

            st.dataframe(simulated_input_df, use_container_width=True)

            if st.button("Generate Future Projections", type="primary"):
                try:
                    simulated_input_scaled = (
                        pd.DataFrame(scaler.transform(simulated_input_df), columns=simulated_input_df.columns)
                        if scaler is not None else simulated_input_df
                    )
                    sim_prediction, _, sim_status = model_prediction_summary(sales_model, simulated_input_scaled)

                    if sim_status == "ok":
                        forecasted_revenue = np.expm1(sim_prediction) + operational_shift
                        st.success("Future inference engine generated projections successfully! 📈")
                        st.metric(
                            label="Forecasted Projected Revenue Output",
                            value=f"Rp {forecasted_revenue:,.2f}",
                            delta=f"Model: {st.session_state.sales_model_name}",
                        )
                    else:
                        st.error(f"Simulation Blocked: {sim_status}")
                except Exception as e:
                    st.error(f"Simulation Execution Failed: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 2: Fraud Prediction
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🕵️ Fraud Prediction")
    st.caption("Score transaction fraud risk using tuned classification ensembles.")

    if not st.session_state.fraud_model_name or st.session_state.fraud_model_name not in fraud_models:
        st.info("No valid fraud classifier selected.")
    else:
        fraud_model = fraud_models.get(st.session_state.fraud_model_name)
        fraud_input_df = build_model_input_from_row(feature_fraud_record, features_fraud_df)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Selected Transaction Record**")
            st.dataframe(pd.DataFrame([selected_record]).T.rename(columns={0: "value"}), use_container_width=True)
        with col2:
            st.markdown("**Classifier Feature Input**")
            st.dataframe(fraud_input_df, use_container_width=True)

        if st.button("Run Fraud Inference", type="primary"):
            pred, confidence, status = model_prediction_summary(fraud_model, fraud_input_df)

            if status == "ok":
                fraud_prob = None
                if hasattr(fraud_model, "predict_proba"):
                    try:
                        proba = fraud_model.predict_proba(fraud_input_df)
                        if proba is not None and len(proba) > 0 and proba.shape[1] > 1:
                            fraud_prob = float(proba[0][1])
                    except Exception:
                        fraud_prob = None

                c1, c2, c3 = st.columns(3)
                c1.metric("Classification Result", "🚨 Fraud Alert" if int(pred) == 1 else "✅ Legitimate")
                c2.metric("Confidence Score", f"{confidence:.2%}" if confidence is not None else "N/A")
                c3.metric("Fraud Probability", f"{fraud_prob:.2%}" if fraud_prob is not None else "N/A")
            else:
                st.error(f"Prediction Failed: {status}")

# -----------------------------------------------------------------------------
# TAB 3: Abuse Detection
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🚨 Customer Abuse & Velocity Detection")

    metric_match = None
    if st.session_state.selected_customer and st.session_state.selected_date is not None:
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
        c1.metric("Daily Orders", int(metric_match.get("daily_orders", 0)))
        c2.metric("Weekly Orders", f"{float(metric_match.get('weekly_orders', 0)):.2f}")
        c3.metric("Monthly Orders", f"{float(metric_match.get('monthly_orders', 0)):.2f}")
        c4.metric("Spike Ratio", f"{float(metric_match.get('spike_ratio', 0)):.2f}")

        st.markdown("**Rule-Based Velocity Score**")
        sc1, sc2 = st.columns(2)
        sc1.metric("Computed Abuse Score", f"{abuse_score:.2f}")
        sc2.metric("Risk Assessment Level", risk_level)

        if int(metric_match.get("velocity_alert_flag", 0)) == 1:
            st.error("⚠️ Velocity Alert Flag: Active")
        else:
            st.success("✅ Velocity Alert Flag: Clear")

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
        fig.update_layout(title="Velocity Distribution", template="plotly_white", height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No matching customer-day metric record found.")

# -----------------------------------------------------------------------------
# TAB 4: Security Agent (Interactive Copilot)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🛡️ Enterprise Threat Intelligence Agent")
    st.caption("Interactive copilot for automated investigation of suspicious activity patterns.")

    metric_match = get_customer_day_match(
        customer_day_metrics,
        st.session_state.selected_customer,
        pd.to_datetime(st.session_state.selected_date),
    ) if st.session_state.selected_customer and st.session_state.selected_date else None

    if metric_match is None:
        st.warning("Select a valid customer and date from the sidebar to inspect velocity telemetry.")
    else:
        context_text = build_velocity_context(metric_match)
        abuse_score = weighted_abuse_score(metric_match)
        risk_level = risk_level_from_score(abuse_score)

        session_id = f"session_{st.session_state.selected_customer}_{st.session_state.selected_date}"

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = {}

        if session_id not in st.session_state.chat_messages:
            initial_summary = llm_abuse_summary(llm, context_text, abuse_score, risk_level)
            st.session_state.chat_messages[session_id] = [
                {
                    "role": "assistant",
                    "content": f"🛡️ **Security Briefing initialized for {st.session_state.selected_customer}:**\n\n{initial_summary}",
                }
            ]

        ui_col_left, ui_col_right = st.columns([1, 1.3])

        with ui_col_left:
            st.markdown("#### Real-Time Metric Telemetry")
            st.code(context_text, language="text")
            st.metric("Computed Anomaly Risk Level", risk_level, delta=f"Score: {abuse_score:.2f}")

        with ui_col_right:
            st.markdown("#### Interactive Investigation Window")

            chat_container = st.container(height=420)

            with chat_container:
                for msg in st.session_state.chat_messages[session_id]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if user_prompt := st.chat_input("Ask for pattern evaluations or lockdown paths..."):
                # Append user prompt to state
                st.session_state.chat_messages[session_id].append({"role": "user", "content": user_prompt})

                # Generate AI response
                with st.spinner("Analyzing risk context..."):
                    if llm is not None:
                        full_prompt = (
                            f"Context:\n{context_text}\nRisk Level: {risk_level}\n\n"
                            f"User Query: {user_prompt}\n"
                            f"Provide a concise, professional security analysis response."
                        )
                        try:
                            assistant_response = llm.invoke(full_prompt)
                        except Exception as err:
                            assistant_response = f"LLM Error: {err}"
                    else:
                        assistant_response = (
                            f"⚠️ LLM Offline Mode.\n\n"
                            f"Regarding your query ('{user_prompt}'): The target profile has a risk level of **{risk_level}** "
                            f"with an abuse score of **{abuse_score:.2f}**. Manual audit recommended."
                        )

                # Append assistant response and refresh UI
                st.session_state.chat_messages[session_id].append({"role": "assistant", "content": assistant_response})
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 5: Saved JSON Results
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("📂 Saved Abuse Detection JSON Audits")
    st.caption("Inspect serialized offline analysis results from background workers.")

    if abuse_df.empty:
        st.info("No saved JSON results found in `app/data/abuse_detection_json/`.")
    else:
        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            if "risk_level" in abuse_df.columns:
                selected_risk = st.multiselect(
                    "Filter by Risk Level",
                    options=abuse_df["risk_level"].unique().tolist(),
                    default=abuse_df["risk_level"].unique().tolist(),
                )
                filtered_abuse_df = abuse_df[abuse_df["risk_level"].isin(selected_risk)]
            else:
                filtered_abuse_df = abuse_df.copy()

        with filter_col2:
            st.metric("Total Cached Audits", len(filtered_abuse_df))

        st.dataframe(filtered_abuse_df, use_container_width=True)

        # Detailed Record Inspection
        st.markdown("#### Record Drilldown")
        if not filtered_abuse_df.empty:
            record_idx = st.number_input(
                "Select Record Index to Expand JSON",
                min_value=0,
                max_value=len(filtered_abuse_df) - 1,
                value=0,
                step=1,
            )
            st.json(filtered_abuse_df.iloc[record_idx].to_dict())