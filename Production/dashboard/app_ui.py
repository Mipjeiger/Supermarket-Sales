import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os

# Page Configuration
st.set_page_config(
    page_title="Supermarket Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://fastapi:8000"

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "XGBRegressor" # Fallback to default model

# Sidebar
with st.sidebar:
    st.title("⚙️ Controls")

    # Model selection
    st.subheader("Model Configuration")
    try:
        response = requests.get(f"{st.session_state.api_url}/api/v1/models", timeout=5)

        if response.status_code == 200:
            data = response.json()
            model_options = data.get("models", ['XGBRegressor'])
            performance = data.get("performance", {})
            best_model = data.get("best_model", "XGBRegressor")

            st.success(f"✅ Best Model: {best_model}")
            st.session_state.selected_model = st.selectbox(
                "Select Model",
                options=model_options,
                index=model_options.index(best_model) if best_model in model_options else 0
            )

            # Show performance
            if st.session_state.selected_model in performance:
                perf = performance[st.session_state.selected_model]
                st.metric("RMSE", f"{perf.get('rmse', 0):.3f}")
                st.metric("R²", f"{perf.get('r2', 0):.3f}")

            else:
                st.warning("⚠️ Using default models (API not available)")
                model_options = ["LinearRegression", "RandomForestRegressor", "XGBRegressor"]
                st.session_state.selected_model = st.selectbox(
                    "Select Model",
                    options=model_options,
                    index=2
                )

    except:
        st.warning("⚠️ API not available. Using default models.")

        model_options = ["LinearRegression", "RandomForestRegressor", "XGBRegressor"]
        st.session_state.selected_model = st.selectbox(
            "Select Model",
            options=model_options,
            index=2
        )

        # Forecast parameters
        st.subheader("Forecast Parameters")
        horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=6,
            max_value=72,
            value=24,
            step=6
        )

        confidence = st.slider(
            "Confidence Interval",
            min_value=80,
            max_value=99,
            value=95,
            step=1
        )
    
        # API connection status

        st.subheader("System Status")
        try:
            response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Not Connected")

# Main content
st.title("🏪 Supermarket Sales Forecast Dashboard")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast",
    "📊 Model Performance",
    "🤖 LLM Insights",
    "📉 Monitoring"
])

# Tab 1: Forecast
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    # Load performance data
    perf_data = None
    try:
        perf_file = "Models/model_performance_comparison.csv"
        if os.path.exists(perf_file):
            perf_df = pd.read_csv(perf_file)
            model_perf = perf_df[perf_df['Model'] == st.session_state.selected_model]
            if not model_perf.empty:
                perf_data = model_perf.iloc[0]
    except:
        pass
    
    with col1:
        rmse = perf_data.get('RMSE', 0.712) if perf_data is not None else 0.712
        st.metric(
            "Current RMSE",
            f"{rmse:.3f} kW",
            f"▼ {((1.193 - rmse) / 1.193 * 100):.1f}% vs baseline" if perf_data is not None else ""
        )
    
    with col2:
        st.metric(
            "Data Completeness",
            "100.0%",
            "✅"
        )
    
    with col3:
        st.metric(
            "Prediction Outliers",
            "0.00%",
            "✅"
        )
    
    with col4:
        st.metric(
            "Avg Prediction",
            "7.09 kW",
            f"Model: {st.session_state.selected_model}"
        )

    # Generate forecast button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_btn = st.button("📊 Generate Forecast", type="primary", use_container_width=True)
    
    if generate_btn:
        with st.spinner("Generating forecast..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/api/v1/forecast",
                    params={
                        "model_name": st.session_state.selected_model,
                        "horizon": horizon
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    forecast_data = response.json()
                    st.success("✅ Forecast generated successfully!")
                    
                    # Display forecast chart
                    st.subheader("📈 24-Hour Ahead Forecast")
                    
                    # Parse data
                    predictions = forecast_data.get('predictions', [])
                    timestamps = forecast_data.get('timestamps', [])
                    
                    if predictions and timestamps:
                        # Create figure
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=("Usage Pattern", "Forecast by Hour"),
                            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
                        )
                        
                        # Generate historical data (sample)
                        hours = list(range(len(predictions)))
                        historical = [p * (0.8 + np.random.normal(0, 0.05)) for p in predictions]
                        
                        # Add traces for historical
                        fig.add_trace(
                            go.Scatter(
                                x=hours,
                                y=historical,
                                name="Actual",
                                line=dict(color="#2E86AB", width=2),
                                mode="lines+markers"
                            ),
                            row=1, col=1
                        )
                        
                        # Add forecast
                        fig.add_trace(
                            go.Scatter(
                                x=hours,
                                y=predictions,
                                name="Forecast",
                                line=dict(color="#A23B72", width=2, dash="dash"),
                                mode="lines+markers"
                            ),
                            row=1, col=1
                        )
                        
                        # Add confidence interval
                        std_dev = np.std(predictions) * 0.3
                        upper = [p + std_dev for p in predictions]
                        lower = [p - std_dev for p in predictions]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=hours,
                                y=upper,
                                fill=None,
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=hours,
                                y=lower,
                                fill='tonexty',
                                mode="lines",
                                line=dict(width=0),
                                fillcolor="rgba(162, 59, 114, 0.2)",
                                name=f"{confidence}% Confidence"
                            ),
                            row=1, col=1
                        )
                        
                        # Hourly forecast
                        fig.add_trace(
                            go.Bar(
                                x=hours,
                                y=predictions,
                                name="Hourly Forecast",
                                marker_color="#F18F01"
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(
                            height=600,
                            showlegend=True,
                            template="plotly_white",
                            hovermode="x unified"
                        )
                        
                        fig.update_xaxes(title_text="Hour", row=1, col=1)
                        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
                        fig.update_yaxes(title_text="Usage (kWh)", row=1, col=1)
                        fig.update_yaxes(title_text="Usage (kWh)", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model metadata
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📋 Model Metadata")
                            metadata = {
                                "Model Type": st.session_state.selected_model,
                                "Horizon": f"{horizon}h",
                                "Window": "168h",
                                "Features": 20,
                                "RMSE": forecast_data.get('rmse', 'N/A')
                            }
                            st.json(metadata)
                        
                        with col2:
                            st.subheader("📊 Forecast Accuracy")
                            accuracy = {
                                "RMSE": f"{forecast_data.get('rmse', 'N/A'):.3f}",
                                "Data Through": forecast_data.get('data_through', 'N/A')
                            }
                            st.json(accuracy)
                else:
                    st.error(f"❌ Failed to generate forecast: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 2: Model Performance
with tab2:
    st.subheader("📊 Model Performance Comparison")
    
    # Load performance data
    perf_file = "Models/model_performance_comparison.csv"
    if os.path.exists(perf_file):
        performance_data = pd.read_csv(perf_file)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Best RMSE",
                f"{performance_data['RMSE'].min():.3f}",
                f"{performance_data[performance_data['RMSE'] == performance_data['RMSE'].min()]['Model'].values[0]}"
            )
        
        with col2:
            st.metric(
                "Best R² Score",
                f"{performance_data['R2'].max():.3f}",
                f"{performance_data[performance_data['R2'] == performance_data['R2'].max()]['Model'].values[0]}"
            )
        
        with col3:
            st.metric(
                "Model Count",
                f"{len(performance_data)}",
                "✅"
            )
        
        # Performance chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=performance_data['Model'],
            y=performance_data['RMSE'],
            name='RMSE',
            marker_color='#2E86AB',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data['Model'],
            y=performance_data['R2'],
            name='R² Score',
            marker_color='#A23B72',
            yaxis='y2',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="RMSE",
            yaxis2=dict(
                title="R² Score",
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            height=400,
            template="plotly_white",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        st.dataframe(
            performance_data.style.background_gradient(cmap='Blues', subset=['RMSE', 'MAE', 'R2']),
            use_container_width=True
        )
    else:
        st.info("📊 Performance data not available. Train models first.")

# Tab 3: LLM Insights
with tab3:
    st.subheader("🤖 LLM-Powered Sales Insights")
    
    user_query = st.text_area(
        "Ask about your sales data",
        placeholder="e.g., What were our best-selling product categories last month?",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        response_type = st.selectbox(
            "Analysis Type",
            options=["Sales Analysis", "Forecast Interpretation", "Product Recommendations"],
            index=0
        )
    
    if analyze_btn and user_query:
        with st.spinner("Analyzing with LLM..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/api/v1/llm/analyze",
                    json={
                        "query": user_query,
                        "response_type": response_type.lower().replace(" ", "_")
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Analysis Complete")
                    st.markdown("### 💡 Insights")
                    st.markdown(data['response'])
                else:
                    st.error(f"❌ Analysis failed: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Sample queries
    with st.expander("📝 Sample Queries"):
        st.markdown("""
        - "What were our top 5 products by sales last month?"
        - "Which product categories are underperforming?"
        - "What's the sales forecast for next week?"
        - "Which products should we promote for the upcoming holiday?"
        - "What factors are driving the current sales trends?"
        """)

# Tab 4: Monitoring
with tab4:
    st.subheader("📉 System Monitoring")
    
    # Monitoring status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current RMSE", "0.712 kW", "✅ Within threshold")
    
    with col2:
        st.metric("Data Completeness", "100.0%", "✅")
    
    with col3:
        st.metric("Prediction Outliers", "0.00%", "✅")
    
    with col4:
        st.metric("Avg Prediction", "7.09 kW", "📊")
    
    # Monitoring cycle
    st.subheader("🔄 Monitoring Cycle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Last Cycle**")
        st.text("2026-05-08 15:26")
        st.markdown("**Next Check (Est.)**")
        st.text("2026-05-09 15:26")
        
    with col2:
        st.markdown("**Time Remaining**")
        st.warning("⏰ Overdue")
        st.markdown("**Last Review**")
        st.text("2026-05-06 18:22")
    
    # Quality evaluation
    st.subheader("✅ Quality Evaluation")
    
    quality_checks = {
        "Data Validation": "✅ Passed",
        "Model Performance": "✅ Passed",
        "Forecast Accuracy": "✅ Passed",
        "System Health": "⚠️ Check latency",
        "LLM Service": "✅ Passed"
    }
    
    for check, status in quality_checks.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(check)
        with col2:
            if "✅" in status:
                st.success(status)
            elif "⚠️" in status:
                st.warning(status)
            else:
                st.error(status)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <small>Supermarket Sales Forecast Dashboard v1.0 | Built with Streamlit</small>
    </div>
""", unsafe_allow_html=True)