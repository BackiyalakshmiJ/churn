# app_ultra.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "catboost_best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODERS_PATH = BASE_DIR / "label_encoders.pkl"
DATA_PATH = BASE_DIR / "TelcoChurn_Preprocessed.csv"

APP_TITLE = "📞 Telco Customer Churn Prediction"
PRIMARY_COLOR = "#2E86C1"

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler_dict = pickle.load(f)
        scaler = scaler_dict["scaler"]
        numeric_cols = scaler_dict["numeric_cols"]

    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    raw_data = pd.read_csv(DATA_PATH)
    return model, scaler, numeric_cols, label_encoders, raw_data


def preprocess_input(input_df, scaler, numeric_cols, label_encoders):
    df = input_df.copy()
    for col, le in label_encoders.items():
        if col in df.columns:
            val = df[col].iloc[0]
            if val not in le.classes_:
                le.classes_ = list(le.classes_) + [val]
            df[col] = le.transform(df[col])
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


def plot_gauge(prob):
    """Plot gauge chart for churn probability."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PRIMARY_COLOR},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "gold"},
                    {"range": [70, 100], "color": "tomato"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
            title={"text": "Churn Probability (%)"},
        )
    )
    fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
    return fig


def shap_explain(model, processed, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed)
    st.write("### 🔍 Local Explanation (Why this prediction?)")
    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        processed.iloc[0, :],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)

    st.write("### 🌍 Global Feature Importance")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, processed, feature_names=feature_names, show=False)
    st.pyplot(fig2)


def download_link(df, filename="prediction_report.csv"):
    """Generate download link for dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Report</a>'


# ---------------- APP ----------------
def main():
    st.set_page_config(page_title="Churn Prediction", layout="wide", page_icon="📊")

    # ---- Custom CSS ----
    st.markdown(
        """
        <style>
        .stMetric {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            backdrop-filter: blur(10px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- HEADER ----
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; border-radius:12px;
                    background:linear-gradient(90deg, {PRIMARY_COLOR}, #154360); color:white;">
            <h1>{APP_TITLE}</h1>
            <p style="font-size:18px; margin-top:-10px;">
                Ultra-Advanced AI-powered churn prediction with explainability
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ---- Load artifacts ----
    try:
        model, scaler, nume
