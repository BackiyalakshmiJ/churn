# app_advanced_target_safe.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "catboost_best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODERS_PATH = BASE_DIR / "label_encoders.pkl"
DATA_PATH = BASE_DIR / "TelcoChurn_Preprocessed.csv"  # raw, human-readable dataset

APP_TITLE = "📞 Telco Customer Churn Prediction"
PRIMARY_COLOR = "#2E86C1"
TARGET_COL = "Churn"

# ---------------- HELPERS ----------------
def _drop_target(cols):
    return [c for c in cols if c != TARGET_COL]

def _safe_numeric_cols(numeric_cols):
    # Just in case the saved scaler list accidentally contained the target
    return [c for c in numeric_cols if c != TARGET_COL]

def _safe_label_encoders(encoders: dict):
    # Remove target encoder if it exists (not needed for inference)
    enc = dict(encoders)
    enc.pop(TARGET_COL, None)
    return enc

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler_dict = pickle.load(f)
        scaler = scaler_dict["scaler"]
        numeric_cols = _safe_numeric_cols(scaler_dict["numeric_cols"])

    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = _safe_label_encoders(pickle.load(f))

    raw_data = pd.read_csv(DATA_PATH)

    # Build feature view from raw_data while excluding the target
    feature_cols = _drop_target(list(raw_data.columns))
    feat_df = raw_data[feature_cols].copy()

    return model, scaler, numeric_cols, label_encoders, raw_data, feat_df, feature_cols


def preprocess_input(input_df, scaler, numeric_cols, label_encoders):
    """Encode categoricals and scale numerics (target excluded)."""
    df = input_df.copy()

    # Encode categoricals using saved label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            val = df[col].iloc[0]
            # Handle unseen categories gracefully
            if val not in le.classes_:
                le.classes_ = list(le.classes_) + [val]
            df[col] = le.transform(df[col])

    # Scale numeric columns
    num_cols = [c for c in numeric_cols if c in df.columns]
    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])

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
