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


# ---------------- APP ----------------
def main():
    st.set_page_config(page_title="Churn Prediction", layout="wide", page_icon="📊")

    # ---- HEADER ----
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; border-radius:12px;
                    background:linear-gradient(90deg, {PRIMARY_COLOR}, #154360); color:white;">
            <h1>{APP_TITLE}</h1>
            <p style="font-size:18px; margin-top:-10px;">
                Advanced AI-powered dashboard to estimate telecom customer churn
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ---- Load artifacts ----
    try:
        (model, scaler, numeric_cols, label_encoders,
         raw_data, feat_df, feature_cols) = load_artifacts()
    except Exception as e:
        st.error(f"❌ Failed to load files: {e}")
        st.stop()

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("ℹ️ About the App")
        st.success("Predicts whether a telecom customer will churn using ML (CatBoost).")
        st.warning("⚠️ Demo app — not for business decisions.")
        st.markdown("### 📊 Dataset Info")
        # Show info against features only (not leaking the target into inputs)
        st.info(f"Rows: {raw_data.shape[0]} | Columns: {len(feature_cols)} features")
        st.markdown("### 🎯 Target")
        if TARGET_COL in raw_data.columns:
            st.text(f"Target column: {TARGET_COL}")
        else:
            st.text("Target column not found in dataset view.")
        st.markdown("### ⚙️ Model Details")
        st.text("• Algorithm: CatBoost Classifier\n• Encoders: LabelEncoder\n• Scaling: StandardScaler")

    # ---- TABS ----
    tabs = st.tabs(["📋 Input Details", "🔮 Prediction Result", "📈 Model Insights"])

    # ---- TAB 1: INPUT ----
    with tabs[0]:
        st.subheader("Enter Customer Details")

        # Build lists from features only
        cat_cols = [c for c in feature_cols if feat_df[c].dtype == "object"]
        num_cols = [c for c in feature_cols if feat_df[c].dtype != "object"]

        input_data = {}
        with st.form("customer_form"):
            st.markdown("#### 🧑 Demographics / Contract")
            c1, c2, c3 = st.columns(3)
            for i, col in enumerate(cat_cols):
                with [c1, c2, c3][i % 3]:
                    input_data[col] = st.selectbox(col, options=sorted(feat_df[col].dropna().unique()))

            st.markdown("#### 💲 Numeric Features")
            c4, c5, c6 = st.columns(3)
            for i, col in enumerate(num_cols):
                with [c4, c5, c6][i % 3]:
                    input_data[col] = st.number_input(
                        col,
                        value=float(feat_df[col].dropna().median()),
                        step=1.0,
                    )

            submitted = st.form_submit_button("🚀 Run Prediction")

    # ---- TAB 2: PREDICTION ----
    with tabs[1]:
        if "input_data" in locals() and submitted:
            try:
                input_df = pd.DataFrame([input_data])

                # Absolute safety: ensure target is not present
                if TARGET_COL in input_df.columns:
                    input_df = input_df.drop(columns=[TARGET_COL])

                processed = preprocess_input(input_df, scaler, numeric_cols, label_encoders)

                pred = model.predict(processed)[0]
                prob = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else None

                st.subheader("Prediction Outcome")

                if pred == 1:
                    st.error("🚨 This customer is **likely to churn**.")
                else:
                    st.success("💚 This customer is **not likely to churn**.")

                if prob is not None:
                    st.plotly_chart(plot_gauge(prob), use_container_width=True)
                    st.metric("Churn Probability", f"{prob:.2%}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.info("⚡ Fill customer details in the **Input tab** and run prediction.")

    # ---- TAB 3: INSIGHTS ----
    with tabs[2]:
        st.subheader("Model & Data Insights")
        st.write(
            """
            - This model predicts the probability that a customer will churn.
            - Inputs exclude the target (**Churn**).
            - Use the Input tab to run a prediction, then view the probability and classification above.
            """
        )
        # Simple, safe exploration: average target by a chosen categorical feature (if target exists)
        if TARGET_COL in raw_data.columns:
            cat_for_view = st.selectbox(
                "View average churn by feature:",
                options=[c for c in raw_data.columns if raw
