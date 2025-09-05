import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "catboost_best_model (1).pkl"   # exact file name
PREPROCESSOR_PATH = BASE_DIR / "preprocessing_tools.pkl"
DATA_PATH = BASE_DIR / "TelcoChurn_Preprocessed.csv"

APP_TITLE = "📞 Telco Customer Churn Prediction"
PRIMARY_COLOR = "#2E86C1"

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_model_and_preprocessor():
    """Load CatBoost model and preprocessing tools."""
    # ---- Load Model ----
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # ---- Load Preprocessor ----
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


@st.cache_data
def load_raw_data():
    """Load dataset before encoding and scaling."""
    return pd.read_csv(DATA_PATH)


def apply_preprocessing(preprocessor, input_df):
    """Apply preprocessing depending on how it was saved."""
    # Case 1: it's a full pipeline
    if hasattr(preprocessor, "transform"):
        return preprocessor.transform(input_df)

    # Case 2: it's a dict of objects
    if isinstance(preprocessor, dict):
        df = input_df.copy()

        # If encoder exists
        if "encoder" in preprocessor and hasattr(preprocessor["encoder"], "transform"):
            df = pd.DataFrame(
                preprocessor["encoder"].transform(df),
                columns=preprocessor["encoder"].get_feature_names_out(),
            )

        # If scaler exists
        if "scaler" in preprocessor and hasattr(preprocessor["scaler"], "transform"):
            numeric_cols = preprocessor.get("numeric_cols", [])
            df[numeric_cols] = preprocessor["scaler"].transform(df[numeric_cols])

        return df

    raise ValueError("❌ Preprocessor format not recognized")


# ---------------- APP ----------------
def main():
    st.set_page_config(page_title="Churn Prediction", layout="wide", page_icon="📊")

    # ---- Header ----
    header_cols = st.columns([1, 3, 1])
    with header_cols[1]:
        st.markdown(
            f"<h1 style='text-align:center;color:{PRIMARY_COLOR};margin:0'>{APP_TITLE}</h1>",
            unsafe_allow_html=True,
        )
    st.markdown("---")

    # ---- Load artifacts ----
    try:
        model, preprocessor = load_model_and_preprocessor()
        raw_data = load_raw_data()
    except Exception as e:
        st.error(f"Failed to load files: {e}")
        st.stop()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app predicts whether a telecom customer is likely to churn.")
        st.warning("⚠️ Demo app — not for business decisions.")

    # ---- Input Section ----
    st.subheader("📋 Enter Customer Details")

    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(raw_data.columns):
        with cols[i % 3]:
            if raw_data[col].dtype == "object":
                val = st.selectbox(col, options=sorted(raw_data[col].dropna().unique()))
            else:
                val = st.number_input(
                    col,
                    value=float(raw_data[col].dropna().median()),
                    step=1.0,
                )
            input_data[col] = val

    # ---- Predict Button ----
    if st.button("🔮 Predict Churn", use_container_width=True):
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply preprocessing (handles both pipeline & dict)
            processed = apply_preprocessing(preprocessor, input_df)

            # Prediction
            pred = model.predict(processed)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(processed)[0][1]

            # ---- Result ----
            st.markdown("### ✅ Prediction Result")
            if pred == 1:
                st.error("🚨 This customer is **likely to churn**.")
            else:
                st.success("💚 This customer is **not likely to churn**.")

            if prob is not None:
                st.info(f"Churn Probability: **{prob:.2%}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()

