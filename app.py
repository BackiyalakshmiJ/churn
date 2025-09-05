# app.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "catboost_best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODERS_PATH = BASE_DIR / "label_encoders.pkl"
DATA_PATH = BASE_DIR / "TelcoChurn_Preprocessed.csv"   # <-- raw dataset

APP_TITLE = "📞 Telco Customer Churn Prediction"
PRIMARY_COLOR = "#2E86C1"


# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_artifacts():
    """Load model, scaler, label encoders, and raw data."""
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
    """Encode categoricals and scale numerics."""
    df = input_df.copy()

    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in df.columns:
            val = df[col].iloc[0]
            # Handle unseen categories gracefully
            if val not in le.classes_:
                le.classes_ = list(le.classes_) + [val]
            df[col] = le.transform(df[col])

    # Scale numeric columns
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


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
        model, scaler, numeric_cols, label_encoders, raw_data = load_artifacts()
    except Exception as e:
        st.error(f"❌ Failed to load files: {e}")
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
            input_df = pd.DataFrame([input_data])
            st.write("🔍 Input DataFrame", input_df)

            # Preprocess
            processed = preprocess_input(input_df, scaler, numeric_cols, label_encoders)
            st.write("🔍 Processed DataFrame", processed)

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


