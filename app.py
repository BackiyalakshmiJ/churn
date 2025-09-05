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
DATA_PATH = BASE_DIR / "TelcoChurn_Preprocessed.csv"

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
    for col, le in label_encoders.items():
        if col in df.columns:
            val = df[col].iloc[0]
            if val not in le.classes_:
                le.classes_ = list(le.classes_) + [val]
            df[col] = le.transform(df[col])
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


# ---------------- APP ----------------
def main():
    st.set_page_config(page_title="Churn Prediction", layout="wide", page_icon="📊")

    # ---- HEADER ----
    st.markdown(
        f"""
        <div style="text-align:center; padding:15px; border-radius:10px;
                    background-color:{PRIMARY_COLOR}; color:white;">
            <h1>{APP_TITLE}</h1>
            <p style="font-size:18px;">AI-powered tool to estimate telecom customer churn</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ---- Load artifacts ----
    try:
        model, scaler, numeric_cols, label_encoders, raw_data = load_artifacts()
    except Exception as e:
        st.error(f"❌ Failed to load files: {e}")
        st.stop()

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("ℹ️ About the App")
        st.success("Predicts whether a telecom customer will churn using ML (CatBoost).")
        st.warning("⚠️ Demo app — not for business decisions.")

        st.markdown("### 📊 Dataset Info")
        st.info(f"Rows: {raw_data.shape[0]} | Columns: {raw_data.shape[1]}")
        st.markdown("### ⚙️ Model Details")
        st.text("• Algorithm: CatBoost Classifier\n• Encoders: LabelEncoder\n• Scaling: StandardScaler")

    # ---- INPUT SECTION ----
    st.markdown("### 📋 Enter Customer Details")
    st.markdown("Fill in customer details below for churn prediction:")

    input_data = {}
    with st.form("customer_form"):
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

        submitted = st.form_submit_button("🔮 Predict Churn")

    # ---- PREDICTION ----
    if submitted:
        try:
            input_df = pd.DataFrame([input_data])
            processed = preprocess_input(input_df, scaler, numeric_cols, label_encoders)

            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else None

            # ---- RESULT SECTION ----
            st.markdown("## ✅ Prediction Result")
            col1, col2 = st.columns([2, 1])

            with col1:
                if pred == 1:
                    st.error("🚨 This customer is **likely to churn**.")
                else:
                    st.success("💚 This customer is **not likely to churn**.")

                if prob is not None:
                    st.markdown(
                        f"""
                        <div style="padding:10px; border:2px solid {PRIMARY_COLOR};
                                    border-radius:8px; text-align:center;">
                            <h3>Churn Probability</h3>
                            <h2 style="color:{PRIMARY_COLOR};">{prob:.2%}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col2:
                if prob is not None:
                    st.metric(label="📊 Churn Probability", value=f"{prob:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
