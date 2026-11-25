import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Change this to the model you selected after Experiment 6.
MODEL_FILE = "logistic_regression.pkl"

st.set_page_config(
    page_title="Heart Disease Decision Support",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.caption("Machine-learning decision-support prototype | College project demo")

st.warning(
    "Educational demonstration only — not a medical diagnosis. "
    "Do not use this app to make real healthcare decisions."
)

@st.cache_resource
def load_model():
    model_path = Path(MODEL_FILE)
    if not model_path.exists():
        return None
    return joblib.load(model_path)

model = load_model()

if model is None:
    st.error(
        f"Model file '{MODEL_FILE}' was not found. Run Experiment 5, "
        "then copy the selected .pkl pipeline into the same folder as app.py."
    )
    st.stop()

st.markdown(
    "Enter the values below. For categorical fields, the options are shown "
    "with readable labels while the app sends the dataset's numeric codes to the model."
)

with st.form("patient_form"):
    st.subheader("Patient information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=29, max_value=77, value=55, help="Observed dataset range: 29–77 years.")
        sex_label = st.selectbox("Sex", ["Female", "Male"])
        cp_label = st.selectbox(
            "Chest pain type",
            [
                "Typical angina",
                "Atypical angina",
                "Non-anginal pain",
                "Asymptomatic"
            ]
        )
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=94, max_value=200, value=130, help="Observed dataset range: 94–200 mm Hg.")
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=126, max_value=564, value=240, help="Observed dataset range: 126–564 mg/dl.")
        fbs_label = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["No", "Yes"])
        restecg_label = st.selectbox(
            "Resting ECG result",
            ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
        )

    with col2:
        thalach = st.number_input("Maximum heart rate achieved", min_value=71, max_value=202, value=150, help="Observed dataset range: 71–202.")
        exang_label = st.selectbox("Exercise-induced angina?", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.2, value=1.0, step=0.1, help="Observed dataset range: 0.0–6.2.")
        slope_label = st.selectbox(
            "Slope of peak exercise ST segment",
            ["Upsloping", "Flat", "Downsloping"]
        )
        ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3])
        thal_label = st.selectbox(
            "Thal result",
            ["Normal", "Fixed defect", "Reversible defect"]
        )

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    # Convert readable labels back to the numeric codes used by this dataset.
    sex = {"Female": 0, "Male": 1}[sex_label]
    cp = {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3
    }[cp_label]
    fbs = {"No": 0, "Yes": 1}[fbs_label]
    restecg = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2
    }[restecg_label]
    exang = {"No": 0, "Yes": 1}[exang_label]
    slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope_label]
    thal = {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}[thal_label]

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = int(model.predict(patient)[0])

    st.divider()
    st.subheader("Prediction result")

    if prediction == 1:
        st.error("Model classification: Class 1 — heart disease present in the dataset label.")
    else:
        st.success("Model classification: Class 0 — no heart disease in the dataset label.")

    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(patient)[0][1])
        st.metric("Model score for Class 1", f"{score:.1%}")
        st.caption(
            "This percentage is the model's estimated probability for class 1, "
            "not a clinically validated personal risk estimate."
        )

    with st.expander("View values sent to the model"):
        st.dataframe(patient, use_container_width=True, hide_index=True)

    st.info(
        "No prediction log is saved. This keeps the demo simple and avoids "
        "recording patient-like input values unnecessarily."
    )

st.divider()
st.caption(
    "Dataset: Cleveland heart-disease dataset (303 records). "
    "Predictions depend on the trained model and dataset coding. "
    "For academic demonstration only."
)
