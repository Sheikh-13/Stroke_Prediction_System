import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🧠",
    layout="centered"
)

# ── Load & preprocess (cached) ────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"]
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df = df.drop_duplicates()
    encoders = {}
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for col in cat_cols:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    clip_bounds = {}
    for col in ["bmi", "avg_glucose_level"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        clip_bounds[col] = (lower, upper)
        df[col] = df[col].clip(lower, upper)
    return df, encoders, clip_bounds

@st.cache_resource
def train_model(df):
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

df, encoders, clip_bounds = load_and_preprocess()
model = train_model(df)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 Stroke Prediction System")
st.markdown("Enter patient details below to predict stroke risk.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    gender        = st.selectbox("Gender", ["Male", "Female"])
    age           = st.slider("Age", 1, 100, 45)
    hypertension  = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    ever_married   = st.selectbox("Ever Married", ["Yes", "No"])
    work_type      = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

with col3:
    avg_glucose = st.slider("Avg Glucose Level (mg/dL)", 50.0, 275.0, 100.0, step=0.5)
    bmi         = st.slider("BMI", 10.0, 60.0, 28.0, step=0.1)
    smoking     = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# ── Encode inputs ─────────────────────────────────────────────────────────────
hypert_enc    = 1 if hypertension == "Yes" else 0
heart_enc     = 1 if heart_disease == "Yes" else 0

gender_enc    = encoders["gender"].transform([gender])[0]
married_enc   = encoders["ever_married"].transform([ever_married])[0]
work_enc      = encoders["work_type"].transform([work_type])[0]
residence_enc = encoders["Residence_type"].transform([residence_type])[0]
smoking_enc   = encoders["smoking_status"].transform([smoking])[0]

bmi_clipped     = np.clip(bmi, *clip_bounds["bmi"])
glucose_clipped = np.clip(avg_glucose, *clip_bounds["avg_glucose_level"])


input_data = pd.DataFrame([{
    "gender":            gender_enc,
    "age":               age,
    "hypertension":      hypert_enc,
    "heart_disease":     heart_enc,
    "ever_married":      married_enc,
    "work_type":         work_enc,
    "Residence_type":    residence_enc,
    "avg_glucose_level": glucose_clipped,
    "bmi":               bmi_clipped,
    "smoking_status":    smoking_enc
}])
# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Predict Stroke Risk", use_container_width=True):
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    THRESHOLD = 0.10
    prediction = 1 if probability[1] >= THRESHOLD else 0
    if prediction == 1:
        st.error("⚠️ **High Stroke Risk Detected**")
        st.markdown("The model predicts this patient is **at risk of stroke**.")
    else:
        st.success("✅ **Low Stroke Risk**")
        st.markdown("The model predicts this patient is **not at risk of stroke**.")

    col1, col2 = st.columns(2)
    col1.metric("Probability — No Stroke", f"{probability[0]:.2%}")
    col2.metric("Probability — Stroke",    f"{probability[1]:.2%}")
    st.markdown("### Prediction Confidence")
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh(["No Stroke", "Stroke"], probability, color=["#378ADD", "#E24B4A"])
    ax.axvline(x=THRESHOLD, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold ({THRESHOLD:.0%})")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    for i, v in enumerate(probability):
        ax.text(v + 0.01, i, f"{v:.2%}", va="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

   