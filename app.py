import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==============================
# Load model, scaler, and dataset
# ==============================
model = jb.load("heart_model.pkl")
scaler = jb.load("scaler.pkl")
df = pd.read_csv("heart.csv")

# Prepare features/labels
X = df.drop("target", axis=1)
y = df["target"]

# Predict for full dataset to get metrics
y_pred = model.predict(scaler.transform(X))

# ==============================
# Calculate metrics
# ==============================
accuracy = accuracy_score(y, y_pred)
class_report = classification_report(
    y, y_pred, target_names=["No Disease", "Heart Disease"], output_dict=False
)

# ==============================
# Function to check abnormal parameters
# ==============================
def check_abnormal_parameters(data):
    abnormal = {}
    if data['cp'] in [2, 3]: abnormal['cp'] = data['cp']
    if data['trestbps'] > 130: abnormal['trestbps'] = data['trestbps']
    if data['chol'] > 240: abnormal['chol'] = data['chol']
    if data['fbs'] == 1: abnormal['fbs'] = 'Fasting > 120 mg/dl'
    if data['restecg'] != 0: abnormal['restecg'] = data['restecg']
    if data['thalach'] < 120: abnormal['thalach'] = data['thalach']
    if data['exang'] == 1: abnormal['exang'] = 'Yes'
    if data['oldpeak'] >= 2.0: abnormal['oldpeak'] = data['oldpeak']
    if data['slope'] != 1: abnormal['slope'] = data['slope']
    if data['ca'] > 0: abnormal['ca'] = data['ca']
    if data['thal'] in [2, 3]: abnormal['thal'] = data['thal']
    return abnormal

# ==============================
# Title
# ==============================
st.title("💓 Heart Disease Prediction App")

# -----------------------
# 📊 Model Performance
# -----------------------
st.subheader("📈 Model Performance Metrics")
st.write(f"**Model Accuracy:** {accuracy:.4f}")
st.text("Classification Report:")
st.text(class_report)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"], ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# Correlation Heatmap
st.subheader("🔍 Correlation Heatmap")
fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
st.pyplot(fig_corr)

# Histograms
st.subheader("📊 Feature Distributions (Histograms)")
df.hist(figsize=(15, 10))
plt.tight_layout()
st.pyplot(plt.gcf())

# -----------------------
# 📝 Prediction Form
# -----------------------
st.subheader("🔮 Heart Disease Prediction Tool")

age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150)
exang = st.radio("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.selectbox("Thalassemia", [1, 2, 3])

if st.button("🔍 Predict"):
    values = np.array([
        age, 1 if sex == "Male" else 0, cp, trestbps, chol,
        fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]).reshape(1, -1)

    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]

    patient_data = {
        'age': age, 'sex': 1 if sex == "Male" else 0, 'cp': cp,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    if prediction == 1:
        st.error("💔 Heart Disease Detected")
        abnormal = check_abnormal_parameters(patient_data)
        if abnormal:
            st.warning("⚠️ Abnormal Parameters Detected:")
            for key, val in abnormal.items():
                st.write(f"- **{key}**: {val}")
    else:
        st.success("❤️ No Heart Disease Detected")
