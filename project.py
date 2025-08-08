
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('heart.csv')  # Ensure heart.csv is in the same folder as this script

# Features and label
X = df.drop('target', axis=1)
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Accuracy score
print(f"\n[OK] Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# Classification report
print("\n[Cl_Report] Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(" Confusion Matrix")
plt.show()

def check_abnormal_parameters(data):
    abnormal = {}
  #  if data['age'] > 50: abnormal['age'] = data['age']
   # if data['sex'] == 1: abnormal['sex'] = 'Male'
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


# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Heart Disease Dataset')
plt.show()

# Histograms of features
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()





import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
 


# Create interactive widgets
age = widgets.IntSlider(description='Age', min=20, max=80, value=50)
sex = widgets.Dropdown(description='Sex', options=[('Female', 0), ('Male', 1)])
cp = widgets.Dropdown(description='Chest Pain', options=[(f'Type {i}', i) for i in range(4)])
trestbps = widgets.IntSlider(description='RestBP', min=80, max=200, value=120)
chol = widgets.IntSlider(description='Cholesterol', min=100, max=400, value=200)
fbs = widgets.ToggleButtons(description='FBS > 120', options=[('No', 0), ('Yes', 1)])
restecg = widgets.Dropdown(description='RestECG', options=[(f'Type {i}', i) for i in range(3)])
thalach = widgets.IntSlider(description='Thalach', min=60, max=210, value=150)
exang = widgets.ToggleButtons(description='ExAngina', options=[('No', 0), ('Yes', 1)])
oldpeak = widgets.FloatSlider(description='Oldpeak', min=0, max=6, step=0.1, value=1.0)
slope = widgets.Dropdown(description='Slope', options=[(f'Type {i}', i) for i in range(3)])
ca = widgets.IntSlider(description='Vessels (ca)', min=0, max=4, value=0)
thal = widgets.Dropdown(description='Thal', options=[('Normal', 1), ('Fixed Defect', 2), ('Reversible Defect', 3)])
family_history = widgets.ToggleButtons(description='Family History', options=[('No', 0), ('Yes', 1)])
 
btn = widgets.Button(description='Predict Heart Disease', button_style='success')
out = widgets.Output()
 
# Prediction function
def predict_heart_disease(btn):
    with out:
        clear_output()
        input_data = {
            'age': age.value,
            'sex': sex.value,
            'cp': cp.value,
            'trestbps': trestbps.value,
            'chol': chol.value,
            'fbs': fbs.value,
            'restecg': restecg.value,
            'thalach': thalach.value,
            'exang': exang.value,
            'oldpeak': oldpeak.value,
            'slope': slope.value,
            'ca': ca.value,
            'thal': thal.value,
            'family_history': family_history.value # Collect the new input
        }
 
        # Note: 'family_history' is not included in the prediction as the model was not trained on this feature.
        values = np.array(list({k: input_data[k] for k in input_data if k != 'family_history'}.values())).reshape(1, -1)
        values_scaled = scaler.transform(values)
        prediction = model.predict(values_scaled)[0]
 
        print("🔍 Prediction Result:", "💔 Heart Disease Detected" if prediction == 1 else "❤️ No Heart Disease")
 
        if prediction == 1:
            # Note: 'family_history' is not included in the abnormal parameters check as the original check_abnormal_parameters function does not include it.
            abnormal = check_abnormal_parameters({k: input_data[k] for k in input_data if k != 'family_history'})
            if abnormal:
                print("\n⚠️ Abnormal Parameters:")
                for key, val in abnormal.items():
                    print(f"  - {key}: {val}")
            else:
                print("✅ No abnormal parameters detected.")
 
btn.on_click(predict_heart_disease)
 
# Combine form items and display (using ipywidgets VBox initially)
form_items = [
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal, family_history, btn, out # Add the new widget
]
display(widgets.VBox(form_items))

import joblib

# Save trained model & scaler
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
