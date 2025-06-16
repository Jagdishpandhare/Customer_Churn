import pandas as pd
import joblib

# Use raw strings to avoid syntax warning
model = joblib.load(r"D:\Portofolio Projects\Customer_Churn\models\logistic_regression.joblib")
scaler = joblib.load(r"D:\Portofolio Projects\Customer_Churn\models\scaler.joblib")

# Sample input — use the same columns you trained the model on
sample_input = pd.DataFrame([{
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 5,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.35,
    'TotalCharges': 350.5
}])

# Step 1: Encode same as training
label_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

from sklearn.preprocessing import LabelEncoder

for col in label_cols:
    le = LabelEncoder()
    # Fit label encoder on the known training classes
    sample_input[col] = le.fit_transform(sample_input[col])

# Step 2: Scale numeric features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
sample_input[numeric_cols] = scaler.transform(sample_input[numeric_cols])

# Step 3: Predict
prediction = model.predict(sample_input)
print("Prediction:", prediction)
