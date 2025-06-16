# Import libraries for data manipulation, model training, and evaluation
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import load_data, clean_data, encode_features, scale_features, split_data


# Load data
df = load_data("d:/Portofolio Projects/Customer_Churn/data/customer_churn.csv")
df = clean_data(df)
df = encode_features(df)
df = scale_features(df)
X_train, X_test, y_train, y_test = split_data(df)

# Separate features and target variable

X = df.drop(columns=['Churn'])
y = df['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE to handle class imbalance in the training set
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define models
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # needs xgboost installed
}

# Train, evaluate and save each model

for model_name, model in models.items():
    print(f"Training {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Evaluating {model_name}...")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")

    # Save the model
    joblib.dump(model, f"d:/Portofolio Projects/Customer_Churn/models/{model_name}.joblib")
# Save the trained models to disk
print("All models trained and saved successfully.")
