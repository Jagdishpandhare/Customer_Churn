import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# === Load Data ===
X_test = pd.read_csv("d:/Portofolio Projects/Customer_Churn/data/X_test.csv")
y_test = pd.read_csv("d:/Portofolio Projects/Customer_Churn/data/y_test.csv").values.ravel()

# === Load Best Model (replace with best model name if known) ===
model_name = "random_forest"
model = joblib.load(f"d:/Portofolio Projects/Customer_Churn/models/{model_name}.joblib")

# === Predict ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# === Classification Report ===
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("d:/Portofolio Projects/Customer_Churn/output/classification_report.csv")
print("\n✅ Classification Report saved.")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {model_name}")
plt.savefig("d:/Portofolio Projects/Customer_Churn/output/confusion_matrix.png")
plt.close()
print("✅ Confusion Matrix saved.")

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("d:/Portofolio Projects/Customer_Churn/output/roc_curve.png")
plt.close()
print("✅ ROC Curve saved.")

# === Final Print ===
print("\n🎯 Evaluation complete.")
print(f"✔ Model: {model_name}")
print(f"✔ AUC Score: {auc_score:.4f}")
