# Customer Churn Prediction Project

This project predicts customer churn using different machine learning algorithms. The dataset comes from a telecom company and includes features like customer demographics, services signed up for, account information, and whether the customer has churned.

## Project Structure

```
Customer_Churn/
|
├── data/
│   ├── customer_churn.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── xgboost.joblib
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│
├── eda.ipynb
├── preprocessing.ipynb
├── modeling.ipynb
├── README.md
```

## Project Phases

### 1. Exploratory Data Analysis (EDA)

* Performed in `eda.ipynb`
* Analyzed distributions, correlations, and churn rates.

### 2. Data Preprocessing

* Cleaned data: handled missing values, converted types
* Encoded categorical variables
* Scaled numerical features
* Used SMOTE to fix class imbalance
* Implemented in `preprocess.py` and `preprocessing.ipynb`

### 3. Model Training

* Trained multiple models:

  * Logistic Regression
  * Random Forest
  * Decision Tree
  * K-Nearest Neighbors
  * XGBoost
* Used cross-validation and saved models with Joblib
* Script: `train_model.py`

### 4. Evaluation

* Evaluated models on:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
* Script: `evaluate.py`

### 5. Model Comparison

* Compared performance both visually and statistically
* Chose the best-performing model
* Notebook: `modeling.ipynb`

## Final Model Performance

**Best Model:** Random Forest / XGBoost (replace with final winner)

| Metric    | Value                                |
| --------- | ------------------------------------ |
| Accuracy  | ~77.5%                               |
| Precision | ~84.6% (Class 0), ~57.5% (Class 1)  |
| Recall    | ~84.8% (Class 0), ~57.2% (Class 1)  |
| F1-Score  | ~84.7% (Class 0), ~57.4% (Class 1)  |

## Dataset

* Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* 7,000+ customer records
* Features include:

  * Customer ID
  * Gender, SeniorCitizen, Partner, Dependents
  * Tenure, MonthlyCharges, TotalCharges
  * InternetService, Contract, PaymentMethod
  * Churn (target variable)

## Next Steps

1. Build a Streamlit web app for churn prediction.
2. Push the entire project to GitHub.
3. Deploy the model using API or cloud.

## How to Run

```bash
# Install dependencies
pip install pandas scikit-learn imbalanced-learn xgboost joblib

# Run training pipeline
python src/train_model.py

# Run evaluation
python src/evaluate.py
```

## Acknowledgments

* Inspired by real-world business churn problems.
* Dataset: IBM Telco Customer Churn (via Kaggle)
* Tools: Python, pandas, scikit-learn, XGBoost, Joblib