# Objective Prepare the data for modeling — clean, encode, scale, and split

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Check for missing values
    if df.isnull().values.any():
        print("Warning: Missing values found in the dataset. Consider handling them before proceeding.")

    # Convert TotalCharges to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill missing values in TotalCharges with the mean
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    return df


# Clean Dataset
def clean_data(df):
    # Drop unnecessary columns
    df.drop(columns=['customerID'], inplace=True)

    # Convert categorical variables to 'category' dtype
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Fix string formatting (like trimming spaces)
    for col in categorical_cols:
        df[col] = df[col].str.strip()

    # Handle duplicate rows
    df.drop_duplicates(inplace=True)


    return df


# encode categorical features (like "Yes/No", "Month-to-month") into numbers

def encode_features(df):
    # Step 1: Label encode binary categorical columns
    binary_cols = ['Churn', 'Partner', 'Dependents']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Step 2: One-hot encode multi-category columns
    multi_category_cols = ['InternetService', 'Contract', 'PaymentMethod', 'SeniorCitizen']
    df = pd.get_dummies(df, columns=multi_category_cols, drop_first=True)

    return df

# Scale numerical features
def scale_features(df):
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Initialize the scaler
    scaler = StandardScaler()

    # Scale the numerical features
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

# Split the dataset into training and testing sets

# Separate features and target variable
def split_data(df, target_col='Churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


# Test the preprocessing pipeline
if __name__ == "__main__":
    df = load_data("d:/Portofolio Projects/Customer_Churn/data/customer_churn.csv")
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train target shape:", y_train.shape)
    print("Test target shape:", y_test.shape)
    print("Preprocessing completed successfully.")

    # Save the processed data to CSV files
    X_train.to_csv("d:/Portofolio Projects/Customer_Churn/data/X_train.csv", index=False)
    X_test.to_csv("d:/Portofolio Projects/Customer_Churn/data/X_test.csv", index=False)
    y_train.to_csv("d:/Portofolio Projects/Customer_Churn/data/y_train.csv", index=False)
    y_test.to_csv("d:/Portofolio Projects/Customer_Churn/data/y_test.csv", index=False)
    print("Processed data saved successfully.")
    print("Data preprocessing pipeline completed.")
# End of preprocessing script

# Create Markdown documentation
"""
# Data Preprocessing Pipeline

This script prepares the customer churn dataset for modeling by performing the following steps:

1. **Load Data**: Reads the dataset from a CSV file.
2. **Clean Data**: 
    - Drops unnecessary columns.
    - Converts categorical variables to 'category' dtype.
    - Fixes string formatting issues.
    - Handles duplicate rows.
3. **Encode Features**:
    - Label encodes binary categorical columns.
    - One-hot encodes multi-category columns.
4. **Scale Features**: Standardizes numerical features using `StandardScaler`.
5. **Split Data**: Divides the dataset into training and testing sets, ensuring stratification based on the target variable.
6. **Save Processed Data**: Saves the processed training and testing datasets to CSV files.

## Usage

Run the script directly to execute the preprocessing pipeline. The processed data will be saved in the specified directory.

## Requirements
- pandas
- scikit-learn
## Note
Ensure that the input CSV file path is correct and that the necessary libraries are installed in your Python environment.
"""

