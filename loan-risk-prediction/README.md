# Loan Risk Prediction

A machine learning project that predicts whether a loan application is likely to be approved or rejected based on applicant information.

## Project Overview

This project uses the Loan Prediction dataset containing 614 loan applications.

The project demonstrates a complete machine learning workflow:

- Data loading and exploration
- Missing value handling
- Feature and target separation
- Categorical data encoding
- Train/test splitting
- Logistic Regression model training
- Model evaluation
- Prediction probability
- Model serialization using Joblib
- Streamlit web application

## Dataset

The dataset contains 614 records and 13 columns.

Important features include:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

Target:

- Loan Status

Where:

- `Y` = Loan Approved
- `N` = Loan Not Approved

## Machine Learning Approach

### 1. Data Preprocessing

Missing values are handled using:

- Mode for categorical features
- Median for numerical features

Categorical features are converted into numerical values using `OneHotEncoder`.

### 2. Train/Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

### 3. Model

The project uses **Logistic Regression** for binary classification.

The complete preprocessing and model are combined into a Scikit-learn Pipeline.

### 4. Evaluation

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

## Project Structure

```text
loan-risk-prediction/
│
├── data/
│   └── full_loan_data.csv
│
├── models/
│   └── loan_pipeline.pkl
│
├── notebooks/
│   ├── loan_prediction.ipynb
│   └── loan_prediction_final.ipynb
│
├── app.py
├── requirements.txt
└── README.md