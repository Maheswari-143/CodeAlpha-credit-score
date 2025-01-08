# CodeAlpha-credit-scoreimport pandas as pd
Bank Marketing Dataset Classifier

This project implements a machine learning model to classify the target variable y from the Bank Marketing dataset. The classifier uses a Random Forest model for predictions. The dataset and feature preprocessing steps are described below.

Table of Contents

Project Overview

Dataset

Requirements

Usage

Loading Data

Preprocessing Data

Training the Model

Evaluating the Model

Results

Project Overview

The goal of this project is to build a machine learning model to predict if a customer will subscribe to a term deposit (y variable). The Random Forest Classifier is used for this classification task. Data preprocessing includes encoding categorical variables and scaling numerical features.

Dataset

The dataset is based on the Bank Marketing dataset. The key features include:

Categorical Columns: job, marital, education, default, housing, loan, contact, month, poutcome, y

Numerical Columns: age, balance, day, duration, campaign, pdays, previous

Columns Description

Column

Description

age

Age of the customer

job

Job type

marital

Marital status

education

Education level

default

Has credit in default? (yes/no)

balance

Average yearly balance

housing

Has a housing loan? (yes/no)

loan

Has a personal loan? (yes/no)

contact

Contact communication type

day

Last contact day of the month

month

Last contact month of the year

duration

Last contact duration in seconds

campaign

Number of contacts performed during campaign

pdays

Days since last contact (-1: no previous)

previous

Number of contacts performed before this

poutcome

Outcome of the previous marketing campaign

y

Target: Subscription to term deposit (yes/no)

Requirements

Install the required Python packages:

pip install pandas scikit-learn

Usage

Loading Data

Make sure the dataset is available as a CSV file (e.g., bank.csv) in the project directory.

file_path = "bank.csv"
data = load_data(file_path)

Preprocessing Data

Categorical variables are encoded, and numerical features are scaled.

data, label_encoders, scaler = preprocess_data(data)

Training the Model

The data is split into training and testing sets (80-20 split). The model is trained using a Random Forest Classifier.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(X_train, y_train)

Evaluating the Model

Model evaluation includes accuracy score, classification report, and confusion matrix.

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

Results

The evaluation metrics of the model include:

Accuracy: Shows how often the model makes correct predictions.

Classification Report: Provides precision, recall, F1-score, and support for each class.

Confusion Matrix: Shows the distribution of true positives, false positives, true negatives, and false negatives.

Example output:

Accuracy: 0.89
Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.92      0.92       800
           1       0.87      0.85      0.86       200

    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000

Confusion Matrix:
[[738  62]
 [ 30 170]]

