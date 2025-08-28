# A project for credit card fraud detection, demonstrating hyperparameter
# tuning with RandomizedSearchCV for a highly imbalanced dataset.

# To install: pip install scikit-learn numpy pandas

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib
import os

# 1. Business Problem Statement:
# A large credit card company wants to build a system that can detect fraudulent
# transactions in real time. The goal is to minimize financial losses for both the
# company and its customers. This is a classification problem, but with a
# significant challenge: the dataset is highly imbalanced, with very few
# fraudulent transactions compared to legitimate ones.

# 2. Use a Real-World Dataset from Kaggle
# We will use the 'creditcard.csv' dataset from Kaggle.
# Please download the dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# and place the 'creditcard.csv' file in the same directory as this script.
print("Loading real-world credit card fraud dataset...")
data_path = 'creditcard.csv'

# Check if the file exists before attempting to load
if not os.path.exists(data_path):
    print(f"Error: The file '{data_path}' was not found.")
    print("Please download 'creditcard.csv' from the link above and place it in this directory.")
    # Exit or provide a placeholder behavior if the data is missing
    raise FileNotFoundError(f"Dataset file '{data_path}' not found.")

df = pd.read_csv(data_path)

# Features (X) and Target (y)
# The features V1-V28 are the result of a PCA transformation.
# 'Time' and 'Amount' are the original features. 'Class' is the target.
X = df.drop('Class', axis=1)
y = df['Class']

# Check the class balance
fraud_count = df['Class'].sum()
legit_count = len(df) - fraud_count
print(f"Dataset loaded. Total transactions: {len(df)}")
print(f"Legitimate transactions: {legit_count} ({legit_count / len(df):.2%})")
print(f"Fraudulent transactions: {fraud_count} ({fraud_count / len(df):.2%})")

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 'stratify=y' ensures the same proportion of fraud cases in both train and test sets.

# 3. Define the Model and the Hyperparameter Search Space
# We will use the Gradient Boosting Classifier again.
model = GradientBoostingClassifier(random_state=42)

# Define the hyperparameter search space using a dictionary.
# We will use RandomizedSearchCV to sample from this space. This is more
# efficient than GridSearchCV for larger search spaces.
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

# 4. Instantiate and Run Randomized Search
# For imbalanced data, accuracy is not a good metric. We'll use 'roc_auc'
# (Area Under the Receiver Operating Characteristic Curve) as it measures
# the model's ability to distinguish between classes, regardless of their
# imbalance.
print("\nStarting Randomized Search for hyperparameter tuning...")
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,                    # Number of random combinations to try
    scoring='roc_auc',            # A better metric for imbalanced data
    cv=5,                         # 5-fold cross-validation
    n_jobs=-1,                    # Use all available CPU cores
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)
print("\nRandomized Search finished.")

# 5. Get and Evaluate the Best Model
print("\nBest hyperparameters found:")
print(random_search.best_params_)

best_model = random_search.best_estimator_

# Make predictions on the unseen test set
# We use predict_proba to get a probability score, which is more useful
# for fraud detection than a simple binary prediction.
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred_binary = best_model.predict(X_test)

# Calculate and print the performance metrics
# We now use metrics that are more sensitive to the minority class (fraud).
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f"\nROC AUC score on test set: {roc_auc:.4f}")
print(f"Precision on test set: {precision:.4f}")
print(f"Recall on test set: {recall:.4f}")
print(f"F1-Score on test set: {f1:.4f}")

# 6. Conceptual Deployment
# In a real-world scenario, the trained model would be used to score
# transactions in real-time. If the model's predicted probability of fraud
# exceeds a certain threshold, the transaction is flagged for human review.
print("\n--- Conceptual Deployment ---")

# Save the final, optimized model to a file using joblib.
model_filename = 'fraud_detection_model.joblib'
print(f"Saving the best model to '{model_filename}'...")
joblib.dump(best_model, model_filename)
print("Model successfully saved.")
print("\nTo use the model in a real application, load it like this:")
print("loaded_model = joblib.load('fraud_detection_model.joblib')")
# Conceptual API Usage:
# An API would be set up to process new transactions.
#
# def check_for_fraud(transaction_data):
#     # The transaction_data would be a dictionary from an API request
#     # e.g., {'transaction_amount': 500, 'time_since_last_transaction': 200, 'location_risk_score': 0.8}
#
#     df_new = pd.DataFrame([transaction_data])
#
#     # Get the probability of the transaction being fraudulent
#     fraud_proba = loaded_model.predict_proba(df_new)[:, 1][0]
#
#     if fraud_proba > 0.7: # A business-defined threshold
#         return {"status": "Flagged for Review", "probability": fraud_proba}
#     else:
#         return {"status": "Approved", "probability": fraud_proba}

