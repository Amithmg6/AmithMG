# A complete project for credit risk modeling, demonstrating hyperparameter
# tuning and conceptual deployment in the financial domain.

# To install: pip install scikit-learn numpy pandas

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Business Problem Statement:
# An online lender wants to automate its loan approval process. The goal is to
# build a predictive model that assesses the credit risk of new applicants.
# A low-risk applicant should be approved, while a high-risk applicant should
# be rejected to minimize financial losses for the lender. This is a
# classification problem.

# 2. Simulate a Realistic-Looking Dataset
# We'll create a synthetic dataset that mimics loan application data.
print("Simulating a dataset for credit risk modeling...")
np.random.seed(42)
num_applicants = 5000
data = {
    'credit_score': np.random.normal(700, 50, num_applicants).astype(int),
    'annual_income': np.random.lognormal(mean=11, sigma=0.8, size=num_applicants).astype(int),
    'loan_amount': np.random.normal(25000, 10000, num_applicants).astype(int),
    'loan_term_years': np.random.choice([2, 3, 4, 5], size=num_applicants),
    # The target variable (0 = No Default, 1 = Default)
    'risk_label': np.random.choice([0, 1], size=num_applicants, p=[0.9, 0.1])
}
df = pd.DataFrame(data)

# Let's add a simple correlation: lower credit score and higher loan amount increase risk
df.loc[df['credit_score'] < 650, 'risk_label'] = 1
df.loc[df['loan_amount'] > 40000, 'risk_label'] = 1

# Features (X) and Target (y)
X = df[['credit_score', 'annual_income', 'loan_amount', 'loan_term_years']]
y = df['risk_label']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset loaded. Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 3. Define the Model and the Hyperparameter Grid
# We will use the Gradient Boosting Classifier, a powerful model for this task.
model = GradientBoostingClassifier(random_state=42)

# Define the hyperparameter grid to search over.
# Grid Search will explore all combinations of these parameters.
# The 'verbose' argument in GridSearchCV will show us the progress.
param_grid = {
    'n_estimators': [100, 200, 300],          # Number of boosting stages
    'learning_rate': [0.05, 0.1, 0.2],        # Step size shrinkage
    'max_depth': [3, 4, 5],                   # Maximum depth of the individual regression estimators
}

# 4. Instantiate and Run Grid Search
print("\nStarting Grid Search for hyperparameter tuning...")
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',   # We want to maximize accuracy
    cv=5,                 # Perform 5-fold cross-validation
    n_jobs=-1,            # Use all available CPU cores
    verbose=2
)

grid_search.fit(X_train, y_train)
print("\nGrid Search finished.")

# 5. Get and Evaluate the Best Model
print("\nBest hyperparameters found:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# Make predictions on the unseen test set
y_pred = best_model.predict(X_test)

# Calculate and print the performance metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['No Default', 'Default'])

print(f"\nAccuracy of the best model on the test set: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# 6. Conceptual Deployment
# In a real-world scenario, the trained 'best_model' would be saved to a file.
# This file would then be used by a production application or API to make
# real-time predictions on new data.
print("\n--- Conceptual Deployment ---")

# Save the final, optimized model to a file using joblib.
model_filename = 'credit_risk_model.joblib'
print(f"Saving the best model to '{model_filename}'...")
joblib.dump(best_model, model_filename)
print("Model successfully saved.")

# Conceptual API Usage:
# In a separate file (e.g., 'api_server.py'), you would set up a simple API
# that loads the model and takes a new applicant's data as input.

# Example of how you would load and use the model in a live application:
# try:
#     loaded_model = joblib.load('credit_risk_model.joblib')
#     print("Model loaded successfully for real-time predictions.")
# except FileNotFoundError:
#     print("Error: Model file not found. Please train and save the model first.")
#
# def assess_credit_risk(applicant_data):
#     # The applicant_data would be a dictionary from an API request, e.g.:
#     # {'credit_score': 720, 'annual_income': 85000, 'loan_amount': 20000, 'loan_term_years': 4}
#     
#     # Convert the new data to a format the model expects (a DataFrame or 2D array)
#     df_new = pd.DataFrame([applicant_data])
#     
#     # Make the prediction
#     prediction = loaded_model.predict(df_new)
#     
#     # The prediction is 0 for 'No Default' and 1 for 'Default'
#     if prediction[0] == 0:
#         return "Low Risk: Approve Loan"
#     else:
#         return "High Risk: Reject Loan"

# To test this function, you could call it with a new data point:
# new_applicant = {'credit_score': 680, 'annual_income': 75000, 'loan_amount': 30000, 'loan_term_years': 3}
# print("\nAssessing a new applicant...")
# print(f"Applicant Data: {new_applicant}")
# # For this example, we'll just show the concept, not run it.
# # prediction = assess_credit_risk(new_applicant)
# # print(f"Prediction: {prediction}")
