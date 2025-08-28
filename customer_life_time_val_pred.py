# A project demonstrating hyperparameter tuning for a regression model
# with a new use case and conceptual deployment details.

# To install: pip install scikit-learn numpy pandas

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Business Use Case: Customer Lifetime Value (CLV) Prediction
# Problem Statement: An e-commerce company wants to predict the total value
# a new customer will bring over their lifetime. This information helps
# them allocate marketing budgets more effectively and identify high-value
# customers early. This is a classic regression problem.

# 1. Simulate the dataset for this business problem
# We'll create a synthetic dataset that mimics real-world e-commerce data.
print("Simulating a dataset for Customer Lifetime Value (CLV) prediction...")
np.random.seed(42)
num_customers = 5000
data = {
    'total_spent_3_months': np.random.lognormal(mean=2, sigma=1, size=num_customers),
    'num_purchases_3_months': np.random.poisson(lam=2, size=num_customers),
    'last_purchase_days_ago': np.random.exponential(scale=30, size=num_customers),
    'customer_clv': np.random.lognormal(mean=3, sigma=1.5, size=num_customers) + np.random.rand(num_customers)*20
}
df = pd.DataFrame(data)

# The target variable is 'customer_clv', which we want to predict.
X = df[['total_spent_3_months', 'num_purchases_3_months', 'last_purchase_days_ago']]
y = df['customer_clv']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset loaded. Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 2. Define the model and the parameter grid
# We will use the Gradient Boosting Regressor for this task.
model = GradientBoostingRegressor(random_state=42)

# Define the hyperparameter grid to search over.
# This grid contains the parameters we want to tune and their possible values.
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4],
}

# 3. Instantiate and run Grid Search
# GridSearchCV will train the model with every possible combination of
# the hyperparameters defined in the grid.
print("\nStarting Grid Search for hyperparameter tuning...")
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print("\nGrid Search finished.")

# 4. Print the best parameters and evaluate the best model
print("\nBest hyperparameters found:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# Make predictions on the unseen test set
y_pred = best_model.predict(X_test)

# Calculate and print the performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE) on test set: {mse:.4f}")
print(f"Root Mean Squared Error (RMSE) on test set: {rmse:.4f}")
print(f"R-squared (R2) score on test set: {r2:.4f}")

# 5. Conceptual Deployment
# In a real-world scenario, once the best model is found, it would be
# saved to a file and deployed as part of a web service or API.
print("\n--- Conceptual Deployment ---")

# Save the final model to a file using joblib.
model_filename = 'clv_predictor.joblib'
print(f"Saving the best model to '{model_filename}'...")
joblib.dump(best_model, model_filename)
print("Model successfully saved.")

# Conceptual API
# In a separate file (e.g., app.py), you would set up a simple web API
# using a framework like Flask or FastAPI. This API would load the saved model
# and use it to make predictions on new data.

# Example of loading the model in a conceptual API:
# try:
#    loaded_model = joblib.load('clv_predictor.joblib')
#    print("Model loaded successfully for deployment.")
# except FileNotFoundError:
#    print("Error: Model file not found. Please train and save the model first.")
#
# def predict_clv(new_customer_data):
#    # The new_customer_data would be a dictionary or DataFrame from an API request.
#    # Example: {'total_spent_3_months': 50, 'num_purchases_3_months': 3, 'last_purchase_days_ago': 15}
#    
#    # Convert the new data to a format the model expects.
#    df_new = pd.DataFrame([new_customer_data])
#    
#    # Make and return the prediction.
#    prediction = loaded_model.predict(df_new)
#    return prediction[0]

