# A project demonstrating hyperparameter tuning for a regression model
# using Grid Search and a real-world dataset.

# To install: pip install scikit-learn numpy

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
# We'll use the California Housing dataset. The business problem is to
# predict house prices based on various features.
print("Loading the California Housing dataset...")
data = fetch_california_housing()
X = data.data
y = data.target

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
    'n_estimators': [100, 200, 300],          # Number of boosting stages
    'learning_rate': [0.05, 0.1, 0.2],        # Step size shrinkage
    'max_depth': [3, 4, 5],                   # Maximum depth of the individual regression estimators
    'min_samples_split': [2, 4],              # Minimum number of samples required to split an internal node
}

# 3. Instantiate and run Grid Search
# GridSearchCV will train the model with every possible combination of
# the hyperparameters defined in the grid.
print("\nStarting Grid Search for hyperparameter tuning...")
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',   # We minimize MSE, so we use its negative
    cv=5,                               # Perform 5-fold cross-validation
    n_jobs=-1,                          # Use all available CPU cores
    verbose=2                           # Provide some output
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

