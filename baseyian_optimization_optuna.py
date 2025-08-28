# A simple example of Bayesian optimization using Optuna

import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 1. Load a real dataset
# We'll use the Iris dataset, which is a classic for classification problems.
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the objective function
# The objective function takes a 'trial' object as input, which is used to suggest hyperparameters.
def objective(trial):
    """
    This function defines the machine learning model and the hyperparameters to tune.
    It returns a metric (e.g., accuracy) that Optuna will try to maximize or minimize.
    """
    # Suggest integer hyperparameters from a range
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)

    # Suggest a categorical hyperparameter
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Initialize the model with the suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )

    # Use cross-validation to evaluate the model's performance on the training data.
    # Optuna will try to maximize this score by default.
    score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=5).mean()

    return score

# 3. Create a study and run the optimization
# A "study" is an optimization session. We set the direction to 'maximize' since we want to
# maximize the cross-validation score.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # Run 50 trials

# 4. Print the best results
print("Number of finished trials:", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 5. Evaluate the best model on the test set
best_params = study.best_params
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

score_on_test = best_model.score(X_test, y_test)
print(f"\nAccuracy of the best model on the test set: {score_on_test:.4f}")
