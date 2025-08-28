# A simple example of Automated Machine Learning (AutoML) with TPOT
# Note: TPOT is more of a full AutoML tool, so it optimizes the entire
# pipeline, including the model and its hyperparameters.

# To install: pip install tpot scikit-learn

from tpot import TPOTClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Load a real dataset
# We'll use the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize and run TPOT
# It will use genetic programming to find the best model and hyperparameters.
# The 'verbosity' argument is no longer supported, so we will remove it.
tpot = TPOTClassifier(
    generations=5,            # Number of iterations to run the genetic algorithm
    population_size=20,       # Number of pipelines to evaluate in each generation
    cv=5,                     # Cross-validation folds
    random_state=42,
    # verbosity=2 is no longer supported, remove this line to fix the TypeError
    log_file='tpot_log.log'   # Use a log file to track progress instead
)

tpot.fit(X_train, y_train)

# 3. Print the results
print("Best pipeline found:")
print(tpot.fitted_pipeline_)

# 4. Evaluate the best model on the test set
print(f"Test set score: {tpot.score(X_test, y_test):.4f}")

# 5. Export the pipeline code
# You can save the best-performing pipeline as a Python script
tpot.export('tpot_best_pipeline.py')
