# A complete example of hyperparameter tuning for a deep learning model
# using Keras Tuner, with a real dataset for demonstration.

# To install: pip install keras-tuner tensorflow scikit-learn

import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Load the dataset
# We'll use the Breast Cancer Wisconsin (Diagnostic) dataset, which is
# a great, simple dataset for a binary classification task.
print("Loading the dataset...")
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets.
# We'll use the training data for the hyperparameter search and
# a separate test set to evaluate the final best model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset loaded. Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 2. Define the model-building function
# This function is the core of Keras Tuner. It takes a 'hp' (hyperparameter)
# object and defines the model's architecture and tunable parameters.
def build_model(hp):
    """
    Builds a neural network model with tunable hyperparameters.
    
    Args:
        hp: The Keras Tuner HyperParameters object.
        
    Returns:
        A compiled Keras Sequential model.
    """
    # Create a Sequential model, which is a linear stack of layers.
    model = keras.Sequential()

    # Define the input layer with the number of features from our dataset.
    model.add(keras.layers.Input(shape=(X_train.shape[1],)))
    
    # Tune the number of hidden layers. We'll search between 1 and 3 layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        # Tune the number of units (neurons) in each dense layer.
        # We search from 32 up to 512, with a step of 32.
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
    
    # Add the output layer.
    # For binary classification, we use a single neuron with a sigmoid activation function.
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Tune the learning rate for the Adam optimizer.
    # We'll search over three distinct values.
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Compile the model with the best optimizer and learning rate.
    # We use 'binary_crossentropy' as the loss function for binary classification.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

# 3. Instantiate and run the Hyperband tuner
# Hyperband is an efficient tuning algorithm that quickly prunes poorly-performing trials.
print("\nStarting the Hyperband search...")
tuner = Hyperband(
    build_model,                     # The model-building function
    objective="val_accuracy",        # The metric to optimize (validation accuracy)
    max_epochs=10,                   # The maximum number of epochs to train for
    factor=3,                        # The reduction factor for trials
    directory="my_tuner_dir",        # Directory to save search logs and models
    project_name="breast_cancer_tuning", # Name of the project
)

# Run the search on the training data.
# We use a validation split to monitor performance during the search.
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
print("\nHyperparameter search finished.")

# 4. Get and evaluate the best model
# Once the search is complete, we get the best model found by the tuner.
best_model = tuner.get_best_models(num_models=1)[0]
print("\nBest model summary:")
best_model.summary()

# Evaluate the best model on the unseen test set to get a final performance score.
print("\nEvaluating the best model on the test set...")
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Accuracy of the best model on the test set: {accuracy:.4f}")

# You can also get the best hyperparameters from the tuner
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest hyperparameters found:")
print(best_hyperparameters.get_config()['values'])

