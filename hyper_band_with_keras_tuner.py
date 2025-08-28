
# An example of Hyperband tuning for a neural network using Keras Tuner

# To install: pip install keras-tuner tensorflow scikit-learn

import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Create a dataset
# We'll use the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define a function that builds the Keras model
# This function takes a 'hp' object to suggest hyperparameters.
def build_model(hp):
    """
    Builds a neural network with tunable hyperparameters.
    """
    # Create a sequential model
    model = keras.Sequential()

    # Tune the number of hidden layers
    for i in range(hp.Int("num_layers", 1, 3)):
        # Tune the number of units in each layer
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
    # Add the output layer
    # Since this is a binary classification problem, we use one neuron
    # and a sigmoid activation function.
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

# 3. Instantiate the Hyperband tuner
tuner = Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="hyperband_dir",
    project_name="hyperband_tuning",
)

# 4. Run the search
# This will start the Hyperband optimization process.
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

# 5. Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
print("Best model found:")
best_model.summary()

# 6. Evaluate the best model on the test set
# It's important to evaluate on the unseen test set to get a final performance metric.
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"\nAccuracy of the best model on the test set: {accuracy:.4f}")
