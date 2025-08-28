# A complete project for sentiment analysis with hyperparameter tuning
# using Keras Tuner and a real-world dataset.

# To install: pip install keras-tuner tensorflow scikit-learn

import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load and prepare the dataset
# We will use the IMDb movie review dataset, a classic for sentiment analysis.
# The dataset contains movie reviews labeled as positive (1) or negative (0).
# Keras has a built-in utility to load this dataset.
print("Loading IMDb movie review dataset...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.imdb.load_data(num_words=10000)

# The data is already pre-tokenized into sequences of integers.
# We need to pad the sequences so they all have the same length.
# This is a critical step for deep learning models on text data.
print("Padding sequences...")
X_train = keras.preprocessing.sequence.pad_sequences(X_train_raw, maxlen=256)
X_test = keras.preprocessing.sequence.pad_sequences(X_test_raw, maxlen=256)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 2. Define the model-building function
# This function is where we define the architecture of our deep learning model
# and the hyperparameters we want to tune.
def build_model(hp):
    """
    Builds a sentiment analysis model with tunable hyperparameters.

    The model uses an Embedding layer to convert word indices to dense vectors,
    followed by an LSTM layer for sequence processing, and a Dropout layer
    for regularization.

    Args:
        hp: The Keras Tuner HyperParameters object.

    Returns:
        A compiled Keras Sequential model.
    """
    # Create a Sequential model
    model = keras.Sequential()
    
    # --- Hyperparameters to Tune ---
    
    # Embeddings are a crucial part of NLP. This layer converts integer-encoded
    # words into dense vectors of a fixed size. We will tune this size.
    # Impact: A larger embedding dimension can capture more semantic meaning,
    # but may lead to overfitting and slower training.
    embedding_dim = hp.Int('embedding_dim', min_value=32, max_value=128, step=32)
    model.add(keras.layers.Embedding(input_dim=10000, output_dim=embedding_dim))
    
    # An LSTM layer is used to process the sequence of word embeddings.
    # We will tune the number of units (neurons) in this layer.
    # Impact: More units can learn more complex patterns, but may lead to
    # a higher risk of overfitting and longer training times.
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
    model.add(keras.layers.LSTM(units=lstm_units))
    
    # Dropout is a regularization technique that randomly sets a fraction of
    # input units to 0 at each update during training time.
    # We will tune the dropout rate.
    # Impact: A higher dropout rate can prevent overfitting, but if it's too high,
    # the model may not be able to learn effectively.
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    model.add(keras.layers.Dropout(rate=dropout_rate))
    
    # Final dense layer for the output. Since this is binary classification
    # (positive or negative), we use a single neuron with a sigmoid activation.
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Tune the learning rate. This is one of the most important hyperparameters.
    # We will choose from a few standard values.
    # Impact: A low learning rate can be slow to converge, while a high learning
    # rate can cause the model to miss the optimal solution.
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Compile the model with the best optimizer and learning rate.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. Instantiate and run the Hyperband tuner
print("\nStarting the Hyperband search...")
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='nlp_tuner_dir',
    project_name='imdb_sentiment_analysis'
)

# The search will now begin, training and evaluating different models with
# different combinations of the hyperparameters defined in `build_model`.
# We use a validation split to monitor performance during the search.
tuner.search(X_train, y_train_raw, epochs=10, validation_split=0.2)
print("\nHyperparameter search finished.")

# 4. Get the best model and evaluate
# Once the search is complete, we retrieve the best model found by the tuner.
best_model = tuner.get_best_models(num_models=1)[0]
print("\nBest model summary:")
best_model.summary()

# Evaluate the best model on the unseen test set to get a final performance score.
print("\nEvaluating the best model on the test set...")
loss, accuracy = best_model.evaluate(X_test, y_test_raw)
print(f"Accuracy of the best model on the test set: {accuracy:.4f}")

# You can also get the best hyperparameters from the tuner
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest hyperparameters found:")
print(best_hyperparameters.get_config()['values'])
