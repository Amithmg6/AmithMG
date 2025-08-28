# A project for sentiment analysis of movie reviews using PyTorch.
# This code will use Optuna for advanced hyperparameter tuning.

# To install: pip install torch numpy torchtext scikit-learn optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
import time
import optuna

# --- 1. Define the Problem ---
# The goal is to classify movie reviews as either positive (1) or negative (0).
# This is a binary text classification problem.

# --- 2. Data Preparation ---
# We will use the IMDB dataset, which comes with torchtext.
# It contains movie reviews labeled as positive or negative.

# The first step is to tokenize the text (break it into words).
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    """
    Yields tokens from the dataset iterator.
    """
    for _, text in data_iter:
        yield tokenizer(text)

# We will build a vocabulary from the training dataset.
print("Loading IMDB dataset and building vocabulary...")
# Split the dataset into train and test iterators
train_iter, test_iter = IMDB(split=('train', 'test'))

# Build the vocabulary
# <unk> is for unknown words, <pad> is for padding sequences to the same length.
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter),
    specials=['<unk>', '<pad>']
)
# Set the default index for unknown words.
vocab.set_default_index(vocab['<unk>'])

# Define a function to convert text to numerical tensors
def text_pipeline(text):
    """
    Converts text to a sequence of integer indices using the vocabulary.
    """
    return vocab(tokenizer(text))

# Define a function to convert a label to an integer
def label_pipeline(label):
    """
    Converts a label string to an integer (0 or 1).
    """
    # 1 is positive, 2 is negative. We map to 1 and 0.
    return int(label) - 1

# Define a function to pad sequences and create batches
def collate_batch(batch):
    """
    Pads sequences in a batch to a fixed length and creates tensors.
    """
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# Set the device for training (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 3. Define the Model Architecture ---
# We will use a simple text classification model with an Embedding Bag layer
# which is very efficient for text.
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        # EmbeddingBag is a good choice for text classification because it is simple
        # and efficient. It computes the mean of embeddings for a text sequence.
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        """
        Initializes the model weights with a uniform distribution.
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        """
        Defines the forward pass of the model.
        """
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# --- 4. Define Training and Evaluation Functions ---
def train_epoch(model, dataloader, optimizer, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_acc, total_loss = 0, 0
    start_time = time.time()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_loss += loss.item()

    end_time = time.time()
    # print(f"Training Time: {(end_time - start_time):.2f}s, "
    #       f"Average Loss: {(total_loss / len(dataloader)):.4f}, "
    #       f"Accuracy: {(total_acc / len(dataloader.dataset)):.4f}")
    return total_loss / len(dataloader), total_acc / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    """
    Evaluates the model on the given dataset.
    """
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for label, text, offsets in dataloader:
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_loss += loss.item()
    return total_loss / len(dataloader), total_acc / len(dataloader.dataset)

# --- 5. Hyperparameter Tuning using Optuna ---
# This objective function is what Optuna will call for each trial.
def objective(trial):
    """
    Objective function for Optuna. It trains a model with a suggested set of
    hyperparameters and returns the validation loss.
    """
    # Define the search space for hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    embed_dim = trial.suggest_int('embed_dim', 32, 256)

    # Set up the model, optimizer, and loss function
    vocab_size = len(vocab)
    num_class = len(set(label_pipeline(label) for label, _ in IMDB(split='train')))
    model = TextClassificationModel(vocab_size, embed_dim, num_class).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_dataloader = DataLoader(
        list(IMDB(split='train')),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        list(IMDB(split='test')),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # Train and evaluate the model for a few epochs
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, test_dataloader, criterion)
        # We can report intermediate results to Optuna.
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

if __name__ == '__main__':
    print("\nStarting Optuna hyperparameter search...")

    # Create an Optuna study. We want to minimize the validation loss.
    study = optuna.create_study(direction='minimize')
    # Run the optimization for a number of trials
    study.optimize(objective, n_trials=20, timeout=600)  # Stop after 20 trials or 600 seconds

    print("\nOptuna search finished.")
    # Print the best trial's parameters and value
    best_trial = study.best_trial
    print("\nBest hyperparameters found:")
    print(best_trial.params)
    print(f"Best validation loss: {best_trial.value:.4f}")
