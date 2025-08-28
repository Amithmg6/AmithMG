# A project for sentiment analysis of movie reviews using PyTorch.
# This code will build a simple neural network and prepare the data.
# We will focus on data preprocessing and model creation first, and then
# move on to hyperparameter tuning.

# To install: pip install torch numpy torchtext scikit-learn

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
from sklearn.model_selection import ParameterGrid
import time

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
    print(f"Training Time: {(end_time - start_time):.2f}s, "
          f"Average Loss: {(total_loss / len(dataloader)):.4f}, "
          f"Accuracy: {(total_acc / len(dataloader.dataset)):.4f}")
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

# --- 5. Hyperparameter Tuning using Grid Search ---
# We will use a simple Grid Search for demonstration purposes. For a real project,
# you would use a more advanced library like Optuna or Hyperopt.

print("\nStarting Grid Search for hyperparameter tuning...")

# Define the hyperparameter search grid
param_grid = {
    'lr': [0.1, 0.01, 0.001],
    'batch_size': [128, 256],
    'embed_dim': [64, 128]
}

best_loss = float('inf')
best_params = {}
best_model_state = None

# Create a ParameterGrid object
for params in ParameterGrid(param_grid):
    print(f"\nTraining with parameters: {params}")

    # Set up the model, optimizer, and loss function
    vocab_size = len(vocab)
    num_class = len(set(label_pipeline(label) for label, _ in train_iter))
    model = TextClassificationModel(vocab_size, params['embed_dim'], num_class).to(device)
    optimizer = optim.SGD(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_dataloader = DataLoader(
        list(IMDB(split='train')),
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        list(IMDB(split='test')),
        batch_size=params['batch_size'],
        shuffle=False,
        collate_fn=collate_batch
    )

    # Train and evaluate the model
    num_epochs = 5  # We'll use a small number of epochs for the search
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, test_dataloader, criterion)
        print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Check if this is the best model so far
    if val_loss < best_loss:
        best_loss = val_loss
        best_params = params
        best_model_state = model.state_dict()

print("\nGrid Search finished.")
print("\nBest hyperparameters found:")
print(best_params)
print(f"Best validation loss: {best_loss:.4f}")

# You can now re-train the model with the best parameters and a larger number of epochs,
# and then save it for deployment.
