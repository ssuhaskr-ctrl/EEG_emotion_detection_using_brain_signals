"""
train_deep.py

Complete Deep Learning Training Pipeline
"""

import os
import pickle
import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from data.data_loader import load_all_subjects
from preprocessing.preprocessor import preprocess_dataset

from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.cnn_lstm_model import CNNLSTMModel
from models.transformer_model import TransformerModel

from models.trainer import Trainer
from models.evaluate import evaluate_model


# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "cnn"

BATCH_SIZE = 32

EPOCHS = 50


# ---------------------------------------------------

def build_model(name):

    name = name.lower()

    if name == "cnn":
        return CNNModel()

    elif name == "lstm":
        return LSTMModel()

    elif name == "cnn_lstm":
        return CNNLSTMModel()

    elif name == "transformer":
        return TransformerModel()

    else:
        raise ValueError("Unknown model")


# ---------------------------------------------------

print("=" * 60)
print("GAMEEMO DEEP LEARNING TRAINING")
print("=" * 60)

print("\nLoading Dataset...\n")

X, y, subjects = load_all_subjects()

print("\nDataset Shape :", X.shape)

print("\nPreprocessing Dataset...\n")

X, y, subjects, pp = preprocess_dataset(
    X,
    y,
    subjects,
    apply_ica=False
)

print("\nProcessed Shape :", X.shape)

# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ---------------------------------------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ---------------------------------------------------

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------------------------------------------

print("\nBuilding Model :", MODEL_NAME)

model = build_model(MODEL_NAME)

print(model)

# ---------------------------------------------------

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    device=DEVICE,
    save_path="results/best_deep_model.pth"
)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

history = trainer.fit(epochs=EPOCHS)

# ---------------------------------------------------
# Save Training History
# ---------------------------------------------------

os.makedirs("results", exist_ok=True)

with open("results/history.pkl", "wb") as f:
    pickle.dump(history, f)

print("\nTraining history saved to:")
print("results/history.pkl")

# ---------------------------------------------------
# Final Evaluation
# ---------------------------------------------------

print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

trainer.load_best_model()

evaluate_model(
    trainer.model,
    test_loader,
    DEVICE
)

# ---------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print("Best Model:")
print("results/best_deep_model.pth")

print()

print("Training History:")
print("results/history.pkl")

print("=" * 60)