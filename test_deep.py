"""
test_deep.py

Test trained deep learning model on GAMEEMO dataset.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from data.data_loader import load_all_subjects
from preprocessing.preprocessor import preprocess_dataset

from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.cnn_lstm_model import CNNLSTMModel
from models.transformer_model import TransformerModel

from models.evaluate import evaluate_model


# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_NAME = "cnn"

BATCH_SIZE = 32
MODEL_PATH = "results/best_deep_model.pth"

# =====================================================


def build_model(name):

    if name.lower() == "cnn":
        return CNNModel()

    elif name.lower() == "lstm":
        return LSTMModel()

    elif name.lower() == "cnn_lstm":
        return CNNLSTMModel()

    elif name.lower() == "transformer":
        return TransformerModel()

    else:
        raise Exception("Invalid model name")


# =====================================================

print("=" * 60)
print("GAMEEMO DEEP MODEL TEST")
print("=" * 60)

print("\nLoading Dataset...\n")

X, y, subjects = load_all_subjects()

print("Dataset Shape :", X.shape)

print("\nPreprocessing Dataset...\n")

X, y, subjects, pp = preprocess_dataset(
    X,
    y,
    subjects,
    apply_ica=False
)

print("Processed Shape :", X.shape)

# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# =====================================================

X_test = torch.tensor(

    X_test,

    dtype=torch.float32

)

y_test = torch.tensor(

    y_test,

    dtype=torch.long

)

# =====================================================

test_dataset = TensorDataset(

    X_test,

    y_test

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)

# =====================================================

print("\nBuilding Model...\n")

model = build_model(MODEL_NAME)

model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=DEVICE

    )

)

model.to(DEVICE)

model.eval()

print("Model Loaded Successfully")

# =====================================================

print()

print("=" * 60)
print("Evaluating Model")
print("=" * 60)

evaluate_model(

    model,

    test_loader,

    DEVICE

)

# =====================================================

print()

print("=" * 60)

print("Prediction Example")

print("=" * 60)

with torch.no_grad():

    sample = X_test[:10].to(DEVICE)

    outputs = model(sample)

    _, pred = torch.max(outputs, 1)

print()

print("True Labels")

print(y_test[:10].numpy())

print()

print("Predicted Labels")

print(pred.cpu().numpy())

print()

print("=" * 60)

print("TEST COMPLETED SUCCESSFULLY")

print("=" * 60)