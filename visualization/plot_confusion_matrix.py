"""
plot_confusion_matrix.py

Plot Confusion Matrix for Deep Learning Model
"""

import os
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from data.data_loader import load_all_subjects
from preprocessing.preprocessor import preprocess_dataset
from models.cnn_model import CNNModel
from config import EMOTION_LABELS


# -------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "results/best_deep_model.pth"

PLOT_DIR = "results/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

CLASS_NAMES = list(EMOTION_LABELS.values())

# -------------------------------------------------

print("=" * 60)
print("CONFUSION MATRIX VISUALIZATION")
print("=" * 60)

print("\nLoading Dataset...\n")

X, y, subjects = load_all_subjects()

X, y, subjects, _ = preprocess_dataset(
    X,
    y,
    subjects,
    apply_ica=False
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=32,
    shuffle=False
)

print("Loading Model...")

model = CNNModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("Model Loaded Successfully\n")

true_labels = []

pred_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(DEVICE)

        outputs = model(X_batch)

        _, pred = torch.max(outputs, 1)

        true_labels.extend(
            y_batch.numpy()
        )

        pred_labels.extend(
            pred.cpu().numpy()
        )

cm = confusion_matrix(
    true_labels,
    pred_labels
)

plt.figure(figsize=(8,6))

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(CLASS_NAMES))

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            color="black"
        )

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.tight_layout()

save_path = os.path.join(
    PLOT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    save_path,
    dpi=300
)

plt.show()

print()

print("Saved :", save_path)

print()

print("=" * 60)
