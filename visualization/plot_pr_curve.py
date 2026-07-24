"""
plot_pr_curve.py

Precision Recall Curve Visualization
"""

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

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from data.data_loader import load_all_subjects

from preprocessing.preprocessor import (
    preprocess_dataset
)

from models.cnn_model import CNNModel
from config import EMOTION_LABELS, N_CLASSES


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "results/best_deep_model.pth"

CLASS_NAMES = list(EMOTION_LABELS.values())

NUM_CLASSES = N_CLASSES

os.makedirs(
    "results/plots",
    exist_ok=True
)

print("=" * 60)
print("PRECISION RECALL CURVE")
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

loader = DataLoader(

    TensorDataset(
        X_test,
        y_test
    ),

    batch_size=32,

    shuffle=False

)

print("Loading Model...\n")

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

all_probs = []
all_labels = []

with torch.no_grad():

    for x, y in loader:

        x = x.to(DEVICE)

        output = model(x)

        probs = torch.softmax(
            output,
            dim=1
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_labels.extend(
            y.numpy()
        )

all_probs = np.array(all_probs)

all_labels = np.array(all_labels)

y_bin = label_binarize(

    all_labels,

    classes=np.arange(NUM_CLASSES)

)

precision = {}
recall = {}
avg_precision = {}

for i in range(NUM_CLASSES):

    precision[i], recall[i], _ = precision_recall_curve(

        y_bin[:, i],

        all_probs[:, i]

    )

    avg_precision[i] = average_precision_score(

        y_bin[:, i],

        all_probs[:, i]

    )

print("=" * 60)
print("AVERAGE PRECISION")
print("=" * 60)

for i in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[i]} : "
        f"{avg_precision[i]:.4f}"
    )

plt.figure(figsize=(8, 6))

colors = [
    "blue",
    "red",
    "green",
    "orange"
]

for i, color in enumerate(colors):

    plt.plot(

        recall[i],

        precision[i],

        color=color,

        lw=2,

        label=(
            f"{CLASS_NAMES[i]}"
            f" (AP={avg_precision[i]:.3f})"
        )

    )

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title(
    "Precision Recall Curve - GAMEEMO"
)

plt.legend()

plt.grid(True)

save_path = (
    "results/plots/pr_curve.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print("Saved :", save_path)

plt.show()

print()
print("=" * 60)
print("PR CURVE COMPLETED")
print("=" * 60)
