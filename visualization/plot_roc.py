"""
plot_roc.py

Plot ROC Curve for GAMEEMO CNN Model
"""

import os
import sys

# ---------------------------------------------------------
# Add Project Root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from data.data_loader import load_all_subjects
from preprocessing.preprocessor import preprocess_dataset
from models.cnn_model import CNNModel
from config import EMOTION_LABELS, N_CLASSES

# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "results/best_deep_model.pth"

SAVE_DIR = "results/plots"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

CLASS_NAMES = list(EMOTION_LABELS.values())

NUM_CLASSES = N_CLASSES

# ---------------------------------------------------------

print("=" * 60)
print("ROC CURVE VISUALIZATION")
print("=" * 60)

print("\nLoading Dataset...\n")

X, y, subjects = load_all_subjects()

print()

print("Dataset Shape :", X.shape)

print()

print("Preprocessing Dataset...\n")

X, y, subjects, _ = preprocess_dataset(
    X,
    y,
    subjects,
    apply_ica=False
)

print()

print("Processed Shape :", X.shape)

print()

# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("Testing Samples :", len(X_test))

print()

# ---------------------------------------------------------

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

test_loader = DataLoader(

    TensorDataset(
        X_test,
        y_test
    ),

    batch_size=32,

    shuffle=False

)

# ---------------------------------------------------------

print("Loading Trained CNN Model...\n")

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

# ---------------------------------------------------------

all_labels = []

all_probs = []

print("Predicting probabilities...\n")

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(DEVICE)

        outputs = model(inputs)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        all_probs.extend(
            probabilities.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

all_probs = np.array(all_probs)

all_labels = np.array(all_labels)

print("Prediction Completed")

print()

print("Probability Matrix Shape :", all_probs.shape)

print()

# ---------------------------------------------------------

# Convert labels to One-vs-Rest format

y_bin = label_binarize(

    all_labels,

    classes=np.arange(NUM_CLASSES)

)

print("One-Hot Labels Shape :", y_bin.shape)

print()

# ---------------------------------------------------------
# PART 2 STARTS FROM HERE
# ---------------------------------------------------------
# ---------------------------------------------------------
# Compute ROC Curve and AUC
# ---------------------------------------------------------

fpr = {}
tpr = {}
roc_auc = {}

for i in range(NUM_CLASSES):

    fpr[i], tpr[i], _ = roc_curve(

        y_bin[:, i],

        all_probs[:, i]

    )

    roc_auc[i] = auc(

        fpr[i],

        tpr[i]

    )

print("=" * 60)
print("AUC SCORES")
print("=" * 60)

for i in range(NUM_CLASSES):

    print(

        f"{CLASS_NAMES[i]:10s}: {roc_auc[i]:.4f}"

    )

print()

# ---------------------------------------------------------
# Plot ROC Curves
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

colors = [

    "blue",

    "red",

    "green",

    "orange"

]

for i, color in enumerate(colors):

    plt.plot(

        fpr[i],

        tpr[i],

        color=color,

        lw=2,

        label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})"

    )

# Random classifier reference

plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    color="black",

    lw=1.5,

    label="Random Guess"

)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - GAMEEMO CNN")

plt.legend(loc="lower right")

plt.grid(True)

save_path = os.path.join(

    SAVE_DIR,

    "roc_curve.png"

)

plt.savefig(

    save_path,

    dpi=300,

    bbox_inches="tight"

)

print("ROC Curve Saved")

print(save_path)

print()

plt.show()

print("=" * 60)
print("ROC VISUALIZATION COMPLETED")
print("=" * 60)
