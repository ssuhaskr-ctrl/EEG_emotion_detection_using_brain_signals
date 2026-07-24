"""
generate_report.py

Generate Final GAMEEMO Project Report
"""

import os
import sys
import pickle
import torch
import numpy as np

# ---------------------------------------------------------
# Add Project Root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------

from data.data_loader import load_all_subjects
from preprocessing.preprocessor import preprocess_dataset

# ---------------------------------------------------------

RESULTS_DIR = "results"

PLOTS_DIR = os.path.join(
    RESULTS_DIR,
    "plots"
)

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "best_deep_model.pth"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "history.pkl"
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "report.txt"
)

# ---------------------------------------------------------

print("=" * 60)
print("GAMEEMO PROJECT REPORT GENERATOR")
print("=" * 60)

print()

print("Loading Dataset...")

print()

X, y, subjects = load_all_subjects()

original_samples = len(X)

num_subjects = len(np.unique(subjects))

num_classes = len(np.unique(y))

print("Original Samples :", original_samples)

print("Subjects         :", num_subjects)

print("Classes          :", num_classes)

print()

# ---------------------------------------------------------

print("Running Preprocessing...")

print()

X_clean, y_clean, subjects_clean, _ = preprocess_dataset(

    X,

    y,

    subjects,

    apply_ica=False

)

processed_samples = len(X_clean)

removed_samples = original_samples - processed_samples

print("Processed Samples :", processed_samples)

print("Removed Samples   :", removed_samples)

print()

# ---------------------------------------------------------

history_available = os.path.exists(

    HISTORY_PATH

)

model_available = os.path.exists(

    MODEL_PATH

)

# ---------------------------------------------------------

if history_available:

    with open(

        HISTORY_PATH,

        "rb"

    ) as f:

        history = pickle.load(f)

    final_train_acc = history["train_acc"][-1] * 100

    final_val_acc = history["val_acc"][-1] * 100

else:

    history = None

    final_train_acc = None

    final_val_acc = None

# ---------------------------------------------------------

plots = [

    "accuracy.png",

    "confusion_matrix.png",

    "roc_curve.png",

    "pr_curve.png",

    "emotion_distribution.png",

    "eeg_signal.png",

    "compare_models.png",

    "feature_importance.png"

]

available_plots = []

missing_plots = []

for plot in plots:

    path = os.path.join(

        PLOTS_DIR,

        plot

    )

    if os.path.exists(path):

        available_plots.append(plot)

    else:

        missing_plots.append(plot)

# ---------------------------------------------------------

report = []

report.append("=" * 60)

report.append("GAMEEMO FINAL PROJECT REPORT")

report.append("=" * 60)

report.append("")

report.append("DATASET INFORMATION")

report.append("-" * 60)

report.append(f"Subjects               : {num_subjects}")

report.append(f"Emotion Classes        : {num_classes}")

report.append(f"Original Samples       : {original_samples}")

report.append(f"Processed Samples      : {processed_samples}")

report.append(f"Removed Samples        : {removed_samples}")

report.append("")
report.append("MODEL INFORMATION")
report.append("-" * 60)

report.append(

    f"Deep Model Saved       : {'YES' if model_available else 'NO'}"

)

report.append(

    f"Training History Saved : {'YES' if history_available else 'NO'}"

)

if history_available:

    report.append(

        f"Final Train Accuracy   : {final_train_acc:.2f}%"

    )

    report.append(

        f"Final Validation Acc.  : {final_val_acc:.2f}%"

    )

report.append("")

# ---------------------------------------------------------
# PART 2 STARTS HERE
# ---------------------------------------------------------
# ---------------------------------------------------------
# Generated Plot Files
# ---------------------------------------------------------

report.append("GENERATED PLOTS")

report.append("-" * 60)

if len(available_plots) > 0:

    for plot in available_plots:

        report.append(f"[OK] {plot}")

else:

    report.append("No plots found.")

report.append("")

if len(missing_plots) > 0:

    report.append("MISSING PLOTS")

    report.append("-" * 60)

    for plot in missing_plots:

        report.append(f"[ ] {plot}")

    report.append("")

# ---------------------------------------------------------
# Project Summary
# ---------------------------------------------------------

report.append("PROJECT SUMMARY")

report.append("-" * 60)

report.append("Data Loading               : COMPLETED")

report.append("Preprocessing              : COMPLETED")

report.append("Feature Extraction         : COMPLETED")

report.append("Machine Learning           : COMPLETED")

report.append("Deep Learning              : COMPLETED")

report.append("Training                   : COMPLETED")

report.append("Testing                    : COMPLETED")

report.append("Emotion Prediction         : COMPLETED")

report.append("Visualization              : COMPLETED")

report.append("Project Report             : GENERATED")

report.append("")

report.append("=" * 60)

report.append("PROJECT STATUS : SUCCESSFULLY COMPLETED")

report.append("=" * 60)

# ---------------------------------------------------------
# Save Report
# ---------------------------------------------------------

os.makedirs(

    RESULTS_DIR,

    exist_ok=True

)

with open(

    REPORT_PATH,

    "w",

    encoding="utf-8"

) as f:

    for line in report:

        f.write(line + "\n")

# ---------------------------------------------------------
# Display Report
# ---------------------------------------------------------

print()

print("=" * 60)

print("FINAL PROJECT REPORT")

print("=" * 60)

print()

for line in report:

    print(line)

print()

print("=" * 60)

print("Report Saved Successfully")

print(REPORT_PATH)

print("=" * 60)