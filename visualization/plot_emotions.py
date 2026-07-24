"""
plot_emotions.py

Emotion Distribution Visualization
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import load_all_subjects
from config import EMOTION_LABELS, N_CLASSES


print("=" * 60)
print("EMOTION DISTRIBUTION")
print("=" * 60)

print()

X, y, subjects = load_all_subjects()

emotion_names = list(EMOTION_LABELS.values())

counts = np.bincount(y, minlength=N_CLASSES)

print("Emotion Counts")

for i in range(len(emotion_names)):
    print(f"{emotion_names[i]} : {counts[i]}")

os.makedirs(
    "results/plots",
    exist_ok=True
)

plt.figure(figsize=(8,6))

bars = plt.bar(
    emotion_names,
    counts
)

plt.title("Emotion Distribution")

plt.xlabel("Emotion")

plt.ylabel("Number of Samples")

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x()+bar.get_width()/2,

        height+50,

        str(int(height)),

        ha="center"

    )

plt.grid(axis="y")

save_path = "results/plots/emotion_distribution.png"

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
print("EMOTION DISTRIBUTION COMPLETED")
print("=" * 60)
