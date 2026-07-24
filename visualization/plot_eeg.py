"""
plot_eeg.py

Visualize EEG Signals from GAMEEMO Dataset
"""

import os
import sys

# ---------------------------------------------------------
# Add project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from data.data_loader import load_all_subjects
from config import EMOTION_LABELS

# ---------------------------------------------------------

print("=" * 60)
print("GAMEEMO EEG SIGNAL VISUALIZATION")
print("=" * 60)

print("\nLoading Dataset...\n")

X, y, subjects = load_all_subjects()

print()

print("Dataset Loaded Successfully")
print("EEG Shape :", X.shape)
print()

emotion_names = list(EMOTION_LABELS.values())

channel_names = [
    "AF3", "F7",
    "F3", "FC5",
    "T7", "P7",
    "O1", "O2",
    "P8", "T8",
    "FC6", "F4",
    "F8", "AF4"
]

# ---------------------------------------------------------
# Subject Selection
# ---------------------------------------------------------

print("=" * 60)
print("AVAILABLE SUBJECTS")
print("=" * 60)

for i in range(1, 29):

    print(f"S{i:02d}", end="    ")

    if i % 7 == 0:
        print()

print("\n")

while True:

    try:

        subject_number = int(

            input("Enter Subject Number (1-28): ")

        )

        if subject_number < 1 or subject_number > 28:

            print("Invalid Subject Number\n")

            continue

        break

    except ValueError:

        print("Please enter an integer.\n")

# Dataset stores 0-27
subject_index = subject_number - 1

indices = np.where(subjects == subject_index)[0]

print()

print(f"Subject S{subject_number:02d}")

print("Total Samples :", len(indices))

print()

# ---------------------------------------------------------
# Sample Selection
# ---------------------------------------------------------

while True:

    try:

        sample_number = int(

            input(

                f"Enter Sample Number (0-{len(indices)-1}): "

            )

        )

        if sample_number < 0 or sample_number >= len(indices):

            print("Invalid Sample Number\n")

            continue

        break

    except ValueError:

        print("Please enter an integer.\n")

sample_index = indices[sample_number]

# ---------------------------------------------------------

eeg = X[sample_index]

emotion = emotion_names[y[sample_index]]

# ---------------------------------------------------------

print()

print("=" * 60)
print("SELECTED SAMPLE")
print("=" * 60)

print(f"Subject        : S{subject_number:02d}")

print("Dataset Index  :", sample_index)

print("Sample Number  :", sample_number)

print("Emotion        :", emotion)

print()

# ---------------------------------------------------------

os.makedirs(

    "results/plots",

    exist_ok=True

)

plt.figure(

    figsize=(16, 12)

)

for channel in range(14):

    plt.subplot(

        7,

        2,

        channel + 1

    )

    plt.plot(

        eeg[channel],

        linewidth=1

    )

    plt.title(

        channel_names[channel]

    )

    plt.xlabel(

        "Time"

    )

    plt.ylabel(

        "Amplitude"

    )

    plt.grid(True)

plt.suptitle(

    f"GAMEEMO EEG SIGNAL\n"

    f"Subject : S{subject_number:02d}    "

    f"Sample : {sample_number}    "

    f"Emotion : {emotion}",

    fontsize=16

)

plt.tight_layout()

save_path = "results/plots/eeg_signal.png"

plt.savefig(

    save_path,

    dpi=300,

    bbox_inches="tight"

)

print("Figure Saved")

print(save_path)

print()

plt.show()

print("=" * 60)

print("EEG VISUALIZATION COMPLETED")

print("=" * 60)
