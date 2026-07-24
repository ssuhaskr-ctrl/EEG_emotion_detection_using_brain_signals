"""
plot_feature_importance.py

Visualize EEG Channel Importance
GAMEEMO Project
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

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------

print("=" * 60)
print("EEG FEATURE IMPORTANCE")
print("=" * 60)

# ---------------------------------------------------------
# EEG Channel Names
# ---------------------------------------------------------

channels = [

    "AF3",

    "F7",

    "F3",

    "FC5",

    "T7",

    "P7",

    "O1",

    "O2",

    "P8",

    "T8",

    "FC6",

    "F4",

    "F8",

    "AF4"

]

# ---------------------------------------------------------
# Feature Importance Values
#
# These are demonstration values.
# Replace them with actual importance values if available
# from your trained classical ML model.
# ---------------------------------------------------------

importance = np.array([

    0.92,

    0.81,

    0.96,

    0.78,

    0.61,

    0.57,

    0.48,

    0.51,

    0.59,

    0.63,

    0.80,

    0.90,

    0.84,

    0.88

])

# ---------------------------------------------------------

order = np.argsort(importance)

sorted_channels = np.array(channels)[order]

sorted_importance = importance[order]

best_channel = sorted_channels[-1]

best_score = sorted_importance[-1]

print()

print("Most Important EEG Channel")

print("--------------------------")

print(best_channel)

print("Importance :", round(best_score,3))

print()

# ---------------------------------------------------------

os.makedirs(

    "results/plots",

    exist_ok=True

)

plt.figure(

    figsize=(10,7)

)

bars = plt.barh(

    sorted_channels,

    sorted_importance,

    color="royalblue"

)

bars[-1].set_color("crimson")

plt.title(

    "EEG Channel Feature Importance",

    fontsize=15,

    fontweight="bold"

)

plt.xlabel(

    "Importance Score"

)

plt.ylabel(

    "EEG Channels"

)

plt.xlim(

    0,

    1.05

)

plt.grid(

    axis="x",

    linestyle="--",

    alpha=0.4

)

# ---------------------------------------------------------
# PART 2 STARTS HERE
# ---------------------------------------------------------
# ---------------------------------------------------------
# Display Importance Values
# ---------------------------------------------------------

for bar, value in zip(bars, sorted_importance):

    plt.text(

        value + 0.02,

        bar.get_y() + bar.get_height() / 2,

        f"{value:.2f}",

        va="center",

        fontsize=10,

        fontweight="bold"

    )

# ---------------------------------------------------------
# Highlight Best Feature
# ---------------------------------------------------------

plt.annotate(

    "Most Important",

    xy=(best_score, len(sorted_channels) - 1),

    xytext=(best_score - 0.25, len(sorted_channels) - 2),

    arrowprops=dict(

        arrowstyle="->",

        color="red",

        lw=2

    ),

    fontsize=11,

    color="red",

    fontweight="bold"

)

# ---------------------------------------------------------
# Save Figure
# ---------------------------------------------------------

save_path = "results/plots/feature_importance.png"

plt.tight_layout()

plt.savefig(

    save_path,

    dpi=300,

    bbox_inches="tight"

)

print("Figure Saved")

print(save_path)

print()

# ---------------------------------------------------------
# Print Ranking
# ---------------------------------------------------------

print("=" * 60)

print("FEATURE RANKING")

print("=" * 60)

print()

print("{:<10} {:>12}".format("Channel", "Importance"))

print("-" * 28)

ranking = sorted(

    zip(channels, importance),

    key=lambda x: x[1],

    reverse=True

)

for channel, score in ranking:

    print(

        "{:<10} {:>10.2f}".format(

            channel,

            score

        )

    )

print()

print("-" * 28)

print(f"Top Channel : {best_channel}")

print(f"Score       : {best_score:.2f}")

print()

plt.show()

print("=" * 60)

print("FEATURE IMPORTANCE VISUALIZATION COMPLETED")

print("=" * 60)