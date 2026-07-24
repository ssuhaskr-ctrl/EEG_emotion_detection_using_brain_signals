"""
compare_models.py

Compare Classical Machine Learning Models
with Deep Learning Models
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

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# ---------------------------------------------------------
# Accuracy Values
#
# Replace these values with your own if you retrain
# ---------------------------------------------------------

models = [

    "Random\nForest",

    "SVM",

    "KNN",

    "Decision\nTree",

    "Logistic\nRegression",

    "CNN"

]

accuracy = [

    23.0,

    24.0,

    26.0,

    22.5,

    28.5,

    95.27

]

# ---------------------------------------------------------

best_index = np.argmax(accuracy)

best_model = models[best_index]

best_accuracy = accuracy[best_index]

print()

print("Best Model :", best_model)

print("Accuracy   :", best_accuracy)

print()

# ---------------------------------------------------------

os.makedirs(

    "results/plots",

    exist_ok=True

)

plt.figure(

    figsize=(10,6)

)

colors = [

    "gray",

    "gray",

    "gray",

    "gray",

    "gray",

    "green"

]

bars = plt.bar(

    models,

    accuracy,

    color=colors

)

plt.title(

    "Performance Comparison of ML and Deep Learning Models",

    fontsize=14,

    fontweight="bold"

)

plt.xlabel(

    "Models"

)

plt.ylabel(

    "Accuracy (%)"

)

plt.ylim(

    0,

    100

)

plt.grid(

    axis="y",

    linestyle="--",

    alpha=0.4

)

# ---------------------------------------------------------
# PART 2 STARTS HERE
# ---------------------------------------------------------
# ---------------------------------------------------------
# Display Accuracy Values
# ---------------------------------------------------------

for bar, value in zip(bars, accuracy):

    plt.text(

        bar.get_x() + bar.get_width() / 2,

        value + 1,

        f"{value:.2f}%",

        ha="center",

        va="bottom",

        fontsize=10,

        fontweight="bold"

    )

# ---------------------------------------------------------
# Highlight Best Model
# ---------------------------------------------------------

plt.annotate(

    "Best Model",

    xy=(best_index, best_accuracy),

    xytext=(best_index, best_accuracy + 10),

    ha="center",

    arrowprops=dict(

        arrowstyle="->",

        lw=2,

        color="red"

    ),

    fontsize=11,

    color="red",

    fontweight="bold"

)

# ---------------------------------------------------------
# Save Figure
# ---------------------------------------------------------

save_path = "results/plots/compare_models.png"

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
# Print Comparison Table
# ---------------------------------------------------------

print("=" * 60)

print("MODEL PERFORMANCE")

print("=" * 60)

print()

print("{:<25} {:>10}".format("Model", "Accuracy"))

print("-" * 38)

for model, acc in zip(models, accuracy):

    print(

        "{:<25} {:>8.2f}%".format(

            model.replace("\n", " "),

            acc

        )

    )

print()

print("-" * 38)

print(

    f"Best Model : {best_model.replace(chr(10),' ')}"

)

print(

    f"Best Accuracy : {best_accuracy:.2f}%"

)

print()

plt.show()

print("=" * 60)

print("MODEL COMPARISON COMPLETED")

print("=" * 60)