"""
plot_history.py

Plot Training Accuracy and Loss Curves
"""

import os
import pickle

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
HISTORY_FILE = os.path.join(RESULTS_DIR, "history.pkl")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")


os.makedirs(PLOT_DIR, exist_ok=True)


# ----------------------------------------------------------


def load_history():

    if not os.path.exists(HISTORY_FILE):
        raise FileNotFoundError(
            f"{HISTORY_FILE} not found.\n"
            "Run train_deep.py first."
        )

    with open(HISTORY_FILE, "rb") as f:
        history = pickle.load(f)

    return history


# ----------------------------------------------------------


def plot_accuracy(history):

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_acc"],
        label="Training Accuracy",
        linewidth=2
    )

    plt.plot(
        history["val_acc"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        "accuracy.png"
    )

    plt.savefig(
        save_path,
        dpi=300
    )

    print("Saved:", save_path)

    plt.show()


# ----------------------------------------------------------


def plot_loss(history):

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        "loss.png"
    )

    plt.savefig(
        save_path,
        dpi=300
    )

    print("Saved:", save_path)

    plt.show()


# ----------------------------------------------------------


def main():

    print("=" * 60)

    print("TRAINING HISTORY VISUALIZATION")

    print("=" * 60)

    history = load_history()

    print()

    print("Epochs :", len(history["train_loss"]))

    print()

    plot_accuracy(history)

    plot_loss(history)

    print()

    print("=" * 60)

    print("Finished")

    print("=" * 60)


# ----------------------------------------------------------

if __name__ == "__main__":

    main()