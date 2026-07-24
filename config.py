
"""
===============================================================================
                GAMEEMO EEG Emotion Detection System
                           Global Configuration
===============================================================================

This file contains all global project settings.

Author : Your Name
Python : 3.10+
===============================================================================
"""

import os

# =============================================================================
# BASE DIRECTORY
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# DATASET SETTINGS
# =============================================================================

# Priority:
# 1. Environment Variable GAMEEMO_ROOT
# 2. GAMEEMO folder inside project
# 3. C:\GAMEEMO
# 4. D:\GAMEEMO

if "GAMEEMO_ROOT" in os.environ:
    DATASET_ROOT = os.environ["GAMEEMO_ROOT"]

elif os.path.exists(os.path.join(BASE_DIR, "GAMEEMO")):
    DATASET_ROOT = os.path.join(BASE_DIR, "GAMEEMO")

elif os.path.exists(r"C:\GAMEEMO"):
    DATASET_ROOT = r"C:\GAMEEMO"

elif os.path.exists(r"D:\GAMEEMO"):
    DATASET_ROOT = r"D:\GAMEEMO"

else:
    DATASET_ROOT = None

# =============================================================================
# DATASET INFORMATION
# =============================================================================

N_SUBJECTS = 28

N_CLASSES = 4

EMOTION_LABELS = {
    0: "Boring",
    1: "Calm",
    2: "Horror",
    3: "Joy"
}

GAME_NAMES = [
    "BoarderLine",
    "HighlandChapel",
    "Minecraft",
    "FerociousPlanet"
]

GAME_FOLDERS = [
    "G1",
    "G2",
    "G3",
    "G4"
]

# =============================================================================
# EEG SETTINGS
# =============================================================================

SAMPLING_RATE = 128

N_CHANNELS = 14

CHANNEL_NAMES = [
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

# =============================================================================
# PREPROCESSING
# =============================================================================

NOTCH_FREQ = 50

BANDPASS_LOW = 0.5

BANDPASS_HIGH = 45

FILTER_ORDER = 5

ICA_N_COMPONENTS = 14

AMPLITUDE_THRESHOLD = 150

# =============================================================================
# WINDOWING
# =============================================================================

WINDOW_SEC = 2

OVERLAP_RATIO = 0.5

WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLING_RATE)

STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP_RATIO))

# =============================================================================
# EEG BANDS
# =============================================================================

FREQ_BANDS = {

    "delta": (0.5, 4),

    "theta": (4, 8),

    "alpha": (8, 13),

    "beta": (13, 30),

    "gamma": (30, 45)

}

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

EXTRACT_TIME_DOMAIN = True

EXTRACT_FREQ_DOMAIN = True

EXTRACT_NONLINEAR = True

EXTRACT_WAVELET = True

# =============================================================================
# TRAINING
# =============================================================================

BATCH_SIZE = 32

EPOCHS = 80

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.4

L2_REG = 1e-4

PATIENCE = 15

RANDOM_SEED = 42

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

FEATURES_DIR = os.path.join(RESULTS_DIR, "features")

for folder in [
    RESULTS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    FEATURES_DIR,
]:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# DEVICE
# =============================================================================

DEVICE = "auto"

# =============================================================================
# CHECK DATASET
# =============================================================================

USE_SYNTHETIC = (
    DATASET_ROOT is None or
    not os.path.exists(DATASET_ROOT)
)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GAMEEMO CONFIGURATION")
    print("=" * 60)

    print(f"Project Folder      : {BASE_DIR}")

    print(f"Dataset Path        : {DATASET_ROOT}")

    print(f"Dataset Exists      : {not USE_SYNTHETIC}")

    print(f"Subjects            : {N_SUBJECTS}")

    print(f"Classes             : {N_CLASSES}")

    print(f"Channels            : {N_CHANNELS}")

    print(f"Sampling Rate       : {SAMPLING_RATE}")

    print(f"Window Samples      : {WINDOW_SAMPLES}")

    print(f"Step Samples        : {STEP_SAMPLES}")

    print(f"Batch Size          : {BATCH_SIZE}")

    print(f"Epochs              : {EPOCHS}")

    print(f"Learning Rate       : {LEARNING_RATE}")

    print(f"Results Folder      : {RESULTS_DIR}")

    if USE_SYNTHETIC:

        print()

        print("WARNING : REAL DATASET NOT FOUND")

        print("The project will use synthetic EEG data.")

    else:

        print()

        print("GAMEEMO dataset detected successfully.")

    print("=" * 60)
