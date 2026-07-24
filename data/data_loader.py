"""
data_loader.py — GAMEEMO Dataset Loader
========================================
Handles both .mat (MATLAB) and .csv layouts of the GAMEEMO dataset.
Produces a unified (n_epochs, n_channels, n_samples) array + labels.

GAMEEMO Kaggle layout reference
────────────────────────────────
GAMEEMO/
  S01/
    G1BoarderLine.mat   ← Boring game
    G2HighlandChapel.mat← Calm game
    G3Minecraft.mat     ← Horror game (actually Labyrinth)
    G4FerociousPlanet.mat← Joy game
  S02/ ... S28/

Each .mat file contains a variable named 'data'  with shape
(14, T)  where T = total EEG samples at 128 Hz.
"""

import os, glob, warnings
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path

# ── Try importing config; fall back to defaults ───────────────────────────────
try:
    from config import (
        DATASET_ROOT, SAMPLING_RATE, N_CHANNELS, CHANNEL_NAMES,
        GAME_FOLDERS, EMOTION_LABELS, N_SUBJECTS,
        WINDOW_SAMPLES, STEP_SAMPLES, RANDOM_SEED
    )
except ImportError:
    DATASET_ROOT   = "./GAMEEMO"
    SAMPLING_RATE  = 128
    N_CHANNELS     = 14
    CHANNEL_NAMES  = ["AF3","F7","F3","FC5","T7","P7","O1",
                      "O2","P8","T8","FC6","F4","F8","AF4"]
    GAME_FOLDERS   = ["G1","G2","G3","G4"]
    EMOTION_LABELS = {0:"Boring",1:"Calm",2:"Horror",3:"Joy"}
    N_SUBJECTS     = 28
    WINDOW_SAMPLES = 256
    STEP_SAMPLES   = 128
    RANDOM_SEED    = 42


# ──────────────────────────────────────────────────────────────────────────────
def _load_mat(filepath: str) -> np.ndarray:
    """
    Load GAMEEMO .mat file.

    Dataset stores each EEG channel separately:
    AF3, F7, F3, FC5, T7, P7, O1,
    O2, P8, T8, FC6, F4, F8, AF4

    Returns:
        ndarray shape (14, T)
    """

    mat = sio.loadmat(filepath)

    eeg_channels = []

    for ch in CHANNEL_NAMES:

        if ch not in mat:
            raise ValueError(f"Channel {ch} missing in {filepath}")

        signal = np.array(mat[ch]).squeeze()

        eeg_channels.append(signal)

    eeg = np.vstack(eeg_channels).astype(np.float32)

    return eeg

def _load_csv(filepath: str) -> np.ndarray:
    """Load EEG matrix from a .csv file → shape (14, T)."""
    df = pd.read_csv(filepath)
    # Drop non-numeric or label columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Keep only 14-channel columns
    ch_cols = [c for c in numeric_cols if c in CHANNEL_NAMES]
    if len(ch_cols) == N_CHANNELS:
        return df[ch_cols].values.T.astype(np.float32)
    # Last resort: first 14 numeric columns
    cols = list(numeric_cols)[:N_CHANNELS]
    return df[cols].values.T.astype(np.float32)


def _segment_signal(eeg: np.ndarray,
                    window: int = WINDOW_SAMPLES,
                    step:   int = STEP_SAMPLES) -> np.ndarray:
    """
    Sliding-window segmentation.
    Input:  eeg  shape (n_ch, T)
    Output: epochs shape (n_epochs, n_ch, window)
    """
    n_ch, T = eeg.shape
    starts  = range(0, T - window + 1, step)
    return np.stack([eeg[:, s:s+window] for s in starts], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
def load_subject(subject_dir: str):

    epochs_list = []
    labels_list = []

    subject_dir = Path(subject_dir)

    preprocessed = subject_dir / "Preprocessed EEG Data" / ".mat format"

    if not preprocessed.exists():
        raise RuntimeError(f"{preprocessed} not found")

    for label, game_folder in enumerate(GAME_FOLDERS):

        pattern = f"*{game_folder}*.mat"

        files = list(preprocessed.glob(pattern))

        if len(files) == 0:
            print("Missing", pattern)
            continue

        eeg = _load_mat(str(files[0]))

        segs = _segment_signal(eeg)

        epochs_list.append(segs)

        labels_list.append(
            np.full(len(segs), label, dtype=int)
        )

    X = np.concatenate(epochs_list)

    y = np.concatenate(labels_list)

    return X, y
def load_all_subjects(dataset_root=DATASET_ROOT, verbose=True):

    root = Path(dataset_root)

    subject_dirs = sorted(root.glob("(S*)"))

    X_all = []
    y_all = []
    sid_all = []

    for sid, folder in enumerate(subject_dirs):

        X, y = load_subject(folder)

        X_all.append(X)

        y_all.append(y)

        sid_all.append(
            np.full(len(y), sid)
        )

        if verbose:
            print(
                f"{folder.name} -> {X.shape[0]} epochs"
            )

    X = np.concatenate(X_all)

    y = np.concatenate(y_all)

    sids = np.concatenate(sid_all)

    print()

    print("Dataset Loaded Successfully")

    print("X :", X.shape)

    print("y :", y.shape)

    print("Subjects :", sids.shape)

    print("Classes :", np.bincount(y))

    return X, y, sids

# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_gameemo(n_subjects: int = 28,
                                srate: int = SAMPLING_RATE,
                                seed: int = RANDOM_SEED
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a realistic synthetic GAMEEMO dataset for testing / CI
    when the real dataset is not yet placed.  Each emotion class gets
    slightly different spectral profiles mimicking real EEG.
    """
    rng   = np.random.default_rng(seed)
    t     = np.linspace(0, WINDOW_SAMPLES / srate, WINDOW_SAMPLES)
    # Band power profiles per emotion (delta/theta/alpha/beta/gamma amplitudes)
    profiles = {
        0: [3.0, 2.0, 1.0, 0.5, 0.2],   # Boring  – high delta
        1: [1.0, 1.5, 3.5, 1.0, 0.3],   # Calm    – high alpha
        2: [1.5, 2.5, 1.0, 3.0, 1.0],   # Horror  – high beta/theta
        3: [1.0, 1.0, 1.5, 2.5, 2.0],   # Joy     – high beta/gamma
    }
    band_freqs = [2, 6, 10, 20, 37]      # representative Hz per band

    X_all, y_all, s_all = [], [], []
    n_epochs_per_game = 60

    for sid in range(n_subjects):
        for label, amps in profiles.items():
            for _ in range(n_epochs_per_game):
                epoch = np.zeros((N_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
                for ch in range(N_CHANNELS):
                    sig = sum(a * np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))
                              for a, f in zip(amps, band_freqs))
                    sig += rng.normal(0, 0.3, WINDOW_SAMPLES)
                    epoch[ch] = sig.astype(np.float32)
                X_all.append(epoch)
                y_all.append(label)
                s_all.append(sid)

    X = np.stack(X_all)
    y = np.array(y_all, dtype=int)
    sids = np.array(s_all, dtype=int)
    print(f"Synthetic GAMEEMO: {X.shape}, classes={np.bincount(y)}")
    return X, y, sids


if __name__ == "__main__":
    print("Testing data loader with synthetic data …")
    X, y, sids = generate_synthetic_gameemo(n_subjects=4)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

# =============================================================================
# MAIN (TEST LOADER)
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GAMEEMO DATA LOADER TEST")
    print("=" * 60)

    print(f"Dataset Path : {DATASET_ROOT}")

    if DATASET_ROOT is not None and os.path.exists(DATASET_ROOT):

        print("\nReal dataset detected.\n")

        try:

            X, y, sids = load_all_subjects(DATASET_ROOT)

            print("\nDataset loaded successfully.")

            print(f"\nX Shape          : {X.shape}")

            print(f"y Shape          : {y.shape}")

            print(f"Subjects         : {len(np.unique(sids))}")

            print(f"Class Distribution : {np.bincount(y)}")

        except Exception as e:

            print("\nFailed to load real dataset.")

            print(e)

            print("\nSwitching to synthetic dataset...\n")

            X, y, sids = generate_synthetic_gameemo(n_subjects=4)

            print(f"X Shape : {X.shape}")

            print(f"y Shape : {y.shape}")

    else:

        print("\nDataset folder not found.")

        print("Using synthetic dataset.\n")

        X, y, sids = generate_synthetic_gameemo(n_subjects=4)

        print(f"X Shape : {X.shape}")

        print(f"y Shape : {y.shape}")

    print("=" * 60)
