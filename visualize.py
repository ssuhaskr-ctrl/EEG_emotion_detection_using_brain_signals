"""
visualize.py — EEG Signal & Results Visualization
==================================================
Standalone visualizer — run after training to generate analysis plots.

Usage:
    python visualize.py --data synthetic
    python visualize.py --data real
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp_signal

sys.path.insert(0, os.path.dirname(__file__))

try:
    from config import (SAMPLING_RATE, FREQ_BANDS, CHANNEL_NAMES,
                        EMOTION_LABELS, PLOTS_DIR, N_CLASSES)
except ImportError:
    SAMPLING_RATE  = 128
    FREQ_BANDS = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),
                  "beta":(13,30),"gamma":(30,45)}
    CHANNEL_NAMES  = ["AF3","F7","F3","FC5","T7","P7","O1",
                      "O2","P8","T8","FC6","F4","F8","AF4"]
    EMOTION_LABELS = {0:"Boring",1:"Calm",2:"Horror",3:"Joy"}
    PLOTS_DIR      = "./results/plots"
    N_CLASSES      = 4

os.makedirs(PLOTS_DIR, exist_ok=True)
CLASS_NAMES  = list(EMOTION_LABELS.values())
BAND_NAMES   = list(FREQ_BANDS.keys())
COLORS       = ["#4C72B0","#DD8452","#55A868","#C44E52","#9B59B6"]
CLASS_COLORS = ["#3498db","#2ecc71","#e74c3c","#f39c12"]


# ──────────────────────────────────────────────────────────────────────────────
def plot_raw_eeg(epoch: np.ndarray, label: int, title_prefix: str = ""):
    """
    Plot all 14 channels of a single epoch in a waterfall layout.
    epoch: (14, 256)
    """
    n_ch, n_s = epoch.shape
    t = np.arange(n_s) / SAMPLING_RATE * 1000   # ms

    fig, ax = plt.subplots(figsize=(14, 9))
    offset_step = np.abs(epoch).max() * 2.5

    for ch in range(n_ch):
        offset = (n_ch - ch) * offset_step
        ax.plot(t, epoch[ch] + offset, linewidth=0.8,
                color=COLORS[ch % len(COLORS)], alpha=0.85)
        ax.text(-15, offset, CHANNEL_NAMES[ch], ha="right", va="center",
                fontsize=8, fontweight="bold")

    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Channels")
    ax.set_title(f"{title_prefix}EEG – Emotion: {EMOTION_LABELS[label]}",
                 fontsize=13, fontweight="bold")
    ax.set_yticks([]); ax.spines[["left","right","top"]].set_visible(False)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/eeg_raw_{EMOTION_LABELS[label]}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved raw EEG plot → {path}")


def plot_psd_per_class(X: np.ndarray, y: np.ndarray):
    """
    Power Spectral Density averaged across channels for each emotion class.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, CLASS_NAMES)):
        mask    = y == cls_idx
        X_cls   = X[mask]                          # (n, 14, 256)
        psd_all = []

        for epoch in X_cls:
            for ch_sig in epoch:
                freqs, psd = sp_signal.welch(ch_sig, fs=SAMPLING_RATE, nperseg=128)
                psd_all.append(psd)

        psd_mean = np.mean(psd_all, axis=0)
        psd_std  = np.std(psd_all, axis=0)

        ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std,
                         alpha=0.25, color=CLASS_COLORS[cls_idx])
        ax.semilogy(freqs, psd_mean, color=CLASS_COLORS[cls_idx],
                    linewidth=2, label=cls_name)

        # Band markers
        for band, (lo, hi) in FREQ_BANDS.items():
            ax.axvspan(lo, hi, alpha=0.07, label=band)

        ax.set_title(cls_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (µV²/Hz)")
        ax.set_xlim(0, 45)
        if cls_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Power Spectral Density per Emotion Class", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/psd_per_class.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved PSD plot → {path}")


def plot_band_power_heatmap(X: np.ndarray, y: np.ndarray):
    """
    Heatmap: mean band power per channel × per emotion class.
    """
    n_bands = len(FREQ_BANDS); n_ch = 14
    # matrix: (n_classes, n_bands, n_ch)
    bp_matrix = np.zeros((N_CLASSES, n_bands, n_ch))

    for cls_idx in range(N_CLASSES):
        mask  = y == cls_idx
        X_cls = X[mask]
        for b_idx, (band, (lo, hi)) in enumerate(FREQ_BANDS.items()):
            for ch in range(n_ch):
                ch_psd = []
                for epoch in X_cls:
                    f, p = sp_signal.welch(epoch[ch], fs=SAMPLING_RATE, nperseg=128)
                    fmask = (f >= lo) & (f <= hi)
                    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
                    ch_psd.append(_trapz(p[fmask], f[fmask]))
                bp_matrix[cls_idx, b_idx, ch] = np.mean(ch_psd)

    fig, axes = plt.subplots(1, N_CLASSES, figsize=(18, 5), sharey=True)
    for cls_idx, (ax, cls_name) in enumerate(zip(axes, CLASS_NAMES)):
        import seaborn as sns
        data = bp_matrix[cls_idx]   # (5 bands, 14 ch)
        vmax = bp_matrix.max()
        sns.heatmap(data, ax=ax, cmap="YlOrRd",
                    xticklabels=CHANNEL_NAMES, yticklabels=BAND_NAMES,
                    vmin=0, vmax=vmax, cbar=(cls_idx==N_CLASSES-1))
        ax.set_title(cls_name, fontsize=11, fontweight="bold")
        ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=7)

    fig.suptitle("Mean Band Power per Channel × Emotion (µV²/Hz)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/band_power_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved band-power heatmap → {path}")


def plot_feature_importance_from_model(model_path: str, X_feat: np.ndarray,
                                        feature_names: list | None = None):
    """
    Extract and plot feature importances from a saved RandomForest model.
    """
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    try:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1][:30]
        labels = [feature_names[i] if feature_names else f"F{i}" for i in idx]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(30), importances[idx], color="#4C72B0")
        ax.set_xticks(range(30)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title("Top-30 Feature Importances (RandomForest)")
        plt.tight_layout()
        path = f"{PLOTS_DIR}/feature_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved feature importance → {path}")
    except Exception as e:
        print(f"  Feature importance plot skipped: {e}")


def plot_tsne_features(X_feat: np.ndarray, y: np.ndarray, n_samples: int = 2000):
    """
    t-SNE 2-D projection of handcrafted features, colored by emotion.
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # Subsample for speed
    idx = np.random.choice(len(X_feat), min(n_samples, len(X_feat)), replace=False)
    X_s = StandardScaler().fit_transform(X_feat[idx])
    y_s = y[idx]

    print("  Running t-SNE (may take ~30 s) …")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                random_state=42)
    emb  = tsne.fit_transform(X_s)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y_s == cls_idx
        ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.6,
                   color=CLASS_COLORS[cls_idx], label=cls_name)
    ax.legend(markerscale=3); ax.set_title("t-SNE of EEG Features by Emotion")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/tsne_features.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved t-SNE plot → {path}")


def run_all_visualizations(X: np.ndarray, y: np.ndarray,
                             X_feat: np.ndarray | None = None):
    print("\n[Visualize] Generating analysis plots …")
    # Raw EEG per class
    for cls_idx in range(N_CLASSES):
        mask = y == cls_idx
        if mask.sum() > 0:
            plot_raw_eeg(X[mask][0], cls_idx)

    plot_psd_per_class(X, y)
    plot_band_power_heatmap(X, y)

    if X_feat is not None:
        plot_tsne_features(X_feat, y)

    print(f"\n  All plots saved to → {PLOTS_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="synthetic",
                        choices=["synthetic", "real"])
    args = parser.parse_args()

    if args.data == "synthetic":
        from data.data_loader import generate_synthetic_gameemo
        X, y, sids = generate_synthetic_gameemo(n_subjects=6)
    else:
        from data.data_loader import load_all_subjects
        X, y, sids = load_all_subjects()

    from features.feature_extractor import extract_features_batch
    X_feat = extract_features_batch(X[:500], verbose=True)
    run_all_visualizations(X, y, X_feat)
