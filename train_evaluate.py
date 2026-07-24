"""
train_evaluate.py — Full Training & Evaluation Pipeline
========================================================
Supports:
  • Within-subject cross-validation (k-fold)
  • Cross-subject (Leave-One-Subject-Out) evaluation
  • Multi-model comparison  (EEGNet, DeepConvLSTM, TransformerEEG, MLP)
  • Confusion matrix, per-class metrics, learning curves
  • Grad-CAM for EEGNet saliency visualization

Run:
    python train_evaluate.py --model eegnet --eval within --subjects 5
"""

import os, sys, argparse, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import (N_CLASSES, EMOTION_LABELS, BATCH_SIZE, EPOCHS,
                    RESULTS_DIR, PLOTS_DIR, MODELS_DIR, FEATURES_DIR,
                    RANDOM_SEED, DATASET_ROOT)

from data.data_loader         import load_all_subjects, generate_synthetic_gameemo
from preprocessing.preprocessor import preprocess_dataset
from features.feature_extractor import extract_features_batch
from models.models             import (build_eegnet, build_deep_conv_lstm,
                                       build_transformer_eeg, build_dense_classifier,
                                       get_callbacks, prepare_for_model)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_NAMES = list(EMOTION_LABELS.values())   # ["Boring","Calm","Horror","Joy"]


# ──────────────────────────────────────────────────────────────────────────────
#  PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name, suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
            axes,
            [cm, cm_norm],
            ["d", ".2f"],
            ["Counts", "Normalized"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{model_name} – {title}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    plt.tight_layout()
    path = f"{PLOTS_DIR}/{model_name}_cm{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → {path}")


def plot_learning_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes,
                                  [("accuracy","val_accuracy"),
                                   ("loss","val_loss")],
                                  ["Accuracy","Loss"]):
        tr_key, val_key = metric
        if tr_key in history and val_key in history:
            ax.plot(history[tr_key],  label="Train")
            ax.plot(history[val_key], label="Val", linestyle="--")
            ax.set_title(f"{model_name} – {title}")
            ax.set_xlabel("Epoch"); ax.legend()

    plt.tight_layout()
    path = f"{PLOTS_DIR}/{model_name}_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved learning curves  → {path}")


def plot_class_distribution(y, title="Class Distribution"):
    counts = np.bincount(y, minlength=N_CLASSES)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_NAMES, counts, color=["#4C72B0","#DD8452","#55A868","#C44E52"])
    ax.set_title(title); ax.set_ylabel("Epoch Count")
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+10, str(c),
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_model_comparison(results_dict):
    """Bar chart comparing all models."""
    models   = list(results_dict.keys())
    accs     = [results_dict[m].get("mean_accuracy", 0) for m in models]
    f1s      = [results_dict[m].get("mean_f1", 0) for m in models]

    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy",  color="#4C72B0")
    b2 = ax.bar(x + w/2, f1s,  w, label="F1 (macro)",color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Model Comparison – GAMEEMO Emotion Detection")
    ax.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved model comparison → {path}")


# ──────────────────────────────────────────────────────────────────────────────
#  WITHIN-SUBJECT K-FOLD EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_within_subject(X_raw, y, sids,
                              model_name: str = "EEGNet",
                              n_folds: int = 5,
                              max_subjects: int | None = None):
    """
    Run k-fold cross-validation independently for each subject,
    then average results.
    """
    subject_ids = np.unique(sids)
    if max_subjects:
        subject_ids = subject_ids[:max_subjects]

    all_metrics = []
    histories   = []

    for sid in subject_ids:
        mask  = sids == sid
        X_s   = X_raw[mask]
        y_s   = y[mask]

        print(f"\n{'='*55}")
        print(f"  Subject {sid+1:02d} | {len(y_s)} epochs | "
              f"dist={np.bincount(y_s)}")
        print(f"{'='*55}")

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        fold_accs, fold_f1s = [], []
        fold_hist = None

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_s, y_s)):
            X_tr, X_val = X_s[tr_idx], X_s[val_idx]
            y_tr, y_val = y_s[tr_idx], y_s[val_idx]

            # Preprocess (fit on train only)
            from preprocessing.preprocessor import EEGPreprocessor
            pp = EEGPreprocessor()
            pp.fit(X_tr)
            X_tr,  _  = pp.transform(X_tr)
            X_val, gm = pp.transform(X_val, reject_artifacts=False)
            y_val_f = y_val   # keep all for evaluation

            # Build model
            model = _build_model(model_name, X_tr)
            X_tr_m  = prepare_for_model(X_tr, model_name)
            X_val_m = prepare_for_model(X_val, model_name)

            hist = model.fit(
                X_tr_m, y_tr,
                validation_data=(X_val_m, y_val),
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                callbacks=get_callbacks(f"{model_name}_s{sid}_f{fold}"),
                verbose=0
            )

            preds = model.predict(X_val_m, verbose=0).argmax(axis=1)
            acc = accuracy_score(y_val, preds)
            f1  = f1_score(y_val, preds, average="macro", zero_division=0)
            fold_accs.append(acc); fold_f1s.append(f1)
            if fold_hist is None: fold_hist = hist.history
            print(f"  Fold {fold+1}: acc={acc:.4f}  f1={f1:.4f}")

        subj_acc = np.mean(fold_accs)
        subj_f1  = np.mean(fold_f1s)
        print(f"  Subject avg: acc={subj_acc:.4f}  f1={subj_f1:.4f}")
        all_metrics.append({"subject": sid+1, "accuracy": subj_acc, "f1": subj_f1})
        if fold_hist: histories.append(fold_hist)

    df = pd.DataFrame(all_metrics)
    mean_acc = df["accuracy"].mean()
    mean_f1  = df["f1"].mean()
    std_acc  = df["accuracy"].std()

    print(f"\n{'='*55}")
    print(f"  {model_name} Within-Subject Results")
    print(f"  Mean Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean F1 Macro : {mean_f1:.4f}")
    print(f"{'='*55}")

    df.to_csv(f"{RESULTS_DIR}/{model_name}_within_subject.csv", index=False)
    if histories: plot_learning_curves(histories[0], f"{model_name}_within")

    return {"mean_accuracy": mean_acc, "mean_f1": mean_f1, "std_accuracy": std_acc,
            "per_subject": df.to_dict(orient="records")}


# ──────────────────────────────────────────────────────────────────────────────
#  CROSS-SUBJECT (LOSO) EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_cross_subject(X_raw, y, sids,
                             model_name: str = "EEGNet",
                             max_subjects: int | None = None):
    """Leave-One-Subject-Out cross-validation."""
    subject_ids = np.unique(sids)
    if max_subjects:
        subject_ids = subject_ids[:max_subjects]

    all_preds, all_true = [], []
    loso_metrics = []

    for test_sid in subject_ids:
        train_mask = sids != test_sid
        test_mask  = sids == test_sid

        X_tr  = X_raw[train_mask]; y_tr = y[train_mask]
        X_te  = X_raw[test_mask];  y_te = y[test_mask]

        print(f"\n  LOSO | Test subject {test_sid+1:02d} | "
              f"train={len(y_tr)} test={len(y_te)}")

        # Preprocess
        from preprocessing.preprocessor import EEGPreprocessor
        pp = EEGPreprocessor()
        pp.fit(X_tr)
        X_tr, _  = pp.transform(X_tr)
        X_te, _  = pp.transform(X_te, reject_artifacts=False)

        model = _build_model(model_name, X_tr)
        X_tr_m = prepare_for_model(X_tr, model_name)
        X_te_m = prepare_for_model(X_te, model_name)

        model.fit(X_tr_m, y_tr,
                  validation_split=0.1,
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=get_callbacks(f"{model_name}_loso_s{test_sid}"),
                  verbose=0)

        preds = model.predict(X_te_m, verbose=0).argmax(axis=1)
        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, average="macro", zero_division=0)
        loso_metrics.append({"subject": test_sid+1, "accuracy": acc, "f1": f1})
        all_preds.extend(preds.tolist()); all_true.extend(y_te.tolist())
        print(f"    acc={acc:.4f}  f1={f1:.4f}")

    all_preds = np.array(all_preds); all_true = np.array(all_true)
    df = pd.DataFrame(loso_metrics)
    mean_acc = df["accuracy"].mean(); mean_f1 = df["f1"].mean()

    print(f"\n  {model_name} LOSO  | acc={mean_acc:.4f}  f1={mean_f1:.4f}")
    plot_confusion_matrix(all_true, all_preds, model_name, suffix="_loso")
    df.to_csv(f"{RESULTS_DIR}/{model_name}_loso.csv", index=False)

    return {"mean_accuracy": mean_acc, "mean_f1": mean_f1,
            "per_subject": df.to_dict(orient="records")}


# ──────────────────────────────────────────────────────────────────────────────
#  ML BASELINE  (scikit-learn)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_ml_baseline(X_feat, y,
                           test_size: float = 0.2):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y, test_size=test_size,
                                                stratify=y, random_state=RANDOM_SEED)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

    results = {}
    for name, clf in [
        ("RandomForest",  RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED)),
        ("SVM_RBF",       SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_SEED)),
        ("GradientBoost", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED)),
    ]:
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, average="macro", zero_division=0)
        elapsed = time.time() - t0
        print(f"  {name:18s} acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)")
        results[name] = {"mean_accuracy": acc, "mean_f1": f1}
        plot_confusion_matrix(y_te, preds, name)

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  HELPER: build model by name
# ──────────────────────────────────────────────────────────────────────────────
def _build_model(model_name: str, X_sample: np.ndarray) -> keras.Model:
    n_ch, n_s = X_sample.shape[1], X_sample.shape[2]
    if model_name == "EEGNet":
        return build_eegnet(n_channels=n_ch, n_samples=n_s)
    elif model_name == "DeepConvLSTM":
        return build_deep_conv_lstm(n_channels=n_ch, n_samples=n_s)
    elif model_name == "TransformerEEG":
        return build_transformer_eeg(n_channels=n_ch, n_samples=n_s)
    elif model_name == "DenseMLP":
        return build_dense_classifier(X_sample.shape[-1])
    raise ValueError(f"Unknown model: {model_name}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GAMEEMO EEG Emotion Detection")
    parser.add_argument("--model",    default="EEGNet",
                        choices=["EEGNet","DeepConvLSTM","TransformerEEG","All"])
    parser.add_argument("--eval",     default="within",
                        choices=["within","loso","ml","all"])
    parser.add_argument("--subjects", type=int, default=None,
                        help="Limit number of subjects (for quick testing)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (real dataset not needed)")
    parser.add_argument("--ica",      action="store_true",
                        help="Apply ICA artifact removal (slow)")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  GAMEEMO EEG Emotion Detection System")
    print("═"*60)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.synthetic:
        print("\n[Data] Using synthetic GAMEEMO data …")
        X_raw, y, sids = generate_synthetic_gameemo(n_subjects=args.subjects or 8)
    else:
        try:
            print(f"\n[Data] Loading from: {DATASET_ROOT}")
            X_raw, y, sids = load_all_subjects(DATASET_ROOT)
        except FileNotFoundError as e:
            print(f"\n⚠  Real dataset not found: {e}")
            print("    Falling back to SYNTHETIC data for demonstration.\n"
                  "    Set GAMEEMO_ROOT env var to your dataset path.")
            X_raw, y, sids = generate_synthetic_gameemo(n_subjects=args.subjects or 6)

    if args.subjects:
        mask = sids < args.subjects
        X_raw, y, sids = X_raw[mask], y[mask], sids[mask]

    plot_class_distribution(y, "Class Distribution (full dataset)")

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_clean, y_clean, sids_clean, pp = preprocess_dataset(
        X_raw, y, sids, apply_ica=args.ica)

    # ── Feature extraction (for ML baselines) ─────────────────────────────────
    print("\n[Features] Extracting handcrafted features …")
    X_feat = extract_features_batch(X_clean, verbose=True)
    np.save(f"{FEATURES_DIR}/X_features.npy", X_feat)
    np.save(f"{FEATURES_DIR}/y_labels.npy", y_clean)
    print(f"  Feature matrix: {X_feat.shape}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    all_results = {}
    deep_models = (["EEGNet","DeepConvLSTM","TransformerEEG"]
                   if args.model == "All" else [args.model])

    if args.eval in ("ml", "all"):
        print("\n[Eval] ML Baselines (handcrafted features) …")
        ml_res = evaluate_ml_baseline(X_feat, y_clean)
        all_results.update(ml_res)

    if args.eval in ("within", "all"):
        for mn in deep_models:
            print(f"\n[Eval] Within-subject 5-fold CV — {mn}")
            res = evaluate_within_subject(X_clean, y_clean, sids_clean,
                                           model_name=mn,
                                           max_subjects=args.subjects)
            all_results[mn] = res

    if args.eval == "loso":
        for mn in deep_models:
            print(f"\n[Eval] LOSO cross-subject — {mn}")
            res = evaluate_cross_subject(X_clean, y_clean, sids_clean,
                                          model_name=mn,
                                          max_subjects=args.subjects)
            all_results[mn] = res

    # ── Summary ───────────────────────────────────────────────────────────────
    plot_all_model_comparison(all_results)

    print("\n" + "═"*60)
    print("  FINAL RESULTS SUMMARY")
    print("═"*60)
    for model_nm, metrics in all_results.items():
        acc = metrics.get("mean_accuracy", 0)
        f1  = metrics.get("mean_f1", 0)
        std = metrics.get("std_accuracy", 0)
        print(f"  {model_nm:20s}  Acc={acc:.4f}±{std:.4f}  F1={f1:.4f}")

    with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  All results saved → {RESULTS_DIR}/")
    print("  Plots saved       → {PLOTS_DIR}/")
    print("  Models saved      → {MODELS_DIR}/")


if __name__ == "__main__":
    main()
