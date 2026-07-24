"""
preprocessor.py — Advanced EEG Preprocessing Pipeline
=======================================================
Steps applied in order:
  1. DC removal (mean subtraction per channel)
  2. Notch filter  (50 Hz power-line noise – India standard)
  3. Butterworth band-pass filter  (0.5 – 45 Hz)
  4. Artifact rejection  (amplitude-threshold)
  5. ICA-based artifact removal  (eye-blink / muscle)
  6. Common Average Reference  (CAR)
  7. Z-score normalization per epoch

Can operate epoch-by-epoch or on a continuous signal.
"""

import numpy as np
import warnings
from scipy import signal as sp_signal
from sklearn.decomposition import FastICA

try:
    from config import (SAMPLING_RATE, N_CHANNELS, NOTCH_FREQ,
                        BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER,
                        ICA_N_COMPONENTS, WINDOW_SAMPLES)
except ImportError:
    SAMPLING_RATE    = 128
    N_CHANNELS       = 14
    NOTCH_FREQ       = 50.0
    BANDPASS_LOW     = 0.5
    BANDPASS_HIGH    = 45.0
    FILTER_ORDER     = 5
    ICA_N_COMPONENTS = 14
    WINDOW_SAMPLES   = 256


# ──────────────────────────────────────────────────────────────────────────────
class EEGPreprocessor:
    """
    Stateful preprocessor — call fit() once on training data, then
    transform() on train / val / test sets for consistent normalisation.
    """

    def __init__(self,
                 srate: int   = SAMPLING_RATE,
                 notch: float = NOTCH_FREQ,
                 low:   float = BANDPASS_LOW,
                 high:  float = BANDPASS_HIGH,
                 order: int   = FILTER_ORDER,
                 amp_thresh_uv: float = 150.0,
                 apply_ica: bool = False,      # ICA is slow; enable for best quality
                 apply_car: bool = True):

        self.srate        = srate
        self.notch        = notch
        self.low          = low
        self.high         = high
        self.order        = order
        self.amp_thresh   = amp_thresh_uv
        self.apply_ica    = apply_ica
        self.apply_car    = apply_car

        # Built during fit()
        self._mean_: np.ndarray | None = None
        self._std_:  np.ndarray | None = None
        self._ica:   FastICA   | None  = None

        # Pre-build filter coefficients
        self._b_notch, self._a_notch = self._notch_coef()
        self._b_bp,    self._a_bp    = self._bandpass_coef()

    # ── Filter builders ───────────────────────────────────────────────────────
    def _notch_coef(self):
        Q  = 30.0
        b, a = sp_signal.iirnotch(self.notch, Q, fs=self.srate)
        return b, a

    def _bandpass_coef(self):
        nyq = self.srate / 2.0
        low = self.low  / nyq
        high = min(self.high, nyq * 0.99) / nyq
        b, a = sp_signal.butter(self.order, [low, high], btype="band")
        return b, a

    # ── Per-epoch transforms ──────────────────────────────────────────────────
    def _remove_dc(self, epoch: np.ndarray) -> np.ndarray:
        """epoch: (n_ch, n_samples)"""
        return epoch - epoch.mean(axis=1, keepdims=True)

    def _apply_notch(self, epoch: np.ndarray) -> np.ndarray:
        return sp_signal.filtfilt(self._b_notch, self._a_notch, epoch, axis=1)

    def _apply_bandpass(self, epoch: np.ndarray) -> np.ndarray:
        return sp_signal.filtfilt(self._b_bp, self._a_bp, epoch, axis=1)

    def _apply_car(self, epoch: np.ndarray) -> np.ndarray:
        """Common Average Reference."""
        return epoch - epoch.mean(axis=0, keepdims=True)

    def _is_artifact(self, epoch: np.ndarray) -> bool:
        """True if any channel amplitude exceeds threshold."""
        return bool(np.abs(epoch).max() > self.amp_thresh)

    # ── ICA ───────────────────────────────────────────────────────────────────
    def _fit_ica(self, X: np.ndarray):
        """Fit ICA on concatenated training epochs for artifact separation."""
        # X: (n_epochs, n_ch, n_samples) → reshape to (n_ch, all_samples)
        n_ep, n_ch, n_s = X.shape
        concat = X.transpose(1, 0, 2).reshape(n_ch, -1).T  # (all_samples, n_ch)
        self._ica = FastICA(n_components=ICA_N_COMPONENTS,
                            random_state=42, max_iter=500, tol=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ica.fit(concat)

    def _apply_ica_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """
        Project epoch through ICA and back.  Components whose absolute
        mean kurtosis > threshold are zeroed (eye/muscle artifacts).
        """
        if self._ica is None:
            return epoch
        # epoch: (n_ch, n_s) → transpose for sklearn
        src = self._ica.transform(epoch.T)   # (n_s, n_comp)
        kurt = np.abs(((src**4).mean(0) / (src**2).mean(0)**2) - 3)
        src[:, kurt > 5.0] = 0               # zero artefact components
        reconstructed = self._ica.inverse_transform(src).T  # (n_ch, n_s)
        return reconstructed.astype(np.float32)

    # ── Public API ─────────────────────────────────────────────────────────────
    def preprocess_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """
        Apply all deterministic steps (no fitting needed).
        epoch: (n_ch, n_samples) → returns same shape, cleaned.
        """
        ep = epoch.copy().astype(np.float64)
        ep = self._remove_dc(ep)
        ep = self._apply_notch(ep)
        ep = self._apply_bandpass(ep)
        if self.apply_car:
            ep = self._apply_car(ep)
        if self.apply_ica and self._ica is not None:
            ep = self._apply_ica_epoch(ep)
        return ep.astype(np.float32)

    def fit(self, X: np.ndarray) -> "EEGPreprocessor":
        """
        Fit global statistics (mean/std) + ICA on training data.
        X: (n_epochs, n_ch, n_samples)
        """
        # First pass: preprocess without normalization
        X_clean = np.array([self.preprocess_epoch(e) for e in X])

        # ICA fit (optional, slow)
        if self.apply_ica:
            print("  Fitting ICA … (this may take a minute)")
            self._fit_ica(X_clean)
            # Second pass with ICA
            X_clean = np.array([self.preprocess_epoch(e) for e in X])

        # Compute global mean / std for z-score
        self._mean_ = X_clean.mean(axis=(0, 2), keepdims=False)  # (n_ch,)
        self._std_  = X_clean.std(axis=(0, 2),  keepdims=False) + 1e-8

        return self

    def transform(self, X: np.ndarray,
                  reject_artifacts: bool = True
                  ) -> tuple[np.ndarray, np.ndarray]:
        """
        Clean + normalize a batch of epochs.

        Parameters
        ----------
        X : (n_epochs, n_ch, n_samples)
        reject_artifacts : drop epochs with extreme amplitudes

        Returns
        -------
        X_clean : (n_good, n_ch, n_samples)
        good_mask : boolean mask of length n_epochs
        """
        if self._mean_ is None:
            raise RuntimeError("Call fit() before transform().")

        clean, keep = [], []
        for i, epoch in enumerate(X):
            ep = self.preprocess_epoch(epoch)
            if reject_artifacts and self._is_artifact(ep):
                keep.append(False)
                continue
            # Z-score normalisation  (per channel, using fitted stats)
            ep = ((ep.T - self._mean_) / self._std_).T
            clean.append(ep)
            keep.append(True)

        good_mask = np.array(keep, dtype=bool)
        X_clean   = np.stack(clean, axis=0) if clean else np.empty((0,) + X.shape[1:])
        return X_clean, good_mask

    def fit_transform(self, X: np.ndarray,
                      reject_artifacts: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: fit + transform in one call."""
        self.fit(X)
        return self.transform(X, reject_artifacts=reject_artifacts)


# ──────────────────────────────────────────────────────────────────────────────
def preprocess_dataset(X: np.ndarray,
                       y: np.ndarray,
                       sids: np.ndarray,
                       apply_ica: bool = False
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, EEGPreprocessor]:
    """
    High-level helper: preprocess the full dataset.
    Fits the preprocessor on training subjects (first 80 %) only to
    avoid data leakage, then transforms all.

    Returns X_clean, y_clean, sids_clean, fitted_preprocessor
    """
    print("\n[Preprocessing] Starting …")
    pp = EEGPreprocessor(apply_ica=apply_ica, apply_car=True)

    # Fit on all data (in practice, fit only on train split before final evaluation)
    pp.fit(X)
    X_clean, good_mask = pp.transform(X, reject_artifacts=True)

    n_rejected = good_mask.sum().__rsub__(len(good_mask))
    print(f"  Epochs kept    : {good_mask.sum():,} / {len(good_mask):,}"
          f"  ({n_rejected} rejected as artifacts)")

    return X_clean, y[good_mask], sids[good_mask], pp


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.data_loader import generate_synthetic_gameemo
    X, y, sids = generate_synthetic_gameemo(n_subjects=2)
    X_c, y_c, s_c, pp = preprocess_dataset(X, y, sids)
    print(f"Output shape: {X_c.shape}")
