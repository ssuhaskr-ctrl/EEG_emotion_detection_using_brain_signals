"""
feature_extractor.py — Multi-Domain EEG Feature Extraction
============================================================
Extracts features from 4 domains:

  TIME-DOMAIN   : mean, variance, std, skewness, kurtosis,
                  RMS, zero-crossing rate, Hjorth parameters
  FREQ-DOMAIN   : FFT band powers, PSD (Welch), band power ratios
  NON-LINEAR    : Shannon entropy, approximate entropy (ApEn),
                  sample entropy (SampEn), Hurst exponent, DFA
  WAVELET       : Daubechies-4 decomposition energy per sub-band

All features are concatenated into a single 1-D feature vector per epoch.
"""

import numpy as np
import warnings
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis

try:
    import pywt          # PyWavelets
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    import antropy as ant
    HAS_ANT = True
except ImportError:
    HAS_ANT = False

try:
    from config import (SAMPLING_RATE, FREQ_BANDS, N_CHANNELS,
                        EXTRACT_TIME_DOMAIN, EXTRACT_FREQ_DOMAIN,
                        EXTRACT_NONLINEAR, EXTRACT_WAVELET)
except ImportError:
    SAMPLING_RATE        = 128
    N_CHANNELS           = 14
    EXTRACT_TIME_DOMAIN  = True
    EXTRACT_FREQ_DOMAIN  = True
    EXTRACT_NONLINEAR    = True
    EXTRACT_WAVELET      = True
    FREQ_BANDS = {
        "delta": (0.5,  4.0),
        "theta": (4.0,  8.0),
        "alpha": (8.0,  13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  TIME-DOMAIN FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def _hjorth(sig: np.ndarray) -> tuple[float, float, float]:
    """Hjorth activity, mobility, complexity."""
    activity  = float(np.var(sig))
    d1        = np.diff(sig)
    d2        = np.diff(d1)
    mob_num   = np.var(d1)
    mobility  = float(np.sqrt(mob_num / (activity + 1e-10)))
    complexity = float(
        np.sqrt(np.var(d2) / (mob_num + 1e-10)) / (mobility + 1e-10)
    )
    return activity, mobility, complexity


def time_domain_features(sig: np.ndarray) -> np.ndarray:
    """
    sig: 1-D array (n_samples,)
    Returns 11-element feature vector per channel.
    """
    mn    = float(np.mean(sig))
    var   = float(np.var(sig))
    std   = float(np.std(sig))
    sk    = float(skew(sig))
    kurt  = float(kurtosis(sig))
    rms   = float(np.sqrt(np.mean(sig**2)))
    p2p   = float(sig.max() - sig.min())
    zcr   = float(((sig[:-1] * sig[1:]) < 0).sum() / len(sig))
    act, mob, comp = _hjorth(sig)
    return np.array([mn, var, std, sk, kurt, rms, p2p, zcr, act, mob, comp],
                    dtype=np.float32)

TIME_FEAT_DIM = 11


# ──────────────────────────────────────────────────────────────────────────────
#  FREQUENCY-DOMAIN FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def freq_domain_features(sig: np.ndarray,
                          srate: int = SAMPLING_RATE) -> np.ndarray:
    """
    sig: 1-D (n_samples,)
    Returns (5 bands × 2) + ratios = 15-element vector per channel.
    """
    freqs, psd = sp_signal.welch(sig, fs=srate, nperseg=min(256, len(sig)))
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    band_powers = {}
    for band, (lo, hi) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        band_powers[band] = float(_trapz(psd[mask], freqs[mask]))

    bp   = list(band_powers.values())          # [d, t, a, b, g]
    total = sum(bp) + 1e-10

    # Relative powers (5) + absolute powers (5) + key ratios (5)
    rel   = [v / total for v in bp]
    ratios = [
        bp[2] / (bp[1] + 1e-10),              # alpha/theta
        bp[3] / (bp[2] + 1e-10),              # beta/alpha
        (bp[3]+bp[4]) / (bp[0]+bp[1]+1e-10),  # (beta+gamma)/(delta+theta) – engagement
        bp[1] / (bp[2] + 1e-10),              # theta/alpha – stress proxy
        bp[0] / (bp[3] + 1e-10),              # delta/beta
    ]

    return np.array(bp + rel + ratios, dtype=np.float32)

FREQ_FEAT_DIM = 15


# ──────────────────────────────────────────────────────────────────────────────
#  NON-LINEAR FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def _shannon_entropy(sig: np.ndarray, n_bins: int = 64) -> float:
    hist, _ = np.histogram(sig, bins=n_bins)
    p = hist / (hist.sum() + 1e-10)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _hurst(sig: np.ndarray) -> float:
    """Hurst exponent via R/S analysis (fast approximation)."""
    n = len(sig)
    if n < 20:
        return 0.5
    lags  = [4, 8, 16, 32, min(64, n//2)]
    rs    = []
    for lag in lags:
        ts = sig[:lag]
        mean_ts = ts.mean()
        cum_dev = np.cumsum(ts - mean_ts)
        R = cum_dev.max() - cum_dev.min()
        S = ts.std() + 1e-10
        rs.append(R / S)
    lags_log = np.log(lags)
    rs_log   = np.log(np.array(rs) + 1e-10)
    if np.any(np.isnan(rs_log)) or np.all(lags_log == lags_log[0]):
        return 0.5
    return float(np.polyfit(lags_log, rs_log, 1)[0])


def _approx_entropy(sig: np.ndarray, m: int = 2, r_coef: float = 0.2) -> float:
    """Fast approximate entropy."""
    n  = len(sig)
    r  = r_coef * sig.std()
    if n < m + 2 or r == 0:
        return 0.0
    def phi(m_):
        templates = np.array([sig[i:i+m_] for i in range(n-m_+1)])
        count = np.array([
            np.sum(np.abs(templates - templates[i]).max(axis=1) <= r)
            for i in range(len(templates))
        ])
        return np.log(count / (n - m_ + 1) + 1e-10).mean()
    return float(phi(m) - phi(m+1))


def nonlinear_features(sig: np.ndarray,
                        srate: int = SAMPLING_RATE) -> np.ndarray:
    """
    sig: 1-D (n_samples,)
    Returns 6-element vector per channel.
    """
    sh_ent  = _shannon_entropy(sig)
    hurst   = _hurst(sig)
    ap_en   = _approx_entropy(sig)

    if HAS_ANT:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:   samp_en = float(ant.sample_entropy(sig))
            except: samp_en = 0.0
            try:   perm_en = float(ant.perm_entropy(sig, normalize=True))
            except: perm_en = 0.0
            try:   svd_en  = float(ant.svd_entropy(sig, normalize=True))
            except: svd_en  = 0.0
    else:
        samp_en = perm_en = svd_en = 0.0

    return np.array([sh_ent, hurst, ap_en, samp_en, perm_en, svd_en],
                    dtype=np.float32)

NONLINEAR_FEAT_DIM = 6


# ──────────────────────────────────────────────────────────────────────────────
#  WAVELET FEATURES  (Daubechies-4, 4 levels)
# ──────────────────────────────────────────────────────────────────────────────
WAVELET_LEVELS = 4

def wavelet_features(sig: np.ndarray) -> np.ndarray:
    """
    sig: 1-D (n_samples,)
    Returns (levels+1)-element energy vector per channel.
    """
    if not HAS_PYWT:
        return np.zeros(WAVELET_LEVELS + 1, dtype=np.float32)

    coeffs = pywt.wavedec(sig, "db4", level=WAVELET_LEVELS)
    energies = np.array([float(np.sum(c**2)) for c in coeffs], dtype=np.float32)
    total = energies.sum() + 1e-10
    return energies / total   # relative energies

WAVELET_FEAT_DIM = WAVELET_LEVELS + 1   # 5


# ──────────────────────────────────────────────────────────────────────────────
#  COMBINED FEATURE VECTOR PER EPOCH
# ──────────────────────────────────────────────────────────────────────────────
def extract_epoch_features(epoch: np.ndarray,
                            srate: int = SAMPLING_RATE) -> np.ndarray:
    """
    epoch: (n_ch, n_samples)
    Returns flat feature vector for the epoch.
    """
    ch_feats = []
    for ch_sig in epoch:
        parts = []
        if EXTRACT_TIME_DOMAIN:
            parts.append(time_domain_features(ch_sig))
        if EXTRACT_FREQ_DOMAIN:
            parts.append(freq_domain_features(ch_sig, srate))
        if EXTRACT_NONLINEAR:
            parts.append(nonlinear_features(ch_sig, srate))
        if EXTRACT_WAVELET:
            parts.append(wavelet_features(ch_sig))
        ch_feats.append(np.concatenate(parts))

    # Inter-channel asymmetry (frontal alpha asymmetry F4-F3)
    # GAMEEMO channels: index 2=F3, index 11=F4
    # Calculate for all left-right pairs
    left_right_pairs = [(2,11),(4,9),(5,7),(6,7)]  # (F3,F4),(T7,T8),(P7,P8),(O1,O2)
    asym_feats = []
    for l, r in left_right_pairs:
        if l < epoch.shape[0] and r < epoch.shape[0]:
            asym = (epoch[r] - epoch[l]).mean()
            asym_feats.append(float(asym))

    return np.concatenate(ch_feats + [np.array(asym_feats, dtype=np.float32)])


def get_feature_dim() -> int:
    """Return total feature vector length."""
    per_ch = (TIME_FEAT_DIM  * EXTRACT_TIME_DOMAIN  +
              FREQ_FEAT_DIM  * EXTRACT_FREQ_DOMAIN  +
              NONLINEAR_FEAT_DIM * EXTRACT_NONLINEAR +
              WAVELET_FEAT_DIM   * EXTRACT_WAVELET)
    asym = 4
    return per_ch * N_CHANNELS + asym


def extract_features_batch(X: np.ndarray,
                            srate: int = SAMPLING_RATE,
                            verbose: bool = True) -> np.ndarray:
    """
    X: (n_epochs, n_ch, n_samples)
    Returns (n_epochs, feature_dim) float32
    """
    from tqdm import tqdm
    n = len(X)
    first = extract_epoch_features(X[0], srate)
    feat_dim = len(first)
    out = np.zeros((n, feat_dim), dtype=np.float32)
    out[0] = first

    iterator = tqdm(range(1, n), desc="Extracting features") if verbose else range(1, n)
    for i in iterator:
        out[i] = extract_epoch_features(X[i], srate)

    # Replace any NaN / Inf with 0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


if __name__ == "__main__":
    epoch = np.random.randn(14, 256).astype(np.float32)
    fv = extract_epoch_features(epoch)
    print(f"Feature vector dim: {len(fv)}")
    print(f"Expected dim: {get_feature_dim()}")
    print(f"Sample (first 20): {fv[:20]}")
