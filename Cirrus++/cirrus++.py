"""
CIRRUS++ / VLDB-Upgrade Prototype
Interactive Streamlit app for domain-aware, quality-index driven preprocessing recommendations
for eye-tracking and related behavioral time-series data.

Run:
    streamlit run cirrus_vldb_upgrade.py
"""

from __future__ import annotations

import io
import json
import math
import re
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from scipy.stats import skew, kurtosis, entropy, median_abs_deviation
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.fft import fft, ifft

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import make_scorer, f1_score, accuracy_score, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CIRRUS++ | Domain-aware Eye-Tracking Preprocessing",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.cirrus-hero {
    border-radius: 22px;
    padding: 1.2rem 1.35rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
    color: white;
    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.18);
    margin-bottom: 1rem;
}
.cirrus-hero h1 {margin-bottom: 0.2rem; font-size: 2.0rem;}
.cirrus-hero p {margin-bottom: 0; color: #cbd5e1;}
.small-card {
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    background: rgba(248, 250, 252, 0.75);
}
.reco-box {
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid rgba(59, 130, 246, 0.25);
    background: rgba(239, 246, 255, 0.75);
}
.warn-box {
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid rgba(245, 158, 11, 0.35);
    background: rgba(255, 251, 235, 0.9);
}
.bad-box {
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid rgba(239, 68, 68, 0.35);
    background: rgba(254, 242, 242, 0.9);
}
.good-box {
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid rgba(34, 197, 94, 0.35);
    background: rgba(240, 253, 244, 0.9);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="cirrus-hero">
<h1>👁️ CIRRUS++</h1>
<p>Domain-aware preprocessing assistant with quality index, adaptive recommendation scoring, multi-feature profiling, pipeline comparison, and optional downstream validation.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Domain profiles
# -----------------------------------------------------------------------------
DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "Auto-detect": {
        "description": "Infer from feature names and quality structure. Still inspect the result manually.",
        "weights": {"completeness": 0.25, "continuity": 0.20, "plausibility": 0.20, "distribution": 0.15, "stability": 0.15, "correlation": 0.05},
        "preferred_mv": ["Linear Interpolation", "LOCF", "KNN Imputation", "MICE"],
        "preferred_outlier": ["MAD", "IQR", "Isolation Forest"],
        "preferred_norm": ["RobustScaler", "StandardScaler", "MinMaxScaler"],
        "smoothing_default": "Savitzky-Golay",
        "risk_note": "Auto-detection is a convenience layer, not a scientific claim.",
    },
    "Clinical / diagnostic pupil data": {
        "description": "Often cleaner, session-controlled, and pupil-centered. Preserve amplitude and avoid unnecessary aggressive cleaning.",
        "weights": {"completeness": 0.22, "continuity": 0.18, "plausibility": 0.18, "distribution": 0.14, "stability": 0.20, "correlation": 0.08},
        "preferred_mv": ["LOCF", "Linear Interpolation", "KNN Imputation"],
        "preferred_outlier": ["Z-Score", "IQR", "MAD"],
        "preferred_norm": ["StandardScaler", "RobustScaler", "MinMaxScaler"],
        "smoothing_default": "Butterworth",
        "risk_note": "Do not erase clinically meaningful pupil variation by over-smoothing.",
    },
    "Academic reading / cognitive task": {
        "description": "Task-dependent gaze shifts, blink clusters, line jumps, and skewed gaze-position distributions are common.",
        "weights": {"completeness": 0.24, "continuity": 0.24, "plausibility": 0.16, "distribution": 0.16, "stability": 0.12, "correlation": 0.08},
        "preferred_mv": ["Linear Interpolation", "KNN Imputation", "MICE", "LOCF"],
        "preferred_outlier": ["MAD", "IQR", "Isolation Forest"],
        "preferred_norm": ["RobustScaler", "MinMaxScaler", "StandardScaler"],
        "smoothing_default": "Savitzky-Golay",
        "risk_note": "Missingness pattern matters: blink clusters are not the same as random dropout.",
    },
    "High-frequency lab tracking": {
        "description": "Potentially millions of rows. Sampling rate, chunked previews, and temporal structure become central.",
        "weights": {"completeness": 0.18, "continuity": 0.25, "plausibility": 0.15, "distribution": 0.12, "stability": 0.22, "correlation": 0.08},
        "preferred_mv": ["Linear Interpolation", "LOCF", "KNN Imputation"],
        "preferred_outlier": ["MAD", "Isolation Forest", "IQR"],
        "preferred_norm": ["RobustScaler", "StandardScaler", "MinMaxScaler"],
        "smoothing_default": "Butterworth",
        "risk_note": "Use sampling-aware filters. Otherwise the UI may look good while the signal is wrong.",
    },
    "Web/mobile remote tracking": {
        "description": "Usually noisier and less controlled. Calibration drift, variable FPS, and missing bursts are expected.",
        "weights": {"completeness": 0.26, "continuity": 0.22, "plausibility": 0.18, "distribution": 0.12, "stability": 0.16, "correlation": 0.06},
        "preferred_mv": ["KNN Imputation", "Linear Interpolation", "MICE"],
        "preferred_outlier": ["Isolation Forest", "MAD", "IQR"],
        "preferred_norm": ["RobustScaler", "MinMaxScaler", "StandardScaler"],
        "smoothing_default": "Savitzky-Golay",
        "risk_note": "Expect heterogeneity. The system should flag, not hide, poor recording quality.",
    },
    "Multimodal behavioral / sensor fusion": {
        "description": "Cross-feature relations matter. KNN/MICE and correlation checks become more useful than univariate-only cleaning.",
        "weights": {"completeness": 0.20, "continuity": 0.16, "plausibility": 0.16, "distribution": 0.12, "stability": 0.16, "correlation": 0.20},
        "preferred_mv": ["KNN Imputation", "MICE", "Linear Interpolation"],
        "preferred_outlier": ["Isolation Forest", "MAD", "IQR"],
        "preferred_norm": ["RobustScaler", "StandardScaler", "MinMaxScaler"],
        "smoothing_default": "None",
        "risk_note": "Feature-wise processing alone is weak here. Use multivariate checks.",
    },
}

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class FeatureProfile:
    feature: str
    n: int
    missing_rate: float
    longest_missing_run: int
    missing_burstiness: float
    outlier_iqr_rate: float
    outlier_mad_rate: float
    skewness: float
    kurtosis_excess: float
    zero_variance: bool
    unique_ratio: float
    drift_score: float
    volatility_score: float
    entropy_score: float
    corr_mean_abs: float
    quality_index: float
    quality_label: str

@dataclass
class PipelineRecommendation:
    missing_method: str
    outlier_method: str
    normalization_method: str
    smoothing_method: str
    confidence: float
    expected_gain: str
    rationale: List[str]
    risk_flags: List[str]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
MISSING_TOKENS = [
    "", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "-", "--", "---", "?",
    "nan", "NaN", "#VALUE!", "Inf", "-Inf", "infinity", "Infinity"
]

ID_PATTERNS = [
    r"^Unnamed", r"^id$", r"participant", r"subject", r"trial", r"session",
    r"timestamp", r"recordingtime", r"export",
    # synthetic/index-like helper columns should not be treated as signals
    r"^row$", r"^index$", r"^sample$", r"^sample_?id$", r"^frame$"
]


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x) or np.isinf(x):
            return default
        return float(x)
    except Exception:
        return default


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def auto_convert_numeric_columns(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(MISSING_TOKENS, np.nan)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype(str).str.strip().replace(MISSING_TOKENS, np.nan)
            s_clean = (
                s.str.replace("\u00A0", "", regex=False)
                 .str.replace(" ", "", regex=False)
                 .str.replace(",", ".", regex=False)
                 .str.replace(r"(?<=\d)([a-zA-Z%]+)$", "", regex=True)
            )
            converted = pd.to_numeric(s_clean, errors="coerce")
            mask = s.notna()
            if mask.sum() == 0:
                continue
            if converted[mask].notna().mean() >= threshold:
                df[col] = converted
    return df


def candidate_numeric_features(df: pd.DataFrame, include_ids: bool = False) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if include_ids:
        return cols
    filtered = []
    for col in cols:
        if any(re.search(p, col, re.IGNORECASE) for p in ID_PATTERNS):
            continue
        filtered.append(col)
    return filtered


def infer_domain(df: pd.DataFrame, features: List[str]) -> str:
    names = " ".join(features).lower()
    n = len(df)
    has_pupil = any(k in names for k in ["pupil", "diameter", "pupill"])
    has_reading = any(k in names for k in ["reading", "word", "sentence", "line", "fixation", "saccade", "gaze x", "gaze_x"])
    has_remote = any(k in names for k in ["web", "browser", "mobile", "fps", "screen"])
    has_multi = len(features) >= 8 or any(k in names for k in ["eda", "eeg", "hr", "heart", "accelerometer", "imu"])
    if n > 500_000:
        return "High-frequency lab tracking"
    if has_multi:
        return "Multimodal behavioral / sensor fusion"
    if has_remote:
        return "Web/mobile remote tracking"
    if has_reading:
        return "Academic reading / cognitive task"
    if has_pupil:
        return "Clinical / diagnostic pupil data"
    return "Academic reading / cognitive task"


def longest_true_run(mask: pd.Series) -> int:
    if len(mask) == 0:
        return 0
    arr = mask.astype(bool).to_numpy()
    best = cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def missing_burstiness(mask: pd.Series) -> float:
    """0=random/none-ish, 1=missingness concentrated in long bursts."""
    m = mask.astype(bool).to_numpy()
    total = int(m.sum())
    if total == 0:
        return 0.0
    runs = []
    cur = 0
    for v in m:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    if not runs:
        return 0.0
    # concentration: mean run length normalized by total missing count and sequence length
    return clamp((np.mean(runs) / max(1, len(m))) * 8 + (max(runs) / max(1, total)) * 0.35)


def outlier_iqr_mask(s: pd.Series, factor: float = 1.5) -> pd.Series:
    x = s.dropna()
    mask = pd.Series(False, index=s.index)
    if len(x) < 4:
        return mask
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= 0 or np.isnan(iqr):
        return mask
    return (s < q1 - factor * iqr) | (s > q3 + factor * iqr)


def outlier_mad_mask(s: pd.Series, thresh: float = 3.5) -> pd.Series:
    x = s.dropna()
    mask = pd.Series(False, index=s.index)
    if len(x) < 4:
        return mask
    med = np.median(x)
    mad = median_abs_deviation(x, nan_policy="omit")
    if mad <= 0 or np.isnan(mad):
        return mask
    robust_z = 0.6745 * (s - med) / mad
    return np.abs(robust_z) > thresh


def drift_score(s: pd.Series, window_ratio: float = 0.08) -> float:
    x = s.dropna().astype(float)
    if len(x) < 30 or x.std() == 0 or np.isnan(x.std()):
        return 0.0
    window = max(10, int(len(x) * window_ratio))
    rolling = x.rolling(window=window, min_periods=max(5, window // 3)).mean()
    if rolling.dropna().empty:
        return 0.0
    # normalized range of rolling mean; high means drift/non-stationarity
    return clamp((rolling.max() - rolling.min()) / (4 * x.std()))


def volatility_score(s: pd.Series) -> float:
    x = s.dropna().astype(float)
    if len(x) < 5 or x.std() == 0 or np.isnan(x.std()):
        return 0.0
    diff_std = x.diff().dropna().std()
    return clamp(diff_std / (3 * x.std()))


def entropy_score(s: pd.Series, bins: int = 30) -> float:
    x = s.dropna().astype(float)
    if len(x) < 10 or x.nunique() <= 1:
        return 0.0
    hist, _ = np.histogram(x, bins=min(bins, max(5, int(np.sqrt(len(x))))), density=False)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return clamp(float(entropy(hist) / np.log(len(hist))))


def quality_label(score: float) -> str:
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Usable / inspect"
    if score >= 40:
        return "Risky"
    return "Critical"


def compute_quality_index(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    missing_rate = metrics["missing_rate"]
    longest_run_ratio = metrics["longest_missing_run"] / max(1, metrics["n"])
    burst = metrics["missing_burstiness"]
    outlier_rate = max(metrics["outlier_iqr_rate"], metrics["outlier_mad_rate"])
    sk = abs(metrics["skewness"])
    ku = max(0.0, metrics["kurtosis_excess"])
    drift = metrics["drift_score"]
    vol = metrics["volatility_score"]
    corr = metrics.get("corr_mean_abs", 0.0)
    unique_ratio = metrics["unique_ratio"]

    completeness = 100 * (1 - clamp(missing_rate / 45.0))
    continuity = 100 * (1 - clamp(0.65 * burst + 0.35 * (longest_run_ratio * 10)))
    plausibility = 100 * (1 - clamp(outlier_rate / 25.0))
    distribution = 100 * (1 - clamp((sk / 3.0) * 0.65 + (ku / 8.0) * 0.35))
    stability = 100 * (1 - clamp(0.60 * drift + 0.40 * vol))
    # correlation is not always required; this rewards some useful multivariate structure but does not punish independence too hard.
    correlation = 100 * clamp(0.45 + corr * 0.80)

    if unique_ratio < 0.01:
        distribution *= 0.55
        stability *= 0.75

    total_weight = sum(weights.values()) or 1.0
    score = (
        weights.get("completeness", 0) * completeness
        + weights.get("continuity", 0) * continuity
        + weights.get("plausibility", 0) * plausibility
        + weights.get("distribution", 0) * distribution
        + weights.get("stability", 0) * stability
        + weights.get("correlation", 0) * correlation
    ) / total_weight
    return round(clamp(score / 100) * 100, 1)


def profile_features(df: pd.DataFrame, features: List[str], weights: Dict[str, float]) -> pd.DataFrame:
    corr = df[features].corr(numeric_only=True).abs() if len(features) > 1 else pd.DataFrame(index=features, columns=features)
    profiles = []
    for f in features:
        s = df[f]
        x = s.dropna().astype(float)
        iqr_mask = outlier_iqr_mask(s)
        mad_mask = outlier_mad_mask(s)
        corr_mean = 0.0
        if len(features) > 1 and f in corr.index:
            vals = corr.loc[f].drop(labels=[f], errors="ignore").replace([np.inf, -np.inf], np.nan).dropna()
            corr_mean = safe_float(vals.mean(), 0.0) if len(vals) else 0.0
        metrics = {
            "feature": f,
            "n": len(s),
            "missing_rate": safe_float(s.isna().mean() * 100),
            "longest_missing_run": longest_true_run(s.isna()),
            "missing_burstiness": missing_burstiness(s.isna()),
            "outlier_iqr_rate": safe_float(iqr_mask.mean() * 100),
            "outlier_mad_rate": safe_float(mad_mask.mean() * 100),
            "skewness": safe_float(skew(x), 0.0) if len(x) > 3 else 0.0,
            "kurtosis_excess": safe_float(kurtosis(x, fisher=True), 0.0) if len(x) > 3 else 0.0,
            "zero_variance": bool(len(x) < 2 or x.std() == 0 or np.isnan(x.std())),
            "unique_ratio": safe_float(s.nunique(dropna=True) / max(1, len(s))),
            "drift_score": drift_score(s),
            "volatility_score": volatility_score(s),
            "entropy_score": entropy_score(s),
            "corr_mean_abs": corr_mean,
        }
        qi = compute_quality_index(metrics, weights)
        metrics["quality_index"] = qi
        metrics["quality_label"] = quality_label(qi)
        profiles.append(metrics)
    return pd.DataFrame(profiles)


def recommend_for_profile(row: pd.Series, domain: str, aggressive: float = 0.50) -> PipelineRecommendation:
    """Adaptive scoring: not a lookup table; each candidate receives a score from metrics + domain priors."""
    profile = DOMAIN_PROFILES[domain]
    mv_candidates = ["None", "Mean Imputation", "LOCF", "Linear Interpolation", "KNN Imputation", "MICE"]
    out_candidates = ["None", "Z-Score", "IQR", "MAD", "Winsorization", "Isolation Forest"]
    norm_candidates = ["None", "StandardScaler", "RobustScaler", "MinMaxScaler"]
    smooth_candidates = ["None", "Butterworth", "Savitzky-Golay", "Fourier"]

    m = float(row["missing_rate"])
    burst = float(row["missing_burstiness"])
    c = max(float(row["outlier_iqr_rate"]), float(row["outlier_mad_rate"]))
    sk = abs(float(row["skewness"]))
    ku = max(0.0, float(row["kurtosis_excess"]))
    drift = float(row["drift_score"])
    vol = float(row["volatility_score"])
    corr = float(row.get("corr_mean_abs", 0.0))

    def prior_score(method: str, preferred: List[str]) -> float:
        if method in preferred:
            return 1.0 - 0.12 * preferred.index(method)
        return 0.35

    mv_scores = {}
    for method in mv_candidates:
        score = 0.0
        if method == "None":
            score = 1.1 if m < 0.5 else max(0.0, 0.35 - m / 40)
        elif method == "Mean Imputation":
            score = (1 - clamp(m / 12)) * (1 - 0.55 * burst) * (1 - 0.25 * corr)
        elif method == "LOCF":
            score = (1 - clamp(m / 18)) * (0.75 + 0.25 * burst) * (1 - 0.20 * drift)
        elif method == "Linear Interpolation":
            score = (1 - clamp(m / 25)) * (0.65 + 0.35 * burst) * (1 - 0.30 * drift)
        elif method == "KNN Imputation":
            score = (1 - clamp((m - 5) / 45)) * (0.50 + 0.70 * corr) * (1 - 0.15 * burst)
        elif method == "MICE":
            score = (1 - clamp((m - 8) / 45)) * (0.45 + 0.70 * corr) * (0.75 + 0.25 * aggressive)
        score = 0.78 * score + 0.22 * prior_score(method, profile["preferred_mv"])
        mv_scores[method] = score

    out_scores = {}
    for method in out_candidates:
        score = 0.0
        if method == "None":
            score = 1.0 if c < 0.5 else max(0.0, 0.40 - c / 30)
        elif method == "Z-Score":
            score = (1 - clamp(c / 15)) * (1 - clamp(sk / 1.5)) * (1 - clamp(ku / 4))
        elif method == "IQR":
            score = (0.55 + clamp(c / 15) * 0.40) * (1 - clamp(sk / 3) * 0.25)
        elif method == "MAD":
            score = (0.50 + clamp(c / 20) * 0.40) * (0.75 + clamp(sk / 2) * 0.35)
        elif method == "Winsorization":
            score = (0.45 + clamp(c / 20) * 0.35) * (1 - 0.20 * aggressive)
        elif method == "Isolation Forest":
            score = (0.35 + clamp(c / 25) * 0.55 + clamp(corr) * 0.15) * (0.75 + 0.25 * aggressive)
        score = 0.78 * score + 0.22 * prior_score(method, profile["preferred_outlier"])
        out_scores[method] = score

    norm_scores = {}
    for method in norm_candidates:
        score = 0.0
        if method == "None":
            score = 0.35 if sk < 0.5 and c < 1 and not row["zero_variance"] else 0.10
        elif method == "StandardScaler":
            score = (1 - clamp(sk / 2.5)) * (1 - clamp(c / 20)) * (1 - clamp(ku / 6) * 0.45)
        elif method == "RobustScaler":
            score = 0.55 + clamp(sk / 2.0) * 0.25 + clamp(c / 20) * 0.25 + clamp(ku / 6) * 0.15
        elif method == "MinMaxScaler":
            score = 0.58 + (1 - clamp(c / 20)) * 0.16 + (1 - clamp(sk / 3)) * 0.10
        score = 0.78 * score + 0.22 * prior_score(method, profile["preferred_norm"])
        norm_scores[method] = score

    smooth_scores = {}
    for method in smooth_candidates:
        if method == "None":
            score = 0.80 if vol < 0.25 and drift < 0.25 else 0.35
        elif method == "Butterworth":
            score = 0.45 + vol * 0.35 + (domain == "High-frequency lab tracking") * 0.18
        elif method == "Savitzky-Golay":
            score = 0.42 + vol * 0.25 + (domain in ["Academic reading / cognitive task", "Web/mobile remote tracking"]) * 0.20
        else:  # Fourier
            score = 0.30 + vol * 0.15
        if method == profile["smoothing_default"]:
            score += 0.12
        smooth_scores[method] = score

    missing_method = max(mv_scores, key=mv_scores.get)
    outlier_method = max(out_scores, key=out_scores.get)
    norm_method = max(norm_scores, key=norm_scores.get)
    smoothing_method = max(smooth_scores, key=smooth_scores.get)

    all_top_scores = [mv_scores[missing_method], out_scores[outlier_method], norm_scores[norm_method], smooth_scores[smoothing_method]]
    confidence = round(clamp(np.mean(all_top_scores)) * 100, 1)

    rationale = []
    risk_flags = []
    if m > 20:
        rationale.append(f"High missingness ({m:.1f}%) requires robust imputation and quality warning.")
        risk_flags.append("Missingness above 20%; inspect recording/sensor failure.")
    elif m > 5:
        rationale.append(f"Moderate missingness ({m:.1f}%) makes simple mean filling too weak.")
    else:
        rationale.append(f"Low missingness ({m:.1f}%) allows conservative imputation.")

    if burst > 0.35:
        rationale.append("Missing values are burst-like; temporal methods are preferred over pure global statistics.")
    if sk > 1.0:
        rationale.append("Strong skewness favors robust outlier handling and robust scaling.")
    if c > 15:
        risk_flags.append("High outlier load; downstream conclusions may be unstable.")
    if row["zero_variance"]:
        risk_flags.append("Near-zero variance feature; consider dropping it.")
    if drift > 0.50:
        risk_flags.append("Strong drift detected; inspect calibration/session effects.")
    rationale.append(f"Domain prior: {DOMAIN_PROFILES[domain]['description']}")

    if row["quality_index"] >= 80:
        gain = "Small, mostly reproducibility/standardization."
    elif row["quality_index"] >= 60:
        gain = "Moderate; should improve stability and comparability."
    else:
        gain = "High potential, but also high risk; validate downstream."

    return PipelineRecommendation(missing_method, outlier_method, norm_method, smoothing_method, confidence, gain, rationale, risk_flags)

# -----------------------------------------------------------------------------
# Preprocessing execution
# -----------------------------------------------------------------------------

def _safe_temporal_imputation(data: pd.DataFrame) -> pd.DataFrame:
    """Scalable fallback for large time-series tables.

    Exact KNNImputer computes pairwise row distances and can explode to O(n²)
    memory on long eye-tracking streams. For large streams we therefore use a
    temporal interpolation + local forward/backward fill + median fallback.
    This keeps the demo usable on 100k+ rows and is defensible for ordered
    gaze/pupil signals.
    """
    out = data.astype(float).interpolate(method="linear", limit_direction="both").ffill().bfill()
    for col in out.columns:
        out[col] = out[col].fillna(out[col].median()).fillna(0.0)
    return out


def apply_missing(df_num: pd.DataFrame, method: str, n_neighbors: int = 5, mice_iter: int = 10) -> pd.DataFrame:
    data = df_num.copy()
    n_rows = len(data)

    # Exact KNN/MICE are fine for small demos, but dangerous on long gaze streams.
    # KNNImputer builds large pairwise-distance chunks; with ~170k rows this can
    # easily allocate hundreds of MB to several GB.
    max_exact_knn_rows = 25000
    max_exact_mice_rows = 30000

    if method == "None":
        return data
    if method == "Mean Imputation":
        return pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(data), columns=data.columns, index=data.index)
    if method == "LOCF":
        return data.ffill().bfill()
    if method == "Linear Interpolation":
        return _safe_temporal_imputation(data)
    if method == "KNN Imputation":
        if n_rows > max_exact_knn_rows:
            return _safe_temporal_imputation(data)
        return pd.DataFrame(KNNImputer(n_neighbors=n_neighbors).fit_transform(data), columns=data.columns, index=data.index)
    if method == "MICE":
        if n_rows > max_exact_mice_rows:
            return _safe_temporal_imputation(data)
        return pd.DataFrame(
            IterativeImputer(random_state=42, max_iter=mice_iter, sample_posterior=False).fit_transform(data),
            columns=data.columns,
            index=data.index,
        )
    return data


def apply_outliers(df_num: pd.DataFrame, method: str, z_thresh: float = 3.0, iqr_factor: float = 1.5,
                   mad_thresh: float = 3.5, contamination: float = 0.05, winsor_q: float = 0.02) -> pd.DataFrame:
    data = df_num.copy()
    if method == "None":
        return data
    for col in data.columns:
        s = data[col].astype(float)
        if method == "Z-Score":
            std = s.std(skipna=True)
            mean = s.mean(skipna=True)
            if std > 0 and not np.isnan(std):
                mask = np.abs((s - mean) / std) > z_thresh
                data.loc[mask, col] = np.nan
        elif method == "IQR":
            mask = outlier_iqr_mask(s, factor=iqr_factor)
            data.loc[mask, col] = np.nan
        elif method == "MAD":
            mask = outlier_mad_mask(s, thresh=mad_thresh)
            data.loc[mask, col] = np.nan
        elif method == "Winsorization":
            lo = s.quantile(winsor_q)
            hi = s.quantile(1 - winsor_q)
            data[col] = s.clip(lo, hi)
        elif method == "Isolation Forest":
            if s.dropna().nunique() > 1:
                filled = s.fillna(s.median()).to_numpy().reshape(-1, 1)
                preds = IsolationForest(contamination=contamination, random_state=42).fit_predict(filled)
                data.loc[preds == -1, col] = np.nan
    return data


def apply_normalization(df_num: pd.DataFrame, method: str) -> pd.DataFrame:
    data = df_num.copy()
    if method == "None":
        return data
    # ensure no NaN for scalers; keep it simple: local median fallback
    filled = data.copy()
    for c in filled.columns:
        filled[c] = filled[c].fillna(filled[c].median()).fillna(0)
    scaler = None
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "RobustScaler":
        scaler = RobustScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scaler is None:
        return data
    arr = scaler.fit_transform(filled)
    return pd.DataFrame(arr, columns=data.columns, index=data.index)


def butter_lowpass_filter(series: pd.Series, cutoff: float, fs: float, order: int = 3) -> pd.Series:
    x = series.astype(float).interpolate(limit_direction="both").ffill().bfill().to_numpy()
    if len(x) < max(10, order * 3 + 1):
        return series
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq if nyq > 0 else 0
    if not (0 < normal_cutoff < 1):
        return series
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    try:
        y = filtfilt(b, a, x)
        return pd.Series(y, index=series.index)
    except Exception:
        return series


def apply_smoothing(df_num: pd.DataFrame, method: str, fs: float = 250.0, cutoff: float = 30.0,
                    sg_window: int = 11, sg_poly: int = 2, fourier_keep_ratio: float = 0.08) -> pd.DataFrame:
    data = df_num.copy()
    if method == "None":
        return data
    for col in data.columns:
        s = data[col].astype(float).interpolate(limit_direction="both").ffill().bfill()
        if method == "Butterworth":
            data[col] = butter_lowpass_filter(s, cutoff=cutoff, fs=fs, order=3)
        elif method == "Savitzky-Golay":
            win = int(sg_window)
            if win % 2 == 0:
                win += 1
            if len(s) >= win and win > sg_poly:
                data[col] = savgol_filter(s.to_numpy(), window_length=win, polyorder=sg_poly)
        elif method == "Fourier":
            x = s.to_numpy()
            vals = fft(x)
            freq = np.fft.fftfreq(len(vals))
            vals[np.abs(freq) > fourier_keep_ratio] = 0
            data[col] = np.real(ifft(vals))
    return data


def execute_cleaning_stage(df: pd.DataFrame, features: List[str], rec: PipelineRecommendation, settings: Dict[str, Any]) -> pd.DataFrame:
    """Execute only missing-value and outlier handling, keeping the original signal scale.

    This stage is useful for ground-truth validation because MAE/RMSE only make
    sense if the processed noisy signal and the clean reference signal are on the
    same scale. Normalization and smoothing are deliberately excluded here.
    """
    data = df[features].copy()
    order = settings.get("operation_order", "Missing → Outlier")

    if order == "Outlier → Missing":
        data = apply_outliers(
            data,
            rec.outlier_method,
            z_thresh=settings["z_thresh"],
            iqr_factor=settings["iqr_factor"],
            mad_thresh=settings["mad_thresh"],
            contamination=settings["if_contamination"],
            winsor_q=settings["winsor_q"],
        )
        data = apply_missing(
            data,
            rec.missing_method if rec.missing_method != "None" else "Linear Interpolation",
            n_neighbors=settings["knn_neighbors"],
            mice_iter=settings["mice_iter"],
        )
    else:
        data = apply_missing(data, rec.missing_method, n_neighbors=settings["knn_neighbors"], mice_iter=settings["mice_iter"])
        data = apply_outliers(
            data,
            rec.outlier_method,
            z_thresh=settings["z_thresh"],
            iqr_factor=settings["iqr_factor"],
            mad_thresh=settings["mad_thresh"],
            contamination=settings["if_contamination"],
            winsor_q=settings["winsor_q"],
        )
        # Impute again after outlier-to-NaN, otherwise normalization/downstream contains gaps.
        data = apply_missing(
            data,
            rec.missing_method if rec.missing_method != "None" else "Linear Interpolation",
            n_neighbors=settings["knn_neighbors"],
            mice_iter=settings["mice_iter"],
        )
    return data


def execute_pipeline(df: pd.DataFrame, features: List[str], rec: PipelineRecommendation, settings: Dict[str, Any]) -> pd.DataFrame:
    data = execute_cleaning_stage(df, features, rec, settings)
    data = apply_normalization(data, rec.normalization_method)
    data = apply_smoothing(data, rec.smoothing_method, fs=settings["sampling_rate"], cutoff=settings["filter_cutoff"],
                           sg_window=settings["sg_window"], sg_poly=settings["sg_poly"], fourier_keep_ratio=settings["fourier_keep_ratio"])
    return data


# -----------------------------------------------------------------------------
# Step-wise audit helpers
# -----------------------------------------------------------------------------

def _safe_prob_hist(values: np.ndarray, bins: int = 10) -> np.ndarray:
    """Return a stable probability vector for 1D/2D binned distributions."""
    values = np.asarray(values)
    if values.size == 0:
        return np.array([], dtype=float)
    hist = values.astype(float)
    total = np.nansum(hist)
    if total <= 0 or not np.isfinite(total):
        return np.array([], dtype=float)
    p = hist.ravel() / total
    return p[p > 0]


def gaze_spatial_distribution_metrics(x: pd.Series, y: pd.Series, bins: int = 8, max_n: int = 8000) -> Dict[str, float]:
    """Spatial/behavioral metrics from Chapter 5.4/10.4 for paired gaze coordinates.

    The function discretizes x/y into a regular grid. It is intentionally lightweight
    for Streamlit: very long streams are sub-sampled deterministically.
    """
    xy = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(xy) < 10:
        return {
            "Spatial Shannon entropy": np.nan,
            "Spatial KL to uniform": np.nan,
            "Spatial Gini": np.nan,
            "Lempel-Ziv complexity": np.nan,
            "Transition entropy": np.nan,
            "Path length": np.nan,
            "Velocity mean": np.nan,
            "Velocity std": np.nan,
            "Effective sample count": np.nan,
        }
    if len(xy) > max_n:
        idx = np.linspace(0, len(xy) - 1, max_n).astype(int)
        xy = xy.iloc[idx]

    # Effective sample count: longest contiguous valid run on the original x/y mask.
    valid_mask = pd.to_numeric(x, errors="coerce").notna() & pd.to_numeric(y, errors="coerce").notna()
    groups = (valid_mask != valid_mask.shift(fill_value=False)).cumsum()
    if valid_mask.any():
        eff = int(valid_mask.groupby(groups).sum().max())
    else:
        eff = 0

    # Min-max to unit square for robust binning across raw/normalized stages.
    xx = xy["x"].to_numpy(dtype=float)
    yy = xy["y"].to_numpy(dtype=float)
    def scale01(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(v, dtype=float)
        return (v - lo) / (hi - lo)
    xs = scale01(xx)
    ys = scale01(yy)

    hist2d, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0, 1], [0, 1]])
    probs_full = hist2d.ravel().astype(float)
    total = probs_full.sum()
    if total <= 0:
        probs_nonzero = np.array([], dtype=float)
    else:
        probs_full = probs_full / total
        probs_nonzero = probs_full[probs_full > 0]

    if len(probs_nonzero) == 0:
        spatial_entropy = 0.0
        kl_uniform = 0.0
        gini = 0.0
    else:
        spatial_entropy = float(-(probs_nonzero * np.log2(probs_nonzero)).sum())
        uniform = 1.0 / (bins * bins)
        kl_uniform = float((probs_nonzero * np.log2(probs_nonzero / uniform)).sum())
        vals = probs_full.copy()
        mean_val = vals.mean()
        if mean_val <= 0:
            gini = 0.0
        else:
            gini = float(np.abs(vals[:, None] - vals[None, :]).sum() / (2 * len(vals) ** 2 * mean_val))

    # Symbol sequence and LZ76-like complexity.
    xi = np.clip(np.floor(xs * bins), 0, bins - 1).astype(int)
    yi = np.clip(np.floor(ys * bins), 0, bins - 1).astype(int)
    symbols = (xi * bins + yi).astype(int).tolist()

    def lz76_complexity(seq: List[int]) -> float:
        if not seq:
            return 0.0
        # Simple incremental parsing via new substrings; good enough for audit UI.
        seen = set()
        i = 0
        c = 0
        n = len(seq)
        while i < n:
            j = i + 1
            while j <= n and tuple(seq[i:j]) in seen:
                j += 1
            seen.add(tuple(seq[i:j]))
            c += 1
            i = j
        denom = n / np.log2(max(n, 2))
        return float(c / denom) if denom > 0 else 0.0

    lz = lz76_complexity(symbols)

    # First-order transition entropy.
    if len(symbols) < 2:
        trans_entropy = 0.0
    else:
        trans = pd.DataFrame({"a": symbols[:-1], "b": symbols[1:]})
        counts = trans.value_counts().to_numpy(dtype=float)
        pp = counts / counts.sum()
        trans_entropy = float(-(pp * np.log2(pp)).sum())

    # Path length / velocity proxy over valid, scaled coordinates.
    dx = np.diff(xs)
    dy = np.diff(ys)
    step = np.sqrt(dx * dx + dy * dy)
    path_length = float(np.nansum(step))
    vel_mean = float(np.nanmean(step)) if len(step) else 0.0
    vel_std = float(np.nanstd(step)) if len(step) else 0.0

    return {
        "Spatial Shannon entropy": safe_float(spatial_entropy),
        "Spatial KL to uniform": safe_float(kl_uniform),
        "Spatial Gini": safe_float(gini),
        "Lempel-Ziv complexity": safe_float(lz),
        "Transition entropy": safe_float(trans_entropy),
        "Path length": safe_float(path_length),
        "Velocity mean": safe_float(vel_mean),
        "Velocity std": safe_float(vel_std),
        "Effective sample count": safe_float(eff),
    }


def find_gaze_companion(feature: str, columns: List[str]) -> Optional[str]:
    """Find likely X/Y partner for gaze-distribution metrics."""
    candidates = []
    f = feature
    replacements = [
        ("Gaze_X", "Gaze_Y"), ("Gaze_Y", "Gaze_X"),
        ("Gaze X", "Gaze Y"), ("Gaze Y", "Gaze X"),
        ("gaze_x", "gaze_y"), ("gaze_y", "gaze_x"),
        ("_X", "_Y"), ("_Y", "_X"),
        (" x", " y"), (" y", " x"),
        ("X", "Y"), ("Y", "X"),
    ]
    for a, b in replacements:
        if a in f:
            candidates.append(f.replace(a, b))
    lower_cols = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None

def _safe_series_delta(a: pd.Series, b: pd.Series) -> float:
    aa = a.astype(float)
    bb = b.astype(float)
    mask = aa.notna() & bb.notna()
    if mask.sum() == 0:
        return 0.0
    return safe_float((aa[mask] - bb[mask]).abs().mean(), 0.0)


def compute_stage_audit_metrics(df_stage: pd.DataFrame, feature: str, weights: Dict[str, float],
                                baseline: Optional[pd.DataFrame] = None,
                                previous: Optional[pd.DataFrame] = None,
                                raw_missing_mask: Optional[pd.Series] = None,
                                spatial_context: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Feature-level metrics for a pipeline stage.

    These metrics intentionally mix the dissertation's quality notions
    (completeness, drift/stability, interpolation impact, plausibility) with
    pragmatic signal diagnostics for the UI.
    """
    one = df_stage[[feature]].copy()
    prof = profile_features(one, [feature], weights).iloc[0].to_dict()
    s = one[feature].astype(float)
    x = s.dropna()

    missing_now = s.isna()
    if raw_missing_mask is None:
        raw_missing_mask = pd.Series(False, index=s.index)
    imputed_mask = raw_missing_mask & s.notna()

    baseline_change = _safe_series_delta(s, baseline[feature]) if baseline is not None and feature in baseline.columns else 0.0
    previous_change = _safe_series_delta(s, previous[feature]) if previous is not None and feature in previous.columns else 0.0

    # Pseudo precision / jitter: inverse of normalized first-difference variability.
    # Not an absolute eye-tracker precision measure, but useful for before/after inspection.
    if len(x) > 4 and x.std() and not np.isnan(x.std()):
        jitter = safe_float(x.diff().dropna().std() / (x.std() + 1e-12), 0.0)
    else:
        jitter = 0.0

    metrics = {
        "Quality Index": safe_float(prof.get("quality_index")),
        "Completeness %": 100.0 - safe_float(prof.get("missing_rate")),
        "Missing %": safe_float(prof.get("missing_rate")),
        "Imputed-from-raw %": safe_float(imputed_mask.mean() * 100),
        "Longest missing run": safe_float(prof.get("longest_missing_run")),
        "Missing burstiness": safe_float(prof.get("missing_burstiness")),
        "IQR outlier %": safe_float(prof.get("outlier_iqr_rate")),
        "MAD outlier %": safe_float(prof.get("outlier_mad_rate")),
        "Skewness": safe_float(prof.get("skewness")),
        "Kurtosis excess": safe_float(prof.get("kurtosis_excess")),
        "Drift score": safe_float(prof.get("drift_score")),
        "Volatility score": safe_float(prof.get("volatility_score")),
        "Entropy score": safe_float(prof.get("entropy_score")),
        "Jitter proxy": jitter,
        "Mean abs change from raw": baseline_change,
        "Mean abs change from previous": previous_change,
        "Std": safe_float(s.std(skipna=True)),
        "Min": safe_float(s.min(skipna=True)),
        "Max": safe_float(s.max(skipna=True)),
    }

    # Behavioral/spatial metrics from dissertation Ch. 5.4 / Ch. 10.4.
    # If feature_focus is a gaze x/y column, combine the processed audit-stage feature
    # with the likely companion coordinate from the context. This makes the audit useful
    # even when the user focuses on one coordinate at a time.
    companion = find_gaze_companion(feature, list(spatial_context.columns) if spatial_context is not None else list(df_stage.columns))
    if companion is not None:
        if spatial_context is not None and companion in spatial_context.columns:
            y_series = spatial_context[companion]
        elif companion in df_stage.columns:
            y_series = df_stage[companion]
        else:
            y_series = None
        if y_series is not None:
            if ("x" in feature.lower()) or ("_x" in feature.lower()) or (" x" in feature.lower()):
                metrics.update(gaze_spatial_distribution_metrics(df_stage[feature], y_series))
            else:
                metrics.update(gaze_spatial_distribution_metrics(y_series, df_stage[feature]))
    else:
        metrics.update({
            "Spatial Shannon entropy": np.nan,
            "Spatial KL to uniform": np.nan,
            "Spatial Gini": np.nan,
            "Lempel-Ziv complexity": np.nan,
            "Transition entropy": np.nan,
            "Path length": np.nan,
            "Velocity mean": np.nan,
            "Velocity std": np.nan,
            "Effective sample count": np.nan,
        })
    return metrics


def build_pipeline_stages(df_source: pd.DataFrame, feature: str, mv_method: str, out_method: str,
                          norm_method: str, smooth_method: str, order: str,
                          settings: Dict[str, Any]) -> List[Tuple[str, pd.DataFrame]]:
    """Return named stage outputs for one feature, preserving intermediate results."""
    stages: List[Tuple[str, pd.DataFrame]] = []
    raw = df_source[[feature]].copy()
    stages.append(("Raw", raw))
    current = raw.copy()

    def do_mv(d: pd.DataFrame, method: str) -> pd.DataFrame:
        return apply_missing(d, method, n_neighbors=settings["knn_neighbors"], mice_iter=settings["mice_iter"])

    def do_out(d: pd.DataFrame, method: str) -> pd.DataFrame:
        return apply_outliers(d, method, z_thresh=settings["z_thresh"], iqr_factor=settings["iqr_factor"],
                              mad_thresh=settings["mad_thresh"], contamination=settings["if_contamination"],
                              winsor_q=settings["winsor_q"])

    if order == "Outlier → Missing":
        current = do_out(current, out_method)
        stages.append((f"1 Outlier: {out_method}", current.copy()))
        mv_eff = mv_method if mv_method != "None" else "Linear Interpolation"
        current = do_mv(current, mv_eff)
        stages.append((f"2 Missing: {mv_eff}", current.copy()))
    else:
        current = do_mv(current, mv_method)
        stages.append((f"1 Missing: {mv_method}", current.copy()))
        current = do_out(current, out_method)
        stages.append((f"2 Outlier: {out_method}", current.copy()))
        # Important: most outlier handlers mark artifacts as NaN. A second fill shows
        # how the chosen MV method and outlier method work together as a stage pair.
        refill_method = mv_method if mv_method != "None" else "Linear Interpolation"
        current = do_mv(current, refill_method)
        stages.append((f"2b Re-fill after outlier: {refill_method}", current.copy()))

    current = apply_normalization(current, norm_method)
    stages.append((f"3 Normalization: {norm_method}", current.copy()))

    current = apply_smoothing(current, smooth_method, fs=settings["sampling_rate"], cutoff=settings["filter_cutoff"],
                              sg_window=settings["sg_window"], sg_poly=settings["sg_poly"],
                              fourier_keep_ratio=settings["fourier_keep_ratio"])
    stages.append((f"4 Smoothing: {smooth_method}", current.copy()))
    return stages


def plot_stage_overlay(stages: List[Tuple[str, pd.DataFrame]], feature: str, max_points: int = 5000) -> go.Figure:
    fig = go.Figure()
    n = len(stages[0][1]) if stages else 0
    if n > max_points:
        idx_local = np.linspace(0, n - 1, max_points).astype(int)
    else:
        idx_local = None
    for name, d in stages:
        s = d[feature].astype(float)
        sp = s.iloc[idx_local] if idx_local is not None else s
        width = 2.2 if name == "Raw" else 1.4
        dash = "dash" if name == "Raw" else None
        fig.add_trace(go.Scatter(x=sp.index, y=sp.values, mode="lines", name=name, line=dict(width=width, dash=dash)))
    fig.update_layout(height=440, margin=dict(l=30, r=20, t=30, b=35), legend=dict(orientation="h"), yaxis_title=feature)
    return fig

# -----------------------------------------------------------------------------
# Downstream validation
# -----------------------------------------------------------------------------

def downstream_validation(df_raw: pd.DataFrame, df_processed: pd.DataFrame, features: List[str], target_col: str) -> Optional[pd.DataFrame]:
    if not target_col or target_col not in df_raw.columns or len(features) < 1:
        return None
    y = df_raw[target_col]
    mask = y.notna()
    if mask.sum() < 30:
        return None
    X_raw = df_raw.loc[mask, features].copy()
    X_proc = df_processed.loc[mask, features].copy()
    y = y.loc[mask]

    # Fill any remaining gaps for model baseline
    X_raw = X_raw.apply(lambda s: s.fillna(s.median()).fillna(0))
    X_proc = X_proc.apply(lambda s: s.fillna(s.median()).fillna(0))

    # Decide classification vs regression
    nunique = y.nunique(dropna=True)
    is_classification = (not pd.api.types.is_numeric_dtype(y)) or nunique <= min(20, max(2, int(len(y) * 0.05)))
    results = []
    if is_classification:
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        if len(np.unique(y_enc)) < 2:
            return None
        cv = StratifiedKFold(n_splits=min(5, np.bincount(y_enc).min() if np.bincount(y_enc).min() >= 2 else 2), shuffle=True, random_state=42)
        models = {
            "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
            "RandomForest": RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced_subsample"),
        }
        scoring = "f1_weighted"
        metric_name = "F1 weighted"
        for name, model in models.items():
            for label, X in [("Raw baseline", X_raw), ("CIRRUS++ pipeline", X_proc)]:
                try:
                    scores = cross_val_score(model, X, y_enc, cv=cv, scoring=scoring)
                    results.append({"Model": name, "Data": label, "Metric": metric_name, "CV mean": scores.mean(), "CV std": scores.std()})
                except Exception:
                    pass
    else:
        y_num = pd.to_numeric(y, errors="coerce")
        mask2 = y_num.notna()
        if mask2.sum() < 30:
            return None
        y_num = y_num[mask2]
        X_raw = X_raw.loc[mask2]
        X_proc = X_proc.loc[mask2]
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        models = {
            "Ridge": make_pipeline(StandardScaler(), Ridge()),
            "RandomForestReg": RandomForestRegressor(n_estimators=120, random_state=42),
        }
        for name, model in models.items():
            for label, X in [("Raw baseline", X_raw), ("CIRRUS++ pipeline", X_proc)]:
                try:
                    scores = cross_val_score(model, X, y_num, cv=cv, scoring="r2")
                    results.append({"Model": name, "Data": label, "Metric": "R²", "CV mean": scores.mean(), "CV std": scores.std()})
                except Exception:
                    pass
    if not results:
        return None
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def plot_feature_series(raw: pd.Series, processed: Optional[pd.Series] = None, max_points: int = 8000) -> go.Figure:
    n = len(raw)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        raw_plot = raw.iloc[idx]
        processed_plot = processed.iloc[idx] if processed is not None else None
    else:
        raw_plot = raw
        processed_plot = processed
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw_plot.index, y=raw_plot.values, mode="lines", name="Raw", line=dict(width=1.4)))
    if processed_plot is not None:
        fig.add_trace(go.Scatter(x=processed_plot.index, y=processed_plot.values, mode="lines", name="Processed", line=dict(width=1.8)))
    fig.update_layout(height=350, margin=dict(l=30, r=20, t=30, b=30), legend=dict(orientation="h"))
    return fig


def plot_quality_radar(profile_row: pd.Series) -> go.Figure:
    cats = ["Completeness", "Continuity", "Plausibility", "Distribution", "Stability", "Correlation"]
    # approximate component values for display
    m = profile_row["missing_rate"]
    continuity = 100 * (1 - clamp(0.65 * profile_row["missing_burstiness"] + 0.35 * (profile_row["longest_missing_run"] / max(1, profile_row["n"]) * 10)))
    vals = [
        100 * (1 - clamp(m / 45.0)),
        continuity,
        100 * (1 - clamp(max(profile_row["outlier_iqr_rate"], profile_row["outlier_mad_rate"]) / 25.0)),
        100 * (1 - clamp((abs(profile_row["skewness"]) / 3.0) * 0.65 + (max(0, profile_row["kurtosis_excess"]) / 8.0) * 0.35)),
        100 * (1 - clamp(0.60 * profile_row["drift_score"] + 0.40 * profile_row["volatility_score"])),
        100 * clamp(0.45 + profile_row.get("corr_mean_abs", 0.0) * 0.80),
    ]
    fig = go.Figure(data=go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself", name="Quality"))
    fig.update_layout(height=350, polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=30, r=30, t=30, b=30))
    return fig


def method_badge(text: str) -> str:
    return f"<span style='display:inline-block;padding:0.25rem 0.55rem;border-radius:999px;background:#e0f2fe;border:1px solid #7dd3fc;margin:0.1rem;font-size:0.86rem'>{text}</span>"

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1) Data & domain")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    delimiter = st.selectbox("CSV delimiter", ["Auto", ",", ";", "\t", "|"])
    decimal = st.selectbox("Decimal format", ["Auto", ".", ","])
    include_id_cols = st.checkbox("Include likely ID/time columns as features", value=False)

    st.header("2) Recommendation settings")
    domain_choice = st.selectbox("Domain profile", list(DOMAIN_PROFILES.keys()), index=0)
    aggressive = st.slider("Recommendation aggressiveness", 0.0, 1.0, 0.50, 0.05, help="Higher values prefer stronger, model-based cleaning.")
    max_plot_points = st.slider("Max preview points", 1000, 25000, 8000, step=1000)
    operation_order = st.selectbox(
        "Pipeline order",
        ["Missing → Outlier", "Outlier → Missing"],
        index=0,
        help="Swap the order to test whether it is better to impute gaps before detecting outliers, or detect artifacts before imputation."
    )

    st.header("3) Method parameters")
    z_thresh = st.slider("Z-score threshold", 1.0, 6.0, 3.0, 0.1)
    iqr_factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1)
    mad_thresh = st.slider("MAD robust-z threshold", 2.0, 6.0, 3.5, 0.1)
    if_contamination = st.slider("Isolation Forest contamination", 0.005, 0.30, 0.05, 0.005)
    winsor_q = st.slider("Winsorization quantile", 0.005, 0.10, 0.02, 0.005)
    knn_neighbors = st.slider("KNN neighbors", 2, 20, 5)
    mice_iter = st.slider("MICE iterations", 3, 25, 10)

    st.header("4) Temporal options")
    sampling_rate = st.number_input("Sampling rate (Hz)", min_value=1.0, max_value=5000.0, value=250.0, step=10.0)
    filter_cutoff = st.number_input("Butterworth cutoff (Hz)", min_value=0.1, max_value=1000.0, value=30.0, step=1.0)
    sg_window = st.slider("Savitzky-Golay window", 5, 101, 11, step=2)
    sg_poly = st.slider("Savitzky-Golay polynomial", 1, 5, 2)
    fourier_keep_ratio = st.slider("Fourier keep ratio", 0.01, 0.50, 0.08, 0.01)

settings = {
    "z_thresh": z_thresh,
    "iqr_factor": iqr_factor,
    "mad_thresh": mad_thresh,
    "if_contamination": if_contamination,
    "winsor_q": winsor_q,
    "knn_neighbors": knn_neighbors,
    "mice_iter": mice_iter,
    "sampling_rate": sampling_rate,
    "filter_cutoff": filter_cutoff,
    "sg_window": sg_window,
    "sg_poly": sg_poly,
    "fourier_keep_ratio": fourier_keep_ratio,
    "operation_order": operation_order,
}

# -----------------------------------------------------------------------------
# Main flow
# -----------------------------------------------------------------------------
if uploaded_file is None:
    st.info("Upload a CSV file to start. The upgraded prototype is designed to answer the review criticism: stronger metrics, domain-aware recommendations, quality score, configurable heuristics, multi-feature execution, and optional downstream validation.")
    with st.expander("What is new compared with the earlier CIRRUS prototype?", expanded=True):
        st.markdown(
            """
- **Domain profile first:** clinical, reading/cognitive, high-frequency, remote/web, multimodal.
- **Quality Index (0–100):** combines completeness, temporal continuity, plausibility, distribution, stability, and multivariate correlation.
- **Better profiling:** longest missing run, missing burstiness, IQR/MAD outliers, skewness, kurtosis, drift, volatility, entropy, correlation.
- **Adaptive recommender scoring:** methods are scored from data + domain priors, not only fixed thresholds.
- **Multi-feature pipeline execution:** apply one recommended or manually modified pipeline to all selected features.
- **Downstream validation:** optional baseline vs. processed model comparison when a target column exists.
- **Scalability-aware UI:** preview downsampling; heavy computations are explicit and controlled.
            """
        )
    st.stop()

# Read CSV robustly
try:
    sep = None if delimiter == "Auto" else delimiter
    dec = "." if decimal == "Auto" else decimal
    if sep is None:
        df = pd.read_csv(uploaded_file, sep=None, engine="python", decimal=dec)
    else:
        df = pd.read_csv(uploaded_file, sep=sep, decimal=dec, low_memory=False)
except Exception as e:
    st.error(f"CSV could not be read: {e}")
    st.stop()

raw_shape = df.shape
start_time = time.time()
df = auto_convert_numeric_columns(df)
read_seconds = time.time() - start_time

features_all = candidate_numeric_features(df, include_ids=include_id_cols)
if not features_all:
    st.error("No numeric feature columns were detected. Check delimiter, decimal separator, and export format.")
    st.stop()

if domain_choice == "Auto-detect":
    inferred_domain = infer_domain(df, features_all)
else:
    inferred_domain = domain_choice
profile = DOMAIN_PROFILES[inferred_domain]

cA, cB, cC, cD = st.columns(4)
cA.metric("Rows", f"{raw_shape[0]:,}")
cB.metric("Columns", f"{raw_shape[1]:,}")
cC.metric("Numeric features", f"{len(features_all):,}")
cD.metric("Domain", inferred_domain.replace(" / ", " /\n"))

st.markdown(f"<div class='small-card'><b>Domain logic:</b> {profile['description']}<br><b>Risk note:</b> {profile['risk_note']}</div>", unsafe_allow_html=True)

with st.expander("Data preview and detected dtypes", expanded=False):
    st.dataframe(df.head(30), use_container_width=True)
    dtype_df = pd.DataFrame({"Column": df.columns, "Detected dtype": df.dtypes.astype(str), "Missing %": df.isna().mean().mul(100).round(2).values})
    st.dataframe(dtype_df, use_container_width=True)

st.subheader("Select features")
default_features = features_all[: min(6, len(features_all))]
selected_features = st.multiselect("Numerical features for profiling and pipeline execution", features_all, default=default_features)

if not selected_features:
    st.warning("Select at least one feature.")
    st.stop()

with st.spinner("Profiling selected features..."):
    profile_df = profile_features(df, selected_features, profile["weights"])

# Quality overview
st.subheader("Quality dashboard")
q_mean = profile_df["quality_index"].mean()
critical_count = (profile_df["quality_index"] < 40).sum()
risky_count = ((profile_df["quality_index"] >= 40) & (profile_df["quality_index"] < 60)).sum()
q1, q2, q3, q4 = st.columns(4)
q1.metric("Mean Quality Index", f"{q_mean:.1f}/100")
q2.metric("Critical features", int(critical_count))
q3.metric("Risky features", int(risky_count))
q4.metric("Profiling time", f"{read_seconds:.2f}s")

show_cols = [
    "feature", "quality_index", "quality_label", "missing_rate", "longest_missing_run", "missing_burstiness",
    "outlier_iqr_rate", "outlier_mad_rate", "skewness", "kurtosis_excess", "drift_score", "volatility_score", "corr_mean_abs"
]
st.dataframe(
    profile_df[show_cols].style.format({
        "quality_index": "{:.1f}", "missing_rate": "{:.1f}", "missing_burstiness": "{:.2f}",
        "outlier_iqr_rate": "{:.1f}", "outlier_mad_rate": "{:.1f}", "skewness": "{:.2f}",
        "kurtosis_excess": "{:.2f}", "drift_score": "{:.2f}", "volatility_score": "{:.2f}", "corr_mean_abs": "{:.2f}"
    }),
    use_container_width=True,
)

# Feature-level details
st.subheader("Feature-level recommendation")
feature_focus = st.selectbox("Inspect feature", selected_features)
row = profile_df.loc[profile_df["feature"] == feature_focus].iloc[0]
rec = recommend_for_profile(row, inferred_domain, aggressive=aggressive)

left, right = st.columns([1.05, 1])
with left:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality Index", f"{row['quality_index']:.1f}/100", row["quality_label"])
    c2.metric("Missing", f"{row['missing_rate']:.1f}%", f"longest run {int(row['longest_missing_run'])}")
    c3.metric("Outliers", f"{max(row['outlier_iqr_rate'], row['outlier_mad_rate']):.1f}%", "IQR/MAD max")
    c4.metric("Skew/Kurt", f"{row['skewness']:.2f}", f"K {row['kurtosis_excess']:.2f}")

    st.markdown(
        f"""
<div class='reco-box'>
<b>Recommended pipeline</b><br>
{method_badge('MV: ' + rec.missing_method)}
{method_badge('Outlier: ' + rec.outlier_method)}
{method_badge('Norm: ' + rec.normalization_method)}
{method_badge('Smooth: ' + rec.smoothing_method)}
<br><br><b>Confidence:</b> {rec.confidence:.1f}% &nbsp; | &nbsp; <b>Expected gain:</b> {rec.expected_gain}
</div>
""",
        unsafe_allow_html=True,
    )
    with st.expander("Recommendation rationale", expanded=True):
        for item in rec.rationale:
            st.markdown(f"- {item}")
        if rec.risk_flags:
            st.markdown("**Risk flags**")
            for item in rec.risk_flags:
                st.markdown(f"- ⚠️ {item}")
with right:
    st.plotly_chart(plot_quality_radar(row), use_container_width=True, key="quality_radar")

# Manual override
st.subheader("Pipeline configuration")
with st.expander("Override recommended methods", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    mv_method = col1.selectbox("Missing values", ["None", "Mean Imputation", "LOCF", "Linear Interpolation", "KNN Imputation", "MICE"], index=["None", "Mean Imputation", "LOCF", "Linear Interpolation", "KNN Imputation", "MICE"].index(rec.missing_method))
    out_method = col2.selectbox("Outliers", ["None", "Z-Score", "IQR", "MAD", "Winsorization", "Isolation Forest"], index=["None", "Z-Score", "IQR", "MAD", "Winsorization", "Isolation Forest"].index(rec.outlier_method))
    norm_method = col3.selectbox("Normalization", ["None", "StandardScaler", "RobustScaler", "MinMaxScaler"], index=["None", "StandardScaler", "RobustScaler", "MinMaxScaler"].index(rec.normalization_method))
    smooth_method = col4.selectbox("Smoothing", ["None", "Butterworth", "Savitzky-Golay", "Fourier"], index=["None", "Butterworth", "Savitzky-Golay", "Fourier"].index(rec.smoothing_method))
    rec = PipelineRecommendation(mv_method, out_method, norm_method, smooth_method, rec.confidence, rec.expected_gain, rec.rationale, rec.risk_flags)

run = st.button("Run CIRRUS++ pipeline", type="primary", use_container_width=True)

# Important: Streamlit keeps st.session_state across reruns. If the inspected feature,
# selected feature set, domain, settings, or manual pipeline changes, the old processed
# dataframe may no longer contain the currently selected feature. This caused e.g.
# KeyError: 'Gaze X' in the signal preview. We therefore use an explicit signature and
# recompute whenever the current UI state no longer matches the cached result.
pipeline_signature = {
    "features": list(selected_features),
    "focus": feature_focus,
    "domain": inferred_domain,
    "recommendation": asdict(rec),
    "settings": settings,
    "raw_shape": raw_shape,
    "columns": list(df.columns),
}

cached_processed = st.session_state.get("processed")
cache_missing = cached_processed is None
cache_incomplete = cache_missing or any(col not in cached_processed.columns for col in selected_features)
signature_changed = st.session_state.get("pipeline_signature") != pipeline_signature

if cache_missing or cache_incomplete or signature_changed or run:
    with st.spinner("Executing pipeline on selected features..."):
        processed_features = execute_pipeline(df, selected_features, rec, settings)
        final_df = df.copy()

        # Do not assign a whole float DataFrame block into possibly int64 columns.
        # Pandas 2.x/3.x can raise TypeError for columns such as Row/index.
        # Replacing columns one-by-one lets pandas safely update the dtype.
        for col in selected_features:
            if col in processed_features.columns:
                final_df[col] = processed_features[col].to_numpy()

        st.session_state["processed"] = processed_features
        st.session_state["final_df"] = final_df
        st.session_state["pipeline"] = asdict(rec) | {"domain": inferred_domain, "settings": settings, "features": selected_features}
        st.session_state["pipeline_signature"] = pipeline_signature

processed_features = st.session_state["processed"]
final_df = st.session_state["final_df"]

# Defensive fallback: if a column still went missing because of an unexpected pandas/sklearn
# edge case, recompute only the focus feature rather than crashing the app.
if feature_focus not in processed_features.columns:
    focus_processed = execute_pipeline(df, [feature_focus], rec, settings)
    processed_features = pd.concat([processed_features, focus_processed], axis=1)
    final_df[feature_focus] = focus_processed[feature_focus].to_numpy()
    st.session_state["processed"] = processed_features
    st.session_state["final_df"] = final_df

# Before/after profile comparison
with st.spinner("Computing before/after quality comparison..."):
    after_profile_df = profile_features(final_df, selected_features, profile["weights"])
compare_df = profile_df[["feature", "quality_index", "missing_rate", "outlier_iqr_rate", "skewness", "kurtosis_excess"]].merge(
    after_profile_df[["feature", "quality_index", "missing_rate", "outlier_iqr_rate", "skewness", "kurtosis_excess"]],
    on="feature",
    suffixes=(" before", " after"),
)
compare_df["QI gain"] = compare_df["quality_index after"] - compare_df["quality_index before"]

st.subheader("Before / after comparison")
col_chart, col_table = st.columns([1.0, 1.1])
with col_chart:
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=compare_df["feature"], y=compare_df["quality_index before"], name="Before"))
    fig_bar.add_trace(go.Bar(x=compare_df["feature"], y=compare_df["quality_index after"], name="After"))
    fig_bar.update_layout(barmode="group", height=360, margin=dict(l=30, r=20, t=30, b=80), yaxis_title="Quality Index")
    st.plotly_chart(fig_bar, use_container_width=True, key="qi_compare")
with col_table:
    st.dataframe(compare_df.style.format({c: "{:.2f}" for c in compare_df.select_dtypes(include=[np.number]).columns}), use_container_width=True)

st.subheader("Signal preview")
st.plotly_chart(plot_feature_series(df[feature_focus], processed_features[feature_focus], max_points=max_plot_points), use_container_width=True, key="signal_preview")


# Shared method lists for all comparison/audit widgets
mv_methods_all = ["Mean Imputation", "LOCF", "Linear Interpolation", "KNN Imputation", "MICE"]
outlier_methods_all = ["Z-Score", "IQR", "MAD", "Winsorization", "Isolation Forest"]
norm_methods_all = ["StandardScaler", "RobustScaler", "MinMaxScaler"]
smooth_methods_all = ["Butterworth", "Savitzky-Golay", "Fourier"]

# -----------------------------------------------------------------------------
# Order-aware step-wise pipeline audit
# -----------------------------------------------------------------------------
st.subheader("Order-aware step-wise pipeline audit")
st.caption(
    "This is the order-sensitive view. It shows the actual intermediate states of the selected pipeline: "
    "raw signal, first operation, second operation, optional re-fill, normalization, and smoothing. "
    "Use this section when you want to see how missing-value handling and outlier handling work together."
)

with st.expander("Configure step-wise audit", expanded=True):
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        audit_order = st.selectbox(
            "Audit order",
            ["Use sidebar order", "Missing → Outlier", "Outlier → Missing"],
            index=0,
            key="audit_order_select",
        )
        audit_order_eff = settings.get("operation_order", "Missing → Outlier") if audit_order == "Use sidebar order" else audit_order
    with a2:
        audit_mv = st.selectbox(
            "Missing method",
            ["None"] + mv_methods_all,
            index=(["None"] + mv_methods_all).index(rec.missing_method if rec.missing_method in mv_methods_all else "None"),
            key="audit_mv_method",
        )
    with a3:
        audit_out = st.selectbox(
            "Outlier method",
            ["None"] + outlier_methods_all,
            index=(["None"] + outlier_methods_all).index(rec.outlier_method if rec.outlier_method in outlier_methods_all else "None"),
            key="audit_out_method",
        )
    with a4:
        audit_norm = st.selectbox(
            "Normalization",
            ["None"] + norm_methods_all,
            index=(["None"] + norm_methods_all).index(rec.normalization_method if rec.normalization_method in norm_methods_all else "None"),
            key="audit_norm_method",
        )
    audit_smooth = st.selectbox(
        "Smoothing/filtering for final stage",
        ["None"] + smooth_methods_all,
        index=(["None"] + smooth_methods_all).index(rec.smoothing_method if rec.smoothing_method in smooth_methods_all else "None"),
        key="audit_smooth_method",
    )

raw_missing_mask_focus = df[feature_focus].isna()
stages = build_pipeline_stages(df, feature_focus, audit_mv, audit_out, audit_norm, audit_smooth, audit_order_eff, settings)
st.plotly_chart(plot_stage_overlay(stages, feature_focus, max_points=max_plot_points), use_container_width=True, key="order_aware_stage_overlay")

stage_rows = []
raw_stage = stages[0][1]
prev_stage = None
for stage_name, stage_df in stages:
    metrics_stage = compute_stage_audit_metrics(
        stage_df,
        feature_focus,
        profile["weights"],
        baseline=raw_stage,
        previous=prev_stage,
        raw_missing_mask=raw_missing_mask_focus,
        spatial_context=df,
    )
    metrics_stage = {"Stage": stage_name, **metrics_stage}
    stage_rows.append(metrics_stage)
    prev_stage = stage_df
stage_df_metrics = pd.DataFrame(stage_rows)

st.markdown("**Stage-wise metrics**")
metric_cols_main = [
    "Stage", "Quality Index", "Completeness %", "Missing %", "Imputed-from-raw %",
    "IQR outlier %", "MAD outlier %", "Drift score", "Volatility score", "Jitter proxy",
    "Mean abs change from previous", "Mean abs change from raw"
]
st.dataframe(
    stage_df_metrics[metric_cols_main].style.format({c: "{:.3f}" for c in metric_cols_main if c != "Stage"}),
    use_container_width=True,
)

m1, m2 = st.columns([1.0, 1.0])
with m1:
    fig_stage_q = px.bar(stage_df_metrics, x="Stage", y="Quality Index", title="Quality Index by pipeline stage")
    fig_stage_q.update_layout(height=330, margin=dict(l=30, r=20, t=50, b=90))
    st.plotly_chart(fig_stage_q, use_container_width=True, key="stage_qi_bar")
with m2:
    delta_metrics = ["Missing %", "IQR outlier %", "MAD outlier %", "Drift score", "Volatility score", "Jitter proxy"]
    stage_long = stage_df_metrics[["Stage"] + delta_metrics].melt(id_vars="Stage", var_name="Metric", value_name="Value")
    fig_stage_lines = px.line(stage_long, x="Stage", y="Value", color="Metric", markers=True, title="Selected quality indicators by stage")
    fig_stage_lines.update_layout(height=330, margin=dict(l=30, r=20, t=50, b=90), legend=dict(orientation="h"))
    st.plotly_chart(fig_stage_lines, use_container_width=True, key="stage_metric_lines")

with st.expander("Full stage metrics: distribution, entropy, run structure", expanded=False):
    full_cols = [
        "Stage", "Quality Index", "Completeness %", "Missing %", "Imputed-from-raw %",
        "Longest missing run", "Missing burstiness", "IQR outlier %", "MAD outlier %",
        "Skewness", "Kurtosis excess", "Drift score", "Volatility score", "Entropy score",
        "Jitter proxy", "Std", "Min", "Max", "Mean abs change from raw", "Mean abs change from previous"
    ]
    st.dataframe(stage_df_metrics[full_cols].style.format({c: "{:.4f}" for c in full_cols if c != "Stage"}), use_container_width=True)

with st.expander("MV × Outlier interaction grid for the selected feature", expanded=False):
    st.caption(
        "This compares how missing-value and outlier methods behave as a pair. "
        "It uses the selected audit order and keeps normalization/smoothing off, so the numbers remain on the original signal scale."
    )
    gi1, gi2, gi3 = st.columns([1, 1, 1])
    with gi1:
        pair_mv = st.multiselect("MV methods for pair grid", mv_methods_all, default=[m for m in [audit_mv, "Linear Interpolation", "LOCF"] if m in mv_methods_all], key="pair_grid_mv")
    with gi2:
        pair_out = st.multiselect("Outlier methods for pair grid", ["None"] + outlier_methods_all, default=[m for m in [audit_out, "None", "IQR", "MAD"] if m in ["None"] + outlier_methods_all], key="pair_grid_out")
    with gi3:
        pair_metric = st.selectbox("Rank by", ["Quality Index", "Missing %", "IQR outlier %", "Drift score", "Volatility score", "Mean abs change from raw"], key="pair_grid_rank_metric")

    if st.button("Run MV × Outlier interaction grid", key="run_pair_grid"):
        pair_rows = []
        for mv in pair_mv:
            for outm in pair_out:
                try:
                    pair_stages = build_pipeline_stages(df, feature_focus, mv, outm, "None", "None", audit_order_eff, settings)
                    # last stage before normalization is either step 2 or 2b; with norm None/smooth None the final duplicates are okay,
                    # so select the last stage whose name starts with 2 or 2b.
                    pair_clean_stage = pair_stages[-3][1] if audit_order_eff == "Missing → Outlier" else pair_stages[-3][1]
                    pair_metrics = compute_stage_audit_metrics(pair_clean_stage, feature_focus, profile["weights"], baseline=raw_stage, previous=None, raw_missing_mask=raw_missing_mask_focus, spatial_context=df)
                    pair_rows.append({"Order": audit_order_eff, "MV": mv, "Outlier": outm, **pair_metrics})
                except Exception as e:
                    pair_rows.append({"Order": audit_order_eff, "MV": mv, "Outlier": outm, "Quality Index": np.nan, "Error": str(e)[:140]})
        if pair_rows:
            pair_df = pd.DataFrame(pair_rows)
            ascending = pair_metric in ["Missing %", "IQR outlier %", "Drift score", "Volatility score", "Mean abs change from raw"]
            pair_df = pair_df.sort_values(pair_metric, ascending=ascending, na_position="last")
            st.dataframe(
                pair_df[["Order", "MV", "Outlier", "Quality Index", "Completeness %", "Missing %", "Imputed-from-raw %", "IQR outlier %", "MAD outlier %", "Drift score", "Volatility score", "Mean abs change from raw"]]
                .style.format({"Quality Index":"{:.2f}", "Completeness %":"{:.2f}", "Missing %":"{:.2f}", "Imputed-from-raw %":"{:.2f}", "IQR outlier %":"{:.2f}", "MAD outlier %":"{:.2f}", "Drift score":"{:.4f}", "Volatility score":"{:.4f}", "Mean abs change from raw":"{:.4f}"}),
                use_container_width=True,
            )
            heat = pair_df.pivot_table(index="MV", columns="Outlier", values="Quality Index", aggfunc="mean")
            if not heat.empty:
                fig_heat = px.imshow(heat, text_auto=".1f", aspect="auto", title="MV × Outlier Quality Index")
                fig_heat.update_layout(height=360, margin=dict(l=30, r=20, t=50, b=40))
                st.plotly_chart(fig_heat, use_container_width=True, key="pair_grid_heatmap")

# -----------------------------------------------------------------------------
# Step-wise method comparison lab
# -----------------------------------------------------------------------------
st.subheader("Method comparison lab — independent single-step comparisons")
st.caption(
    "This older lab compares method families mostly in isolation; it is not the best place to inspect order effects. Use the order-aware audit above for actual intermediate pipeline states. "
    "This is intentionally step-wise: first missing-value methods on the raw feature, "
    "then outlier methods after the selected imputation, then normalization after MV+outlier, "
    "and finally smoothing after MV+outlier+normalization."
)

focus_raw_df = df[[feature_focus]].copy()
focus_raw_series = focus_raw_df[feature_focus].astype(float)


tab_mv, tab_out, tab_norm, tab_smooth, tab_grid = st.tabs([
    "1 Missing values", "2 Outliers", "3 Normalization", "4 Smoothing", "Pipeline grid"
])

with tab_mv:
    st.markdown("**Question:** How would each imputation method reconstruct gaps in the selected feature?")
    mv_compare = st.multiselect(
        "Compare missing-value methods",
        mv_methods_all,
        default=[m for m in [rec.missing_method, "Linear Interpolation", "KNN Imputation"] if m in mv_methods_all],
        key="mv_compare_methods",
    )
    fig_mv_cmp = go.Figure()
    raw_plot = focus_raw_series
    if len(raw_plot) > max_plot_points:
        idx = np.linspace(0, len(raw_plot) - 1, max_plot_points).astype(int)
        raw_plot = raw_plot.iloc[idx]
    fig_mv_cmp.add_trace(go.Scatter(x=raw_plot.index, y=raw_plot.values, mode="lines", name="Raw", line=dict(width=1.3, dash="dot")))
    mv_summary_rows = []
    for method in mv_compare:
        try:
            tmp = apply_missing(focus_raw_df, method, n_neighbors=settings["knn_neighbors"], mice_iter=settings["mice_iter"])[feature_focus]
            tmp_plot = tmp.iloc[idx] if len(tmp) > max_plot_points else tmp
            fig_mv_cmp.add_trace(go.Scatter(x=tmp_plot.index, y=tmp_plot.values, mode="lines", name=method, line=dict(width=1.4)))
            mv_summary_rows.append({
                "Method": method,
                "Missing after %": float(tmp.isna().mean() * 100),
                "Std after": float(tmp.std(skipna=True)),
                "Mean abs change vs raw": float((tmp - focus_raw_series).abs().mean(skipna=True)),
            })
        except Exception as e:
            mv_summary_rows.append({"Method": method, "Missing after %": np.nan, "Std after": np.nan, "Mean abs change vs raw": np.nan, "Error": str(e)[:120]})
    fig_mv_cmp.update_layout(height=390, margin=dict(l=30, r=20, t=30, b=35), legend=dict(orientation="h"))
    st.plotly_chart(fig_mv_cmp, use_container_width=True, key="mv_method_comparison_plot")
    if mv_summary_rows:
        st.dataframe(pd.DataFrame(mv_summary_rows).style.format({"Missing after %":"{:.2f}", "Std after":"{:.4f}", "Mean abs change vs raw":"{:.4f}"}), use_container_width=True)

with tab_out:
    st.markdown("**Question:** Which values would each outlier method remove or clip?")
    out_base_mv = st.selectbox("Base imputation before outlier comparison", ["None"] + mv_methods_all, index=(["None"] + mv_methods_all).index(rec.missing_method if rec.missing_method in mv_methods_all else "None"), key="out_base_mv")
    out_compare = st.multiselect(
        "Compare outlier methods",
        outlier_methods_all,
        default=[m for m in [rec.outlier_method, "IQR", "MAD", "Isolation Forest"] if m in outlier_methods_all],
        key="out_compare_methods",
    )
    out_base = apply_missing(focus_raw_df, out_base_mv, n_neighbors=settings["knn_neighbors"], mice_iter=settings["mice_iter"])
    base_series = out_base[feature_focus].astype(float)
    fig_out_cmp = go.Figure()
    base_plot = base_series.iloc[idx] if len(base_series) > max_plot_points else base_series
    fig_out_cmp.add_trace(go.Scatter(x=base_plot.index, y=base_plot.values, mode="lines", name=f"Base after {out_base_mv}", line=dict(width=1.3, dash="dot")))
    out_summary_rows = []
    for method in out_compare:
        try:
            tmp_df = apply_outliers(out_base, method, z_thresh=settings["z_thresh"], iqr_factor=settings["iqr_factor"], mad_thresh=settings["mad_thresh"], contamination=settings["if_contamination"], winsor_q=settings["winsor_q"])
            tmp = tmp_df[feature_focus]
            tmp_plot = tmp.iloc[idx] if len(tmp) > max_plot_points else tmp
            fig_out_cmp.add_trace(go.Scatter(x=tmp_plot.index, y=tmp_plot.values, mode="lines", name=method, line=dict(width=1.4)))
            removed = int(tmp.isna().sum() - base_series.isna().sum())
            changed = int(((tmp.fillna(np.inf) != base_series.fillna(np.inf)) & ~(tmp.isna() & base_series.isna())).sum())
            out_summary_rows.append({
                "Method": method,
                "New NaNs / removed": removed,
                "Changed values": changed,
                "Outlier/changed %": changed / max(1, len(tmp)) * 100,
                "Std after": float(tmp.std(skipna=True)),
            })
        except Exception as e:
            out_summary_rows.append({"Method": method, "New NaNs / removed": np.nan, "Changed values": np.nan, "Outlier/changed %": np.nan, "Std after": np.nan, "Error": str(e)[:120]})
    fig_out_cmp.update_layout(height=390, margin=dict(l=30, r=20, t=30, b=35), legend=dict(orientation="h"))
    st.plotly_chart(fig_out_cmp, use_container_width=True, key="out_method_comparison_plot")
    if out_summary_rows:
        st.dataframe(pd.DataFrame(out_summary_rows).style.format({"Outlier/changed %":"{:.2f}", "Std after":"{:.4f}"}), use_container_width=True)

with tab_norm:
    st.markdown("**Question:** How do scaling methods reshape the same cleaned signal?")
    norm_base_mv = st.selectbox("Base imputation", ["None"] + mv_methods_all, index=(["None"] + mv_methods_all).index(rec.missing_method if rec.missing_method in mv_methods_all else "None"), key="norm_base_mv")
    norm_base_out = st.selectbox("Base outlier handling", ["None"] + outlier_methods_all, index=(["None"] + outlier_methods_all).index(rec.outlier_method if rec.outlier_method in outlier_methods_all else "None"), key="norm_base_out")
    norm_compare = st.multiselect("Compare normalization methods", norm_methods_all, default=[m for m in [rec.normalization_method, "StandardScaler", "RobustScaler", "MinMaxScaler"] if m in norm_methods_all], key="norm_compare_methods")
    norm_base = apply_missing(focus_raw_df, norm_base_mv, n_neighbors=settings["knn_neighbors"], mice_iter=settings["mice_iter"])
    norm_base = apply_outliers(norm_base, norm_base_out, z_thresh=settings["z_thresh"], iqr_factor=settings["iqr_factor"], mad_thresh=settings["mad_thresh"], contamination=settings["if_contamination"], winsor_q=settings["winsor_q"])
    norm_base = apply_missing(norm_base, norm_base_mv if norm_base_mv != "None" else "Linear Interpolation", n_neighbors=settings["knn_neighbors"], mice_iter=settings["mice_iter"])
    fig_norm_cmp = go.Figure()
    norm_base_series = norm_base[feature_focus].astype(float)
    norm_base_plot = norm_base_series.iloc[idx] if len(norm_base_series) > max_plot_points else norm_base_series
    fig_norm_cmp.add_trace(go.Scatter(x=norm_base_plot.index, y=norm_base_plot.values, mode="lines", name="Base cleaned", line=dict(width=1.3, dash="dot")))
    norm_summary_rows = []
    for method in norm_compare:
        try:
            tmp = apply_normalization(norm_base, method)[feature_focus]
            tmp_plot = tmp.iloc[idx] if len(tmp) > max_plot_points else tmp
            fig_norm_cmp.add_trace(go.Scatter(x=tmp_plot.index, y=tmp_plot.values, mode="lines", name=method, line=dict(width=1.4)))
            norm_summary_rows.append({
                "Method": method,
                "Mean": float(tmp.mean(skipna=True)),
                "Std": float(tmp.std(skipna=True)),
                "Min": float(tmp.min(skipna=True)),
                "Max": float(tmp.max(skipna=True)),
            })
        except Exception as e:
            norm_summary_rows.append({"Method": method, "Mean": np.nan, "Std": np.nan, "Min": np.nan, "Max": np.nan, "Error": str(e)[:120]})
    fig_norm_cmp.update_layout(height=390, margin=dict(l=30, r=20, t=30, b=35), legend=dict(orientation="h"))
    st.plotly_chart(fig_norm_cmp, use_container_width=True, key="norm_method_comparison_plot")
    if norm_summary_rows:
        st.dataframe(pd.DataFrame(norm_summary_rows).style.format({"Mean":"{:.4f}", "Std":"{:.4f}", "Min":"{:.4f}", "Max":"{:.4f}"}), use_container_width=True)

with tab_smooth:
    st.markdown("**Question:** How aggressively do the smoothing/filtering methods change the selected signal?")
    smooth_compare = st.multiselect("Compare smoothing methods", smooth_methods_all, default=[m for m in [rec.smoothing_method, "Butterworth", "Savitzky-Golay"] if m in smooth_methods_all], key="smooth_compare_methods")
    smooth_base = execute_pipeline(df, [feature_focus], PipelineRecommendation(rec.missing_method, rec.outlier_method, rec.normalization_method, "None", rec.confidence, rec.expected_gain, rec.rationale, rec.risk_flags), settings)
    smooth_base_series = smooth_base[feature_focus].astype(float)
    fig_smooth_cmp = go.Figure()
    smooth_base_plot = smooth_base_series.iloc[idx] if len(smooth_base_series) > max_plot_points else smooth_base_series
    fig_smooth_cmp.add_trace(go.Scatter(x=smooth_base_plot.index, y=smooth_base_plot.values, mode="lines", name="Base before smoothing", line=dict(width=1.3, dash="dot")))
    smooth_summary_rows = []
    for method in smooth_compare:
        try:
            tmp = apply_smoothing(smooth_base, method, fs=settings["sampling_rate"], cutoff=settings["filter_cutoff"], sg_window=settings["sg_window"], sg_poly=settings["sg_poly"], fourier_keep_ratio=settings["fourier_keep_ratio"])[feature_focus]
            tmp_plot = tmp.iloc[idx] if len(tmp) > max_plot_points else tmp
            fig_smooth_cmp.add_trace(go.Scatter(x=tmp_plot.index, y=tmp_plot.values, mode="lines", name=method, line=dict(width=1.4)))
            smooth_summary_rows.append({
                "Method": method,
                "Mean abs change": float((tmp - smooth_base_series).abs().mean(skipna=True)),
                "Variance reduction %": float((1 - (tmp.var(skipna=True) / smooth_base_series.var(skipna=True))) * 100) if smooth_base_series.var(skipna=True) else np.nan,
                "Std after": float(tmp.std(skipna=True)),
            })
        except Exception as e:
            smooth_summary_rows.append({"Method": method, "Mean abs change": np.nan, "Variance reduction %": np.nan, "Std after": np.nan, "Error": str(e)[:120]})
    fig_smooth_cmp.update_layout(height=390, margin=dict(l=30, r=20, t=30, b=35), legend=dict(orientation="h"))
    st.plotly_chart(fig_smooth_cmp, use_container_width=True, key="smooth_method_comparison_plot")
    if smooth_summary_rows:
        st.dataframe(pd.DataFrame(smooth_summary_rows).style.format({"Mean abs change":"{:.4f}", "Variance reduction %":"{:.2f}", "Std after":"{:.4f}"}), use_container_width=True)

with tab_grid:
    st.markdown("**Fast local grid:** compare several complete pipelines on the selected feature using the intrinsic Quality Index.")
    grid_mv = st.multiselect("Grid MV", mv_methods_all, default=[m for m in [rec.missing_method, "Linear Interpolation"] if m in mv_methods_all], key="grid_mv")
    grid_out = st.multiselect("Grid outlier", ["None"] + outlier_methods_all, default=[m for m in [rec.outlier_method, "None", "IQR"] if m in ["None"] + outlier_methods_all], key="grid_out")
    grid_norm = st.multiselect("Grid normalization", ["None"] + norm_methods_all, default=[m for m in [rec.normalization_method, "RobustScaler"] if m in ["None"] + norm_methods_all], key="grid_norm")
    if st.button("Run selected mini-grid", key="run_mini_grid"):
        grid_rows = []
        for mv in grid_mv:
            for out in grid_out:
                for norm in grid_norm:
                    try:
                        tmp_rec = PipelineRecommendation(mv, out, norm, "None", rec.confidence, rec.expected_gain, rec.rationale, rec.risk_flags)
                        tmp_processed = execute_pipeline(df, [feature_focus], tmp_rec, settings)
                        tmp_df = df.copy()
                        tmp_df[feature_focus] = tmp_processed[feature_focus]
                        tmp_prof = profile_features(tmp_df, [feature_focus], profile["weights"]).iloc[0]
                        grid_rows.append({
                            "MV": mv,
                            "Outlier": out,
                            "Norm": norm,
                            "Quality Index": float(tmp_prof["quality_index"]),
                            "Missing %": float(tmp_prof["missing_rate"]),
                            "Outlier IQR %": float(tmp_prof["outlier_iqr_rate"]),
                            "Skewness": float(tmp_prof["skewness"]),
                            "Kurtosis": float(tmp_prof["kurtosis_excess"]),
                        })
                    except Exception as e:
                        grid_rows.append({"MV": mv, "Outlier": out, "Norm": norm, "Quality Index": np.nan, "Error": str(e)[:120]})
        if grid_rows:
            grid_df = pd.DataFrame(grid_rows).sort_values("Quality Index", ascending=False, na_position="last")
            st.dataframe(grid_df.style.format({"Quality Index":"{:.2f}", "Missing %":"{:.2f}", "Outlier IQR %":"{:.2f}", "Skewness":"{:.2f}", "Kurtosis":"{:.2f}"}), use_container_width=True)

with st.expander("Distribution preview", expanded=False):
    d1, d2 = st.columns(2)
    with d1:
        fig_hist_raw = px.histogram(df, x=feature_focus, nbins=60, title="Raw distribution")
        fig_hist_raw.update_layout(height=320, margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig_hist_raw, use_container_width=True, key="hist_raw")
    with d2:
        tmp_dist = pd.DataFrame({feature_focus: processed_features[feature_focus]})
        fig_hist_proc = px.histogram(tmp_dist, x=feature_focus, nbins=60, title="Processed distribution")
        fig_hist_proc.update_layout(height=320, margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig_hist_proc, use_container_width=True, key="hist_proc")

# Optional downstream validation
st.subheader("Optional downstream validation")
non_feature_cols = [c for c in df.columns if c not in selected_features]
target_col = st.selectbox("Target column for baseline vs. CIRRUS++ comparison", ["None"] + non_feature_cols)
if target_col != "None":
    with st.spinner("Running lightweight downstream validation..."):
        val_df = downstream_validation(df, final_df, selected_features, target_col)
    if val_df is None or val_df.empty:
        st.warning("Validation could not be computed. Need a target column with enough non-missing labels/values and enough samples.")
    else:
        st.dataframe(val_df.style.format({"CV mean": "{:.4f}", "CV std": "{:.4f}"}), use_container_width=True)
        fig_val = px.bar(val_df, x="Model", y="CV mean", color="Data", barmode="group", error_y="CV std", facet_col="Metric")
        fig_val.update_layout(height=360, margin=dict(l=30, r=20, t=40, b=40))
        st.plotly_chart(fig_val, use_container_width=True, key="downstream_validation")
else:
    st.caption("Select a target column if the dataset contains labels/classes/scores. This directly addresses the reviewer request for predictive before/after evidence.")

# Ground-truth validation
st.subheader("Ground-truth validation")
st.caption(
    "Upload the clean ground-truth CSV that matches the noisy file. "
    "CIRRUS++ then compares the processed noisy signal against the clean signal using MAE, RMSE, and correlation. "
    "For MAE/RMSE, the default comparison uses the reconstruction stage before normalization so the scale stays meaningful."
)

gt_file = st.file_uploader("Optional clean / ground-truth CSV", type=["csv"], key="ground_truth_csv")
if gt_file is not None:
    try:
        sep_gt = None if delimiter == "Auto" else delimiter
        dec_gt = "." if decimal == "Auto" else decimal
        if sep_gt is None:
            gt_df = pd.read_csv(gt_file, sep=None, engine="python", decimal=dec_gt)
        else:
            gt_df = pd.read_csv(gt_file, sep=sep_gt, decimal=dec_gt, low_memory=False)
        gt_df = auto_convert_numeric_columns(gt_df)
    except Exception as e:
        st.error(f"Ground-truth CSV could not be read: {e}")
        gt_df = None

    if gt_df is not None:
        available_gt_features = [c for c in selected_features if c in gt_df.columns]
        if not available_gt_features:
            st.warning("No selected signal feature exists in the ground-truth file. Check column names.")
        else:
            cgt1, cgt2, cgt3 = st.columns([1, 1, 1])
            with cgt1:
                possible_keys = [c for c in ["Sample_ID", "sample_id", "Row", "Time_s"] if c in df.columns and c in gt_df.columns]
                align_key = st.selectbox("Alignment", ["Row order"] + possible_keys, index=1 if "Sample_ID" in possible_keys else 0)
            with cgt2:
                gt_features = st.multiselect("Features to compare", available_gt_features, default=available_gt_features[: min(6, len(available_gt_features))])
            with cgt3:
                comparison_stage = st.selectbox(
                    "Compared CIRRUS stage",
                    ["Reconstruction before normalization", "Final processed output"],
                    index=0,
                    help="Use reconstruction before normalization for MAE/RMSE. Final output is useful only when scale changes are intentional or normalization is None."
                )

            if gt_features:
                with st.spinner("Computing ground-truth validation..."):
                    if comparison_stage == "Reconstruction before normalization":
                        pred_stage = execute_cleaning_stage(df, gt_features, rec, settings)
                    else:
                        pred_stage = final_df[gt_features].copy()

                    if align_key == "Row order":
                        n_align = min(len(gt_df), len(pred_stage))
                        clean_aligned = gt_df.iloc[:n_align][gt_features].reset_index(drop=True)
                        pred_aligned = pred_stage.iloc[:n_align][gt_features].reset_index(drop=True)
                        raw_aligned = df.iloc[:n_align][gt_features].reset_index(drop=True)
                    else:
                        clean_tmp = gt_df[[align_key] + gt_features].copy()
                        pred_tmp = pd.concat([df[[align_key]].copy(), pred_stage[gt_features].copy()], axis=1)
                        raw_tmp = df[[align_key] + gt_features].copy()
                        merged_pred = clean_tmp.merge(pred_tmp, on=align_key, suffixes=("_clean", "_processed"))
                        merged_raw = clean_tmp.merge(raw_tmp, on=align_key, suffixes=("_clean", "_raw"))
                        clean_aligned = pd.DataFrame({f: merged_pred[f + "_clean"] for f in gt_features})
                        pred_aligned = pd.DataFrame({f: merged_pred[f + "_processed"] for f in gt_features})
                        raw_aligned = pd.DataFrame({f: merged_raw[f + "_raw"] for f in gt_features})

                    rows_gt = []
                    for f in gt_features:
                        clean_s = pd.to_numeric(clean_aligned[f], errors="coerce")
                        proc_s = pd.to_numeric(pred_aligned[f], errors="coerce")
                        raw_s = pd.to_numeric(raw_aligned[f], errors="coerce")

                        mask_proc = clean_s.notna() & proc_s.notna()
                        mask_raw = clean_s.notna() & raw_s.notna()

                        def _metrics(a, b):
                            if len(a) < 3:
                                return np.nan, np.nan, np.nan
                            err = b - a
                            mae = float(np.mean(np.abs(err)))
                            rmse = float(np.sqrt(np.mean(err ** 2)))
                            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
                            return mae, rmse, corr

                        mae_p, rmse_p, corr_p = _metrics(clean_s[mask_proc].to_numpy(), proc_s[mask_proc].to_numpy())
                        mae_r, rmse_r, corr_r = _metrics(clean_s[mask_raw].to_numpy(), raw_s[mask_raw].to_numpy())
                        rows_gt.append({
                            "Feature": f,
                            "Raw MAE": mae_r,
                            "CIRRUS MAE": mae_p,
                            "MAE gain %": ((mae_r - mae_p) / mae_r * 100) if mae_r and not np.isnan(mae_r) else np.nan,
                            "Raw RMSE": rmse_r,
                            "CIRRUS RMSE": rmse_p,
                            "RMSE gain %": ((rmse_r - rmse_p) / rmse_r * 100) if rmse_r and not np.isnan(rmse_r) else np.nan,
                            "Raw corr": corr_r,
                            "CIRRUS corr": corr_p,
                            "Compared rows": int(mask_proc.sum()),
                        })

                    gt_result_df = pd.DataFrame(rows_gt)
                    st.dataframe(gt_result_df.style.format({
                        "Raw MAE":"{:.5f}", "CIRRUS MAE":"{:.5f}", "MAE gain %":"{:.2f}",
                        "Raw RMSE":"{:.5f}", "CIRRUS RMSE":"{:.5f}", "RMSE gain %":"{:.2f}",
                        "Raw corr":"{:.4f}", "CIRRUS corr":"{:.4f}"
                    }), use_container_width=True)

                    fig_gt = go.Figure()
                    fig_gt.add_trace(go.Bar(x=gt_result_df["Feature"], y=gt_result_df["Raw RMSE"], name="Raw vs clean RMSE"))
                    fig_gt.add_trace(go.Bar(x=gt_result_df["Feature"], y=gt_result_df["CIRRUS RMSE"], name="CIRRUS vs clean RMSE"))
                    fig_gt.update_layout(barmode="group", height=360, margin=dict(l=30, r=20, t=35, b=80), yaxis_title="RMSE")
                    st.plotly_chart(fig_gt, use_container_width=True, key="ground_truth_rmse_plot")

                    st.caption(
                        "Interpretation: positive gain means CIRRUS is closer to the clean ground truth than the noisy raw signal. "
                        "If Final processed output is normalized, MAE/RMSE are only meaningful if the clean reference is transformed the same way."
                    )
else:
    st.caption("For synthetic experiments: upload the clean CSV here while the noisy CSV remains the main uploaded dataset.")

# Export
st.subheader("Export")
export_col1, export_col2, export_col3 = st.columns(3)
with export_col1:
    csv_bytes = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download processed CSV", csv_bytes, file_name="cirrus_processed.csv", mime="text/csv", use_container_width=True)
with export_col2:
    pipeline_json = json.dumps(st.session_state["pipeline"], indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button("Download pipeline JSON", pipeline_json, file_name="cirrus_pipeline.json", mime="application/json", use_container_width=True)
with export_col3:
    report = {
        "domain": inferred_domain,
        "domain_profile": profile,
        "pipeline": st.session_state["pipeline"],
        "before_profile": profile_df.to_dict(orient="records"),
        "after_profile": after_profile_df.to_dict(orient="records"),
        "comparison": compare_df.to_dict(orient="records"),
    }
    st.download_button("Download quality report JSON", json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"), file_name="cirrus_quality_report.json", mime="application/json", use_container_width=True)

with st.expander("Reviewer-facing upgrade summary", expanded=False):
    st.markdown(
        """
**What this prototype now demonstrates more strongly:**

1. **Technical depth:** recommendation is method scoring based on missingness pattern, outlier structure, skewness/kurtosis, drift, volatility, correlation, and domain priors.
2. **Quantitative evaluation:** Quality Index before/after and optional downstream validation against a target column.
3. **Scalability awareness:** preview downsampling, explicit method execution, and high-frequency domain profile. For a full VLDB version, this should still be backed by DuckDB/Polars/Dask or chunked Arrow storage.
4. **Domain adaptation:** domain profile influences metric weights and method priors.
5. **Interactivity:** users can modify all major thresholds, domain assumptions, and pipeline methods.
        """
    )
