"""
WindBench Rich Feature Engineering.

Implements SOTA feature engineering pipeline from Hybrid Autoencoder 2025
(arxiv 2510.15010) and related works:
  - Rolling statistics (mean, std, skew, kurtosis) over multiple windows
  - First and second derivatives
  - FFT-based frequency-domain features
  - Cross-correlations between sensor pairs

For each raw sensor, we generate ~15 derived features. A dataset with
20 base features expands to ~150-300 engineered features.

Usage:
    from src.benchmark.features import build_rich_features
    X = build_rich_features(df, feature_cols=['wind_speed_ms', ...])
"""

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import rfft, rfftfreq


# Rolling windows in number-of-observations (assume hourly data by default)
ROLLING_WINDOWS_HOURS = [6, 24, 72]


def rolling_stats(series: pd.Series, window: int) -> Dict[str, pd.Series]:
    """Compute rolling mean, std, skew, kurtosis for one series."""
    rolled = series.rolling(window, min_periods=max(2, window // 4))
    return {
        f"{series.name}_ma_{window}": rolled.mean(),
        f"{series.name}_std_{window}": rolled.std(),
        f"{series.name}_skew_{window}": rolled.skew(),
        f"{series.name}_kurt_{window}": rolled.kurt(),
    }


def derivative_features(series: pd.Series) -> Dict[str, pd.Series]:
    """First and second finite differences."""
    d1 = series.diff()
    d2 = d1.diff()
    return {
        f"{series.name}_diff1": d1,
        f"{series.name}_diff2": d2,
        f"{series.name}_abs_diff1": d1.abs(),
    }


def fft_features(series: pd.Series, window: int = 24) -> Dict[str, np.ndarray]:
    """
    Sliding-window FFT features: total spectral power, and power in
    low/mid/high frequency bands.
    """
    vals = series.fillna(series.mean()).to_numpy()
    n = len(vals)
    if n < window:
        return {}

    feat_total = np.full(n, np.nan)
    feat_low = np.full(n, np.nan)
    feat_mid = np.full(n, np.nan)
    feat_high = np.full(n, np.nan)
    feat_dom = np.full(n, np.nan)

    for i in range(window, n):
        chunk = vals[i - window:i]
        fft_out = rfft(chunk - chunk.mean())
        power = np.abs(fft_out) ** 2

        # Split spectrum into low/mid/high bands
        k = len(power)
        low = power[: max(1, k // 3)].sum()
        mid = power[k // 3 : 2 * k // 3].sum()
        high = power[2 * k // 3 :].sum()
        total = power.sum()

        feat_total[i] = total
        feat_low[i] = low / max(total, 1e-9)
        feat_mid[i] = mid / max(total, 1e-9)
        feat_high[i] = high / max(total, 1e-9)
        # Dominant frequency index (normalized)
        if len(power) > 1:
            feat_dom[i] = np.argmax(power[1:]) / len(power)

    return {
        f"{series.name}_fft_total_{window}": feat_total,
        f"{series.name}_fft_low_{window}": feat_low,
        f"{series.name}_fft_mid_{window}": feat_mid,
        f"{series.name}_fft_high_{window}": feat_high,
        f"{series.name}_fft_dom_freq_{window}": feat_dom,
    }


def cross_correlations(df: pd.DataFrame, pairs: List[tuple],
                        window: int = 24) -> Dict[str, np.ndarray]:
    """Rolling correlation between sensor pairs (captures coupling breakdowns)."""
    out = {}
    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        s_a = df[a].fillna(df[a].mean())
        s_b = df[b].fillna(df[b].mean())
        rolling_corr = s_a.rolling(window, min_periods=5).corr(s_b)
        out[f"corr_{a}__{b}_{window}"] = rolling_corr.to_numpy()
    return out


DEFAULT_CORRELATION_PAIRS = [
    ("wind_speed_ms", "active_power_kw"),        # power curve
    ("wind_speed_ms", "rotor_speed_rpm"),         # speed follows wind
    ("rotor_speed_rpm", "active_power_kw"),      # RPM → power (P ∝ ω³)
    ("gearbox_oil_temp_c", "active_power_kw"),    # thermal response to load
    ("gearbox_oil_temp_c", "ambient_temp_c"),     # thermal baseline
    ("main_bearing_temp_c", "ambient_temp_c"),
    ("gearbox_oil_temp_c", "main_bearing_temp_c"),  # thermal coupling
]


def build_rich_features(
    df: pd.DataFrame,
    base_features: Optional[List[str]] = None,
    include_fft: bool = True,
    include_correlations: bool = True,
    correlation_pairs: Optional[List[tuple]] = None,
) -> pd.DataFrame:
    """
    Build the full rich feature set from a canonical WindBench DataFrame.

    Args:
        df: DataFrame with 'event_ts', 'turbine_id', and canonical features.
        base_features: Which columns to expand. Defaults to all numeric canonicals
                       except id/timestamp.
        include_fft: Whether to compute FFT features (slower but powerful).
        include_correlations: Whether to compute cross-sensor correlations.
        correlation_pairs: Custom pairs for cross-correlation; defaults provided.

    Returns:
        DataFrame with all engineered features. Preserves event_ts, turbine_id.
    """
    df = df.sort_values(["turbine_id", "event_ts"]).reset_index(drop=True).copy()

    if base_features is None:
        from src.benchmark.schema import CANONICAL_RAW_FEATURES
        base_features = [c for c in CANONICAL_RAW_FEATURES
                          if c in df.columns and df[c].notna().sum() > 100]

    if correlation_pairs is None:
        correlation_pairs = DEFAULT_CORRELATION_PAIRS

    # Per-turbine grouping so rolling computations don't cross turbines
    new_frames = []
    for tid, g in df.groupby("turbine_id", sort=False):
        g = g.sort_values("event_ts").reset_index(drop=True)
        new_feats = {}

        for feat in base_features:
            if feat not in g.columns:
                continue
            s = g[feat]
            if s.notna().sum() < 20:
                continue

            # Rolling statistics across multiple windows
            for w in ROLLING_WINDOWS_HOURS:
                for name, vals in rolling_stats(s, w).items():
                    new_feats[name] = vals.values

            # Derivatives
            for name, vals in derivative_features(s).items():
                new_feats[name] = vals.values

            # FFT (slower — only on key sensors if needed)
            if include_fft:
                for name, vals in fft_features(s, window=24).items():
                    new_feats[name] = vals

        # Correlations (computed over full DF since we have grouped g)
        if include_correlations:
            corrs = cross_correlations(g, correlation_pairs, window=24)
            for name, vals in corrs.items():
                new_feats[name] = vals

        # Combine
        feat_df = pd.DataFrame(new_feats)
        feat_df.index = g.index
        combined = pd.concat([g, feat_df], axis=1)
        new_frames.append(combined)

    result = pd.concat(new_frames, ignore_index=True)
    return result


def engineered_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of column names produced by build_rich_features."""
    patterns = ["_ma_", "_std_", "_skew_", "_kurt_",
                "_diff1", "_diff2", "_abs_diff1",
                "_fft_", "corr_"]
    return [c for c in df.columns if any(p in c for p in patterns)]


if __name__ == "__main__":
    # Demo
    import sys
    sys.path.insert(0, ".")

    df = pd.read_parquet("data/benchmark/harmonized/care_farm_a.parquet")
    # Subsample for speed in demo
    df = df.head(5000)
    print(f"Input: {df.shape}")
    enriched = build_rich_features(df)
    print(f"Output: {enriched.shape}")
    eng = engineered_feature_columns(enriched)
    print(f"Added {len(eng)} engineered features:")
    for c in eng[:20]:
        print(f"  {c}")
    print(f"  ... and {len(eng) - 20} more")
