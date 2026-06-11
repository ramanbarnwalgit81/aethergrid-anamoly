"""
Fault-Type-Specific Models.

NOVEL CONTRIBUTION: Instead of one universal anomaly detector, train a
specialized model for each fault subsystem (gearbox, bearing, hydraulic,
transformer). At inference, route predictions through a fault-classifier
that weights the specialist models.

Hypothesis: Different fault types have different feature signatures
  - Gearbox faults → temperature + vibration subspace
  - Generator bearing → generator temperature + RPM subspace
  - Hydraulic → pitch/pressure subspace
  - Transformer → electrical subspace

A universal detector struggles when features relevant to fault X dilute
those relevant to fault Y. Specialists can focus.

Usage:
    python -m src.benchmark.fault_specific
"""

from pathlib import Path
import json
import os, sys

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

# IMPORTANT: torch MUST be imported before numpy/pandas on Windows to avoid
# an OpenMP DLL initialization conflict (libiomp5md.dll).
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from src.benchmark.features import build_rich_features
from src.benchmark.vae_baseline import train_vae, vae_anomaly_score
from src.benchmark.care_sota import build_event_features, ewma_smooth

CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")
DATASETS_DIR = CARE_DIR / "datasets"
EVENT_INFO = CARE_DIR / "event_info.csv"
RESULTS_DIR = Path("docs/results")


# Fault-subsystem feature subsets (from CARE Farm A semantic mappings)
FAULT_FEATURE_SUBSETS = {
    "Gearbox failure": {
        "primary": ["gearbox_oil_temp_c", "gearbox_bearing_temp_c"],
        "supporting": ["active_power_kw", "rotor_speed_rpm",
                        "main_bearing_temp_c", "ambient_temp_c"],
    },
    "Generator bearing failure": {
        "primary": ["generator_bearing_de_temp_c", "generator_bearing_nde_temp_c",
                     "generator_winding_temp_c"],
        "supporting": ["rotor_speed_rpm", "active_power_kw", "ambient_temp_c"],
    },
    "Hydraulic group": {
        "primary": ["pitch_angle_deg"],
        "supporting": ["active_power_kw", "rotor_speed_rpm", "wind_speed_ms"],
    },
    "Transformer failure": {
        "primary": ["reactive_power_kvar", "active_power_kw"],
        "supporting": ["grid_frequency_hz", "generator_winding_temp_c"],
    },
}


def get_fault_subset_columns(df: pd.DataFrame, fault_type: str) -> list:
    """Return engineered feature columns relevant to a specific fault."""
    subset = FAULT_FEATURE_SUBSETS.get(fault_type, {})
    all_cols = subset.get("primary", []) + subset.get("supporting", [])

    # Include engineered features derived from these base cols
    matching = []
    for col in df.columns:
        for base in all_cols:
            if col.startswith(base + "_") or col == base:
                matching.append(col)
                break
    return matching


def train_fault_specialist(X_train: np.ndarray, fault_type: str) -> dict:
    """Train a VAE specialist model on features relevant to one fault type."""
    return train_vae(X_train, n_epochs=50, patience=10, hidden=64, latent=8,
                      beta=0.1)


def score_fault_specialist(specialist_bundle: dict, X_test: np.ndarray) -> np.ndarray:
    """Score test data with a specialist model."""
    model = specialist_bundle["model"]
    return vae_anomaly_score(model, torch.FloatTensor(X_test.astype(np.float32)))


def evaluate_fault_specific(event_row) -> dict:
    """Evaluate with fault-type-specific model, then with universal model.

    Hypothesis: specialist model outperforms universal on matching fault types.
    """
    event_id = int(event_row["event_id"])
    fault_type = str(event_row.get("event_description", ""))

    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return {"event_id": event_id, "error": "csv missing"}

    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    event_start_idx = int(event_row["event_start_id"])
    event_end_idx = int(event_row["event_end_id"])

    enriched = build_event_features(df, use_fft=False)  # skip FFT for speed

    # Get fault-specific columns
    fault_cols = get_fault_subset_columns(enriched, fault_type)
    # Remove non-numeric
    fault_cols = [c for c in fault_cols if c in enriched.select_dtypes("number").columns]

    if len(fault_cols) < 5:
        return {"event_id": event_id, "error": f"only {len(fault_cols)} fault-specific features"}

    # Labels
    n = len(enriched)
    is_anomaly = np.zeros(n, dtype=int)
    if event_row["event_label"] == "anomaly":
        is_anomaly[event_start_idx:event_end_idx + 1] = 1

    # Train on first 60% (normal only)
    train_end = int(n * 0.6)
    if is_anomaly[:train_end].sum() > 0:
        # Fault starts early — truncate training further
        first_anomaly = np.argmax(is_anomaly > 0)
        train_end = max(1000, first_anomaly - 100)

    # Feature matrix
    X_all = enriched[fault_cols].fillna(0).to_numpy(dtype=np.float32)

    # Clean NaN/Inf
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    X_train = X_all[:train_end]
    if len(X_train) < 200:
        return {"event_id": event_id, "error": "insufficient training data"}

    scaler = RobustScaler().fit(X_train)
    X_train_s = np.clip(scaler.transform(X_train), -10, 10).astype(np.float32)
    X_all_s = np.clip(scaler.transform(X_all), -10, 10).astype(np.float32)

    # Specialist
    bundle = train_fault_specialist(X_train_s, fault_type)
    test_scores = score_fault_specialist(bundle, X_all_s)
    test_scores_smooth = ewma_smooth(test_scores, alpha=0.3)

    # Evaluate on test portion
    y_test = is_anomaly[train_end:]
    s_test = test_scores_smooth[train_end:]

    result = {
        "event_id": event_id,
        "fault_type": fault_type,
        "event_type": event_row["event_label"],
        "n_fault_cols": len(fault_cols),
        "n_train": int(train_end),
        "n_test": int(len(y_test)),
    }

    if len(np.unique(y_test)) >= 2:
        result["auc_specialist"] = round(float(roc_auc_score(y_test, s_test)), 4)

    return result


def main():
    print("=" * 80)
    print("  FAULT-TYPE-SPECIFIC MODELS — CARE Farm A")
    print("  Novel contribution: specialist VAE per fault subsystem")
    print("=" * 80)

    events = pd.read_csv(EVENT_INFO, sep=";")
    anomaly_events = events[events["event_label"] == "anomaly"]
    print(f"\n[FS] {len(anomaly_events)} anomaly events to evaluate")

    results = []
    print(f"\n{'ID':>4} {'Fault':<30} {'#Cols':>6} {'AUC-Specialist':>15}")
    print("-" * 60)

    for _, row in anomaly_events.iterrows():
        r = evaluate_fault_specific(row)
        if "error" in r:
            print(f"{r['event_id']:>4}: ERROR — {r['error']}")
            continue
        results.append(r)
        auc_str = f"{r.get('auc_specialist', 0):.4f}" if r.get("auc_specialist") else "—"
        print(f"{r['event_id']:>4} {r['fault_type'][:28]:<30} "
              f"{r['n_fault_cols']:>6} {auc_str:>15}")

    # Aggregate per fault type
    print(f"\n{'='*80}")
    print(f"  AGGREGATE BY FAULT TYPE (Fault-Specialist VAE)")
    print(f"{'='*80}")

    valid = [r for r in results if r.get("auc_specialist") is not None]
    by_fault = {}
    for r in valid:
        by_fault.setdefault(r["fault_type"], []).append(r["auc_specialist"])

    for fault, aucs in sorted(by_fault.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {fault:<35} n={len(aucs):>2}  mean AUC={np.mean(aucs):.4f}")

    all_aucs = [r["auc_specialist"] for r in valid]
    print(f"\n  OVERALL mean AUC across all faults: {np.mean(all_aucs):.4f}")
    print(f"  OVERALL median:                       {np.median(all_aucs):.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "care_fault_specific.json").open("w") as f:
        json.dump({"per_event": results,
                    "mean_auc": float(np.mean(all_aucs)) if all_aucs else 0,
                    "by_fault": {k: float(np.mean(v)) for k, v in by_fault.items()}},
                   f, indent=2, default=str)
    print(f"\n[OK] Saved: {RESULTS_DIR / 'care_fault_specific.json'}")


if __name__ == "__main__":
    main()
