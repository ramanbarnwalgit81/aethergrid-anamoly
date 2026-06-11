"""Physics-constant sensitivity sweep for the PINN-v2 stacker (CARE Farm A).

Perturbs the IEC-inspired physics constants (power-curve cut-in/rated wind and
the thermal-model offsets/load coefficients of Eqs. 1-2) by a multiplicative
factor c, re-runs the seven-channel stacker leave-one-event-out under the
event-window label, and records the mean per-event AUC at each c. A small AUC
range across the sweep substantiates the claim that the stacker is insensitive
to the exact constants (per-event robust-z-scoring + XGBoost reweighting absorb
their location and scale).

Run from repo root:  python -m scripts.physics_sensitivity_sweep
Output: docs/results/physics_sensitivity.json
"""
import json
from pathlib import Path
import numpy as np
import src.benchmark.pinn_stacker as ps

ORIG_CIN, ORIG_RATED = ps.CUT_IN_WIND, ps.RATED_WIND
RPK = ps.RATED_POWER_KW
ORIG_RESID = ps.physics_residuals_input


def make_residuals(c: float):
    """physics_residuals_input with all model constants scaled by c."""
    def f(df):
        wind = df["wind_speed_3_avg"].to_numpy(dtype=np.float32)
        power = df["power_30_avg"].to_numpy(dtype=np.float32)
        ambient = df["sensor_0_avg"].to_numpy(dtype=np.float32)
        rpm = df["sensor_18_avg"].to_numpy(dtype=np.float32)
        gearbox_oil = df["sensor_12_avg"].to_numpy(dtype=np.float32)
        gen_bearing_de = df["sensor_13_avg"].to_numpy(dtype=np.float32)
        nde = df["sensor_14_avg"].to_numpy(dtype=np.float32) if "sensor_14_avg" in df.columns else None
        wnd = df["sensor_15_avg"].to_numpy(dtype=np.float32) if "sensor_15_avg" in df.columns else None
        nac = df["sensor_7_avg"].to_numpy(dtype=np.float32) if "sensor_7_avg" in df.columns else None

        expected_power = ps.iec_power_curve(wind)           # uses patched globals
        r_power = (power - expected_power) / RPK
        power_ratio = np.clip(power / RPK, 0.0, 1.2)
        expected_gbx = ambient + 40.0 * c + 25.0 * c * power_ratio
        r_thermal_gbx = (gearbox_oil - expected_gbx) / 30.0
        r_thermal_brg_de = (gen_bearing_de - (gearbox_oil + 10.0 * c)) / 20.0
        rpm_ratio = np.clip(rpm / 1600.0, 0.0, 1.2)
        r_rpm_power = (power - RPK * rpm_ratio ** 3) / RPK
        out = {"r_power": r_power, "r_thermal_gbx": r_thermal_gbx,
               "r_thermal_brg_de": r_thermal_brg_de, "r_rpm_power": r_rpm_power}
        if nde is not None:
            out["r_thermal_brg_nde"] = (nde - (gearbox_oil + 8.0 * c)) / 20.0
        if wnd is not None:
            out["r_thermal_winding"] = (wnd - (ambient + 25.0 * c + 40.0 * c * power_ratio)) / 40.0
        if nac is not None:
            out["r_thermal_nacelle"] = (nac - (ambient + 5.0 * c + 10.0 * c * power_ratio)) / 15.0
        return out
    return f


def main():
    rows = []
    for c in [0.8, 0.9, 1.0, 1.1, 1.2]:
        ps.CUT_IN_WIND = ORIG_CIN * c
        ps.RATED_WIND = ORIG_RATED * c
        ps.physics_residuals_input = make_residuals(c)
        agg = ps.run_stacker("y_event")["aggregate"]
        rows.append({
            "scale": c,
            "mean_stacker_auc": agg["mean_pinn_stacker_auc"],
            "mean_delta_auc": agg["mean_delta_auc"],
            "pooled_auc": agg["pooled_pinn_stacker_auc"],
        })
        print(f"c={c:.1f}  mean_stacker_auc={agg['mean_pinn_stacker_auc']}  "
              f"delta={agg['mean_delta_auc']}  pooled={agg['pooled_pinn_stacker_auc']}")

    ps.CUT_IN_WIND, ps.RATED_WIND = ORIG_CIN, ORIG_RATED
    ps.physics_residuals_input = ORIG_RESID

    aucs = [r["mean_stacker_auc"] for r in rows]
    summary = {
        "label": "y_event",
        "sweep": rows,
        "auc_min": round(min(aucs), 4),
        "auc_max": round(max(aucs), 4),
        "auc_range": round(max(aucs) - min(aucs), 4),
        "baseline_auc_c1": [r["mean_stacker_auc"] for r in rows if r["scale"] == 1.0][0],
    }
    Path("docs/results/physics_sensitivity.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAUC range across +/-20% constant perturbation: {summary['auc_range']} "
          f"(min {summary['auc_min']}, max {summary['auc_max']})")


if __name__ == "__main__":
    main()
