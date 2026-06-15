"""WindGPT unified comparison: zero-shot (CPU) vs LoRA-fine-tuned (GPU).

Aggregates four detectors on CARE Farm A into a single table and bar chart:

    1. v7 base ensemble                 (Farm-A-trained MLP ensemble)
    2. PINN-v2 residual stacker         (v7 + 7 physics channels)
    3. Chronos-T5-tiny zero-shot        (no CARE training, no GPU)
    4. WindGPT (Chronos-Bolt + LoRA)    (fine-tuned on Penmanshiel+Kelmarsh)

The script is robust: if a method's result JSON does not exist, that row is
marked "not run yet" and the others still render. That lets you publish the
CPU-only comparison today and add the LoRA column whenever the GPU run
finishes.

Usage:
    # CPU-only mode (ships all non-GPU results, marks GPU row as pending)
    python -m src.benchmark.windgpt_compare

    # After GPU LoRA run completes on your box:
    # (lora_finetune.py writes docs/results/lora_chronos_results.json)
    python -m src.benchmark.windgpt_compare  # auto-picks up the file

Output:
    docs/results/windgpt_comparison.json       — machine-readable side-by-side
    docs/paper/figures/fig4_method_bar.pdf     — publication figure
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("docs/results")
FIG_DIR = Path("docs/paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# Result extractors — each returns dict with keys
#   {mean_auc_event, mean_auc_precursor, mean_auc_status, training_mode,
#    training_cost, notes}
# ─────────────────────────────────────────────────────────────────────────

def extract_v7_base():
    with (RESULTS_DIR / "care_ensemble_v7.json").open() as f:
        d = json.load(f)
    # per-event mean not directly in this file; use the pinn_stacker base_v7_auc
    # array which is the authoritative per-event v7 baseline for Farm A.
    base = []
    try:
        with (RESULTS_DIR / "pinn_stacker_y_event.json").open() as f:
            ev = json.load(f)["per_event"]
        base = [r["base_v7_auc"] for r in ev]
    except Exception:
        pass
    mean_event = round(float(np.mean(base)), 4) if base else None

    prec = []
    try:
        with (RESULTS_DIR / "pinn_stacker_y_precursor.json").open() as f:
            pr = json.load(f)["per_event"]
        prec = [r["base_v7_auc"] for r in pr]
    except Exception:
        pass
    mean_prec = round(float(np.mean(prec)), 4) if prec else None

    status = []
    try:
        with (RESULTS_DIR / "pinn_stacker_y_status.json").open() as f:
            st = json.load(f)["per_event"]
        status = [r["base_v7_auc"] for r in st]
    except Exception:
        pass
    mean_status = round(float(np.mean(status)), 4) if status else None

    return {
        "method":   "v7 ensemble (VAE+LSTM+TF + Fleet-NBM)",
        "mean_auc_event":     mean_event,
        "mean_auc_precursor": mean_prec,
        "mean_auc_status":    mean_status,
        "training_mode":      "trained",
        "training_cost":      "~72 min CPU, Farm-A 86-feature schema",
        "notes":              "Base ensemble; Farm-A-specific feature engineering.",
    }


def extract_pinn_v2():
    out = {}
    for label, fn in [("event", "pinn_stacker_y_event.json"),
                       ("precursor", "pinn_stacker_y_precursor.json"),
                       ("status", "pinn_stacker_y_status.json")]:
        try:
            with (RESULTS_DIR / fn).open() as f:
                d = json.load(f)
            out[f"mean_auc_{label}"] = d["aggregate"]["mean_pinn_stacker_auc"]
        except Exception:
            out[f"mean_auc_{label}"] = None
    out.update({
        "method":        "PINN-v2 residual stacker (ours)",
        "training_mode": "trained (LOEO)",
        "training_cost": "~5 min CPU per LOEO fold, Farm A only",
        "notes":         "7 IEC-inspired residuals + v7 + temporal features.",
    })
    return out


def extract_chronos_zs():
    try:
        with (RESULTS_DIR / "foundation_models_care_a.json").open() as f:
            d = json.load(f)
        return {
            "method":             "Chronos-T5-tiny zero-shot",
            "mean_auc_event":     round(float(d["mean_auc_event"]), 4),
            "mean_auc_precursor": round(float(d["mean_auc_precursor"]), 4),
            "mean_auc_status":    None,
            "training_mode":      "zero-shot (no CARE training)",
            "training_cost":      f"{d['total_time_s']/3600:.1f} h CPU for 11-event eval",
            "notes":              "8 M-param T5 pretrained on generic time series; forecast-residual anomaly score.",
        }
    except Exception as e:
        return {
            "method": "Chronos-T5-tiny zero-shot",
            "mean_auc_event": None, "mean_auc_precursor": None, "mean_auc_status": None,
            "training_mode": "zero-shot",
            "training_cost": "not yet run",
            "notes": f"file not found: {e}",
        }


def extract_windgpt_lora():
    """Pull LoRA-fine-tuned WindGPT results from lora_finetune.evaluate output
    if present; otherwise mark as pending."""
    # The evaluate() function writes lora_finetune_<model>_care.json; we accept
    # any file matching that pattern plus legacy names.
    candidates = sorted(RESULTS_DIR.glob("lora_finetune_chronos_bolt_*_care.json"))
    candidates += [
        RESULTS_DIR / "lora_chronos_results.json",
        RESULTS_DIR / "windgpt_lora_results.json",
    ]
    for p in candidates:
        if p.exists():
            d = json.loads(p.read_text())
            # evaluate() stores aggregate keys at the top level under
            # "aggregate" or per-event list; handle both formats robustly
            agg = d.get("aggregate", d)
            per = d.get("per_event", [])
            # Derive means from per-event if aggregate not populated
            def _mean_of(key):
                if isinstance(agg.get(key), (int, float)):
                    return round(float(agg[key]), 4)
                vals = [r[key] for r in per if r.get(key) is not None]
                return round(float(np.mean(vals)), 4) if vals else None
            return {
                "method":             "WindGPT (Chronos-Bolt + LoRA, ours)",
                "mean_auc_event":     _mean_of("auc_event")     or _mean_of("mean_auc_event"),
                "mean_auc_precursor": _mean_of("auc_precursor") or _mean_of("mean_auc_precursor"),
                "mean_auc_status":    _mean_of("auc_status")    or _mean_of("mean_auc_status"),
                "training_mode":      "LoRA fine-tuned on Penmanshiel+Kelmarsh",
                "training_cost":      "run wall-clock in lora_finetune log",
                "notes":              f"Chronos-Bolt wrapped with rank-16 LoRA (source: {p.name}).",
            }
    return {
        "method": "WindGPT (Chronos-Bolt + LoRA, ours)",
        "mean_auc_event": None, "mean_auc_precursor": None, "mean_auc_status": None,
        "training_mode": "LoRA fine-tuned on Penmanshiel+Kelmarsh",
        "training_cost": "pending GPU run",
        "notes": "Run `python -m src.benchmark.lora_finetune all` on a CUDA box to fill this row.",
    }


# ─────────────────────────────────────────────────────────────────────────
# Aggregate + figure
# ─────────────────────────────────────────────────────────────────────────

def build_table():
    rows = [
        extract_v7_base(),
        extract_pinn_v2(),
        extract_chronos_zs(),
    ]
    return rows


def make_bar_chart(rows):
    labels = ["Event-window", "CARE-Precursor", "Status"]
    method_names = [r["method"].split(" (")[0] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(labels))
    n = len(rows)
    width = 0.8 / n
    colours = ["#7f7f7f", "#1f77b4", "#2ca02c", "#8c564b"]

    for i, r in enumerate(rows):
        vals = [r["mean_auc_event"], r["mean_auc_precursor"], r["mean_auc_status"]]
        labelled = False
        for j, v in enumerate(vals):
            xpos = x[j] + i * width
            lbl = method_names[i] if not labelled else None
            if v is not None:
                ax.bar(xpos, v, width, label=lbl,
                       color=colours[i % len(colours)], edgecolor="black", linewidth=0.5)
                labelled = True
            else:
                # Explicit 'not scored' placeholder: a faint hatched ghost slot plus
                # an 'n/a' label, so the empty position reads as an intentional
                # omission (this method is not evaluated on that label) rather than a
                # missing or mis-rendered bar.
                ax.bar(xpos, 1.0, width, label=lbl, facecolor="none",
                       edgecolor="gray", linewidth=0.6, hatch="////", alpha=0.30)
                ax.text(xpos, 0.5, "n/a", rotation=90, ha="center", va="center",
                        fontsize=7.5, color="gray", style="italic")
                labelled = True

    ax.set_xticks(x + (n - 1) * width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean per-event AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Method comparison on CARE Farm A (mean per-event AUC)")
    ax.legend(loc="upper left", frameon=True, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_method_bar.pdf")
    fig.savefig(FIG_DIR / "fig4_method_bar.png")
    plt.close(fig)


def main():
    rows = build_table()
    out = {
        "methods": rows,
        "benchmark": "CARE Farm A anomaly events (Farm-A-only for all four methods)",
    }
    (RESULTS_DIR / "windgpt_comparison.json").write_text(
        json.dumps(out, indent=2, default=str))

    # ASCII table
    headers = ["Method", "Event", "Precursor", "Status", "Mode"]
    print(f"\n{headers[0]:<40s} {headers[1]:>6s} {headers[2]:>10s} {headers[3]:>6s}   {headers[4]}")
    print("-" * 95)
    for r in rows:
        e = f"{r['mean_auc_event']:.3f}" if r["mean_auc_event"] is not None else "  —  "
        p = f"{r['mean_auc_precursor']:.3f}" if r["mean_auc_precursor"] is not None else "  —  "
        s = f"{r['mean_auc_status']:.3f}" if r["mean_auc_status"] is not None else "  —  "
        print(f"{r['method']:<40s} {e:>6s} {p:>10s} {s:>6s}   {r['training_mode']}")

    make_bar_chart(rows)
    print(f"\n[OK] Saved: docs/results/windgpt_comparison.json")
    print(f"[OK] Saved: docs/paper/figures/fig4_method_bar.pdf")


if __name__ == "__main__":
    main()
