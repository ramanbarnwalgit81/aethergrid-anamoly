"""WindGPT Kaggle runner — single-cell script for a free T4/P100 notebook.

Paste the entire contents of this file into the first code cell of a new
Kaggle notebook. The notebook must have:
  - Accelerator: GPU T4 x2 (or GPU P100)
  - Internet: On
  - Input dataset attached: "windgpt-bundle" (the zip you uploaded per
    docs/KAGGLE_SETUP.md)

The script:
  1. Unpacks the bundle into /kaggle/working/aethergrid-anomaly/
  2. Installs peft
  3. Validates the GPU
  4. Builds the pre-training corpus if not already present
  5. LoRA-fine-tunes Chronos-Bolt on Penmanshiel + Kelmarsh
  6. Evaluates on CARE Farm A
  7. Regenerates the comparison table + Fig. 4
  8. Packages everything into /kaggle/working/windgpt_results.zip for download
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Configuration — edit these if you need to change behaviour
# ─────────────────────────────────────────────────────────────────────────
MODEL_SIZE = "chronos-bolt-base"     # tiny | small | base (VRAM: 1 / 4 / 8 GB)
EPOCHS = 3
LORA_RANK = 16
BATCH_SIZE = 16                         # drop to 8 if OOM
MAX_WINDOWS_PER_SIGNAL = 2000           # controls training set size


# ─────────────────────────────────────────────────────────────────────────
# 1. Locate the input dataset and set up the working tree
# ─────────────────────────────────────────────────────────────────────────
WORK = Path("/kaggle/working/aethergrid-anomaly")
KAGGLE_INPUT = Path("/kaggle/input")

print("=" * 70)
print("WindGPT Kaggle runner")
print("=" * 70)

# Find the dataset, either as a zip (not yet extracted) or as flat files.
# Kaggle auto-extracts zips most of the time, but the extracted tree may be
# nested (e.g. /kaggle/input/<slug>/<zipname>/src, or even /kaggle/input/
# datasets/<owner>/<slug>/src) depending on how Kaggle processes the upload.
# We search for the unique file src/benchmark/lora_finetune.py at any depth.
bundle_zip = None
repo_root = None

# 0) Try a small list of known paths first — fastest and avoids rglob overhead
for known in (
    Path("/kaggle/input/windgpt-bundle"),
    Path("/kaggle/input/datasets/ramanbarn/windgpt-bundle"),
):
    if known.exists() and (known / "src/benchmark/lora_finetune.py").exists():
        repo_root = known
        print(f"[fast-path] found repo at {repo_root}")
        break

# 1) If fast-path didn't find it, look for a zip with a matching name anywhere
if repo_root is None:
    for p in KAGGLE_INPUT.rglob("*.zip"):
        nm = p.name.lower()
        if any(k in nm for k in ("bundle", "windgpt", "aethergrid")):
            bundle_zip = p
            break

# 2) If still nothing, search for the unique script `lora_finetune.py` at any depth
if bundle_zip is None and repo_root is None:
    for py in KAGGLE_INPUT.rglob("lora_finetune.py"):
        # repo_root is three parents up (repo/src/benchmark/lora_finetune.py)
        if py.parent.name == "benchmark" and py.parent.parent.name == "src":
            repo_root = py.parent.parent.parent
            break

# 3) Both failed -> print a diagnostic tree so we can see what's really there
if bundle_zip is None and repo_root is None:
    print("[DIAGNOSTIC] /kaggle/input contents (up to depth 4):")
    for root, dirs, files in os.walk(str(KAGGLE_INPUT)):
        depth = root.replace(str(KAGGLE_INPUT), "").count("/")
        if depth > 4:
            dirs[:] = []
            continue
        pad = "  " * depth
        print(f"{pad}{os.path.basename(root) or '/kaggle/input'}/")
        for f in files[:15]:
            print(f"{pad}  {f}")
        if len(files) > 15:
            print(f"{pad}  ... +{len(files)-15} more files")
    raise SystemExit(
        "Could not locate the repo in /kaggle/input/. Check the tree above "
        "and ensure you attached the dataset (Sidebar -> + Add input -> "
        "Datasets -> ramanbarn/windgpt-bundle)."
    )

# 4) Materialise the repo into /kaggle/working/aethergrid-anomaly
if bundle_zip is not None:
    print(f"[1/8] Unpacking {bundle_zip.name} ({bundle_zip.stat().st_size/1e6:.0f} MB)...")
    if not WORK.exists():
        WORK.mkdir(parents=True)
        shutil.unpack_archive(str(bundle_zip), str(WORK))
    # Flatten any nested aethergrid-anomaly/ folder
    nested = WORK / "aethergrid-anomaly"
    if nested.exists() and not (WORK / "src").exists():
        for child in nested.iterdir():
            shutil.move(str(child), str(WORK))
        nested.rmdir()
else:
    print(f"[1/8] Found flat-file repo at {repo_root}")
    if not WORK.exists():
        shutil.copytree(str(repo_root), str(WORK))

# Verify the materialised tree has what we need
required = [WORK / "src/benchmark/lora_finetune.py",
              WORK / "src/benchmark/windgpt_pretrain.py"]
missing = [str(p.relative_to(WORK)) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"[ERR] Missing required files in {WORK}: {missing}")

# ── Forceful in-place patch of the Chronos-Bolt loader ──────────────────
# This always runs, regardless of which dataset version is attached.  It
# overwrites prepare_chronos_bolt_model with a known-good Chronos-Bolt +
# peft (FEATURE_EXTRACTION) implementation that:
#   - uses ChronosBoltPipeline (skips the broken AutoTokenizer call), and
#   - uses task_type=FEATURE_EXTRACTION (skips the prepare_inputs_for_generation call).
import re as _re
_lf = WORK / "src/benchmark/lora_finetune.py"
_fixed_loader = '''def prepare_chronos_bolt_model(model_size: str, lora_rank: int, device: str,
                                  dtype):
    """Load Chronos-Bolt via chronos-forecasting and wrap with LoRA (peft)."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from chronos import ChronosBoltPipeline
    model_map = {
        "chronos-bolt-tiny":  "amazon/chronos-bolt-tiny",
        "chronos-bolt-small": "amazon/chronos-bolt-small",
        "chronos-bolt-base":  "amazon/chronos-bolt-base",
    }
    model_name = model_map[model_size]
    print(f"[LOAD] {model_name} via chronos-forecasting")
    pipeline = ChronosBoltPipeline.from_pretrained(
        model_name, device_map=device, torch_dtype=dtype)
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        target_modules=["q", "k", "v", "o", "wi", "wo"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    wrapped = get_peft_model(pipeline.model, lora_config)
    wrapped.print_trainable_parameters()
    return wrapped.to(device), None
'''
_content = _lf.read_text()
# Replace the function from `def prepare_chronos_bolt_model(` up to the
# next top-level `\ndef ` or `\nclass ` — robust to either old or new source.
_pat = _re.compile(
    r"def prepare_chronos_bolt_model\b.*?(?=\ndef |\nclass )",
    _re.DOTALL,
)
_new = _pat.sub(_fixed_loader.rstrip() + "\n\n", _content, count=1)
if _new != _content:
    _lf.write_text(_new)
    print("[force-patch] rewrote prepare_chronos_bolt_model (FEATURE_EXTRACTION + ChronosBoltPipeline)")
else:
    print("[force-patch] no replacement made — function pattern not found, "
          "training may fail")

os.chdir(WORK)
sys.path.insert(0, str(WORK))
print(f"[OK] working dir = {WORK}")
print(f"     contents: {sorted(p.name for p in WORK.iterdir())}")


# ─────────────────────────────────────────────────────────────────────────
# 2. Install GPU-only dependencies (peft for LoRA)
# ─────────────────────────────────────────────────────────────────────────
print("\n[2/8] Installing peft (LoRA support)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "peft>=0.10", "bitsandbytes", "chronos-forecasting"], check=True)


# ─────────────────────────────────────────────────────────────────────────
# 3. GPU check
# ─────────────────────────────────────────────────────────────────────────
print("\n[3/8] GPU check...")
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU detected. Right sidebar -> Accelerator: "
                      "GPU T4 x2 or P100, then Save + Run All again.")
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"[OK] GPU: {gpu_name}  VRAM: {vram_gb:.1f} GB")

# Auto-downgrade the model if VRAM is tight
if vram_gb < 14 and MODEL_SIZE == "chronos-bolt-base":
    print(f"[warn] {vram_gb:.1f} GB VRAM is tight for chronos-bolt-base; "
          f"switching to chronos-bolt-small for safety")
    MODEL_SIZE = "chronos-bolt-small"
if vram_gb < 6:
    print(f"[warn] {vram_gb:.1f} GB VRAM — using chronos-bolt-tiny")
    MODEL_SIZE = "chronos-bolt-tiny"


# ─────────────────────────────────────────────────────────────────────────
# 4. Build pre-training corpus if missing
# ─────────────────────────────────────────────────────────────────────────
corpus_path = WORK / "data/benchmark/windgpt/pretrain_corpus.parquet"
if not corpus_path.exists():
    print("\n[4/8] Building Penmanshiel+Kelmarsh pre-training corpus...")
    subprocess.run([sys.executable, "-m", "src.benchmark.windgpt_pretrain"],
                   check=True, cwd=str(WORK))
else:
    print(f"\n[4/8] Corpus already present: {corpus_path.name} "
          f"({corpus_path.stat().st_size/1e6:.0f} MB) — skipping build")


# ─────────────────────────────────────────────────────────────────────────
# 5. LoRA fine-tuning on Penmanshiel+Kelmarsh
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[5/8] LoRA fine-tune {MODEL_SIZE}, {EPOCHS} epochs, rank {LORA_RANK}")
t0 = time.time()
subprocess.run([
    sys.executable, "-m", "src.benchmark.lora_finetune", "train",
    "--model", MODEL_SIZE,
    "--epochs", str(EPOCHS),
    "--lora-rank", str(LORA_RANK),
    "--batch-size", str(BATCH_SIZE),
    "--max-windows", str(MAX_WINDOWS_PER_SIGNAL),
], check=True, cwd=str(WORK))
print(f"[OK] training elapsed: {(time.time()-t0)/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────
# 6. Evaluate on CARE Farm A
# ─────────────────────────────────────────────────────────────────────────
ckpt_dir = WORK / "models/lora_chronos"
ckpts = sorted(ckpt_dir.glob(f"{MODEL_SIZE}_lora_rank{LORA_RANK}_epoch*.pt"))
if not ckpts:
    raise SystemExit(f"No checkpoints found in {ckpt_dir}")

# Use the last TWO epoch checkpoints as an ensemble (upgrade #3).  If only
# one epoch was saved, fall back to that single checkpoint.
ensemble_ckpts = ckpts[-2:] if len(ckpts) >= 2 else ckpts
ckpt_arg = ",".join(str(c) for c in ensemble_ckpts)
print(f"\n[6/8] Evaluating with {len(ensemble_ckpts)}-checkpoint ensemble "
       f"({[c.name for c in ensemble_ckpts]}) + multi-scale/multi-stride TTA...")
t0 = time.time()
subprocess.run([
    sys.executable, "-m", "src.benchmark.lora_finetune", "eval",
    "--checkpoint", ckpt_arg,
    "--model", MODEL_SIZE,
    "--lora-rank", str(LORA_RANK),
], check=True, cwd=str(WORK))
print(f"[OK] evaluation elapsed: {(time.time()-t0)/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────
# 7. Rebuild comparison table + figure (NON-FATAL: baseline JSONs may be
#    missing from the Kaggle bundle, in which case we skip and let the
#    user run the comparator locally where all baselines are present.)
# ─────────────────────────────────────────────────────────────────────────
print("\n[7/8] Regenerating comparison table + Fig. 4...")
_compare = subprocess.run(
    [sys.executable, "-m", "src.benchmark.windgpt_compare"],
    cwd=str(WORK),
)
if _compare.returncode != 0:
    print("[warn] comparison step failed (baseline JSONs likely not bundled "
          "for Kaggle). The LoRA result is still saved; run windgpt_compare.py "
          "locally to merge with v7 / PINN-v2 / Chronos baselines.")


# ─────────────────────────────────────────────────────────────────────────
# 8. Package everything for download
# ─────────────────────────────────────────────────────────────────────────
print("\n[8/8] Packaging results for download...")
OUT_ZIP = Path("/kaggle/working/windgpt_results.zip")
if OUT_ZIP.exists():
    OUT_ZIP.unlink()

import zipfile
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    # Result JSONs
    for p in (WORK / "docs/results").glob("*.json"):
        zf.write(p, f"docs/results/{p.name}")
    # Updated figure
    for p in (WORK / "docs/paper/figures").glob("fig4*"):
        zf.write(p, f"docs/paper/figures/{p.name}")
    # The fine-tuned LoRA checkpoint (user may want to publish it)
    for p in ckpt_dir.glob("*.pt"):
        zf.write(p, f"models/lora_chronos/{p.name}")
print(f"[OK] wrote {OUT_ZIP}  ({OUT_ZIP.stat().st_size/1e6:.1f} MB)")

# Always print the LoRA result we just produced — even if step 7 failed,
# the AUC numbers are saved in lora_finetune_<model>_care.json
import json
print("\n" + "=" * 70)
print("WindGPT LoRA RESULT (CARE Farm A)")
print("=" * 70)
for p in sorted((WORK / "docs/results").glob("lora_finetune*_care.json")):
    d = json.loads(p.read_text())
    print(f"  source:             {p.name}")
    print(f"  model:              {d.get('model')}")
    print(f"  checkpoints used:   {len(d.get('checkpoints_used', []))}")
    print(f"  multi-scale TTA:    {d.get('multi_scale')}")
    print(f"  multi-stride TTA:   {d.get('multi_stride')}")
    print(f"  rolling-std fuse:   {d.get('ensemble_rolling_std')}")
    print(f"  Mean AUC event:     {d.get('mean_auc_event')}")
    print(f"  Mean AUC precursor: {d.get('mean_auc_precursor')}")
    print(f"  Mean AUC status:    {d.get('mean_auc_status')}")

# Also print the comparison table if it was rebuilt successfully
_cmp = WORK / "docs/results/windgpt_comparison.json"
if _cmp.exists():
    print("\n" + "=" * 70)
    print("FULL COMPARISON")
    print("=" * 70)
    d = json.loads(_cmp.read_text())
    hdr = f"{'Method':<42s} {'Event':>7s} {'Precursor':>10s} {'Status':>7s}"
    print(hdr); print("-" * len(hdr))
    for r in d["methods"]:
        e = f"{r['mean_auc_event']:.3f}"     if r["mean_auc_event"]     is not None else "  —  "
        p = f"{r['mean_auc_precursor']:.3f}" if r["mean_auc_precursor"] is not None else "  —  "
        s = f"{r['mean_auc_status']:.3f}"    if r["mean_auc_status"]    is not None else "  —  "
        print(f"{r['method'][:42]:<42s} {e:>7s} {p:>10s} {s:>7s}")

print("\nAll done. Download windgpt_results.zip from the Output tab on the right.")
