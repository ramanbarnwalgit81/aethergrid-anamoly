"""
WindBench Leaderboard Generator.

Produces:
  1. Static HTML leaderboard (docs/benchmark/index.html)
  2. Markdown tables for paper/README
  3. JSON export for leaderboard API

Usage:  python -m src.benchmark.leaderboard
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from src.benchmark.evaluate import collect_results
from src.benchmark.tasks import TASKS
from src.benchmark.registry import REGISTRY

DOCS_DIR = Path("docs/benchmark")


def generate_markdown() -> str:
    """Leaderboard in Markdown (for README)."""
    df = collect_results()
    if df.empty:
        return "No results.\n"

    lines = ["# WindBench Leaderboard\n",
             f"_Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n"]

    for task_id in sorted(df["task_id"].dropna().unique()):
        task = TASKS.get(task_id)
        sub = df[df["task_id"] == task_id].sort_values("auc_roc", ascending=False)

        lines.append(f"\n## {task_id}\n")
        if task:
            lines.append(f"**Description:** {task.description}\n")
            lines.append(f"**Primary metric:** `{task.primary_metric}`\n")

        cols = ["model_name", "auc_roc", "auc_pr", "recall_at_fpr_0.10",
                "f1_best", "n_faults"]
        cols = [c for c in cols if c in sub.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in sub.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def generate_html() -> str:
    """Full HTML leaderboard page with embedded CSS."""
    df = collect_results()
    if df.empty:
        return "<html><body><h1>WindBench</h1><p>No results yet.</p></body></html>"

    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WindBench — Wind Turbine CMS Benchmark</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 2rem; color: #222; line-height: 1.6; background: #fafafa; }}
  h1 {{ color: #1a4d7a; border-bottom: 3px solid #1a4d7a; padding-bottom: 0.4rem; }}
  h2 {{ color: #2a6ea1; margin-top: 2.5rem; }}
  .subtitle {{ color: #666; font-size: 1.1rem; }}
  .timestamp {{ color: #999; font-size: 0.9rem; }}
  .badge {{ display: inline-block; background: #2a6ea1; color: white; padding: 0.25rem 0.6rem;
            border-radius: 4px; font-size: 0.85rem; margin-right: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: white;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-radius: 4px; overflow: hidden; }}
  th, td {{ padding: 0.7rem 1rem; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #2a6ea1; color: white; font-weight: 600; }}
  tr.top-3 {{ background: #fff7e0; }}
  tr.top-1 {{ background: #ffe8a0; font-weight: 700; }}
  tr:hover {{ background: #f5f5f5; }}
  .metric-primary {{ color: #1a4d7a; font-weight: 600; }}
  .task-desc {{ background: #eef5fc; padding: 0.8rem 1rem; border-left: 4px solid #2a6ea1;
                margin: 1rem 0; border-radius: 0 4px 4px 0; }}
  footer {{ margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #ddd;
            color: #666; font-size: 0.9rem; }}
  code {{ background: #f0f0f0; padding: 0.15rem 0.4rem; border-radius: 3px;
          font-family: 'Menlo', 'Consolas', monospace; }}
</style>
</head>
<body>

<h1>WindBench</h1>
<p class="subtitle">Wind Turbine Condition Monitoring Benchmark Suite</p>
<p class="timestamp">Last updated: {ts}</p>

<p>
  <span class="badge">Datasets: {len(REGISTRY)}</span>
  <span class="badge">Tasks: {len(TASKS)}</span>
  <span class="badge">Models: {df['model_name'].nunique()}</span>
  <span class="badge">Results: {len(df)}</span>
</p>

<h2>Overview</h2>
<p>
  WindBench is the first comprehensive benchmark suite for wind turbine
  condition monitoring. It unifies multiple public datasets
  (Kelmarsh, Penmanshiel, CARE Farms A/B/C) under a canonical schema
  with five standardized tasks and reproducible baselines.
</p>
<p>
  All datasets are harmonized to the same feature schema with 20 canonical
  SCADA features. Every baseline is evaluated on every applicable task
  with bootstrap 95% confidence intervals.
</p>

<h2>Datasets</h2>
<table>
<tr><th>Dataset</th><th>Turbines</th><th>Rows</th><th>Fault Labels</th><th>Date Range</th><th>License</th></tr>
"""

    for did, manifest in REGISTRY.items():
        parquet_exists = Path(manifest.canonical_parquet_path).exists()
        if not parquet_exists:
            continue
        html += (f"<tr><td><b>{did}</b></td>"
                 f"<td>{manifest.n_turbines}</td>"
                 f"<td>{manifest.n_rows:,}</td>"
                 f"<td>{manifest.fault_label_type.value}</td>"
                 f"<td>{manifest.date_min[:10]} – {manifest.date_max[:10]}</td>"
                 f"<td>{manifest.license}</td></tr>")

    html += "</table>\n"

    # Per-task leaderboards
    for task_id in sorted(df["task_id"].dropna().unique()):
        task = TASKS.get(task_id)
        sub = df[df["task_id"] == task_id].sort_values("auc_roc", ascending=False)

        html += f"<h2>{task_id}</h2>\n"
        if task:
            html += (f'<div class="task-desc">'
                     f'<b>Type:</b> {task.task_type.value}<br>'
                     f'<b>Description:</b> {task.description}<br>'
                     f'<b>Primary metric:</b> <code>{task.primary_metric}</code>'
                     f'</div>\n')

        html += "<table>\n<tr>"
        html += ("<th>Rank</th><th>Model</th><th>AUC-ROC</th><th>AUC-PR</th>"
                 "<th>Recall@FPR=0.10</th><th>F1-best</th><th>n_faults</th></tr>\n")

        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            cls = ""
            if rank == 1:
                cls = "top-1"
            elif rank <= 3:
                cls = "top-3"
            def fmt(x):
                if pd.isna(x):
                    return "—"
                if isinstance(x, float):
                    return f"{x:.4f}"
                return str(int(x)) if x == int(x) else str(x)
            html += (f'<tr class="{cls}"><td>{rank}</td>'
                     f'<td><b>{row["model_name"]}</b></td>'
                     f'<td class="metric-primary">{fmt(row.get("auc_roc"))}</td>'
                     f'<td>{fmt(row.get("auc_pr"))}</td>'
                     f'<td>{fmt(row.get("recall_at_fpr_0.10"))}</td>'
                     f'<td>{fmt(row.get("f1_best"))}</td>'
                     f'<td>{fmt(row.get("n_faults"))}</td>'
                     f'</tr>\n')
        html += "</table>\n"

    html += """
<footer>
<p>
  <b>How to submit a model:</b> Implement a scorer function conforming to
  <code>def scorer(train_df, test_df) -> np.ndarray</code> and add it to
  <code>src/benchmark/baselines.py</code>. Run
  <code>python -m src.benchmark.run_all</code>.
</p>
<p>
  <b>Citation:</b> Barnwal, R. (2026). "WindBench: A Comprehensive Benchmark
  for Wind Turbine Condition Monitoring." <i>In preparation.</i>
</p>
<p>
  <b>Data sources:</b>
  Kelmarsh (Zenodo 5841834, CC-BY-4.0) ·
  Penmanshiel (Zenodo 5946808, CC-BY-4.0) ·
  CARE (Zenodo 14006163, CC-BY-SA-4.0)
</p>
</footer>

</body>
</html>
"""
    return html


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # HTML
    html = generate_html()
    html_path = DOCS_DIR / "index.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[OK] HTML leaderboard: {html_path}")

    # Markdown
    md = generate_markdown()
    md_path = DOCS_DIR / "leaderboard.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[OK] Markdown leaderboard: {md_path}")

    # JSON export (for API/programmatic access)
    df = collect_results()
    if not df.empty:
        json_path = DOCS_DIR / "results.json"
        df.to_json(json_path, orient="records", indent=2)
        print(f"[OK] JSON export: {json_path}")

    print(f"\n[OK] Leaderboard written to {DOCS_DIR}/")


if __name__ == "__main__":
    main()
