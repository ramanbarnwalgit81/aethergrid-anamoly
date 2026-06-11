"""
Labeling Convention Decision Tree — figure generator for IEEE paper.

Produces a flowchart (PNG + PDF) guiding practitioners on which ground-truth
labeling convention to use for SCADA anomaly detection.

Usage:
    python -m src.benchmark.labeling_decision_tree

Output:
    docs/figures/labeling_decision_tree.png   (300 dpi, for paper)
    docs/figures/labeling_decision_tree.pdf   (vector, for submission)
"""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path("docs/figures")

# Color palette
C_Q    = "#1E4D8C"   # dark blue  — decision node
C_A    = "#1A6B3A"   # dark green — recommended action
C_W    = "#9C4221"   # rust       — caution
C_G    = "#4A3500"   # dark gold  — best-practice
C_S    = "#2D3748"   # near-black — start node
C_BG   = "#FFFFFF"   # white
C_E    = "#2D3748"   # arrow / edge color


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _rect(ax, cx, cy, w, h, lines, fc,
          fontsize=8.0, tc="white", lw=1.5):
    """Rounded rectangle centered at (cx, cy); lines is a list of text rows."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.12",
        linewidth=lw, edgecolor=C_E, facecolor=fc, zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, "\n".join(lines),
            ha="center", va="center",
            fontsize=fontsize, color=tc,
            fontweight="bold", multialignment="center", zorder=4)


def _diamond(ax, cx, cy, w, h, lines, fc, fontsize=8.0, tc="white", lw=1.5):
    """Diamond centered at (cx, cy); lines is a list of text rows."""
    hw, hh = w / 2, h / 2
    xs = [cx, cx + hw, cx, cx - hw, cx]
    ys = [cy + hh, cy, cy - hh, cy, cy + hh]
    ax.fill(xs, ys, color=fc, zorder=3)
    ax.plot(xs, ys, color=C_E, linewidth=lw, zorder=4)
    ax.text(cx, cy, "\n".join(lines),
            ha="center", va="center",
            fontsize=fontsize, color=tc,
            fontweight="bold", multialignment="center", zorder=5)


def _arrow(ax, x0, y0, x1, y1, lw=1.6):
    """Straight arrow from (x0,y0) to (x1,y1)."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_E,
                                lw=lw, mutation_scale=16), zorder=2)


def _elbow_h_v(ax, x0, y0, x1, y1, lw=1.6):
    """L-shaped connector: horizontal from (x0,y0) to (x1,y0), then arrow down to (x1,y1)."""
    ax.plot([x0, x1], [y0, y0], color=C_E, lw=lw, zorder=2)
    _arrow(ax, x1, y0, x1, y1, lw=lw)


def _label(ax, x, y, text, side="right", fontsize=8.5):
    """Small YES/NO label next to a connector."""
    ha = "left" if side == "right" else "right"
    dx = 0.10 if side == "right" else -0.10
    ax.text(x + dx, y, text,
            ha=ha, va="center",
            fontsize=fontsize, fontstyle="italic", color="#444444",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
            zorder=7)


# ── Figure builder ─────────────────────────────────────────────────────────────

def build_figure() -> plt.Figure:
    # Canvas: 13 wide × 18 tall (coordinate units == inches here)
    fig, ax = plt.subplots(figsize=(13, 18))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 18)
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)
    ax.axis("off")

    # ── Grid constants ─────────────────────────────────────────────────────────
    # X centres
    XC  = 6.5          # center column
    XL  = 1.9          # far-left  (A_PLC, A_EVT_L)
    XCL = 5.2          # center-left (Q4, A_EVT_L2)
    XCR = 8.3          # center-right (A_EVT_R)
    XR  = 11.3         # far-right (A_PROXY)
    XQ2 = 3.0          # Q2 diamond center
    XQ3 = 10.0         # Q3 diamond center

    # Y centres (top → bottom)
    Y_TITLE = 17.35
    Y_START = 16.4
    Y_Q1    = 14.8
    Y_Q23   = 12.5
    Y_ROW3  = 10.0
    Y_ROW4  =  7.7
    Y_BEST  =  5.1
    Y_REF   =  3.1

    # Box sizes
    DW, DH   = 4.8, 1.4    # decision diamond width/height
    AW, AH   = 3.0, 1.5    # action rect width/height
    BW, BH   = 12.0, 1.9   # best-practice rect
    SW, SH   = 5.5, 0.8    # start rect

    # Derived edge y-coords
    Q1_bot     = Y_Q1  - DH / 2
    Q23_top    = Y_Q23 + DH / 2
    Q23_bot    = Y_Q23 - DH / 2
    Q2_left_x  = XQ2  - DW / 2     # 3.0 - 2.4 = 0.6
    Q2_right_x = XQ2  + DW / 2     # 5.4
    Q3_left_x  = XQ3  - DW / 2     # 7.6
    Q3_right_x = XQ3  + DW / 2     # 12.4
    ROW3_top   = Y_ROW3 + AH / 2
    ROW3_bot   = Y_ROW3 - AH / 2
    ROW4_top   = Y_ROW4 + AH / 2
    BEST_top   = Y_BEST + BH / 2

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.text(XC, Y_TITLE,
            "Practitioner Decision Tree: Selecting a Ground-Truth\n"
            "Labeling Convention for SCADA Anomaly Detection",
            ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=C_S, multialignment="center")

    # ── START ──────────────────────────────────────────────────────────────────
    _rect(ax, XC, Y_START, SW, SH,
          ["START — Label your SCADA anomaly dataset"],
          C_S, fontsize=10)

    # ── Q1: PLC columns available? ────────────────────────────────────────────
    _diamond(ax, XC, Y_Q1, DW, DH,
             ["Q1: Are PLC / SCADA status-code",
              "columns available and populated?"],
             C_Q, fontsize=9)
    _arrow(ax, XC, Y_START - SH / 2, XC, Y_Q1 + DH / 2)

    # ── Q1 → Q2 (YES, left) ───────────────────────────────────────────────────
    # from Q1 left corner → horizontal to XQ2 → down to Q2 top
    _elbow_h_v(ax, XC - DW / 2, Y_Q1, XQ2, Q23_top)
    _label(ax, (XC - DW / 2 + XQ2) / 2, Y_Q1, "YES", side="left")

    # ── Q1 → Q3 (NO, right) ───────────────────────────────────────────────────
    # from Q1 right corner → horizontal to XQ3 → down to Q3 top
    _elbow_h_v(ax, XC + DW / 2, Y_Q1, XQ3, Q23_top)
    _label(ax, (XC + DW / 2 + XQ3) / 2, Y_Q1, "NO", side="right")

    # ── Q2: PLC reliable? ─────────────────────────────────────────────────────
    _diamond(ax, XQ2, Y_Q23, DW, DH,
             ["Q2: Does the PLC code reliably",
              "fire during faults?",
              "(check: LAS-data < 0.30)"],
             C_Q, fontsize=8.5)

    # Q2 YES (left) → A_PLC
    _elbow_h_v(ax, Q2_left_x, Y_Q23, XL, ROW3_top)
    _label(ax, (Q2_left_x + XL) / 2, Y_Q23, "YES", side="left")

    # Q2 NO (right) → Q4
    _elbow_h_v(ax, Q2_right_x, Y_Q23, XCL, ROW3_top)
    _label(ax, (Q2_right_x + XCL) / 2, Y_Q23, "NO", side="right")

    # ── A_PLC: USE PLC labels ─────────────────────────────────────────────────
    _rect(ax, XL, Y_ROW3, AW, AH,
          ["USE: PLC status-code labels",
           "(status_type_id ≠ 0)",
           "Expect high AUC.",
           "Disclose convention in paper."],
          C_A, fontsize=8)

    # ── Q4: Compute LAS-data ──────────────────────────────────────────────────
    _rect(ax, XCL, Y_ROW3, AW, AH,
          ["Compute LAS-data (Jaccard)",
           "between PLC codes and",
           "maintenance log.",
           "LAS-data > 0.30 → DO NOT",
           "use PLC labels alone."],
          C_W, fontsize=7.5)

    # Q4 → A_EVT_L
    _arrow(ax, XCL, ROW3_bot, XCL, ROW4_top)

    # ── A_EVT_L: USE event-window (from Q2 NO path) ───────────────────────────
    _rect(ax, XCL, Y_ROW4, AW, AH,
          ["USE: Event-window labels",
           "(event_start_id →",
           " event_end_id)",
           "Lower AUC expected;",
           "report honestly."],
          C_A, fontsize=7.8)

    # ── Q3: Maintenance log available? ────────────────────────────────────────
    _diamond(ax, XQ3, Y_Q23, DW, DH,
             ["Q3: Is a maintenance log /",
              "event_start_id column",
              "available?"],
             C_Q, fontsize=8.5)

    # Q3 YES (left) → A_EVT_R
    _elbow_h_v(ax, Q3_left_x, Y_Q23, XCR, ROW3_top)
    _label(ax, (Q3_left_x + XCR) / 2, Y_Q23, "YES", side="left")

    # Q3 NO (right) → A_PROXY
    _elbow_h_v(ax, Q3_right_x, Y_Q23, XR, ROW3_top)
    _label(ax, (Q3_right_x + XR) / 2, Y_Q23, "NO", side="right")

    # ── A_EVT_R: USE event-window (from Q3 YES) ───────────────────────────────
    _rect(ax, XCR, Y_ROW3, AW, AH,
          ["USE: Event-window labels",
           "(event_start_id →",
           " event_end_id)",
           "Conservative; often lower",
           "AUC. Report LAS-data too."],
          C_A, fontsize=7.8)

    # ── A_PROXY: USE proxy labels ─────────────────────────────────────────────
    _rect(ax, XR, Y_ROW3, AW, AH,
          ["USE: Proxy labels only",
           "(rolling-score threshold",
           " or expert annotation)",
           "High risk of label leakage",
           "— caveat results strongly."],
          C_W, fontsize=7.8)

    # ── Convergence arrows to BEST PRACTICE ───────────────────────────────────
    # A_PLC bottom → BEST top
    _arrow(ax, XL,   ROW3_bot, XL,   BEST_top)
    # A_EVT_L bottom → BEST top
    _arrow(ax, XCL,  Y_ROW4 - AH / 2, XCL,  BEST_top)
    # A_EVT_R bottom → BEST top
    _arrow(ax, XCR,  ROW3_bot, XCR,  BEST_top)
    # A_PROXY bottom → BEST top
    _arrow(ax, XR,   ROW3_bot, XR,   BEST_top)

    # ── BEST PRACTICE ─────────────────────────────────────────────────────────
    _rect(ax, XC, Y_BEST, BW, BH,
          ["BEST PRACTICE — Report AUC under BOTH conventions + LAS-model score.",
           "LAS-model > 0.30: CRITICAL — dual reporting mandatory; results unreliable without disclosure.",
           "LAS-model 0.10–0.30: HIGH — flag as high ambiguity in abstract or Section II.",
           "Mandatory: state labeling rule explicitly in every paper."],
          C_G, fontsize=8.5, lw=2.0)

    # ── Empirical reference panel ──────────────────────────────────────────────
    ref = (
        "Empirical reference (this paper — CARE dataset):\n"
        "  Farm B: LAS-model = 0.463  →  CRITICAL   |   "
        "Farm C: LAS-model = 0.296  →  HIGH   |   "
        "Farm A: LAS-model = 0.011  →  SAFE (null result)"
    )
    ax.text(XC, Y_REF, ref,
            ha="center", va="center",
            fontsize=8.5, fontstyle="italic", color="#1A365D",
            multialignment="center",
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="#EBF8FF", edgecolor="#2B6CB0",
                      linewidth=1.5))

    # ── Legend ─────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=C_Q, label="Decision node"),
        mpatches.Patch(color=C_A, label="Recommended action"),
        mpatches.Patch(color=C_W, label="Caution / conditional"),
        mpatches.Patch(color=C_G, label="Best practice (gold standard)"),
    ]
    ax.legend(handles=handles,
              loc="lower left", bbox_to_anchor=(0.01, 0.005),
              fontsize=8.5, framealpha=0.95, edgecolor="#AAAAAA")

    fig.tight_layout(pad=0.2)
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()

    png_path = FIG_DIR / "labeling_decision_tree.png"
    pdf_path = FIG_DIR / "labeling_decision_tree.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=C_BG)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    print(f"[OK] PNG: {png_path}")
    print(f"[OK] PDF: {pdf_path}")
    print("Caption suggestion:")
    print("  'Practitioner decision tree for selecting a ground-truth labeling")
    print("   convention in SCADA-based wind turbine anomaly detection. LAS-data")
    print("   and LAS-model thresholds are calibrated to CARE Farm A/B/C results.")
    print("   High LAS-model (>0.10) mandates dual-convention reporting.'")


if __name__ == "__main__":
    main()
