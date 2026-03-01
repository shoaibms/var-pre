#!/usr/bin/env python3
r"""
Figure 6: The VAD Diagnostic Framework -- Practitioner Toolkit
===============================================================

Panel layout (asymmetric 12-col GridSpec):
  Row 1:  a (7 cols) Topographic safety landscape + marginal rug plots
        | b (5 cols) Raincloud validation by zone
  Row 2:  c (5 cols) Dual correlation subpanels
        | d (7 cols) Decision flowchart diagram

All data read from files. Missing file or column = crash.

Output:
  C:\Users\ms\Desktop\var-pre\outputs\figures\figure_6.png
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# COLOURS  (centralised palette)
# =====================================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colourlist import (
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    TEXT_PRIMARY, TEXT_SECONDARY, SPINE_COLOR, GRID_COLOR, BG_WHITE,
    GREY, GREY_LIGHTER, GREY_LIGHT, GREY_PALE,
    DARK_TURQUOISE, SPRING_GREEN, TEAL,
    DS_MARKERS, DS_SHORT, DS_DISPLAY, FONT,
    vad_zones, vad_models,
    bugreen, ylgreen, greens,
)

# =====================================================================
# DESIGN SYSTEM
# =====================================================================

TEXT_TERTIARY    = "#52796f"
TEXT_LIGHT       = "#6b8f71"

ZONE_SAFE        = vad_zones["safe"]
ZONE_SAFE_BG     = vad_zones["safe_bg"]
ZONE_UNCERTAIN   = vad_zones["uncertain"]
ZONE_UNCERTAIN_BG = vad_zones["uncertain_bg"]
ZONE_RISKY       = vad_zones["risky"]
ZONE_RISKY_BG    = vad_zones["risky_bg"]

DEEP_GREEN       = COUPLED_GREEN
XGB_GREEN        = vad_models["xgb"]
RF_GREEN         = vad_models["rf"]

NEUTRAL_MARKER   = "#6E7681"

DI_CMAP = LinearSegmentedColormap.from_list("di_green", [
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN,
    NEUTRAL_GREEN, LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
], N=256)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)


# =====================================================================
# ZONE HELPERS
# =====================================================================

ZONE_MAP = {
    "GREEN_SAFE":           ("GREEN (safe)", ZONE_SAFE,      ZONE_SAFE_BG,      0),
    "YELLOW_INCONCLUSIVE":  ("YELLOW (inconclusive)", ZONE_UNCERTAIN, ZONE_UNCERTAIN_BG, 1),
    "RED_HARMFUL":          ("RED (harmful)",   ZONE_RISKY,     ZONE_RISKY_BG,     2),
}


def _zone_label(raw: str) -> str:
    return ZONE_MAP[raw][0]

def _zone_color(raw: str) -> str:
    return ZONE_MAP[raw][1]

def _zone_bg(raw: str) -> str:
    return ZONE_MAP[raw][2]

def _zone_order(raw: str) -> int:
    return ZONE_MAP[raw][3]


# =====================================================================
# STYLE
# =====================================================================

def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": FONT["base"],
        "axes.titlesize": FONT["title"],
        "axes.titleweight": "bold",
        "axes.labelsize": FONT["label"],
        "axes.labelweight": "medium",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": SPINE_COLOR,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "xtick.color": SPINE_COLOR,
        "ytick.color": SPINE_COLOR,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": FONT["legend"],
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": GRID_COLOR,
        "legend.fancybox": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# =====================================================================
# HELPERS
# =====================================================================

def panel_label(ax, letter, subtitle="", x=-0.02, y=1.08):
    if letter.strip():
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=FONT["panel"], fontweight="black", color=TEXT_PRIMARY,
                va="bottom", ha="left",
                path_effects=[pe.withStroke(linewidth=3, foreground=BG_WHITE)])


def subtle_grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID_COLOR, linewidth=0.4, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def require_col(df: pd.DataFrame, col: str, context: str):
    assert col in df.columns, (
        f"MISSING COLUMN '{col}' in {context}. "
        f"Available: {list(df.columns)}"
    )


def delta_to_pp(df: pd.DataFrame, col: str, context: str) -> np.ndarray:
    require_col(df, col, context)
    raw = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(raw))
    assert np.isfinite(max_abs), f"{context}: all values NaN/inf for '{col}'"
    assert max_abs <= 1.0 + 1e-9, (
        f"{context}: '{col}' max|value|={max_abs:.3f} > 1.0. "
        "Already in pp? Remove x100 conversion."
    )
    return raw * 100.0


def assert_zone_consistency(overview, ablation, model):
    z = overview[["dataset", "view", "predicted_zone"]].rename(
        columns={"predicted_zone": "predicted_zone_overview"})
    m = ablation.merge(z, on=["dataset", "view"], how="left")
    assert m["predicted_zone_overview"].notna().all(), (
        f"{model}: missing overview zones for some (dataset,view) keys")
    bad = m[m["predicted_zone"] != m["predicted_zone_overview"]]
    assert bad.empty, (
        f"{model}: zone mismatch: "
        + ", ".join((bad["dataset"] + "/" + bad["view"]).tolist()))


# =====================================================================
# DATA LOADING
# =====================================================================

SEC5 = Path("results") / "main_results" / "section_5_diagnostic"


def load_overview(outputs_dir):
    p = outputs_dir / SEC5 / "vad_metric_overview.csv"
    assert p.exists(), f"MISSING FILE: {p}"
    df = pd.read_csv(p)
    assert len(df) == 14, f"Expected 14 rows in overview, got {len(df)}"
    for col in ["dataset", "view", "vsa_mean", "eta_es_mean", "pcla_mean",
                "predicted_zone", "DI_10pct_consensus"]:
        require_col(df, col, "vad_metric_overview.csv")
    valid_zones = set(ZONE_MAP.keys())
    actual_zones = set(df["predicted_zone"].unique())
    bad = actual_zones - valid_zones
    assert not bad, f"Unknown zones: {bad}. Expected: {valid_zones}"
    print(f"  overview: {len(df)} rows, zones: {df['predicted_zone'].value_counts().to_dict()}")
    return df


def load_ablation(outputs_dir, model):
    p = outputs_dir / SEC5 / "source_tables" / f"vad_vs_ablation__{model}.csv"
    assert p.exists(), f"MISSING FILE: {p}"
    df = pd.read_csv(p)
    assert len(df) == 14, f"Expected 14 rows in ablation/{model}, got {len(df)}"
    for col in ["dataset", "view", "delta_var_minus_random_mean",
                "perf_var_mean", "perf_random_mean", "predicted_zone",
                "vsa_mean", "pcla_mean"]:
        require_col(df, col, f"vad_vs_ablation__{model}.csv")
    print(f"  ablation ({model}): {len(df)} rows")
    return df


def load_validation(outputs_dir):
    p = outputs_dir / SEC5 / "vad_validation_by_model.csv"
    assert p.exists(), f"MISSING FILE: {p}"
    df = pd.read_csv(p)
    for col in ["model", "metric_col", "rho", "p_value"]:
        require_col(df, col, "vad_validation_by_model.csv")
    print(f"  validation: {len(df)} rows")
    return df


# =====================================================================
# PANEL A: Zone Map with Gradient Fills + Marginal Rug Plots
# =====================================================================

def panel_A_zonemap(ax_main, ax_top, ax_right, overview):
    """
    Scatter over smooth gradient zone backgrounds (no model implied).
    Marginal rug plots (honest for n=14, no density estimation).
    """
    print("    Panel A: Zone map with gradient fills + rug plots...")

    vsa = overview["vsa_mean"].values.astype(float)
    eta = overview["eta_es_mean"].values.astype(float)
    di  = overview["DI_10pct_consensus"].values.astype(float)
    zones = overview["predicted_zone"].values
    datasets = overview["dataset"].values
    views = overview["view"].values

    # Axis limits
    vsa_pad = max(0.08, (vsa.max() - vsa.min()) * 0.18)
    eta_pad = max(0.08, (eta.max() - eta.min()) * 0.18)
    xlim = (-0.4, max(vsa.max(), 0.05) + vsa_pad)
    ylim = (-2.0, max(eta.max(), 1.15) + eta_pad)
    vsa_range = xlim[1] - xlim[0]
    eta_range = ylim[1] - ylim[0]

    # --- Gradient zone backgrounds (purely definitional, no fitting) ---
    # Build a colour image: each quadrant gets a flat tint that fades
    # gently toward the threshold lines (pure visual polish, not a model).
    grid_res = 300
    xg = np.linspace(xlim[0], xlim[1], grid_res)
    yg = np.linspace(ylim[0], ylim[1], grid_res)
    Xg, Yg = np.meshgrid(xg, yg)

    # RGBA image: start with white, blend zone colours near corners
    img = np.ones((grid_res, grid_res, 4), dtype=float)

    # Distance from thresholds (normalised 0-1 across range)
    dx = (Xg - 0) / vsa_range          # positive = right of VSA=0
    dy = (Yg - 1.0) / eta_range        # positive = above eta=1

    # Quadrant masks (soft via clipping, no sigmoid)
    safe_strength    = np.clip(dx * 4, 0, 1) * np.clip(dy * 4, 0, 1)
    risky_strength   = np.clip(-dx * 4, 0, 1) * np.clip(-dy * 4, 0, 1)
    uncert1_strength = np.clip(dx * 4, 0, 1) * np.clip(-dy * 4, 0, 1)
    uncert2_strength = np.clip(-dx * 4, 0, 1) * np.clip(dy * 4, 0, 1)

    # Blend zone colours (low alpha → subtle tint)
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i+2], 16)/255 for i in (0, 2, 4)])

    safe_rgb   = hex_to_rgb(ZONE_SAFE_BG)
    risky_rgb  = hex_to_rgb(ZONE_RISKY_BG)
    uncert_rgb = hex_to_rgb(ZONE_UNCERTAIN_BG)

    for c in range(3):
        img[:, :, c] = (1.0
                        - safe_strength * (1.0 - safe_rgb[c]) * 0.55
                        - risky_strength * (1.0 - risky_rgb[c]) * 0.55
                        - uncert1_strength * (1.0 - uncert_rgb[c]) * 0.35
                        - uncert2_strength * (1.0 - uncert_rgb[c]) * 0.35)
    img = np.clip(img, 0, 1)

    ax_main.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   origin="lower", aspect="auto", zorder=0, interpolation="bilinear")

    # Threshold lines -- layered glow + dashed
    for lw, alpha in [(4.0, 0.08), (2.0, 0.15), (1.0, 0.4)]:
        ax_main.axvline(0, color=TEXT_SECONDARY, linewidth=lw, alpha=alpha, zorder=2)
        ax_main.axhline(1.0, color=TEXT_SECONDARY, linewidth=lw, alpha=alpha, zorder=2)
    ax_main.axvline(0, color=TEXT_PRIMARY, linewidth=1.2, linestyle="--",
                    alpha=0.5, zorder=3)
    ax_main.axhline(1.0, color=TEXT_PRIMARY, linewidth=1.2, linestyle="--",
                    alpha=0.5, zorder=3)

    # Zone labels
    ax_main.text(0.96, 0.84, "GREEN (safe)\nVariance filtering\npreserves signal",
                 transform=ax_main.transAxes, fontsize=8.5, fontweight="bold",
                 color=ZONE_SAFE, alpha=0.8, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.75, edgecolor=ZONE_SAFE, linewidth=0.8))
    ax_main.text(0.04, 0.04, "RED (harmful)\nVariance filtering\ndestroys signal",
                 transform=ax_main.transAxes, fontsize=8.5, fontweight="bold",
                 color=ZONE_RISKY, alpha=0.8, ha="left", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.75, edgecolor=ZONE_RISKY, linewidth=0.8))
    ax_main.text(0.96, 0.04, "YELLOW (inconclusive)",
                 transform=ax_main.transAxes, fontsize=8, fontweight="bold",
                 color=ZONE_UNCERTAIN, alpha=0.55, ha="right", va="bottom")
    ax_main.text(0.04, 0.96, "YELLOW (inconclusive)",
                 transform=ax_main.transAxes, fontsize=8, fontweight="bold",
                 color=ZONE_UNCERTAIN, alpha=0.55, ha="left", va="top")

    # Collision detection for labels
    label_positions = [
        (xlim[0] + 0.87 * vsa_range, ylim[0] + 0.92 * eta_range),
        (xlim[0] + 0.13 * vsa_range, ylim[0] + 0.10 * eta_range),
    ]

    # --- Points with glow halos ---
    for glow_s, glow_a in [(280, 0.05), (180, 0.10), (120, 0.20)]:
        for i in range(len(overview)):
            c = DI_CMAP(DI_NORM(di[i]))
            m = DS_MARKERS[datasets[i]]
            ax_main.scatter(vsa[i], eta[i], s=glow_s, color=c,
                            alpha=glow_a, marker=m, edgecolors="none", zorder=4)

    for i in range(len(overview)):
        c = DI_CMAP(DI_NORM(di[i]))
        m = DS_MARKERS[datasets[i]]
        ax_main.scatter(vsa[i], eta[i], s=80, color=c, marker=m,
                        edgecolors="white", linewidths=1.2, zorder=6)

    # Labels for safe & risky
    order = np.argsort(-np.sqrt(vsa ** 2 + (eta - 1.0) ** 2))
    for idx in order:
        zone_raw = zones[idx]
        if zone_raw == "YELLOW_INCONCLUSIVE":
            continue
        lbl = f"{DS_SHORT[datasets[idx]]}/{views[idx]}"
        x_pt, y_pt = vsa[idx], eta[idx]
        x_off = 0.025 * vsa_range if x_pt >= 0 else -0.025 * vsa_range
        y_off = 0.025 * eta_range
        ha = "left" if x_pt >= 0 else "right"

        for attempt in range(12):
            x_txt = x_pt + x_off
            y_txt = y_pt + y_off
            collision = any(
                abs(lx - x_txt) < 0.12 * vsa_range
                and abs(ly - y_txt) < 0.08 * eta_range
                for lx, ly in label_positions
            )
            if not collision:
                break
            if attempt % 2 == 0:
                y_off += 0.04 * eta_range
            else:
                x_off += (0.03 if x_pt >= 0 else -0.03) * vsa_range

        ax_main.annotate(lbl, (x_pt, y_pt),
                         xytext=(x_txt, y_txt),
                         fontsize=6.8, color=TEXT_PRIMARY, fontweight="medium",
                         ha=ha, va="bottom",
                         arrowprops=dict(arrowstyle="-", color=TEXT_LIGHT,
                                        linewidth=0.4, alpha=0.4,
                                        connectionstyle="arc3,rad=0.1"),
                         bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                                   alpha=0.85, edgecolor="none"),
                         zorder=7)
        label_positions.append((x_txt, y_txt))

    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    ax_main.set_xlabel("Variance-Signal Alignment (VSA)", fontweight="medium")
    ax_main.set_ylabel(r"Signal enrichment ratio ($\eta_{ES}$)", fontweight="medium")

    # Threshold annotations
    ax_main.text(0.03, ylim[0] + eta_range * 0.02, "VSA = 0",
                 fontsize=7.5, color=TEXT_TERTIARY, ha="center", va="bottom",
                 rotation=90, style="italic", alpha=0.7)
    ax_main.text(xlim[0] + vsa_range * 0.02, 1.015, r"$\eta_{ES}=1.0$",
                 fontsize=7.5, color=TEXT_TERTIARY, ha="left", va="bottom",
                 style="italic", alpha=0.7)

    # Dataset shape legend
    present = set(datasets)
    ds_handles = [
        Line2D([0], [0], marker=DS_MARKERS[ds], color="none",
               markerfacecolor=NEUTRAL_MARKER, markeredgecolor="white",
               markersize=8, label=DS_DISPLAY[ds])
        for ds in DS_MARKERS if ds in present
    ]
    leg = ax_main.legend(handles=ds_handles, loc="upper center",
                         bbox_to_anchor=(0.5, 1.0),
                         fontsize=7.5, framealpha=0.88, edgecolor=GRID_COLOR,
                         title="Dataset (shape)", title_fontsize=7.5,
                         handletextpad=0.4, borderpad=0.4)
    leg.get_title().set_fontweight("bold")

    ax_main.text(0.5, -0.16, "n = 14 views at K = 10%  |  colour = DI",
                 transform=ax_main.transAxes, fontsize=8, ha="center",
                 color=TEXT_TERTIARY, style="italic")

    # --- Marginal rug plots (honest for n=14) ---
    # Top marginal: VSA rug
    ax_top.set_xlim(xlim)
    for i in range(len(vsa)):
        zone_raw = zones[i]
        c = _zone_color(zone_raw)
        ax_top.plot([vsa[i], vsa[i]], [0, 1], color=c,
                    linewidth=1.8, alpha=0.7, solid_capstyle="round")
        # Small coloured dot at top of tick
        ax_top.scatter(vsa[i], 0.85, s=18, color=DI_CMAP(DI_NORM(di[i])),
                       edgecolors="white", linewidths=0.5, zorder=5)
    ax_top.axvline(0, color=TEXT_PRIMARY, linewidth=0.8, linestyle="--", alpha=0.35)
    ax_top.set_ylim(0, 1.1)
    ax_top.set_yticks([])
    for sp in ax_top.spines.values():
        sp.set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)

    # Right marginal: etaES rug
    ax_right.set_ylim(ylim)
    for i in range(len(eta)):
        zone_raw = zones[i]
        c = _zone_color(zone_raw)
        ax_right.plot([0, 1], [eta[i], eta[i]], color=c,
                      linewidth=1.8, alpha=0.7, solid_capstyle="round")
        ax_right.scatter(0.85, eta[i], s=18, color=DI_CMAP(DI_NORM(di[i])),
                         edgecolors="white", linewidths=0.5, zorder=5)
    ax_right.axhline(1.0, color=TEXT_PRIMARY, linewidth=0.8, linestyle="--", alpha=0.35)
    ax_right.set_xlim(0, 1.1)
    ax_right.set_xticks([])
    for sp in ax_right.spines.values():
        sp.set_visible(False)
    ax_right.tick_params(labelleft=False, left=False)

    panel_label(ax_main, "a", "Variance-filtering safety landscape", y=1.18)


# =====================================================================
# PANEL B: Raincloud Plot -- Validation by Zone
# =====================================================================

def panel_B_raincloud(ax, abl_xgb, abl_rf):
    """
    Raincloud: half-violin 'cloud' + jittered individual points 'rain'
    + bold median bar. XGB and RF shown as paired points.
    """
    print("    Panel B: Raincloud validation by zone...")

    rows = []
    for model_label, df in [("XGBoost", abl_xgb), ("RF", abl_rf)]:
        delta_pp = delta_to_pp(df, "delta_var_minus_random_mean", f"panel B {model_label}")
        for i in range(len(df)):
            rows.append({
                "model": model_label,
                "delta_pp": delta_pp[i],
                "zone_raw": df["predicted_zone"].values[i],
                "zone_label": _zone_label(df["predicted_zone"].values[i]),
                "zone_order": _zone_order(df["predicted_zone"].values[i]),
                "dataset": df["dataset"].values[i],
                "view": df["view"].values[i],
            })

    plot_df = pd.DataFrame(rows)
    zones_sorted = (plot_df.drop_duplicates("zone_raw")
                    .sort_values("zone_order")["zone_raw"].tolist())
    n_zones = len(zones_sorted)
    assert n_zones == 3, f"Expected 3 zones, got {n_zones}"

    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.4, zorder=1)

    model_colors = {"XGBoost": XGB_GREEN, "RF": RF_GREEN}
    rng = np.random.default_rng(42)

    for i, z in enumerate(zones_sorted):
        zone_data = plot_df[plot_df["zone_raw"] == z]
        all_pp = zone_data["delta_pp"].values
        zone_c = _zone_color(z)

        # --- Half-violin (cloud) on the LEFT side ---
        if len(all_pp) >= 3 and np.std(all_pp) > 1e-9:
            kde = gaussian_kde(all_pp, bw_method=0.4)
            y_eval = np.linspace(all_pp.min() - 2, all_pp.max() + 2, 150)
            density = kde(y_eval)
            # Normalize density so max width ~ 0.35
            density_norm = density / density.max() * 0.32
            # Draw on left side of position i
            ax.fill_betweenx(y_eval, i - density_norm, i,
                             color=_zone_bg(z), alpha=0.7, zorder=2,
                             edgecolor=zone_c, linewidth=0.8)

        # --- Rain drops (individual points) on the RIGHT side ---
        for model in ["XGBoost", "RF"]:
            md = zone_data[zone_data["model"] == model]
            if len(md) == 0:
                continue
            model_offset = 0.08 if model == "XGBoost" else 0.20
            jitter = rng.uniform(-0.03, 0.03, size=len(md))
            xs = i + model_offset + jitter
            ys = md["delta_pp"].values

            # Glow
            ax.scatter(xs, ys, s=100, color=model_colors[model],
                       alpha=0.10, edgecolors="none", zorder=3)
            # Main dots
            ax.scatter(xs, ys, s=50, color=model_colors[model],
                       edgecolors="white", linewidths=0.7, zorder=5, alpha=0.85)

        # --- Paired lines connecting XGB & RF for same dataset-view ---
        for dv in zone_data["dataset"].unique():
            for vv in zone_data[zone_data["dataset"] == dv]["view"].unique():
                xgb_row = zone_data[(zone_data["dataset"] == dv) &
                                    (zone_data["view"] == vv) &
                                    (zone_data["model"] == "XGBoost")]
                rf_row = zone_data[(zone_data["dataset"] == dv) &
                                   (zone_data["view"] == vv) &
                                   (zone_data["model"] == "RF")]
                if len(xgb_row) == 1 and len(rf_row) == 1:
                    ax.plot([i + 0.08, i + 0.20],
                            [xgb_row["delta_pp"].values[0], rf_row["delta_pp"].values[0]],
                            color=zone_c, linewidth=0.6, alpha=0.25, zorder=4)

        # --- Bold median bar ---
        med = np.median(all_pp)
        ax.plot([i - 0.05, i + 0.30], [med, med],
                color=zone_c, linewidth=3.0, alpha=0.85,
                zorder=6, solid_capstyle="round")
        ax.text(i + 0.32, med, f"{med:+.1f} pp",
                fontsize=7.5, fontweight="bold", color=zone_c,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          alpha=0.85, edgecolor="none"),
                zorder=7)

    ax.set_xticks(range(n_zones))
    ax.set_xticklabels([_zone_label(z) for z in zones_sorted],
                       fontsize=9, fontweight="bold")
    for tick_label, z in zip(ax.get_xticklabels(), zones_sorted):
        tick_label.set_color(_zone_color(z))

    ax.set_ylabel("$\\Delta$(TopVar $-$ Random) (pp)")
    ax.set_xlabel("")
    subtle_grid(ax, axis="y")

    # "Below 0 = harm" callout
    ax.text(0.97, 0.03, "Below 0 = harm", fontsize=7.5,
            transform=ax.transAxes, ha="right", va="bottom",
            color=TEXT_TERTIARY, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=ZONE_RISKY_BG,
                      alpha=0.5, edgecolor="none"))

    # Model legend
    model_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=XGB_GREEN, markeredgecolor="white",
               markersize=7, label="XGBoost"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=RF_GREEN, markeredgecolor="white",
               markersize=7, label="Random Forest"),
        Line2D([0], [0], color=TEXT_LIGHT, linewidth=0.8,
               alpha=0.5, label="Paired view"),
    ]
    ax.legend(handles=model_handles, loc="upper center",
              bbox_to_anchor=(0.5, 0.995), fontsize=7.5,
              framealpha=0.88, edgecolor=GRID_COLOR)

    panel_label(ax, "b", "Zones predict real harm")


# =====================================================================
# PANEL C: Two Correlation Subpanels (C1: XGB/PCLA, C2: RF/VSA)
# =====================================================================

def _draw_correlation_subpanel(ax, x_vals, y_vals, di_vals, datasets,
                               x_label, model_label, model_color,
                               marker_shape, rho, p_val, panel_id):
    """Shared helper for C1 and C2 correlation scatter with CI band + glow."""

    # Regression
    coeffs = np.polyfit(x_vals, y_vals, 1)
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)

    # 95% CI band
    y_pred = np.polyval(coeffs, x_vals)
    resid = y_vals - y_pred
    ss_x = np.sum((x_vals - x_vals.mean()) ** 2)
    se = np.std(resid) * np.sqrt(
        1.0 / len(x_vals) + (x_fit - x_vals.mean()) ** 2 / max(ss_x, 1e-12))
    ci = 1.96 * se

    ax.fill_between(x_fit, y_fit - ci, y_fit + ci,
                    color=model_color, alpha=0.08, zorder=1)
    ax.plot(x_fit, y_fit, color=model_color, linewidth=2.0,
            alpha=0.65, zorder=3)

    ax.axhline(0, color=TEXT_LIGHT, linewidth=0.8, linestyle=":", alpha=0.4, zorder=1)

    # Points with glow
    for i in range(len(x_vals)):
        c = DI_CMAP(DI_NORM(di_vals[i]))
        m = DS_MARKERS.get(datasets[i], marker_shape)
        ax.scatter(x_vals[i], y_vals[i], s=120, color=c,
                   alpha=0.12, marker=m, edgecolors="none", zorder=4)
        ax.scatter(x_vals[i], y_vals[i], s=60, color=c, marker=m,
                   edgecolors=model_color, linewidths=0.9, zorder=6)

    # Spearman annotation
    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    ax.text(0.04, 0.96,
            f"Spearman \u03C1 = {rho:+.2f}\n{p_str}",
            transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            color=TEXT_PRIMARY, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.88, edgecolor=GRID_COLOR, linewidth=0.8))

    ax.set_xlabel(x_label, fontweight="medium", color=TEXT_PRIMARY)
    ax.set_ylabel("$\\Delta$(TopVar $-$ Random) (pp)", fontsize=FONT["label"] - 1)
    subtle_grid(ax, axis="y")

    # Dataset shape legend
    present = set(datasets)
    ds_handles = [
        Line2D([0], [0], marker=DS_MARKERS[ds], color="none",
               markerfacecolor=NEUTRAL_MARKER, markeredgecolor=model_color,
               markersize=6, label=DS_DISPLAY[ds])
        for ds in DS_MARKERS if ds in present
    ]
    ax.legend(handles=ds_handles, loc="lower right", fontsize=6.5,
              framealpha=0.88, edgecolor=GRID_COLOR,
              handletextpad=0.3, borderpad=0.3)

    ax.text(0.5, 1.01, model_label, transform=ax.transAxes,
            fontsize=9.0, fontweight="bold", color=TEXT_PRIMARY,
            ha="center", va="bottom")


def panel_C_correlations(ax_c1, ax_c2, abl_xgb, abl_rf, validation):
    """Two side-by-side correlation subpanels (C1: XGB/PCLA, C2: RF/VSA)."""
    print("    Panel C: Two correlation subpanels...")

    # --- C1: XGBoost / PCLA ---
    pcla_xgb = abl_xgb["pcla_mean"].values.astype(float)
    delta_xgb = delta_to_pp(abl_xgb, "delta_var_minus_random_mean", "panel C1 XGB")
    di_xgb = abl_xgb["DI_10pct_consensus"].values.astype(float)
    ds_xgb = abl_xgb["dataset"].values

    row_xgb = validation[(validation["model"] == "xgb_bal") &
                         (validation["metric_col"] == "pcla_mean")]
    assert len(row_xgb) == 1, f"Expected 1 xgb_bal/pcla_mean row, got {len(row_xgb)}"
    rho_xgb = float(row_xgb.iloc[0]["rho"])
    p_xgb = float(row_xgb.iloc[0]["p_value"])

    _draw_correlation_subpanel(
        ax_c1, pcla_xgb, delta_xgb, di_xgb, ds_xgb,
        "PCLA (XGBoost)", "XGBoost", XGB_GREEN, "D",
        rho_xgb, p_xgb, "c\u2081")

    # --- C2: RF / VSA ---
    vsa_rf = abl_rf["vsa_mean"].values.astype(float)
    delta_rf = delta_to_pp(abl_rf, "delta_var_minus_random_mean", "panel C2 RF")
    di_rf = abl_rf["DI_10pct_consensus"].values.astype(float)
    ds_rf = abl_rf["dataset"].values

    row_rf = validation[(validation["model"] == "rf") &
                        (validation["metric_col"] == "vsa_mean")]
    assert len(row_rf) == 1, f"Expected 1 rf/vsa_mean row, got {len(row_rf)}"
    rho_rf = float(row_rf.iloc[0]["rho"])
    p_rf = float(row_rf.iloc[0]["p_value"])

    _draw_correlation_subpanel(
        ax_c2, vsa_rf, delta_rf, di_rf, ds_rf,
        "VSA (Random Forest)", "Random Forest", RF_GREEN, "o",
        rho_rf, p_rf, "c\u2082")

    ax_c1.text(0.02, 1.08, "c", transform=ax_c1.transAxes,
               fontsize=FONT["panel"], fontweight="black", color=TEXT_PRIMARY,
               ha="left", va="bottom",
               path_effects=[pe.withStroke(linewidth=3, foreground=BG_WHITE)])


# =====================================================================
# PANEL D: Metro-Map Decision Diagram
# =====================================================================

def panel_D_flowchart(ax):
    """
    Decision flowchart.
    Boxes + diamond decision nodes + curved arrows.
    """
    print("    Panel D: Flowchart...")

    FLOW_COMPUTE = "#488976"
    FLOW_CHECK   = "#2e8b57"
    FLOW_SAFE    = "#008080"
    FLOW_RISKY   = "#ed9c77"
    FLOW_UNCERTAIN = "#F0EB93"
    FLOW_BG      = "#E0F7FA"
    FLOW_ARROW   = "#34495e"

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 0.2), 9.6, 9.6,
        boxstyle="round,pad=0.2", facecolor=FLOW_BG, alpha=0.35,
        edgecolor=GRID_COLOR, linewidth=0.6))

    def draw_box(x, y, w, h, text, color, text_color="white",
                 fontsize=8.5, fontweight="bold"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.18", facecolor=color, alpha=0.92,
            edgecolor=to_rgba(color, 0.7), linewidth=1.5, zorder=3))
        ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight,
                color=text_color, ha="center", va="center", zorder=4,
                linespacing=1.35)

    def draw_diamond(x, y, size, text, color, fontsize=7):
        ax.add_patch(plt.Polygon(
            [(x, y + size), (x + size * 0.9, y), (x, y - size), (x - size * 0.9, y)],
            facecolor=color, edgecolor=to_rgba(color, 0.7),
            linewidth=1.5, alpha=0.92, zorder=3))
        ax.text(x, y, text, fontsize=fontsize, fontweight="bold",
                color="white", ha="center", va="center", zorder=4,
                linespacing=1.15)

    def draw_arrow(x1, y1, x2, y2, label="", label_side="right", curved=False):
        style = "arc3,rad=0.15" if curved else "arc3,rad=0"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=FLOW_ARROW,
                                   linewidth=1.8, shrinkA=4, shrinkB=4,
                                   connectionstyle=style),
                    zorder=2)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            off = (0.3, 0) if label_side == "right" else (-0.3, 0)
            ax.text(mx + off[0], my + off[1], label,
                    fontsize=7.5, color=FLOW_ARROW, fontweight="bold",
                    ha="left" if label_side == "right" else "right",
                    va="center", style="italic",
                    bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                              alpha=0.85, edgecolor="none"),
                    zorder=5)

    cx = 5.0

    draw_box(cx, 9.0, 4.5, 0.75,
             "1  Use training fold / split only", FLOW_COMPUTE)
    draw_arrow(cx, 8.62, cx, 8.15)

    draw_box(cx, 7.65, 4.5, 0.85,
             "2  Compute VSA & $\\eta_{ES}$\n    (< 1 second)", FLOW_COMPUTE)
    draw_arrow(cx, 7.22, cx, 6.65)

    draw_diamond(cx, 5.8, 0.85,
                "VSA > 0\n&\n$\\eta_{ES} > 1$?",
                 FLOW_CHECK, fontsize=7.5)

    draw_box(8.2, 3.8, 3.0, 1.15,
             "GREEN (safe)\nProceed with\nTopVar filtering",
             FLOW_SAFE, fontsize=9)
    draw_arrow(cx + 0.76, 5.8, 8.2, 4.4,
               label="Both YES", label_side="right", curved=True)

    draw_diamond(cx, 3.8, 0.75,
                 "VSA < 0\n&\n$\\eta_{ES} < 1$?",
                 "#2e8b57", fontsize=6.5)
    draw_arrow(cx, 4.95, cx, 4.55, label="No", label_side="left")

    draw_box(1.8, 1.8, 3.0, 1.15,
            "RED (harmful)\nUse importance\n(e.g. SHAP)",
             FLOW_RISKY, text_color="white", fontsize=9)
    draw_arrow(cx - 0.68, 3.8, 1.8, 2.4,
               label="Both YES", label_side="left", curved=True)

    draw_box(8.2, 1.8, 3.0, 1.15,
             "YELLOW (inconclusive)\nPilot ablation\nor random baseline",
             FLOW_UNCERTAIN, text_color="white", fontsize=9)
    draw_arrow(cx + 0.68, 3.8, 8.2, 2.4,
               label="Mixed", label_side="right", curved=True)

    ax.text(cx, 0.55, "No model training required",
            fontsize=8.5, color=TEXT_TERTIARY, ha="center", va="center",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=ZONE_SAFE_BG,
                      alpha=0.5, edgecolor=GRID_COLOR, linewidth=0.5))

    panel_label(ax, "d", "Practitioner decision recipe")


# =====================================================================
# MAIN ASSEMBLY
# =====================================================================

overview_ref: pd.DataFrame = pd.DataFrame()


def create_figure(outputs_dir: Path, output_path: Path):
    global overview_ref

    apply_style()

    print("=" * 70)
    print("FIGURE 6: VAD Diagnostic Framework")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/4] Loading data (crash on missing)...")
    overview = load_overview(outputs_dir)
    abl_xgb  = load_ablation(outputs_dir, "xgb")
    abl_rf   = load_ablation(outputs_dir, "rf")
    validation = load_validation(outputs_dir)

    assert_zone_consistency(overview, abl_xgb, "XGBoost")
    assert_zone_consistency(overview, abl_rf,  "RandomForest")

    # Merge DI from overview onto ablation tables
    di_lookup = overview[["dataset", "view", "DI_10pct_consensus"]].copy()
    abl_xgb = abl_xgb.merge(di_lookup, on=["dataset", "view"], how="left",
                              suffixes=("", "_ov"))
    abl_rf  = abl_rf.merge(di_lookup, on=["dataset", "view"], how="left",
                             suffixes=("", "_ov"))
    if "DI_10pct_consensus_ov" in abl_xgb.columns:
        abl_xgb["DI_10pct_consensus"] = abl_xgb["DI_10pct_consensus_ov"]
        abl_xgb = abl_xgb.drop(columns=["DI_10pct_consensus_ov"])
    if "DI_10pct_consensus_ov" in abl_rf.columns:
        abl_rf["DI_10pct_consensus"] = abl_rf["DI_10pct_consensus_ov"]
        abl_rf = abl_rf.drop(columns=["DI_10pct_consensus_ov"])

    require_col(abl_xgb, "DI_10pct_consensus", "abl_xgb after DI merge")
    require_col(abl_rf, "DI_10pct_consensus", "abl_rf after DI merge")
    assert abl_xgb["DI_10pct_consensus"].notna().all(), "NaN DI in abl_xgb"
    assert abl_rf["DI_10pct_consensus"].notna().all(), "NaN DI in abl_rf"

    overview_ref = overview

    # --- Layout ---
    print("\n[2/4] Building layout...")
    fig = plt.figure(figsize=(11, 9))
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    # Outer grid: 2 rows
    gs_outer = GridSpec(
        2, 12, figure=fig,
        height_ratios=[1.3, 1.0],
        hspace=0.38, wspace=1.0,
        left=0.05, right=0.97, top=0.96, bottom=0.05,
    )

    # --- Row 1: Panel A (with marginals) + Panel B ---
    # Panel A needs sub-grid for main + marginal KDEs
    gs_a = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer[0, 0:7],
        width_ratios=[7, 1], height_ratios=[1, 7],
        hspace=0.05, wspace=0.05,
    )
    ax_a_top   = fig.add_subplot(gs_a[0, 0])  # top marginal
    ax_a_main  = fig.add_subplot(gs_a[1, 0])  # main scatter
    ax_a_right = fig.add_subplot(gs_a[1, 1])  # right marginal

    ax_b = fig.add_subplot(gs_outer[0, 7:12])

    # --- Row 2: Panel C1 + C2 + Panel D ---
    gs_c = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[1, 0:5],
        wspace=0.40,
    )
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    ax_d = fig.add_subplot(gs_outer[1, 5:12])

    # Row-1 spacing tweak: shrink panel B by ~5%, anchored to right edge.
    pos_b = ax_b.get_position()
    b_new_w = pos_b.width * 0.95
    ax_b.set_position([pos_b.x1 - b_new_w, pos_b.y0, b_new_w, pos_b.height])

    # Row-2 rebalance: reduce panel D width by ~20%, allocate to panel C.
    pos_d = ax_d.get_position()
    d_new_w = pos_d.width * 0.80
    d_delta = pos_d.width - d_new_w
    d_new_h = pos_d.height * 1.22
    d_new_y0 = pos_d.y1 - d_new_h  # keep top aligned; extend downward to fill gap
    ax_d.set_position([pos_d.x1 - d_new_w, d_new_y0, d_new_w, d_new_h])

    pos_c1 = ax_c1.get_position()
    pos_c2 = ax_c2.get_position()
    c_left = min(pos_c1.x0, pos_c2.x0)
    c_right = max(pos_c1.x1, pos_c2.x1)
    c_old_w = c_right - c_left
    c_new_w = c_old_w + d_delta

    for ax_c in [ax_c1, ax_c2]:
        pos = ax_c.get_position()
        rel_x0 = (pos.x0 - c_left) / max(c_old_w, 1e-12)
        rel_x1 = (pos.x1 - c_left) / max(c_old_w, 1e-12)
        new_x0 = c_left + rel_x0 * c_new_w
        new_x1 = c_left + rel_x1 * c_new_w
        ax_c.set_position([new_x0, pos.y0, new_x1 - new_x0, pos.height])

    # --- Draw panels ---
    print("\n[3/4] Drawing panels...")
    panel_A_zonemap(ax_a_main, ax_a_top, ax_a_right, overview)
    panel_B_raincloud(ax_b, abl_xgb, abl_rf)
    panel_C_correlations(ax_c1, ax_c2, abl_xgb, abl_rf, validation)
    panel_D_flowchart(ax_d)

    # DI colorbar at bottom
    cax = fig.add_axes([0.10, -0.035, 0.35, 0.008])
    sm = ScalarMappable(cmap=DI_CMAP, norm=DI_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Decoupling Index (DI)", fontsize=9, labelpad=3,
                   color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    cbar.ax.xaxis.set_major_locator(mticker.FixedLocator(
        [0.65, 0.80, 0.95, 1.0, 1.05, 1.10]))
    cbar.outline.set_edgecolor(GRID_COLOR)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.text(0.15, 1.65, "Coupled", transform=cbar.ax.transAxes,
                 fontsize=8, ha="center", va="bottom",
                 color=DEEP_GREEN, fontweight="bold")
    cbar.ax.text(0.85, 1.65, "Anti-aligned", transform=cbar.ax.transAxes,
                 fontsize=8, ha="center", va="bottom",
                 color=ANTI_YLGREEN, fontweight="bold")

    # --- Save ---
    print("\n[4/4] Saving...")
    png_path = output_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, transparent=False,
                facecolor="white", edgecolor="white")
    print(f"  PNG: {png_path}")

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, transparent=False,
                facecolor="white", edgecolor="white")
    print(f"  PDF: {pdf_path}")

    # Inventory
    support_dir = png_path.parent / "figure_6"
    support_dir.mkdir(parents=True, exist_ok=True)
    inv = {
        "figure": str(png_path),
        "figure_pdf": str(pdf_path),
        "version": "publication",
        "data_files": {
            "overview": str(outputs_dir / SEC5 / "vad_metric_overview.csv"),
            "ablation_xgb": str(outputs_dir / SEC5 / "source_tables" / "vad_vs_ablation__xgb.csv"),
            "ablation_rf": str(outputs_dir / SEC5 / "source_tables" / "vad_vs_ablation__rf.csv"),
            "validation": str(outputs_dir / SEC5 / "vad_validation_by_model.csv"),
        },
        "zone_distribution": overview["predicted_zone"].value_counts().to_dict(),
        "design_notes": {
            "panel_a": "Gradient zone fills (no model implied) + marginal rug plots (n=14 honest)",
            "panel_b": "Raincloud: half-violin + beeswarm + paired model lines",
            "panel_c": "Two clean subpanels (C1: XGB/PCLA, C2: RF/VSA) with CI bands + glow",
            "panel_d": "Flowchart (boxes + diamonds + curved arrows)",
        },
    }
    inv_path = support_dir / "figure_6_inventory.json"
    inv_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    print(f"  Inventory: {inv_path}")

    plt.close(fig)
    print("\nDone!")
    print("=" * 70)
    return png_path


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 6: VAD Diagnostic Framework")
    parser.add_argument(
        "--outputs-dir", type=str,
        default=r"C:\Users\ms\Desktop\var-pre\outputs",
    )
    parser.add_argument(
        "--output", type=str,
        default=r"C:\Users\ms\Desktop\var-pre\outputs\figures\figure_6.png",
    )
    args = parser.parse_args()
    create_figure(Path(args.outputs_dir), Path(args.output))