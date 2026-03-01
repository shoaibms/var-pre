#!/usr/bin/env python3
r"""
Figure 5: Hidden Biomarkers & Biological Interpretation
========================================================

Panels:
  a  Rank-rank scatter with hexbin density and Q4 overlay (anti-aligned exemplar)
  b  Rank-rank scatter (coupled exemplar, contrast panel)
  c  Stacked proportion bars: Q4 vs rest, ordered by DI
  d  Exemplar Q4 features: gradient-filled eta-squared bars
  e  Gene vs Pathway divergence: dumbbell plot

Colors: drawn from colourlist.py palette.

Data sources:
  - section_1_paradox_discovery/rank_rank_exemplars.csv   -> a, b
  - section_4_consequences/hidden_biomarkers_by_regime.csv -> c
  - section_3_mechanism/variance_decomposition_exemplars.csv -> d
  - section_4_consequences/pathway_convergence_by_regime.csv -> e
  - section_1_paradox_discovery/regime_map.csv             -> DI coloring

Output:
  outputs/figures/figure_5.png
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================================
# Import colourlist.py -- single source of truth for all colours
# =====================================================================
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colourlist import (
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    TEXT_PRIMARY, TEXT_SECONDARY, SPINE_COLOR, GRID_COLOR, BG_WHITE,
    SIGNAL_GREEN, VARIANCE_COLOR,
    Q4_GLOW, Q4_DEEP, Q4_FILL,
    GREY, GREY_LIGHTER, GREY_LIGHT, GREY_PALE,
    DARK_TURQUOISE, SPRING_GREEN, TEAL,
    DS_MARKERS, DS_SHORT, DS_DISPLAY, DS_LABEL_COLORS, FONT,
    bugreen, ylgreen, greens,
)


# =====================================================================
# DESIGN SYSTEM -- sourced from colourlist.py
# =====================================================================

# --- Canvas ---
BG_PANEL      = "#FFFFFF"
BG_CARD       = "#F6F8FA"

# --- Green family (from colourlist palettes) ---
DEEP_GREEN    = bugreen[7]
FOREST_GREEN  = bugreen[6]
MEDIUM_GREEN  = greens[5]
LIGHT_GREEN   = bugreen[4]
PALE_GREEN    = bugreen[3]
MINT          = greens[2]
SOFT_MINT     = greens[1]

# --- Accents ---
TEAL_BRIGHT   = "#00CED1"
BLUE_STEEL    = "#4682B4"
CORNFLOWER    = "#6495ED"
NATURE_GREEN  = "#00A087"
SEA_GREEN     = "#2E8B57"
CADET_BLUE    = "#5F9EA0"

# --- Yellow-green family ---
YLGREEN       = "#94CB64"
YLGREEN_LIGHT = ylgreen[3]
YLGREEN_MID   = ylgreen[4]

# --- Neutrals ---
TEXT_TERTIARY  = "#6E7781"
WHITE_GLOW     = "#FFFFFF"
GREY_MID       = "#6E7681"
LIGHT_GREY     = "#D3D3D3"
TEXT_DARK      = "#1A1A2E"
TEXT_MID       = "#4A4A5A"
TEXT_LIGHT     = "#8A8A9A"

# --- Q4 colours ---
Q4_COLOR_GLOW = Q4_GLOW
Q4_COLOR_FILL = Q4_FILL

# --- Quadrant colours ---
Q1_COLOR = SIGNAL_GREEN
Q2_COLOR = LIGHT_GREEN
Q3_COLOR = GREY_MID
Q4_COLOR = Q4_COLOR_GLOW

# --- DI Colormap ---
DI_CMAP = LinearSegmentedColormap.from_list(
    "di_green",
    [
        (0.00, DEEP_GREEN),
        (0.25, FOREST_GREEN),
        (0.45, LIGHT_GREEN),
        (0.55, PALE_GREEN),
        (0.70, YLGREEN_LIGHT),
        (0.85, YLGREEN),
        (1.00, ANTI_YLGREEN),
    ],
)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)

# Hexbin colormap for density (panels a, b)
HEX_CMAP = LinearSegmentedColormap.from_list(
    "hex_density",
    [BG_PANEL, "#0a2f1a", "#144d2e", FOREST_GREEN, SIGNAL_GREEN, LIGHT_GREEN],
)

# Glow effects (DISABLED for print)
GLOW = []
GLOW_Q4 = []
GLOW_STRONG = []


# =====================================================================
# STYLE
# =====================================================================

def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelweight": "medium",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_SECONDARY,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
    })


# =====================================================================
# HELPERS
# =====================================================================

SEC1 = Path("results/main_results/section_1_paradox_discovery")
SEC3 = Path("results/main_results/section_3_mechanism")
SEC4 = Path("results/main_results/section_4_consequences")


def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def draw_glow_scatter(ax, x, y, color, s=70, marker="o", zorder=5):
    """Scatter point with glow halo."""
    ax.scatter(x, y, color=color, s=s * 2.5, alpha=0.08, marker=marker,
               zorder=zorder - 2, edgecolors="none")
    ax.scatter(x, y, color=color, s=s * 1.5, alpha=0.15, marker=marker,
               zorder=zorder - 1, edgecolors="none")
    ax.scatter(x, y, color=color, s=s, alpha=0.95, marker=marker,
               zorder=zorder, edgecolors=WHITE_GLOW, linewidths=0.4)


def draw_glow_line(ax, x0, y0, x1, y1, color, lw=2.0, alpha=0.9, glow_layers=3):
    """Draw a line with soft glow."""
    for i in range(glow_layers, 0, -1):
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw + i * 1.5,
                alpha=alpha * 0.08 / i, solid_capstyle="round", zorder=1)
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=alpha,
            solid_capstyle="round", zorder=2)


def panel_label(ax, letter, subtitle="", x=-0.02, y=1.08):
    """Panel label with bold letter and italic subtitle."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=16, fontweight="black", color=TEXT_PRIMARY,
            va="bottom", ha="left",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_WHITE)])
    if subtitle:
        ax.text(x + 0.04, y - 0.005, subtitle, transform=ax.transAxes,
                fontsize=9.5, fontweight="medium", color=TEXT_SECONDARY,
                va="bottom", ha="left", style="italic")


def ds_label_color(ds):
    """Dataset-specific label colour."""
    return {
        "mlomics": SIGNAL_GREEN,
        "ibdmdb": TEAL_BRIGHT,
        "ccle": CORNFLOWER,
        "tcga_gbm": PALE_GREEN,
    }.get(ds, TEXT_SECONDARY)


def ds_label_color_v5(ds):
    """Dataset colour coding for panels d and e."""
    return {
        "mlomics": FOREST_GREEN,
        "ibdmdb": TEAL,
        "ccle": BLUE_STEEL,
        "tcga_gbm": CADET_BLUE,
    }.get(ds, TEXT_MID)


# =====================================================================
# DATA LOADING
# =====================================================================

def load_regime_map(base: Path) -> pd.DataFrame:
    p = base / SEC1 / "regime_map.csv"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_csv(p)
    renames = {}
    for old, new in [("consensus_regime", "regime"), ("spearman_rho_consensus", "rho")]:
        if old in df.columns and new not in df.columns:
            renames[old] = new
    if renames:
        df = df.rename(columns=renames)
    # Match Results: prefer uncertainty DI@K (DI_mean at K for prefer_model), then consensus DI.
    di_col = _find_col(df, ["DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean", "DI"])
    if di_col is None:
        raise ValueError(f"No DI column found in regime_map.csv. cols={list(df.columns)}")
    df["DI"] = df[di_col]
    print(f"  regime_map:           {len(df)} views")
    return df


def load_rank_rank(base: Path) -> pd.DataFrame:
    p = base / SEC1 / "rank_rank_exemplars.csv"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_csv(p)
    print(f"  rank_rank_exemplars:  {len(df)} features, cols: {list(df.columns)[:10]}")
    return df


def load_hidden_biomarkers(base: Path) -> pd.DataFrame:
    p = base / SEC4 / "hidden_biomarkers_by_regime.csv"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_csv(p)
    print(f"  hidden_biomarkers:    {len(df)} rows, cols: {list(df.columns)[:10]}")
    return df


def load_decomp_exemplars(base: Path) -> pd.DataFrame:
    p = base / SEC3 / "variance_decomposition_exemplars.csv"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_csv(p)
    print(f"  decomp_exemplars:     {len(df)} features, cols: {list(df.columns)[:10]}")
    return df


def load_pathway_convergence(base: Path) -> pd.DataFrame:
    p = base / SEC4 / "pathway_convergence_by_regime.csv"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_csv(p)
    print(f"  pathway_convergence:  {len(df)} rows, cols: {list(df.columns)[:10]}")
    return df


# =====================================================================
# PANEL A -- Rank-rank with hexbin density + Q4 overlay
# =====================================================================

def _draw_rank_rank(ax, rr: pd.DataFrame, ds: str, vw: str, label: str,
                    regime: pd.DataFrame, hidden: pd.DataFrame = None, hero: bool = True):
    """Rank-rank scatter with hexbin density base and luminous Q4 overlay."""
    sub = rr[(rr["dataset"] == ds) & (rr["view"] == vw)].copy()

    var_col = _find_col(sub, ["var_rank_pct", "var_rank", "v_rank"])
    pred_col = _find_col(sub, ["pred_rank_pct", "pred_rank", "p_consensus_rank_int",
                                "p_consensus_rank_pct"])

    print(f"    {label}: {ds}/{vw}, {len(sub)} features, var={var_col}, pred={pred_col}")

    if len(sub) == 0 or not var_col or not pred_col:
        ax.text(0.5, 0.5, f"No data for {ds}/{vw}\nCols: {list(rr.columns)[:8]}",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color=TEXT_SECONDARY)
        return

    x_raw = pd.to_numeric(sub[var_col], errors="coerce").values
    y_raw = pd.to_numeric(sub[pred_col], errors="coerce").values

    # Normalise to 0-1 if needed
    if np.nanmax(x_raw) > 1.5:
        x = x_raw / max(1.0, np.nanmax(x_raw))
    else:
        x = x_raw
    if np.nanmax(y_raw) > 1.5:
        y = y_raw / max(1.0, np.nanmax(y_raw))
    else:
        y = y_raw

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    thresh = 0.5  # median split

    # --- HEXBIN density base layer (all points) ---
    gridsize = 25 if hero else 18
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=HEX_CMAP, mincnt=1,
                   edgecolors="none", alpha=0.85, zorder=1, linewidths=0.0)
    # Rasterize only the density layer for smaller PDFs (keep text/Q4 vector)
    hb.set_rasterized(True)

    # --- Q4 zone highlight (luminous rectangle) ---
    q4_rect = mpatches.FancyBboxPatch(
        (-0.02, thresh), thresh + 0.02, 1.04 - thresh,
        boxstyle="round,pad=0.01",
        facecolor=Q4_COLOR, alpha=0.04 if hero else 0.03,
        edgecolor=Q4_COLOR, linewidth=1.2 if hero else 0.6,
        linestyle="--", zorder=2)
    ax.add_patch(q4_rect)

    # --- Assign quadrant ---
    q = np.full(len(x), 3, dtype=int)
    q[(x >= thresh) & (y >= thresh)] = 2
    q[(x >= thresh) & (y < thresh)] = 1
    q[(x < thresh) & (y >= thresh)] = 4

    # --- Draw Q4 points (crisp overlay; print-safe) ---
    q4_mask = q == 4
    if q4_mask.sum() > 0:
        xq4, yq4 = x[q4_mask], y[q4_mask]
        ax.scatter(
            xq4, yq4, s=8 if hero else 6.4, alpha=0.9,
            color=Q4_COLOR, edgecolors="white", linewidths=0.25, zorder=5
        )

    # --- Q2 points (aligned; subtle) ---
    q2_mask = q == 2
    if q2_mask.sum() > 0 and hero:
        ax.scatter(x[q2_mask], y[q2_mask], s=3.9, alpha=0.3,
                   color=SIGNAL_GREEN, edgecolors="none", zorder=3)

    # --- Quadrant labels ---
    fs = 10 if hero else 8
    quad_bbox = dict(boxstyle="round,pad=0.15", facecolor=BG_WHITE, edgecolor="none", alpha=0.78)
    ax.text(0.75, 0.97, "Q2\naligned", transform=ax.transAxes, fontsize=fs,
            ha="center", va="top", color=TEXT_PRIMARY, fontweight="bold", alpha=0.9,
            zorder=10, bbox=quad_bbox, clip_on=False)
    ax.text(0.75, 0.05, "Q1\nnoisy var.", transform=ax.transAxes, fontsize=fs - 1,
            ha="center", va="bottom", color=TEXT_PRIMARY, alpha=0.8,
            zorder=10, bbox=quad_bbox, clip_on=False)
    ax.text(0.25, 0.05, "Q3\nbackground", transform=ax.transAxes, fontsize=fs - 1,
            ha="center", va="bottom", color=TEXT_PRIMARY, alpha=0.8,
            zorder=10, bbox=quad_bbox, clip_on=False)
    ax.text(0.25, 0.97, "Q4\nHIDDEN", transform=ax.transAxes, fontsize=fs + 1 if hero else fs,
            ha="center", va="top", color=TEXT_PRIMARY, fontweight="black",
            alpha=0.95, path_effects=GLOW_Q4 if hero else None,
            zorder=10, bbox=quad_bbox, clip_on=False)

    # --- Reference lines ---
    ax.axhline(thresh, color=TEXT_TERTIARY, linewidth=0.6, linestyle=":", alpha=0.35)
    ax.axvline(thresh, color=TEXT_TERTIARY, linewidth=0.6, linestyle=":", alpha=0.35)
    ax.plot([0, 1], [0, 1], color=TEXT_TERTIARY, linewidth=0.5, linestyle="--", alpha=0.2)

    # --- Stats box ---
    counts = {qi: (q == qi).sum() for qi in [1, 2, 3, 4]}
    n_total = len(x)

    # Default: what the scatter shows
    q4_count_disp = int(counts[4])
    q4_pct_disp = (counts[4] / max(1, n_total) * 100.0)

    # Canonical: override from Section 4 table if available (matches Results + panel c)
    if hidden is not None:
        hsub = hidden[(hidden["dataset"] == ds) & (hidden["view"] == vw)]
        if len(hsub) > 0:
            if "Q4_count" in hsub.columns:
                q4_count_disp = int(pd.to_numeric(hsub["Q4_count"].iloc[0], errors="coerce"))
            if "Q4_pct" in hsub.columns:
                q4_pct_disp = float(pd.to_numeric(hsub["Q4_pct"].iloc[0], errors="coerce"))

    reg_row = regime[(regime["dataset"] == ds) & (regime["view"] == vw)]
    di_val = float(reg_row["DI"].iloc[0]) if len(reg_row) > 0 else None
    rho_val = reg_row["rho"].iloc[0] if len(reg_row) > 0 and "rho" in reg_row.columns else None

    info = f"{DS_SHORT.get(ds, ds)}/{vw}"
    if di_val is not None:
        info += f"  |  DI = {di_val:.2f}"
    if rho_val is not None and pd.notna(rho_val):
        info += f",  rho = {rho_val:.2f}"
    info += f"\nQ4 (hidden): {q4_count_disp:,}/{n_total:,} ({q4_pct_disp:.1f}%)"

    box_edge = Q4_COLOR if q4_pct_disp > 15 else GRID_COLOR
    # Keep both panels' stats boxes at right-middle for readability.
    stats_x, stats_y = (0.97, 0.50)
    stats_ha, stats_va = ("right", "center")
    ax.text(stats_x, stats_y, info, transform=ax.transAxes, fontsize=8.5,
            ha=stats_ha, va=stats_va, color=TEXT_PRIMARY,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_CARD,
                      edgecolor=box_edge, alpha=0.92, linewidth=0.8))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Variance rank (percentile)", fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylabel("Importance rank (percentile)", fontsize=10, color=TEXT_PRIMARY)
    ax.tick_params(axis="both", colors=TEXT_PRIMARY)


def panel_A_antialigned(ax, rr: pd.DataFrame, regime: pd.DataFrame, hidden: pd.DataFrame):
    """Anti-aligned exemplar: Q4 features dominate -- the hidden signal glows."""
    views_in_rr = rr.groupby(["dataset", "view"]).size().reset_index(name="n")
    regime_cols = ["dataset", "view", "DI"] + (["regime"] if "regime" in regime.columns else [])
    views_merged = views_in_rr.merge(regime[regime_cols], on=["dataset", "view"], how="left")

    ds, vw = "mlomics", "methylation"
    if len(views_merged) > 0:
        # Prefer regime label, DI as tie-break
        if "regime" in views_merged.columns:
            anti = views_merged[views_merged["regime"].astype(str).str.upper().str.contains("ANTI", na=False)]
        else:
            anti = views_merged.iloc[0:0]

        if len(anti) > 0:
            di_num = pd.to_numeric(anti["DI"], errors="coerce")
            best = anti.loc[di_num.idxmax()] if di_num.notna().any() else anti.iloc[0]
        else:
            di_num = pd.to_numeric(views_merged["DI"], errors="coerce")
            best = views_merged.loc[di_num.idxmax()] if di_num.notna().any() else views_merged.iloc[0]
        ds, vw = best["dataset"], best["view"]

    _draw_rank_rank(ax, rr, ds, vw, "Panel a (anti-aligned)", regime, hidden=hidden, hero=True)
    panel_label(ax, "a")


def panel_B_coupled(ax, rr: pd.DataFrame, regime: pd.DataFrame, hidden: pd.DataFrame):
    """Coupled exemplar: Q1/Q3 dominate, few Q4."""
    views_in_rr = rr.groupby(["dataset", "view"]).size().reset_index(name="n")
    regime_cols = ["dataset", "view", "DI"] + (["regime"] if "regime" in regime.columns else [])
    views_merged = views_in_rr.merge(regime[regime_cols], on=["dataset", "view"], how="left")

    ds, vw = "ibdmdb", "MGX"
    if len(views_merged) > 0:
        # Prefer regime label, DI as tie-break
        if "regime" in views_merged.columns:
            coup = views_merged[views_merged["regime"].astype(str).str.upper().str.contains("COUP", na=False)]
        else:
            coup = views_merged.iloc[0:0]

        if len(coup) > 0:
            di_num = pd.to_numeric(coup["DI"], errors="coerce")
            best = coup.loc[di_num.idxmin()] if di_num.notna().any() else coup.iloc[0]
        else:
            di_num = pd.to_numeric(views_merged["DI"], errors="coerce")
            best = views_merged.loc[di_num.idxmin()] if di_num.notna().any() else views_merged.iloc[0]
        ds, vw = best["dataset"], best["view"]

    _draw_rank_rank(ax, rr, ds, vw, "Panel b (coupled)", regime, hidden=hidden, hero=False)
    panel_label(ax, "b")


# =====================================================================
# PANEL C -- Stacked proportion bars: Q4 composition
# =====================================================================

def panel_C_q4_composition(ax, q4: pd.DataFrame, regime: pd.DataFrame):
    """Q4 features are ~18% of all features -- the hidden signal is substantial."""
    df = q4.copy()

    if "DI" not in df.columns:
        df = df.merge(regime[["dataset", "view", "DI"]], on=["dataset", "view"], how="left")

    q4_col = _find_col(df, ["Q4_pct", "Q4_fraction", "q4_pct", "q4_fraction",
                             "pct_Q4", "frac_Q4"])
    print(f"    Panel c: q4_col={q4_col}, cols={list(df.columns)[:12]}")

    if not q4_col:
        n_q4 = _find_col(df, ["n_Q4", "n_q4", "Q4_count"])
        n_total = _find_col(df, ["n_features", "n_total", "total_features"])
        if n_q4 and n_total:
            df["Q4_pct"] = pd.to_numeric(df[n_q4], errors="coerce") / pd.to_numeric(df[n_total], errors="coerce")
            q4_col = "Q4_pct"
        else:
            ax.text(0.5, 0.5, f"No Q4 column\n{list(df.columns)[:10]}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9, color=TEXT_SECONDARY)
            return

    if "dataset" in df.columns and "view" in df.columns:
        df = df.groupby(["dataset", "view"], as_index=False).first()

    df["q4_val"] = pd.to_numeric(df[q4_col], errors="coerce")
    if df["q4_val"].max() <= 1.0:
        df["q4_frac"] = df["q4_val"]
        df["q4_val"] = df["q4_val"] * 100
    else:
        df["q4_frac"] = df["q4_val"] / 100

    df = df.dropna(subset=["q4_val"]).sort_values("DI", ascending=True).reset_index(drop=True)
    n = len(df)
    y = np.arange(n)

    # --- Full-width stacked bars: rest + glowing Q4 ---
    bar_height = 0.65

    # Rest (non-Q4) -- subtle background
    ax.barh(y, 100, height=bar_height, color=BG_CARD, edgecolor=GRID_COLOR,
            linewidth=0.3, zorder=1)

    # Q4 portion -- LUMINOUS
    for i, (_, row) in enumerate(df.iterrows()):
        q4v = row["q4_val"]
        di_val = row.get("DI", 1.0)
        di_val = di_val if np.isfinite(di_val) else 1.0

        # Gradient fill: brighter Q4 = brighter bar
        intensity = min(q4v / 35, 1.0)
        bar_alpha = 0.5 + 0.5 * intensity

        # Glow under the bar (disabled for print)
        # ax.barh(i, q4v, height=bar_height + 0.15, color=Q4_COLOR,
        #         alpha=0.08, zorder=1.5, edgecolor="none")
        # Actual bar
        ax.barh(i, q4v, height=bar_height, color=Q4_COLOR,
                alpha=bar_alpha, zorder=2, edgecolor=Q4_COLOR, linewidth=0.3)

        # Value label
        if q4v > 8:
            ax.text(q4v - 0.5, i, f"{q4v:.1f}%", va="center", ha="right",
                    fontsize=8.5, color=TEXT_PRIMARY, fontweight="normal", zorder=3)
        else:
            ax.text(q4v + 0.8, i, f"{q4v:.1f}%", va="center", ha="left",
                    fontsize=8, color=TEXT_PRIMARY, fontweight="normal", zorder=3)

    # --- Y-axis labels with dataset colouring ---
    labels = []
    for _, row in df.iterrows():
        labels.append(f"{DS_SHORT.get(row['dataset'], row['dataset'])}/{row['view']}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.get_yticklabels()[i].set_color(TEXT_PRIMARY)

    # --- Mean reference line ---
    mean_q4 = df["q4_val"].mean()
    ax.axvline(mean_q4, color=Q4_COLOR, linewidth=1.2, linestyle="--", alpha=0.4)
    ax.text(mean_q4 + 0.5, n + 0.10, f"Mean: {mean_q4:.1f}%",
            fontsize=8.5, color=TEXT_PRIMARY, fontweight="bold", va="bottom", ha="left",
            path_effects=GLOW_Q4)

    ax.set_xlabel("Feature composition (%)", fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylim(-0.6, n + 0.3)
    ax.set_xlim(0, 105)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", colors=TEXT_PRIMARY)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "c")


# =====================================================================
# PANEL D -- Exemplar Q4 features with eta-squared
# =====================================================================

def panel_D_exemplar_features(ax, decomp: pd.DataFrame, regime: pd.DataFrame):
    """Hidden features are among the most biologically discriminative."""
    df = decomp.copy()

    print(f"    Panel d: decomp cols={list(df.columns)[:15]}")

    eta_col = _find_col(df, ["eta_sq", "eta_squared", "eta2", "R_j", "R_snr",
                              "r_snr", "fisher_ratio", "between_frac"])
    feat_col = _find_col(df, ["feature", "feature_name", "gene", "probe"])
    between_col = _find_col(df, ["between_var", "var_between", "between_frac"])
    within_col = _find_col(df, ["within_var", "var_within", "within_frac"])
    total_col = _find_col(df, ["total_var", "var_total"])
    pred_col = _find_col(df, ["importance", "shap_mean", "p_consensus_score",
                                "p_xgb_bal_score", "pred_rank", "pred_rank_pct"])
    var_col = _find_col(df, ["var_rank", "var_rank_pct", "v_rank", "v_score"])
    quadrant_col = _find_col(df, ["quadrant", "quad", "q"])

    if not feat_col:
        ax.text(0.5, 0.5, f"No feature col\n{list(df.columns)[:12]}",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color=TEXT_MID)
        return

    # Compute eta_sq from between/total if not present
    if not eta_col and between_col and total_col:
        b = pd.to_numeric(df[between_col], errors="coerce")
        t = pd.to_numeric(df[total_col], errors="coerce")
        df["eta_sq_computed"] = b / t.replace(0, np.nan)
        eta_col = "eta_sq_computed"

    # If R_j, convert to eta_sq
    if eta_col and eta_col in ["R_j", "R_snr", "r_snr"]:
        rj_vals = pd.to_numeric(df[eta_col], errors="coerce")
        df["eta_sq_converted"] = rj_vals / (1 + rj_vals)
        eta_col = "eta_sq_converted"

    # Filter to Q4 features if possible
    if quadrant_col:
        q4_df = df[df[quadrant_col].astype(str).str.contains("4|Q4|q4", na=False)].copy()
        if len(q4_df) > 0:
            df = q4_df

    if len(df) > 20 and var_col and pred_col:
        v = pd.to_numeric(df[var_col], errors="coerce")
        p = pd.to_numeric(df[pred_col], errors="coerce")
        if v.max() > 1.5:
            q4_mask = (v > v.median()) & (p < p.median())
        else:
            q4_mask = (v < 0.5) & (p > 0.5)
        q4_features = df[q4_mask]
        if len(q4_features) > 5:
            df = q4_features

    # Sort by eta_sq
    sort_col = eta_col if eta_col else pred_col
    if sort_col:
        df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")
        df = df.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False)

    df = df.head(15).reset_index(drop=True)
    df = df.iloc[::-1].reset_index(drop=True)  # reverse for horizontal bars
    n = len(df)

    if n == 0:
        ax.text(0.5, 0.5, "No Q4 exemplar features",
                transform=ax.transAxes, ha="center", va="center", color=TEXT_MID)
        return

    y_pos = np.arange(n)

    if eta_col:
        vals = pd.to_numeric(df[eta_col], errors="coerce").values

        # --- Gradient-filled bars (colour intensity scales with value) ---
        for i, v in enumerate(vals):
            if np.isfinite(v):
                intensity = min(v / 0.7, 1.0)
                bar_color = to_rgba(Q4_COLOR_FILL, 0.3 + 0.6 * intensity)
                ax.barh(i, v, color=bar_color, height=0.6,
                        edgecolor=Q4_COLOR_FILL, linewidth=0.3, zorder=2)
                # Value annotation
                ax.text(v + 0.01, i, f"{v:.2f}", va="center", ha="left",
                        fontsize=7.5, color=TEXT_PRIMARY, fontweight="medium")

        ax.set_xlabel("Signal fraction (η²)", fontsize=9.5, color=TEXT_PRIMARY)
    elif pred_col:
        vals = pd.to_numeric(df[pred_col], errors="coerce").values
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.barh(i, v, color=Q4_COLOR_FILL, height=0.6,
                        alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("SHAP importance score", fontsize=9.5, color=TEXT_PRIMARY)

    # --- Feature labels ---
    labels = []
    for _, row in df.iterrows():
        feat = str(row[feat_col])[:22]
        # Keep labels compact to avoid cross-panel overlap with panel c.
        labels.append(feat)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.0)
    for lab in ax.get_yticklabels():
        lab.set_clip_on(True)

    # --- Highlight known biomarkers ---
    known_markers = {"MIA", "CHI3L1", "DNM3", "YKL-40"}
    for i, (_, row) in enumerate(df.iterrows()):
        feat = str(row[feat_col])
        if any(m.lower() in feat.lower() for m in known_markers):
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color(TEXT_PRIMARY)
            # Star annotation
            if eta_col:
                v = pd.to_numeric(row.get(eta_col), errors="coerce")
                if pd.notna(v):
                    ax.scatter(v, i, marker="*", s=80, color=YLGREEN,
                               edgecolors=Q4_COLOR_FILL, linewidths=0.5, zorder=5)

    ax.set_ylim(-0.6, n + 0.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", colors=TEXT_PRIMARY)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.3, color=GRID_COLOR)
    panel_label(ax, "d")


# =====================================================================
# PANEL E -- Gene vs Pathway divergence: dumbbell plot
# =====================================================================

def panel_E_divergence(ax, pathways: pd.DataFrame, regime: pd.DataFrame):
    """Different genes but partially overlapping pathways."""
    df = pathways.copy()

    if "DI" not in df.columns:
        df = df.merge(regime[["dataset", "view", "DI"]], on=["dataset", "view"], how="left")

    gene_col = _find_col(df, ["gene_jaccard", "jaccard_gene", "gene_overlap",
                                "jaccard_genes", "gene_level_jaccard"])
    path_col = _find_col(df, ["pathway_jaccard", "jaccard_pathway", "pathway_overlap",
                                "jaccard_pathways", "pathway_level_jaccard"])
    cr_col = _find_col(df, ["convergence_ratio", "CR", "cr"])

    print(f"    Panel e: gene={gene_col}, pathway={path_col}, cr={cr_col}")

    if not gene_col and not path_col:
        ax.text(0.5, 0.5, f"No Jaccard cols\n{list(df.columns)[:12]}",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color=TEXT_MID)
        return

    if "dataset" in df.columns and "view" in df.columns:
        df = df.groupby(["dataset", "view"], as_index=False).first()

    # Canonicalise IDs (do NOT trust pre-existing view_id formatting)
    df["view_id"] = df["dataset"].astype(str) + ":" + df["view"].astype(str)
    reg = regime.copy()
    reg["view_id"] = reg["dataset"].astype(str) + ":" + reg["view"].astype(str)

    # Prefer anti-aligned definition from the pathways table itself if available,
    # otherwise fall back to regime_map. Fail open if it collapses to <2 rows.
    df_anti = None
    if "consensus_regime" in df.columns:
        m = df["consensus_regime"].astype(str).str.upper().str.contains("ANTI", na=False)
        df_anti = df[m].copy()
    elif "regime" in reg.columns:
        anti_ids = set(reg.loc[reg["regime"].astype(str).str.upper().str.contains("ANTI", na=False), "view_id"])
        df_anti = df[df["view_id"].isin(anti_ids)].copy()

    if df_anti is not None and len(df_anti) >= 2:
        df = df_anti
    else:
        print(f"    Panel e: anti-aligned filter yielded n={0 if df_anti is None else len(df_anti)}; plotting all valid views instead.")

    df = df.sort_values("DI", ascending=True).reset_index(drop=True)

    # Keep only rows with actual Jaccard values
    if gene_col:
        df[gene_col] = pd.to_numeric(df[gene_col], errors="coerce")
    if path_col:
        df[path_col] = pd.to_numeric(df[path_col], errors="coerce")
    need = [c for c in [gene_col, path_col] if c]
    df = df.dropna(subset=need).copy()

    if len(df) == 0:
        ax.text(0.5, 0.5, "No valid gene/pathway Jaccard rows after filtering",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color=TEXT_MID)
        return

    n = len(df)
    y = np.arange(n)

    labels = [f"{DS_SHORT.get(row['dataset'], row['dataset'])}/{row['view']}"
              for _, row in df.iterrows()]

    if gene_col and path_col:
        g_vals = pd.to_numeric(df[gene_col], errors="coerce").values * 100
        p_vals = pd.to_numeric(df[path_col], errors="coerce").values * 100

        # --- Dumbbell: thick connector + styled dots ---
        for i in range(n):
            gv, pv = g_vals[i], p_vals[i]
            if np.isfinite(gv) and np.isfinite(pv):
                # Connector line (thicker, with gradient-like styling)
                ax.plot([gv, pv], [i, i], color=PALE_GREEN, linewidth=3.0,
                        solid_capstyle="round", alpha=0.5, zorder=1)
                ax.plot([gv, pv], [i, i], color=LIGHT_GREEN, linewidth=1.2,
                        solid_capstyle="round", alpha=0.6, zorder=2)

            # Gene dot (circle, DI-coloured)
            di_val = df["DI"].values[i] if "DI" in df.columns else 1.0
            if np.isfinite(gv):
                ax.scatter(gv, i, color=DI_CMAP(DI_NORM(di_val)) if np.isfinite(di_val) else GREY,
                           s=55, zorder=4, edgecolors="white", linewidths=0.8,
                           marker="o")
            # Pathway dot (diamond, teal)
            if np.isfinite(pv):
                ax.scatter(pv, i, color=TEAL, s=60, zorder=4,
                           edgecolors="white", linewidths=0.8, marker="D")

            # CR annotation
            if cr_col and np.isfinite(gv) and np.isfinite(pv):
                cr = pd.to_numeric(df.iloc[i].get(cr_col), errors="coerce")
                if pd.notna(cr) and cr > 0:
                    x_pos = max(gv, pv) + 1.0
                    cr_color = TEXT_PRIMARY
                    ax.text(x_pos, i, f"{cr:.1f}x",
                            fontsize=7, va="center", ha="left",
                            color=cr_color, fontweight="bold")

    elif gene_col:
        g_vals = pd.to_numeric(df[gene_col], errors="coerce").values * 100
        for i in range(n):
            if np.isfinite(g_vals[i]):
                ax.scatter(g_vals[i], i, color=FOREST_GREEN, s=50, zorder=3,
                           edgecolors="white", linewidths=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.get_yticklabels()[i].set_color(TEXT_PRIMARY)

    # --- Summary stats box ---
    summary_parts = []
    if gene_col:
        g_mean = pd.to_numeric(df[gene_col], errors="coerce").mean() * 100
        summary_parts.append(f"Gene Jaccard: {g_mean:.1f}%")
    if path_col:
        p_mean = pd.to_numeric(df[path_col], errors="coerce").mean() * 100
        summary_parts.append(f"Pathway Jaccard: {p_mean:.1f}%")
    if cr_col and cr_col in df.columns:
        cr_mean = pd.to_numeric(df[cr_col], errors="coerce").mean()
        if pd.notna(cr_mean):
            summary_parts.append(f"Convergence: {cr_mean:.1f}x")

    if summary_parts:
        ax.text(0.97, 0.97, "\n".join(summary_parts),
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                color=TEXT_PRIMARY,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_WHITE,
                          edgecolor=SPINE_COLOR, alpha=0.92, linewidth=0.6))

    # --- Legend ---
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=FOREST_GREEN,
                   markeredgecolor="white", markersize=7, label="Gene overlap"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=TEAL,
                   markeredgecolor="white", markersize=7, label="Pathway overlap"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=8)
    for txt in leg.get_texts():
        txt.set_color(TEXT_PRIMARY)

    ax.set_xlabel("Jaccard overlap (%)", fontsize=9.5, color=TEXT_PRIMARY, labelpad=2)
    ax.set_ylim(-0.6, n + 0.3)

    # --- Tighten x-axis to the observed Jaccard range (still anchored at 0) ---
    x_arrays = []
    if gene_col:
        x_arrays.append(pd.to_numeric(df[gene_col], errors="coerce").values * 100.0)
    if path_col:
        x_arrays.append(pd.to_numeric(df[path_col], errors="coerce").values * 100.0)

    if x_arrays:
        x_max = np.nanmax(np.concatenate(x_arrays))
        pad = 3.0  # a little breathing room so rightmost labels/markers don't touch the frame
        x_hi = np.ceil((x_max + pad) / 10.0) * 10.0   # round up to a clean 10% tick
        x_hi = float(min(100.0, max(30.0, x_hi)))     # cap at 100; avoid overly-tight axes
    else:
        x_hi = 100.0

    ax.set_xlim(0, x_hi)
    ax.set_xticks(np.arange(0, x_hi + 0.1, 10.0))
    ax.tick_params(axis="x", colors=TEXT_PRIMARY)
    ax.grid(axis="x", alpha=0.3, color=GRID_COLOR)
    panel_label(ax, "e")


# =====================================================================
# MAIN ASSEMBLY
# =====================================================================

def create_figure(outputs_dir: Path, output_path: Path):
    apply_style()

    print("=" * 70)
    print("FIGURE 5: Hidden Biomarkers & Biological Interpretation")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    regime = load_regime_map(outputs_dir)
    rr = load_rank_rank(outputs_dir)
    q4 = load_hidden_biomarkers(outputs_dir)
    decomp = load_decomp_exemplars(outputs_dir)
    pathways = load_pathway_convergence(outputs_dir)

    print("\n[2/5] Building asymmetric layout...")
    fig = plt.figure(figsize=(10, 9.5))
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    gs = GridSpec(
        3, 12, figure=fig,
        height_ratios=[1.3, 1.1, 0.70],
        hspace=0.38, wspace=0.8,
        left=0.06, right=0.97, top=0.96, bottom=0.06,
    )

    # Row 1: a (HERO, 7 cols) + b (5 cols)
    ax_a = fig.add_subplot(gs[0, 0:7])
    ax_b = fig.add_subplot(gs[0, 7:12])

    # Reduce panel b width by ~5% without redistributing: keep right edge fixed,
    # shrink from the left to open spacing between panels a and b.
    pos_b = ax_b.get_position()
    new_w_b = pos_b.width * 0.95
    new_x0_b = pos_b.x0 + (pos_b.width - new_w_b)
    ax_b.set_position([new_x0_b, pos_b.y0, new_w_b, pos_b.height])

    # Row 2: c (5 cols) + d (7 cols)
    ax_c = fig.add_subplot(gs[1, 0:5])
    ax_d = fig.add_subplot(gs[1, 5:12])

    # Reduce panel widths without redistributing panel positions.
    pos_c = ax_c.get_position()
    new_w_c = pos_c.width * 0.85
    ax_c.set_position([pos_c.x0, pos_c.y0, new_w_c, pos_c.height])
    pos_d = ax_d.get_position()
    new_w_d = pos_d.width * 0.95
    ax_d.set_position([pos_d.x0, pos_d.y0, new_w_d, pos_d.height])

    # Row 3: e (full width)
    ax_e = fig.add_subplot(gs[2, :])

    print("\n[3/5] Drawing panels...")
    print("  Panel a: Anti-aligned rank-rank scatter (hexbin)...")
    panel_A_antialigned(ax_a, rr, regime, q4)

    print("  Panel b: Coupled rank-rank scatter...")
    panel_B_coupled(ax_b, rr, regime, q4)

    print("  Panel c: Q4 composition (stacked bars)...")
    panel_C_q4_composition(ax_c, q4, regime)

    print("  Panel d: Exemplar Q4 features (gradient eta-sq)...")
    panel_D_exemplar_features(ax_d, decomp, regime)

    print("  Panel e: Gene vs Pathway divergence (dumbbell)...")
    panel_E_divergence(ax_e, pathways, regime)

    # --- Shared DI colorbar ---
    cax = fig.add_axes([0.30, 0.006, 0.40, 0.010])
    sm = ScalarMappable(cmap=DI_CMAP, norm=DI_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Decoupling Index (DI)", fontsize=9, labelpad=3, color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    cbar.ax.xaxis.set_major_locator(mticker.FixedLocator([0.65, 0.80, 0.95, 1.0, 1.05, 1.10]))
    cbar.outline.set_edgecolor(GRID_COLOR)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.text(0.15, 1.9, "Coupled", transform=cbar.ax.transAxes,
                 fontsize=8, ha="center", color=DEEP_GREEN, fontweight="bold")
    cbar.ax.text(0.85, 1.9, "Anti-aligned", transform=cbar.ax.transAxes,
                 fontsize=8, ha="center", color=YLGREEN, fontweight="bold")

    # --- Save ---
    print("\n[4/5] Saving...")
    png_path = output_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=300,
        transparent=False,
        facecolor="white",
        edgecolor="white",
    )
    print(f"  Saved: {png_path}")

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        transparent=False,
        facecolor="white",
        edgecolor="white",
    )
    print(f"  Saved: {pdf_path}")

    # Supporting inventory
    support_dir = png_path.parent / "figure_5"
    support_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "figure": str(png_path),
        "figure_pdf": str(pdf_path),
        "design": "Five-panel layout: hexbin rank-rank, Q4 composition, eta-sq bars, dumbbell",
        "data_sources": {
            "rank_rank_exemplars": str(outputs_dir / SEC1 / "rank_rank_exemplars.csv"),
            "hidden_biomarkers": str(outputs_dir / SEC4 / "hidden_biomarkers_by_regime.csv"),
            "decomp_exemplars": str(outputs_dir / SEC3 / "variance_decomposition_exemplars.csv"),
            "pathway_convergence": str(outputs_dir / SEC4 / "pathway_convergence_by_regime.csv"),
            "regime_map": str(outputs_dir / SEC1 / "regime_map.csv"),
        },
        "panel_mapping": {
            "a": "Rank-rank scatter (anti-aligned, hexbin + Q4 overlay)",
            "b": "Rank-rank scatter (coupled, hexbin)",
            "c": "Q4 composition (stacked bars)",
            "d": "Exemplar Q4 features (gradient eta-sq bars)",
            "e": "Gene vs Pathway Jaccard divergence (dumbbell)",
        },
        "color_source": "colourlist.py",
    }
    inv_path = support_dir / "figure_5_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"  Inventory: {inv_path}")

    plt.close(fig)
    print("\n[5/5] Done!")
    print("=" * 70)
    return png_path


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 5: Hidden Biomarkers & Biological Interpretation")
    parser.add_argument(
        "--outputs-dir", type=str,
        default=r"C:\Users\ms\Desktop\var-pre\outputs",
    )
    parser.add_argument(
        "--output", type=str,
        default=r"C:\Users\ms\Desktop\var-pre\outputs\figures\figure_5.png",
    )
    args = parser.parse_args()
    create_figure(Path(args.outputs_dir), Path(args.output))