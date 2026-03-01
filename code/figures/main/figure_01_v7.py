#!/usr/bin/env python3
r"""
Figure 1: The Variance-Prediction Paradox
==========================================
Composite figure with five panels:

  a  Gradient-filled diverging waterfall (all 14 views)
  b  Hex-density contour rank-rank (two contrasting views)
  c  Cleveland dot plot (within-dataset DI spread)
  d  DI(K) gradient ribbon trajectories with K=10 pct marker
  e  DI vs rho scatter with marginal KDE and regression CI

Data sources:
  - results/main_results/section_1_paradox_discovery/regime_map.csv
  - results/main_results/section_1_paradox_discovery/di_scores_all_views.csv
  - results/main_results/section_1_paradox_discovery/rank_rank_exemplars.csv
  - results/main_results/section_1_paradox_discovery/source_tables/joined_vp/

Output:
  figures/figure_1.png  |  figures/figure_1.pdf

Usage:
  python figure_01_v7.py --outputs-dir "C:\Users\ms\Desktop\var-pre\outputs"
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================================
# IMPORT COLOURS FROM colourlist.py
# =====================================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colourlist import (
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    TEXT_PRIMARY, TEXT_SECONDARY, SPINE_COLOR, GRID_COLOR, BG_WHITE,
    GREY, GREY_LIGHTER, GREY_PALE, GREY_LIGHT,
    TEAL, NATURE_GREEN, DARK_TURQUOISE, SPRING_GREEN,
    REGIME_COUPLED, REGIME_MIXED, REGIME_ANTI_ALIGNED,
    DS_MARKERS, DS_SHORT, DS_DISPLAY, DS_LABEL_COLORS, FONT,
    bugreen, ylgreen, greens,
)

# =====================================================================
# DESIGN SYSTEM
# =====================================================================

C_TEXT     = TEXT_PRIMARY
C_SPINE    = SPINE_COLOR
C_GRID     = GRID_COLOR
C_OVERLAP  = "#B2DFDB"

DARK_SEAGREEN = TEXT_SECONDARY

# DI colormap (standard across all figures)
DI_CMAP = LinearSegmentedColormap.from_list("di_green", [
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN,
    NEUTRAL_GREEN, LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
], N=256)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)

# Hex density colormaps (Panel B)
HEX_CMAP_COUPLED = LinearSegmentedColormap.from_list(
    "hex_coupled", ["#ffffff", "#B2DFDB", "#48D1CC", "#00CED1", "#008080", COUPLED_GREEN], N=256)
HEX_CMAP_ANTI = LinearSegmentedColormap.from_list(
    "hex_anti", ["#ffffff", "#F0F4C3", "#CDDC39", "#9ACD32", "#7CB342", "#3d5a00"], N=256)


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
        "axes.edgecolor": C_SPINE,
        "axes.labelcolor": C_TEXT,
        "text.color": C_TEXT,
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "xtick.color": C_SPINE,
        "ytick.color": C_SPINE,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": FONT["legend"],
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": C_GRID,
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

SEC1 = Path("results/main_results/section_1_paradox_discovery")


def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def panel_label(ax, letter, x=-0.08, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=FONT["panel"], fontweight="bold", color=C_TEXT,
            va="bottom", ha="left")


def subtle_grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=C_GRID, linewidth=0.4, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# =====================================================================
# DATA LOADING -- v6 style (panels A, C, E)
# =====================================================================

def load_regime_map(base: Path) -> pd.DataFrame:
    """Load regime_map.csv for v6-style panels (A, C, E)."""
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
    # Prefer uncertainty DI@K (matches Results), then consensus. Keep original cols; just create alias "DI".
    di_col = _find_col(df, ["DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean", "DI"])
    if di_col is None:
        raise ValueError(f"No DI column found in regime_map.csv. cols={list(df.columns)}")
    df["DI"] = df[di_col]
    print(f"  regime_map: {len(df)} views, cols: {list(df.columns)[:12]}")
    return df


def load_di_scores(base: Path) -> pd.DataFrame:
    """Load DI scores across K values."""
    p = base / SEC1 / "di_scores_all_views.csv"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_csv(p)
    print(f"  di_scores:  {len(df)} rows")
    return df


def load_rank_rank(base: Path) -> pd.DataFrame:
    p = base / SEC1 / "rank_rank_exemplars.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    print(f"  rank_rank:  {len(df)} rows")
    return df


# =====================================================================
# DATA LOADING -- v4 style (panels B, D)
# =====================================================================

def load_regime_consensus(base: Path) -> pd.DataFrame:
    """Load regime_consensus.csv for v4-style panels (B, D)."""
    path = (
        base
        / "results"
        / "main_results"
        / "section_1_paradox_discovery"
        / "source_tables"
        / "regime_consensus.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"regime_consensus.csv not found: {path}")
    df = pd.read_csv(path)
    print(f"  regime_consensus: {len(df)} views, cols={list(df.columns)[:8]}...")
    return df


def load_vp_joined(base: Path, dataset: str, view: str) -> pd.DataFrame:
    """Load per-view joined VP data for hex-density panel B."""
    path = (
        base
        / "results"
        / "main_results"
        / "section_1_paradox_discovery"
        / "source_tables"
        / "joined_vp"
        / f"vp_joined__{dataset}__{view}.csv.gz"
    )
    if not path.exists():
        raise FileNotFoundError(f"vp_joined not found: {path}")
    df = pd.read_csv(path)
    print(f"  vp_joined {dataset}:{view}: {len(df)} features")
    return df


# =====================================================================
# PANEL A (v6) -- Gradient-filled diverging waterfall
# =====================================================================

def panel_A_waterfall(ax, regime: pd.DataFrame):
    """Diverging waterfall bars from DI=1, with layered gradient fill."""
    df = regime.sort_values("DI", ascending=True).reset_index(drop=True)
    n = len(df)

    # Extend RHS so the ρ column never overlaps DI value text.
    # (Some views can have DI very close to the previous hard max=1.14.)
    x_min = 0.42
    x_max = max(1.22, float(df["DI"].max()) + 0.20)

    # Zone shading (blue-green for coupled, yellow-green for anti-aligned)
    ax.axvspan(0.42, 1.0, alpha=0.12, color=DARK_TURQUOISE, zorder=0)
    ax.axvspan(1.0, x_max, alpha=0.12, color=SPRING_GREEN, zorder=0)

    # Reference line
    ax.axvline(1.0, color=GREY, linewidth=1.4, linestyle="-", alpha=0.5, zorder=1)

    for i, (_, row) in enumerate(df.iterrows()):
        di = row["DI"]
        color = DI_CMAP(DI_NORM(di))
        bar_width = di - 1.0

        # 3-layer gradient bar
        for layer, (w_scale, alpha) in enumerate([(1.0, 0.18), (0.75, 0.38), (0.5, 0.88)]):
            ax.barh(i, bar_width, left=1.0, color=color,
                    height=0.65 * w_scale, alpha=alpha,
                    edgecolor="none", zorder=2 + layer)

        # End cap dot
        ax.scatter(di, i, color=color, s=55, zorder=6,
                   edgecolors="white", linewidths=0.6)

        # DI value
        offset = 0.008 if di >= 1.0 else -0.008
        ha = "left" if di >= 1.0 else "right"
        ax.text(di + offset, i, f"{di:.2f}", va="center", ha=ha,
                fontsize=FONT["annot"], color="black", fontweight="bold")

    # Y labels
    labels = [f"{DS_SHORT.get(row['dataset'], row['dataset'])}/{row['view']}"
              for _, row in df.iterrows()]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=FONT["tick"])

    # Rho annotations
    for i, (_, row) in enumerate(df.iterrows()):
        rho_val = row.get("rho", np.nan)
        if pd.notna(rho_val):
            rho_color = COUPLED_GREEN if rho_val > 0.5 else (ANTI_YLGREEN if rho_val < 0 else GREY)
            ax.text(x_max - 0.01, i, rf"$\rho$ = {rho_val:+.2f}", va="center", ha="right",
                    fontsize=7.5, color=rho_color, fontweight="medium")

    # Zone labels
    ax.text(0.72, n + 0.1, "Variance aids selection",
            fontsize=9, color=MID_BLUEGREEN, fontweight="medium",
            style="italic", va="bottom", ha="center")
    ax.text(1.06, n + 0.1, "Variance\nharms",
            fontsize=9, color=MID_YLGREEN, fontweight="medium",
            style="italic", va="bottom", ha="center")
    # Place below axis in Axes coords to avoid colliding with x tick labels.
    ax.text(0.5, -0.18, "DI = 1.0 (random overlap)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT["annot"], color=GREY, style="italic",
            clip_on=False)

    ax.set_xlabel("Decoupling Index (DI) at K = 10%", fontsize=FONT["label"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.8, n + 0.8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", pad=2)
    subtle_grid(ax, axis="x")
    panel_label(ax, "a")


# =====================================================================
# PANEL B (v4) -- Hex-density contour rank-rank
# =====================================================================

def _hex_contour_subplot(ax, df, dataset, view, di_val, rho_val, cmap, k_pct=10, box_side="right"):
    """Render one hex-density + KDE contour rank-rank plot (from v4)."""
    from scipy import stats
    from scipy.ndimage import gaussian_filter

    n = len(df)
    k_n = max(1, int(n * k_pct / 100))

    v_col = _find_col(df, ["v_rank_pct", "var_rank_pct"])
    p_col = _find_col(df, ["p_consensus_rank_pct", "pred_rank_pct"])

    if v_col and p_col:
        x = df[v_col].values * 100
        y_raw = df[p_col].values
        y = y_raw * 100 if y_raw.max() <= 1.5 else y_raw
    else:
        v_rank_col = _find_col(df, ["v_rank", "var_rank"])
        p_rank_col = _find_col(df, ["p_consensus_rank_int", "pred_rank"])
        x = df[v_rank_col].values / n * 100
        y = df[p_rank_col].values / n * 100

    print(f"    Panel b ({dataset}:{view}): n={n}, x_range=[{x.min():.1f},{x.max():.1f}], "
          f"y_range=[{y.min():.1f},{y.max():.1f}]")

    hb = ax.hexbin(x, y, gridsize=35, cmap=cmap, mincnt=1,
                   linewidths=0.1, edgecolors="white", alpha=0.85, zorder=1)

    try:
        idx = np.random.default_rng(42).choice(len(x), min(3000, len(x)), replace=False)
        xs, ys = x[idx], y[idx]
        kde = stats.gaussian_kde(np.vstack([xs, ys]), bw_method=0.15)
        xg = np.linspace(0, 100, 80)
        yg = np.linspace(0, 100, 80)
        Xg, Yg = np.meshgrid(xg, yg)
        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        Z = gaussian_filter(Z, sigma=1.5)
        ax.contour(Xg, Yg, Z, levels=5, colors=DARK_SEAGREEN,
                   linewidths=0.7, alpha=0.5, zorder=2)
    except Exception as e:
        print(f"      KDE contour failed: {e}")

    ax.axhline(y=k_pct, color=DARK_SEAGREEN, linewidth=1.0, alpha=0.4, linestyle="-", zorder=3)
    ax.axvline(x=k_pct, color=DARK_SEAGREEN, linewidth=1.0, alpha=0.4, linestyle="-", zorder=3)
    ax.plot([0, 100], [0, 100], color=GREY, linestyle=":", linewidth=1.0, alpha=0.4, zorder=3)

    title_color = COUPLED_GREEN if di_val < 1.0 else ANTI_YLGREEN
    ax.set_title(f"{DS_DISPLAY.get(dataset, dataset)}:{view}",
                 fontsize=9, fontweight="bold", color=title_color, pad=4)

    box_text = f"DI = {di_val:.2f}\n$\\rho$ = {rho_val:.2f}\nn = {n:,}"
    box_side = "left" if str(box_side).lower().startswith("l") else "right"
    bx = 0.03 if box_side == "left" else 0.97
    bha = "left" if box_side == "left" else "right"
    ax.text(bx, 0.97, box_text, transform=ax.transAxes, fontsize=7,
            ha=bha, va="top", color=DARK_SEAGREEN,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY_LIGHTER, alpha=0.9))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.spines["left"].set_color(C_SPINE)
    ax.spines["bottom"].set_color(C_SPINE)


def panel_B_hexdensity(fig, gs_slot, outputs_dir, regime_v4):
    """Two stacked hex-density rank-rank plots (from v4)."""
    # Ensure the two stacked axes don't collide (titles/ticks) while keeping
    # enough height for equal-aspect plots.
    inner = gs_slot.subgridspec(2, 1, hspace=0.38)
    ax_top = fig.add_subplot(inner[0])
    ax_bot = fig.add_subplot(inner[1])

    # Differentiate background tint per subplot (bluish-green vs yellowish-green).
    ax_top.set_facecolor(matplotlib.colors.to_rgba(DARK_TURQUOISE, 0.10))
    ax_bot.set_facecolor(matplotlib.colors.to_rgba(SPRING_GREEN, 0.10))

    # regime_v4 is regime_map-derived; "DI" is the uncertainty DI alias after load_regime_map().
    di_col  = _find_col(regime_v4, ["DI", "DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus"])
    rho_col = _find_col(regime_v4, ["rho", "spearman_rho_consensus"])
    if di_col is None or rho_col is None:
        raise ValueError(f"Missing DI/rho columns. cols={list(regime_v4.columns)}")

    df_mgx = load_vp_joined(outputs_dir, "ibdmdb", "MGX")
    mgx_row = regime_v4[(regime_v4["dataset"] == "ibdmdb") & (regime_v4["view"] == "MGX")]
    if len(mgx_row) == 0:
        raise ValueError("Missing regime row for ibdmdb/MGX (check MGX vs MGX_func naming).")

    di_mgx = float(mgx_row[di_col].iloc[0])
    rho_mgx = float(mgx_row[rho_col].iloc[0])
    _hex_contour_subplot(ax_top, df_mgx, "ibdmdb", "MGX", di_mgx, rho_mgx, HEX_CMAP_COUPLED, box_side="left")
    ax_top.set_xlabel("")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel("Importance rank (percentile)", fontsize=8)

    df_meth = load_vp_joined(outputs_dir, "mlomics", "methylation")
    meth_row = regime_v4[(regime_v4["dataset"] == "mlomics") & (regime_v4["view"] == "methylation")]
    if len(meth_row) == 0:
        raise ValueError("Missing regime row for mlomics/methylation.")
    di_meth = float(meth_row[di_col].iloc[0])
    rho_meth = float(meth_row[rho_col].iloc[0])
    _hex_contour_subplot(ax_bot, df_meth, "mlomics", "methylation", di_meth, rho_meth, HEX_CMAP_ANTI, box_side="left")
    ax_bot.set_xlabel("Variance rank (percentile)", fontsize=8)
    ax_bot.set_ylabel("Importance rank (percentile)", fontsize=8)

    panel_label(ax_top, "b", x=-0.2, y=1.1)


# =====================================================================
# PANEL C (v6) -- Cleveland dot plot: within-dataset DI spread
# =====================================================================

def panel_C_cleveland(ax, regime: pd.DataFrame):
    datasets = sorted(regime["dataset"].unique())
    y_pos = 0

    for ds in datasets:
        sub = regime[regime["dataset"] == ds].sort_values("DI")
        n_views = len(sub)

        # Dataset band (green)
        ax.axhspan(y_pos - 0.4, y_pos + n_views - 0.6,
                   alpha=0.09, color=LIGHT_BLUEGREEN, zorder=0)

        for j, (_, row) in enumerate(sub.iterrows()):
            yj = y_pos + j
            di = row["DI"]
            color = DI_CMAP(DI_NORM(di))
            marker = DS_MARKERS.get(ds, "o")

            ax.plot([1.0, di], [yj, yj], color=color, linewidth=1.2,
                    alpha=0.3, solid_capstyle="round", zorder=1)
            ax.scatter(di, yj, color=color, marker=marker, s=55,
                       zorder=5, edgecolors=C_SPINE, linewidths=0.4)

            label_x = di - 0.008 if di < 1.0 else di + 0.008
            label_ha = "right" if di < 1.0 else "left"
            ax.text(label_x, yj, row["view"], fontsize=7, ha=label_ha,
                    va="center", color=GREY, fontweight="bold")

        y_center = y_pos + (n_views - 1) / 2
        ax.text(0.56, y_center, DS_DISPLAY.get(ds, ds),
                ha="right", va="center", fontsize=FONT["tick"],
                fontweight="normal", color=GREY)

        di_min, di_max = sub["DI"].min(), sub["DI"].max()
        span = di_max - di_min
        if span > 0.04 and n_views > 1:
            ax.annotate("", xy=(di_max, y_pos + n_views - 1), xytext=(di_min, y_pos),
                        arrowprops=dict(arrowstyle="|-|", color=GREY, lw=0.6,
                                        shrinkA=6, shrinkB=6))
            ax.text((di_min + di_max) / 2, y_center - 0.55,
                    f"\u0394 = {span:.2f}", fontsize=7, ha="center", va="top",
                    color=GREY, style="italic")

        y_pos += n_views + 1.0

    ax.axvline(1.0, color=GREY, linewidth=1.0, linestyle="-", alpha=0.4, zorder=1)
    ax.text(1.002, y_pos - 0.5, "DI = 1", fontsize=7.5, color=GREY,
            style="italic", va="top")

    ax.set_xlim(0.58, 1.14)
    ax.set_ylim(-0.8, y_pos)
    ax.set_xlabel("DI at K = 10%", fontsize=FONT["label"], fontweight="normal")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    subtle_grid(ax, axis="x")
    panel_label(ax, "c")


# =====================================================================
# PANEL D (v4 ribbons + v6 K=10% marker) -- DI(K) trajectories
# =====================================================================

def panel_D_ribbons_with_marker(ax, di_curves: pd.DataFrame, regime_v4: pd.DataFrame):
    """
    DI trajectories across K = 1, 5, 10, 20% (from v4 ribbon style).
    Each view: gradient-filled ribbon showing CI band.
    Hero views emphasized (thicker, labeled).
    PLUS: K=10% vertical marker line from v6.
    """
    di_col = _find_col(di_curves, ["DI_mean"])
    lo_col = _find_col(di_curves, ["DI_pctl_2_5", "DI_pctl_2.5"])
    hi_col = _find_col(di_curves, ["DI_pctl_97_5", "DI_pctl_97.5"])
    k_col = _find_col(di_curves, ["k_pct"])
    model_col = _find_col(di_curves, ["model"])
    print(f"    Panel d: di={di_col}, lo={lo_col}, hi={hi_col}, k={k_col}, model={model_col}")

    df = di_curves.copy()
    if model_col and "xgb_bal" in df[model_col].unique():
        df = df[df[model_col] == "xgb_bal"]

    # Zone bands
    ax.axhspan(0.55, 0.95, color=DARK_TURQUOISE, alpha=0.10, zorder=0)
    ax.axhspan(1.005, 1.10, color=SPRING_GREEN, alpha=0.10, zorder=0)
    ax.axhline(y=1.0, color=GREY, linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)
    # Lift the label so it doesn't collide with line/labels near DI=1.
    ax.text(19.2, 1.04, "DI = 1", fontsize=7, color=GREY,
            va="bottom", style="italic", zorder=6)

    # *** K=10% vertical marker from v6 ***
    ax.axvline(10, color=MID_YLGREEN, linewidth=0.6, linestyle=":", alpha=0.5)
    ax.text(10, 0.47, "K = 10%", fontsize=7, color=GREY,
            ha="center", va="bottom", style="italic")

    # Get DI consensus for coloring
    di_consensus_col = _find_col(regime_v4, ["DI", "DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus"])
    if di_consensus_col is None:
        raise ValueError(f"No DI column for coloring. cols={list(regime_v4.columns)}")
    reg_di = dict(zip(
        regime_v4["dataset"] + ":" + regime_v4["view"],
        regime_v4[di_consensus_col]
    ))

    heroes = {"mlomics:methylation", "ibdmdb:MPX", "ccle:mRNA", "ibdmdb:MGX"}

    views = df.groupby(["dataset", "view"])
    for (dataset, view), grp in views:
        grp = grp.sort_values(k_col)
        k_vals = grp[k_col].values
        di_vals = grp[di_col].values
        key = f"{dataset}:{view}"
        di_10 = reg_di.get(key, 1.0)
        color = DI_CMAP(DI_NORM(di_10))
        is_hero = key in heroes

        if lo_col and hi_col:
            lo_vals = grp[lo_col].values
            hi_vals = grp[hi_col].values
            ax.fill_between(k_vals, lo_vals, hi_vals,
                           color=color, alpha=0.12 if is_hero else 0.06, zorder=2)

        lw = 2.5 if is_hero else 1.0
        alpha = 0.9 if is_hero else 0.35
        ax.plot(k_vals, di_vals, color=color, linewidth=lw, alpha=alpha,
                zorder=3 if is_hero else 2, solid_capstyle="round")

        if is_hero:
            marker = DS_MARKERS.get(dataset, "o")
            ax.scatter(k_vals[-1], di_vals[-1], marker=marker, s=60,
                       c=[color], edgecolors="white", linewidths=0.8, zorder=5)
            offset_y = 0.01 if di_vals[-1] < 1.0 else -0.015
            ax.text(k_vals[-1] + 0.8, di_vals[-1] + offset_y, view,
                    fontsize=7, color=color, fontweight="bold", va="center")

    ax.set_xlabel("K (% features selected)", fontsize=FONT["label"])
    ax.set_ylabel("Decoupling Index (DI)", fontsize=FONT["label"])
    ax.set_xlim(0, 22)
    ax.set_ylim(0.45, 1.10)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.spines["left"].set_color(C_SPINE)
    ax.spines["bottom"].set_color(C_SPINE)
    subtle_grid(ax, axis="both")
    panel_label(ax, "d")


# =====================================================================
# PANEL E (v6) -- DI vs rho scatter with marginals + regression CI
# =====================================================================

def panel_E_scatter_marginals(ax, regime: pd.DataFrame):
    rho_col = _find_col(regime, ["rho", "spearman_rho_consensus"])
    if not rho_col:
        ax.text(0.5, 0.5, "\u03C1 data not available",
                transform=ax.transAxes, ha="center", va="center")
        return

    x = regime[rho_col].astype(float).values
    y_vals = regime["DI"].astype(float).values
    valid = np.isfinite(x) & np.isfinite(y_vals)
    x, y_vals = x[valid], y_vals[valid]

    if valid.sum() == len(regime):
        datasets = regime["dataset"].values
        views = regime["view"].values
    else:
        datasets = regime.loc[valid, "dataset"].values
        views = regime.loc[valid, "view"].values

    ax.axhline(1.0, color=GREY, linewidth=0.8, linestyle="-", alpha=0.4)
    ax.axvline(0.0, color=GREY, linewidth=0.6, linestyle=":", alpha=0.3)
    ax.axhspan(1.0, 1.15, alpha=0.10, color=SPRING_GREEN, zorder=0)

    # Regression + bootstrap CI
    if len(x) >= 5:
        z = np.polyfit(x, y_vals, 1)
        xfit = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
        yfit = np.polyval(z, xfit)

        rng = np.random.default_rng(42)
        boot_fits = np.zeros((500, len(xfit)))
        for b in range(500):
            idx = rng.choice(len(x), size=len(x), replace=True)
            zb = np.polyfit(x[idx], y_vals[idx], 1)
            boot_fits[b] = np.polyval(zb, xfit)
        ci_lo = np.percentile(boot_fits, 2.5, axis=0)
        ci_hi = np.percentile(boot_fits, 97.5, axis=0)

        ax.fill_between(xfit, ci_lo, ci_hi, color=LIGHT_BLUEGREEN, alpha=0.15, zorder=1)
        ax.plot(xfit, yfit, color=MID_BLUEGREEN, linewidth=1.2,
                linestyle="--", alpha=0.5, zorder=2)

    for ds in np.unique(datasets):
        mask = datasets == ds
        marker = DS_MARKERS.get(ds, "o")
        colors = [DI_CMAP(DI_NORM(yi)) for yi in y_vals[mask]]
        ax.scatter(x[mask], y_vals[mask], c=colors, marker=marker, s=70,
                   edgecolors=C_SPINE, linewidths=0.45, zorder=4, alpha=0.9,
                   label=DS_DISPLAY.get(ds, ds))

    for xi, yi, ds, vw in zip(x, y_vals, datasets, views):
        if yi > 1.01 or yi < 0.72 or xi > 0.85 or xi < -0.15:
            color = DI_CMAP(DI_NORM(yi))
            ax.annotate(vw, (xi, yi), fontsize=7, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points",
                        color=GREY, alpha=0.85)

    if len(x) >= 3:
        try:
            from scipy.stats import pearsonr
            r, p = pearsonr(x, y_vals)
            sig = " *" if p < 0.05 else ""
            ax.text(0.03, 0.04, f"r = {r:.2f}{sig}\nn = {len(x)}",
                    transform=ax.transAxes, fontsize=FONT["tick"], ha="left", va="bottom",
                    color=GREY,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor=C_SPINE, alpha=0.92, linewidth=0.6))
        except ImportError:
            pass

    legend_greens = [COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, DARK_TURQUOISE]
    handles = []
    for i, (ds, m) in enumerate(DS_MARKERS.items()):
        if ds in np.unique(datasets):
            lc = legend_greens[i] if i < len(legend_greens) else DARK_SEAGREEN
            handles.append(
                plt.Line2D([0], [0], marker=m, color="none",
                           markerfacecolor=lc, markeredgecolor=C_SPINE, markeredgewidth=0.45,
                           markersize=7, label=DS_DISPLAY.get(ds, ds))
            )
    ax.legend(handles=handles, loc="upper right", fontsize=FONT["annot"],
              handletextpad=0.3, borderpad=0.5, labelspacing=0.4)

    ax.set_xlabel(f"Spearman \u03C1 (variance rank vs. SHAP importance rank)", fontsize=FONT["label"])
    ax.set_ylabel("DI at K = 10%", fontsize=FONT["label"])
    ax.set_xlim(-0.45, 1.05)
    ax.set_ylim(0.55, 1.12)
    subtle_grid(ax, axis="both")

    ax.text(-0.03, 1.08, "e", transform=ax.transAxes,
            fontsize=FONT["panel"], fontweight="black", color=C_TEXT)

    # Marginals
    divider = make_axes_locatable(ax)

    ax_top = divider.append_axes("top", size="15%", pad=0.06)
    _marginal_kde(ax_top, x, axis="x", xlim=(-0.45, 1.05))
    ax_top.tick_params(labelbottom=False, labelleft=False, length=0)
    for sp in ax_top.spines.values():
        sp.set_visible(False)

    ax_right = divider.append_axes("right", size="15%", pad=0.06)
    _marginal_kde(ax_right, y_vals, axis="y", xlim=(0.55, 1.12))
    ax_right.axhline(1.0, color=GREY, linewidth=0.5, linestyle="-", alpha=0.3)
    ax_right.tick_params(labelbottom=False, labelleft=False, length=0)
    for sp in ax_right.spines.values():
        sp.set_visible(False)


def _marginal_kde(ax, values, axis="x", xlim=None, n_pts=100):
    """Filled KDE marginal using colourlist.py greens."""
    valid = values[np.isfinite(values)]
    if len(valid) < 3:
        return

    ax.set_facecolor("white")
    grid = np.linspace(xlim[0], xlim[1], n_pts) if xlim else \
           np.linspace(valid.min() - 0.1, valid.max() + 0.1, n_pts)

    bw = 0.15 * (valid.max() - valid.min()) / max(len(valid) ** 0.2, 1)
    if bw <= 0:
        bw = 0.05
    density = np.zeros(n_pts)
    for v in valid:
        density += np.exp(-0.5 * ((grid - v) / bw) ** 2)
    density /= (bw * np.sqrt(2 * np.pi) * len(valid))

    if axis == "x":
        ax.fill_between(grid, 0, density, color=C_OVERLAP, alpha=0.35)
        ax.plot(grid, density, color=LIGHT_BLUEGREEN, linewidth=1.0, alpha=0.7)
        if xlim: ax.set_xlim(xlim)
        ax.set_ylim(0, None)
    else:
        ax.fill_betweenx(grid, 0, density, color=C_OVERLAP, alpha=0.35)
        ax.plot(density, grid, color=LIGHT_BLUEGREEN, linewidth=1.0, alpha=0.7)
        if xlim: ax.set_ylim(xlim)
        ax.set_xlim(0, None)


# =====================================================================
# MAIN ASSEMBLY
# =====================================================================

def create_figure(outputs_dir: Path, output_path: Path):
    apply_style()

    print("=" * 70)
    print("FIGURE 1: The Variance-Prediction Paradox")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1/5] Loading data...")

    # v6-style data (panels A, C, E)
    regime_v6 = load_regime_map(outputs_dir)
    di_scores = load_di_scores(outputs_dir)

    # Panels B/D must use the same DI definition as Results (uncertainty DI@K from regime_map).
    regime_v4 = regime_v6.copy()

    print(f"\n  v6 regime: {len(regime_v6)} views, DI range: "
          f"{regime_v6['DI'].min():.2f} -- {regime_v6['DI'].max():.2f}")
    if "rho" in regime_v6.columns:
        rho = regime_v6["rho"].dropna()
        print(f"  rho range: {rho.min():.2f} -- {rho.max():.2f}")

    # ---- Build layout ----
    print("\n[2/5] Building layout...")
    fig = plt.figure(figsize=(10, 9))

    gs = GridSpec(3, 12, figure=fig,
                  height_ratios=[1.35, 1.0, 0.75],
                  hspace=0.38, wspace=0.6,
                  left=0.07, right=0.96, top=0.97, bottom=0.09)

    ax_a = fig.add_subplot(gs[0, 0:9])        # Panel A: v6 waterfall (1 col wider → uses gap)
    gs_b = gs[0, 9:12]                         # Panel B: v4 hex-density (shifted right by 1 col)
    ax_c = fig.add_subplot(gs[1, 0:5])         # Panel C: v6 Cleveland
    # Leave a 1-column spacer between C and D to prevent overlap.
    ax_d = fig.add_subplot(gs[1, 6:12])        # Panel D: v4 ribbons + v6 marker
    ax_e = fig.add_subplot(gs[2, 0:12])        # Panel E: v6 scatter (full width ~20% wider)

    # ---- Draw panels ----
    print("\n[3/5] Drawing panels...")

    print("  Panel a (v6): Gradient waterfall...")
    panel_A_waterfall(ax_a, regime_v6)

    print("  Panel b (v4): Hex-density rank-rank...")
    panel_B_hexdensity(fig, gs_b, outputs_dir, regime_v4)

    print("  Panel c (v6): Cleveland dot plot...")
    panel_C_cleveland(ax_c, regime_v6)

    print("  Panel d (v4+v6): DI(K) ribbon trajectories + K=10% marker...")
    panel_D_ribbons_with_marker(ax_d, di_scores, regime_v4)

    print("  Panel e (v6): DI vs rho scatter + marginals...")
    panel_E_scatter_marginals(ax_e, regime_v6)

    # ---- Colorbar ----
    print("\n[4/5] Adding colorbar...")
    cax = fig.add_axes([0.30, 0.018, 0.40, 0.008])
    sm = ScalarMappable(cmap=DI_CMAP, norm=DI_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Decoupling Index (DI)", fontsize=FONT["tick"], labelpad=3, color=C_TEXT)
    cbar.ax.tick_params(labelsize=FONT["annot"], colors=C_SPINE)
    cbar.ax.xaxis.set_major_locator(mticker.FixedLocator([0.65, 0.80, 0.95, 1.0, 1.05, 1.10]))
    cbar.outline.set_edgecolor(C_SPINE)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.text(-0.05, -2.8, "Coupled", transform=cbar.ax.transAxes,
                 fontsize=FONT["tick"], ha="center", color=COUPLED_GREEN, fontweight="bold")
    cbar.ax.text(0.88, -2.8, "Anti-aligned", transform=cbar.ax.transAxes,
                 fontsize=FONT["tick"], ha="center", color=MID_YLGREEN, fontweight="bold")

    # ---- Save ----
    print("\n[5/5] Saving...")
    png_path = output_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, facecolor="white", edgecolor="none")
    print(f"  Saved: {png_path}")

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, facecolor="white", edgecolor="none")
    print(f"  Saved: {pdf_path}")

    # Inventory
    support_dir = png_path.parent / "figure_1"
    support_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "figure": str(png_path),
        "figure_pdf": str(pdf_path),
        "design": "Composite layout, colourlist.py palette",
        "data_sources": {
            "regime_map_v6": str(outputs_dir / SEC1 / "regime_map.csv"),
            "regime_consensus_v4": str(outputs_dir / "04_importance" / "aggregated" / "regime_consensus.csv"),
            "di_scores_all_views": str(outputs_dir / SEC1 / "di_scores_all_views.csv"),
            "vp_joined_ibdmdb_MGX": str(outputs_dir / "04_importance" / "joined_vp" / "vp_joined__ibdmdb__MGX.csv.gz"),
            "vp_joined_mlomics_methylation": str(outputs_dir / "04_importance" / "joined_vp" / "vp_joined__mlomics__methylation.csv.gz"),
        },
        "panel_mapping": {
            "a": "Gradient-filled diverging waterfall with rho annotations",
            "b": "Hex-density contour rank-rank (IBDMDB:MGX vs MLOmics:methylation)",
            "c": "Cleveland dot plot with dataset grouping bands",
            "d": "DI(K) ribbon trajectories with K=10 pct vertical marker",
            "e": "DI vs rho scatter with bootstrap CI and KDE marginals",
        },
    }
    inv_path = support_dir / "figure_1_composite_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"  Inventory: {inv_path}")

    plt.close(fig)
    print("\nDone!")
    print("=" * 70)
    return png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 1: The Variance-Prediction Paradox")
    parser.add_argument("--outputs-dir", type=str, default=r"C:\Users\ms\Desktop\var-pre\outputs")
    parser.add_argument("--output", type=str, default=r"C:\Users\ms\Desktop\var-pre\outputs\figures\figure_1.png")
    args = parser.parse_args()
    create_figure(Path(args.outputs_dir), Path(args.output))