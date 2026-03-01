#!/usr/bin/env python3
r"""
Figure 4: Downstream Consequences of Variance Filtering
========================================================

Panels:
  a = Strategy comparison strip at K=10%
  b = Delta-vs-K trajectories (pp scale, view labels)
  c = DI vs Delta scatter
  d = Unsupervised clustering ΔARI (TopVar − Random) at K=10%
  e = Regime-stratified Δ(TopVar − Random) bars at K=10%

Data sources:
  section_1_paradox_discovery/regime_map.csv
  section_4_consequences/ablation_by_regime.csv
  section_4_consequences/ablation_across_k.csv
  section_4_consequences/di_vs_delta_scatter_k10.csv
  section_4_consequences/unsupervised_clustering_table.csv
  04_importance/aggregated/regime_consensus.csv

Usage:
  python figure_04_v4.py --outputs-dir "C:\Users\ms\Desktop\var-pre\outputs"
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =====================================================================
# IMPORT COLOURS FROM colourlist.py
# =====================================================================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colourlist import (
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    TEXT_PRIMARY, TEXT_SECONDARY, SPINE_COLOR, GRID_COLOR, BG_WHITE,
    STRATEGY_TOPVAR, STRATEGY_RANDOM, STRATEGY_TOPSHAP, STRATEGY_ALL,
    GREY, GREY_LIGHTER, GREY_LIGHT, GREY_PALE,
    DS_MARKERS, DS_SHORT, DS_DISPLAY, FONT,
)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# =====================================================================
# DESIGN SYSTEM
# =====================================================================

DARK_SEAGREEN   = TEXT_SECONDARY
FOREST_GREEN    = TEXT_PRIMARY
TEAL_GREEN      = "#00A087"
LIGHT_GREY      = GREY_LIGHTER
PALE_GREY       = GREY_PALE
Q4_COLOR        = "#3d5a00"

COLOR_TOPVAR  = STRATEGY_TOPVAR
COLOR_RANDOM  = STRATEGY_RANDOM
COLOR_TOPSHAP = STRATEGY_TOPSHAP
COLOR_ALL     = STRATEGY_ALL

DI_CMAP = LinearSegmentedColormap.from_list("di_green", [
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN,
    NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
], N=256)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)


# =====================================================================
# STYLE
# =====================================================================

def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT["base"],
        "axes.titlesize": FONT["title"],
        "axes.titleweight": "bold",
        "axes.labelsize": FONT["label"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"],
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "hatch.linewidth": 1.2,
    })


# =====================================================================
# UTILITY
# =====================================================================

def _find_col(df, candidates):
    """Find first matching column from candidates list."""
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower().replace(" ", "_") == c.lower().replace(" ", "_"):
                return col
    return None


def _get_di(row, di_map, fallback=1.0):
    """Get DI value for a row."""
    vid = row.get("view_id", "")
    if not vid and "dataset" in row and "view" in row:
        vid = f"{row['dataset']}:{row['view']}"
    return di_map.get(vid, fallback)


def _panel_label(ax, letter, subtitle="", x=-0.06, y=1.08, color=FOREST_GREEN):
    """Add bold panel letter only (no subplot subtitle text)."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=FONT["panel"], fontweight="bold", va="bottom", ha="left",
            color=color)


def _spine_style(ax, color=DARK_SEAGREEN):
    """Style spines consistently."""
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(color)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=DARK_SEAGREEN, width=0.6)


def _get_di_map(regime_df: pd.DataFrame) -> Dict[str, float]:
    """Return dict of view_id -> DI value."""
    di_col = _find_col(regime_df, ["DI", "DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean"])
    if di_col is None:
        return {}
    return dict(zip(regime_df['view_id'], regime_df[di_col]))


# =====================================================================
# DATA LOADING
# =====================================================================

SEC1 = Path("results/main_results/section_1_paradox_discovery")
SEC4 = Path("results/main_results/section_4_consequences")


def load_regime_map(base: Path) -> pd.DataFrame:
    p = base / SEC1 / "regime_map.csv"
    df = pd.read_csv(p)
    di_col = _find_col(df, ["DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean", "DI"])
    if di_col is None:
        raise ValueError(f"No DI column found in regime_map.csv. cols={list(df.columns)}")
    df["DI"] = df[di_col]
    # Canonical view_id used across all results tables
    df["view_id"] = df["dataset"].astype(str) + ":" + df["view"].astype(str)

    print(f"  regime_map:           {len(df)} views")
    return df


def load_regime_consensus(base: Path) -> pd.DataFrame:
    """Load regime consensus (Phase 4 aggregated). Used by panel b."""
    p = base / "04_importance" / "aggregated" / "regime_consensus.csv"
    df = pd.read_csv(p)
    print(f"  regime_consensus:     {len(df)} rows")
    return df


def load_ablation_by_regime(base: Path) -> pd.DataFrame:
    p = base / SEC4 / "ablation_by_regime.csv"
    df = pd.read_csv(p)
    print(f"  ablation_by_regime:   {len(df)} rows")
    return df


def load_ablation_across_k(base: Path) -> pd.DataFrame:
    p = base / SEC4 / "ablation_across_k.csv"
    df = pd.read_csv(p)
    print(f"  ablation_across_k:    {len(df)} rows, K={sorted(df['k_pct'].unique())}")
    return df


def load_di_vs_delta(base: Path) -> pd.DataFrame:
    p = base / SEC4 / "di_vs_delta_scatter_k10.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required file: {p}\n"
            "Re-run Section 4 results compilation to generate di_vs_delta_scatter_k10.csv "
            "(per view × model)."
        )
    df = pd.read_csv(p)
    print(f"  di_vs_delta:          {len(df)} rows")
    return df


def load_unsupervised_clustering_table(base: Path) -> pd.DataFrame:
    p = base / SEC4 / "unsupervised_clustering_table.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required file: {p}\n"
            "Run section_4_result.py to generate unsupervised_clustering_table.csv."
        )
    df = pd.read_csv(p)
    print(f"  unsupervised_table:   {len(df)} rows")
    return df


# =====================================================================
# PANEL A -- HERO: Strategy comparison at K=10%
# =====================================================================

def panel_A_strategy(ax, abl: pd.DataFrame, regime: pd.DataFrame):
    """
    Connected strip plot: TopVar / Random / TopSHAP.
    Each view = one line connecting its performance across three strategies.
    Lines colored by DI gradient. Shows variance filtering is a coin flip
    while importance-based selection wins consistently.
    """
    # Build DI map
    di_map = {}
    for _, r in regime.iterrows():
        vid = r.get("view_id", f"{r['dataset']}:{r['view']}")
        di_val = r.get("DI", r.get("DI_10pct_consensus", r.get("DI_mean", 1.0)))
        di_map[vid] = di_val

    # Filter to balanced accuracy, XGB
    df = abl[abl["metric"] == "balanced_accuracy"].copy()
    if "model" in df.columns:
        xgb = df[df["model"] == "xgb_bal"]
        df = xgb if len(xgb) > 0 else df.drop_duplicates(subset="view_id", keep="first")
    else:
        df = df.drop_duplicates(subset="view_id", keep="first")

    if "view_id" not in df.columns:
        df["view_id"] = df["dataset"] + ":" + df["view"]
    df["di"] = df["view_id"].map(di_map)
    df = df.sort_values("di", ascending=True).reset_index(drop=True)

    strategies = ["topvar", "random", "topshap"]
    strat_labels = ["TopVar\n(variance)", "Random\n(baseline)", "TopSHAP\n(importance)"]
    strat_x = [0, 1, 2]

    # Check available columns
    available = [s for s in strategies if s in df.columns]
    if len(available) < 2:
        ax.text(0.5, 0.5, "Ablation data\nnot available", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color=GREY)
        _panel_label(ax, "a", "Strategy comparison at K = 10%")
        return

    # --- Background zones ---
    ax.axhspan(0.5, 0.65, color="#E0F7FA", alpha=0.5, zorder=0)
    ax.axhspan(0.65, 0.8, color="#00CED1", alpha=0.10, zorder=0)
    ax.axhspan(0.8, 1.0, color="#B2DFDB", alpha=0.3, zorder=0)

    # --- Connecting lines (each view) ---
    for _, row in df.iterrows():
        di = row["di"] if pd.notna(row.get("di")) else 1.0
        color = DI_CMAP(DI_NORM(np.clip(di, 0.6, 1.1)))
        alpha = 0.6 if 0.95 <= di <= 1.05 else 0.85

        vals = [row.get(s, np.nan) for s in strategies]
        valid_x = [strat_x[i] for i, v in enumerate(vals) if pd.notna(v)]
        valid_y = [v for v in vals if pd.notna(v)]

        if len(valid_x) >= 2:
            ax.plot(valid_x, valid_y, color=color, alpha=alpha * 0.5,
                    linewidth=0.8, zorder=2)

    # --- Scatter dots (with jitter for overlap) ---
    for si, s in enumerate(strategies):
        if s not in df.columns:
            continue
        vals = df[s].values
        dis = df["di"].values

        # Compute jitter to reduce overlap
        jitters = np.zeros(len(vals))
        sorted_idx = np.argsort(vals)
        for rank, idx in enumerate(sorted_idx):
            nearby = [j for j in sorted_idx[:rank]
                      if abs(vals[j] - vals[idx]) < 0.012]
            if nearby:
                n = len(nearby)
                direction = 1 if n % 2 == 0 else -1
                jitters[idx] = direction * min(((n + 1) // 2) * 0.06, 0.22)

        for i in range(len(vals)):
            if pd.isna(vals[i]):
                continue
            di = dis[i] if pd.notna(dis[i]) else 1.0
            color = DI_CMAP(DI_NORM(np.clip(di, 0.6, 1.1)))
            ds = df.iloc[i].get("dataset", "")
            marker = DS_MARKERS.get(ds, "o")
            ax.scatter(strat_x[si] + jitters[i], vals[i],
                       c=[color], s=50, marker=marker,
                       edgecolors="white", linewidth=0.5, zorder=4, alpha=0.9)

    # --- Strategy means ---
    for si, s in enumerate(strategies):
        if s not in df.columns:
            continue
        mean_val = df[s].dropna().mean()
        ax.plot([strat_x[si] - 0.3, strat_x[si] + 0.3], [mean_val, mean_val],
                color=MID_BLUEGREEN, linewidth=2.5, zorder=5, solid_capstyle="round")
        ax.text(strat_x[si], mean_val + 0.015, f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=FONT["annot"],
                fontweight="bold", color=MID_BLUEGREEN, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.15))

    # --- Annotations ---
    if "topvar" in df.columns and "topshap" in df.columns:
        gap = df["topshap"].dropna().mean() - df["topvar"].dropna().mean()
        sign = "+" if gap > 0 else ""
        ax.annotate("", xy=(2, df["topshap"].mean()),
                     xytext=(0, df["topvar"].mean()),
                     arrowprops=dict(arrowstyle="->", color=COUPLED_GREEN,
                                     lw=1.5, connectionstyle="arc3,rad=0.15"))

    # Dataset marker legend
    legend_handles = [
        plt.Line2D([0], [0], marker=m, color="none", markerfacecolor=DARK_SEAGREEN,
                    markeredgecolor="white", markersize=6,
                    label=DS_DISPLAY.get(d, d))
        for d, m in DS_MARKERS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              handletextpad=0.3, borderpad=0.4, labelspacing=0.3)

    ax.set_xticks(strat_x)
    ax.set_xticklabels(strat_labels, fontsize=FONT["tick"])
    ax.set_ylabel("Balanced accuracy", fontsize=FONT["label"])
    ax.set_xlim(-0.5, 2.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#B2DFDB", alpha=0.5, linewidth=0.5)
    _spine_style(ax)
    _panel_label(ax, "a", "Strategy comparison at K = 10%")


# =====================================================================
# PANEL B -- Delta-vs-K Trajectories
# =====================================================================

def panel_B_delta_k(ax, across_k_df: pd.DataFrame, regime_df: pd.DataFrame):
    """Spaghetti plot: Delta(Var-Random) across K for each view.
    Uses regime_consensus for DI mapping. Deltas in percentage points.
    View labels at line endpoints."""
    print("    Panel b: Delta-vs-K trajectories")

    di_col = _find_col(regime_df, ["DI", "DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean"])
    if di_col is None:
        raise ValueError(f"No DI column for panel B. cols={list(regime_df.columns)}")
    di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df[di_col]
    ))

    views = across_k_df['view_id'].unique()

    for vid in views:
        sub = across_k_df[across_k_df['view_id'] == vid].sort_values('k_pct')
        di = di_map.get(vid, 1.0)
        color = DI_CMAP(DI_NORM(di))
        ds = vid.split(':')[0]
        marker = DS_MARKERS.get(ds, 'o')

        k_vals = sub['k_pct'].values
        deltas = sub['delta_var_minus_random'].values * 100  # pp

        ax.plot(k_vals, deltas, color=color, linewidth=1.8, alpha=0.5, zorder=2)
        ax.scatter(k_vals, deltas, marker=marker, s=30, c=[color],
                   edgecolors='white', linewidths=0.5, zorder=3)

        if len(deltas) > 0:
            parts = vid.split(':')
            label = f"{DS_SHORT.get(parts[0], parts[0])}:{parts[1]}"
            ax.text(k_vals[-1] + 0.5, deltas[-1], label, fontsize=5.5,
                    va='center', ha='left', color=color, alpha=0.8)

    ax.axhline(0, color=GREY, linewidth=1, alpha=0.4, linestyle='--')
    ax.fill_between([0, 25], 0, -25, color="#00FF7F", alpha=0.10, zorder=0)
    ax.text(1.5, -1.5, 'TopVar hurts', fontsize=7, color=Q4_COLOR, alpha=0.8, style='italic')
    ax.fill_between([0, 25], 0, 15, color="#00CED1", alpha=0.12, zorder=0)
    ax.text(1.5, 1.5, 'TopVar helps', fontsize=7, color=COUPLED_GREEN, alpha=0.8, style='italic')

    ax.set_xlabel('K (% features selected)', fontsize=FONT['label'])
    ax.set_ylabel('$\\Delta$(TopVar $-$ Random) (pp)', fontsize=FONT['label'])
    ax.set_xticks([1, 5, 10, 20])
    _spine_style(ax)
    _panel_label(ax, "b", "Harm amplifies at tight K")


# =====================================================================
# PANEL C -- DI vs Delta scatter
# =====================================================================

def panel_C_di_scatter(ax, di_delta: pd.DataFrame, regime: pd.DataFrame):
    """
    Scatter of DI vs Delta(TopVar-Random) at K=10%.
    Continuous DI gradient coloring. Regression line overlay.
    Shows DI predicts harm magnitude.
    """
    df_all = di_delta.copy()

    def _spearman_for(model_name: str, di_col: str, delta_col: str):
        sub = df_all[df_all["model"].astype(str) == model_name].copy()
        xx = pd.to_numeric(sub[di_col], errors="coerce")
        yy = pd.to_numeric(sub[delta_col], errors="coerce")
        ok = np.isfinite(xx) & np.isfinite(yy)
        from scipy import stats as sp_stats
        rho, p = sp_stats.spearmanr(xx[ok], yy[ok])
        return int(ok.sum()), float(rho), float(p)

    # Plot XGB points, but annotate XGB + RF correlations (as reported in Results).
    df = df_all.copy()
    if "model" in df.columns:
        df = df[df["model"].astype(str) == "xgb_bal"].copy()

    di_col = _find_col(df, [
        "DI_10pct_model",              # <-- MUST be first to match Results XGB/RF correlations
        "DI_10pct_consensus",
        "DI_mean",
        "DI",
        "DI_10pct_consensus_mean",
        "DI_10pct_uncertainty_xgb_bal",
    ])

    # Prefer per-view delta (not averaged) to match reported ρ.
    delta_col = _find_col(df, [
        "delta_var_minus_random",
        "delta_var_minus_random_mean",
        "delta_J_10pct_consensus",
        "delta_J_10pct_consensus_mean",
    ])

    if di_col is None or delta_col is None:
        ax.text(0.5, 0.5, "DI vs Delta data\nnot available", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color=GREY)
        _panel_label(ax, "c", "DI predicts harm magnitude")
        return

    x = pd.to_numeric(df[di_col], errors="coerce")
    y = pd.to_numeric(df[delta_col], errors="coerce") * 100.0  # fraction -> pp
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid].values, y[valid].values

    if len(x) < 3:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color=GREY)
        _panel_label(ax, "c", "DI predicts harm magnitude")
        return

    # Reference lines
    ax.axhline(0, color=GREY, linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)
    ax.axvline(1.0, color=GREY, linewidth=0.7, linestyle=":", alpha=0.4, zorder=1)

    # Background quadrant shading
    xlim = (min(x.min() - 0.02, 0.6), max(x.max() + 0.02, 1.12))
    ylim = (min(y.min() - 2, -20), max(y.max() + 2, 10))

    # Regression line
    coef = np.polyfit(x, y, 1)
    x_fit = np.linspace(xlim[0], xlim[1], 100)
    y_fit = np.polyval(coef, x_fit)
    ax.plot(x_fit, y_fit, color=MID_BLUEGREEN, linewidth=1.8, alpha=0.7,
            linestyle="-", zorder=3)

    # Confidence band via bootstrap
    from scipy import stats as sp_stats
    r, p = sp_stats.pearsonr(x, y)
    rho_s, p_s = sp_stats.spearmanr(x, y)

    # Scatter points
    colors = [DI_CMAP(DI_NORM(np.clip(xi, 0.6, 1.1))) for xi in x]

    # Dataset markers
    if "dataset" in df.columns:
        datasets = df.loc[valid.values if hasattr(valid, 'values') else valid, "dataset"].values
        for ds_name, marker in DS_MARKERS.items():
            mask = datasets == ds_name
            if mask.any():
                ax.scatter(x[mask], y[mask],
                           c=[colors[i] for i in range(len(mask)) if mask[i]],
                           s=65, marker=marker, edgecolors="white", linewidth=0.6,
                           zorder=5, alpha=0.9)
    else:
        ax.scatter(x, y, c=colors, s=65, edgecolors="white", linewidth=0.6,
                   zorder=5, alpha=0.9)

    # Stats annotation: XGB (plotted) + RF (computed from df_all)
    n_xgb, rho_xgb, p_xgb = len(x), float(rho_s), float(p_s)

    txt = f"XGB: Spearman $\\rho$ = {rho_xgb:.2f}\n(n={n_xgb}, p={p_xgb:.2f})"
    if "model" in df_all.columns:
        n_rf, rho_rf, p_rf = _spearman_for("rf", di_col, delta_col)
        txt += f"\nRF:  Spearman $\\rho$ = {rho_rf:.2f}\n(n={n_rf}, p={p_rf:.2f})"

    ax.text(0.05, 0.05, txt,
            transform=ax.transAxes, fontsize=FONT["annot"],
            va="bottom", ha="left", color=FOREST_GREEN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#B2DFDB", alpha=0.9))

    ax.set_xlabel("Decoupling Index (DI)", fontsize=FONT["label"])
    ax.set_ylabel("$\\Delta$(TopVar $-$ Random)\nat K = 10% (pp)", fontsize=FONT["label"])
    ax.grid(color="#B2DFDB", alpha=0.4, linewidth=0.5)
    _spine_style(ax)
    _panel_label(ax, "c", "DI predicts harm magnitude")


# =====================================================================
# PANEL D -- Unsupervised clustering harm (ΔARI)
# =====================================================================

def panel_D_unsupervised_delta_ari(ax, unsup: pd.DataFrame, regime: pd.DataFrame):
    """
    Horizontal lollipop: ΔARI(TopVar − Random) per view (K=10%),
    ordered by DI. Negative values indicate variance pre-filtering
    degrades unsupervised recovery vs random.
    """
    df = unsup.copy()
    if "view_id" not in df.columns:
        if "dataset" in df.columns and "view" in df.columns:
            df["view_id"] = df["dataset"].astype(str) + ":" + df["view"].astype(str)

    dcol = _find_col(df, ["delta_ARI_TopVar_Random", "delta_ari_topvar_random"])
    if dcol is None:
        vcol = _find_col(df, ["ARI_TopVar", "ari_topvar"])
        rcol = _find_col(df, ["ARI_Random", "ari_random"])
        if vcol is None or rcol is None:
            raise ValueError("Unsupervised table missing ΔARI or ARI_TopVar/ARI_Random columns.")
        df["delta_ARI_TopVar_Random"] = pd.to_numeric(df[vcol], errors="coerce") - pd.to_numeric(df[rcol], errors="coerce")
        dcol = "delta_ARI_TopVar_Random"

    di_map = _get_di_map(regime.assign(view_id=regime["dataset"].astype(str) + ":" + regime["view"].astype(str)))
    df["DI"] = df["view_id"].map(di_map)

    x = pd.to_numeric(df[dcol], errors="coerce")
    ok = np.isfinite(x.values)
    df = df.loc[ok].copy()
    df["delta"] = x.loc[ok].values
    df = df.sort_values("DI", ascending=True).reset_index(drop=True)

    if df.empty:
        ax.text(0.5, 0.5, "No unsupervised data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color=GREY)
        _panel_label(ax, "d")
        return

    y = np.arange(len(df))
    ax.axvline(0, color=GREY, lw=1.0, ls="--", alpha=0.5, zorder=1)

    for i, row in df.iterrows():
        di = row["DI"] if pd.notna(row["DI"]) else 1.0
        color = DI_CMAP(DI_NORM(np.clip(di, 0.6, 1.1)))
        val = row["delta"]
        ax.plot([0, val], [y[i], y[i]], color=color, lw=2.0, alpha=0.85, solid_capstyle="round")
        ax.scatter(val, y[i], c=[color], s=55, edgecolors="white", linewidth=0.6, zorder=3)

    labels = []
    for _, r in df.iterrows():
        ds = r.get("dataset", r.get("view_id", "").split(":")[0])
        vw = r.get("view", r.get("view_id", "").split(":")[1] if ":" in r.get("view_id", "") else "")
        labels.append(f"{DS_SHORT.get(ds, ds)}:{vw}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()

    from scipy import stats as sp_stats
    xx = pd.to_numeric(df["DI"], errors="coerce")
    yy = pd.to_numeric(df["delta"], errors="coerce")
    mask = np.isfinite(xx) & np.isfinite(yy)
    rho, p = sp_stats.spearmanr(xx[mask], yy[mask])
    hurts = int((yy[mask] < 0).sum())
    n = int(mask.sum())
    ax.text(0.98, 0.05, f"TopVar hurts: {hurts}/{n}\nSpearman ρ={rho:.2f}, p={p:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT["annot"],
            color="black", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B2DFDB", alpha=0.9))

    ax.set_xlabel("ΔARI (TopVar − Random) at K = 10%", fontsize=FONT["label"], color="black")
    ax.grid(axis="x", color="#B2DFDB", alpha=0.4, lw=0.5)
    _spine_style(ax)
    ax.tick_params(colors="black", width=0.6)
    _panel_label(ax, "d", color="black")


# =====================================================================
# PANEL E -- Regime-bracketed diverging bars
# =====================================================================

def panel_E_regime_bars(ax, abl: pd.DataFrame, regime: pd.DataFrame):
    """
    Fig 3c-style diverging bars: Δ(TopVar − Random) (pp) at K=10%, grouped by regime.
    """
    df = abl.copy()

    # keep consistent with other panels
    if "model" in df.columns:
        df = df[df["model"].astype(str) == "xgb_bal"].copy()
    if "metric" in df.columns:
        df = df[df["metric"].astype(str) == "balanced_accuracy"].copy()

    # required columns
    for c in ["delta_var_minus_random", "consensus_regime", "dataset", "view"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' in ablation_by_regime.csv")

    df["delta_pp"] = pd.to_numeric(df["delta_var_minus_random"], errors="coerce") * 100.0
    df = df.dropna(subset=["delta_pp"]).copy()
    df["view_id"] = df["dataset"].astype(str) + ":" + df["view"].astype(str)

    # DI map (safe fallback to 1.0)
    reg = regime.copy()
    reg["view_id"] = reg["dataset"].astype(str) + ":" + reg["view"].astype(str)
    di_col = next((c for c in ["DI", "di", "di_k10", "DI_k10"] if c in reg.columns), None)
    di_map = dict(zip(reg["view_id"], pd.to_numeric(reg[di_col], errors="coerce"))) if di_col else {}
    df["DI"] = df["view_id"].map(di_map).fillna(1.0)

    # regime order
    def _rk(s):
        su = str(s).upper()
        if "COUP" in su or su.startswith("C"): return 0
        if "MIX"  in su or su.startswith("M"): return 1
        if "ANTI" in su or su.startswith("A"): return 2
        return 99

    df["rk"] = df["consensus_regime"].map(_rk)
    df = df.sort_values(["rk", "delta_pp"]).reset_index(drop=True)
    y = np.arange(len(df))

    # colors from DI
    colors = [DI_CMAP(DI_NORM(np.clip(float(di), 0.6, 1.1))) for di in df["DI"].values]

    # bars
    ax.axvline(0, color=GREY, lw=1.0, ls="--", alpha=0.6, zorder=1)
    ax.barh(y, df["delta_pp"].values, color=colors, edgecolor="white", linewidth=0.6, height=0.75, zorder=2)

    # value labels (pp)
    for i, v in enumerate(df["delta_pp"].values):
        ha = "left" if v >= 0 else "right"
        x = v + (0.3 if v >= 0 else -0.3)
        ax.text(x, y[i], f"{v:+.1f}", va="center", ha=ha, fontsize=7, color="black")

    # y labels
    labels = [f"{DS_SHORT.get(r['dataset'], r['dataset'])}:{r['view']}" for _, r in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()

    # regime separators + labels
    bounds = []
    for k in sorted(df["rk"].unique()):
        idx = np.where(df["rk"].values == k)[0]
        if len(idx): bounds.append((k, idx.min(), idx.max(), len(idx)))
    for (k, lo, hi, n) in bounds:
        ax.axhline(lo - 0.5, color="#B2DFDB", lw=0.8, alpha=0.9)
        name = {0:"Coupled",1:"Mixed",2:"Anti-aligned"}.get(k,"Other")
        ax.text(0.01, (lo+hi)/2, f"{name} (n={n})",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#B2DFDB", alpha=0.9))

    # semantic side labels (optional but very Fig 3c-like)
    ax.text(0.02, 0.96, "TopVar harms", transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="black", alpha=0.8)
    ax.text(0.98, 0.96, "TopVar helps", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="black", alpha=0.8)

    ax.set_xlabel("Δ(TopVar − Random) (pp) at K = 10%", fontsize=FONT["label"], labelpad=0, color="black")
    ax.grid(axis="x", color="#B2DFDB", alpha=0.35, lw=0.6)
    _spine_style(ax)
    ax.tick_params(colors="black", width=0.6)
    _panel_label(ax, "e", color="black")


# =====================================================================
# MAIN ASSEMBLY
# =====================================================================

def create_figure(outputs_dir: Path, output_path: Path):
    apply_style()

    print("=" * 70)
    print("FIGURE 4: Downstream Consequences of Variance Filtering")
    print("  a = Strategy comparison          b = Delta-vs-K")
    print("  c = DI vs Delta scatter          d = Unsupervised ΔARI harm")
    print("  e = Regime-stratified Δ at K=10%")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    regime = load_regime_map(outputs_dir)
    regime_v2 = load_regime_consensus(outputs_dir)
    abl = load_ablation_by_regime(outputs_dir)
    abl_k = load_ablation_across_k(outputs_dir)
    di_delta = load_di_vs_delta(outputs_dir)
    unsup = load_unsupervised_clustering_table(outputs_dir)

    # ---------------------------------------------------------------------
    # Enforce canonical view set (14 views) for downstream consequence tables.
    # The canonical set is defined by di_vs_delta_scatter_k10.csv (14 views × 2 models).
    # This removes derived views such as ibdmdb:MGX_CLR and tcga_gbm:methylation_Mval.
    # ---------------------------------------------------------------------
    canon = set(di_delta["dataset"].astype(str) + ":" + di_delta["view"].astype(str))

    def _filter_to_canon(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df = df.copy()
        if "view_id" not in df.columns:
            if "dataset" in df.columns and "view" in df.columns:
                df["view_id"] = df["dataset"].astype(str) + ":" + df["view"].astype(str)
            else:
                return df
        before = len(df)
        df = df[df["view_id"].isin(canon)].copy()
        after = len(df)
        if after != before:
            print(f"  filtered {name}: {before} -> {after} rows (canonical views)")
        return df

    unsup = _filter_to_canon(unsup, "unsupervised_table")

    print("\n[2/5] Building asymmetric layout...")
    fig = plt.figure(figsize=(11, 9.3))

    gs = GridSpec(
        3, 12, figure=fig,
        height_ratios=[1.25, 0.80, 0.65],
        hspace=0.38, wspace=0.9,
        left=0.07, right=0.96, top=0.96, bottom=0.07,
    )

    # Row 1: a (HERO, 7 cols) + b (5 cols)
    ax_a = fig.add_subplot(gs[0, 0:7])
    ax_b = fig.add_subplot(gs[0, 7:12])

    # Slightly narrow panel a (~5%) to increase gap before panel b.
    _pa = ax_a.get_position()
    ax_a.set_position([_pa.x0, _pa.y0, _pa.width * 0.95, _pa.height])

    # Row 2: c unchanged (5 cols), d narrowed (~14%) with a spacer column in between
    ax_c = fig.add_subplot(gs[1, 0:5])
    ax_d = fig.add_subplot(gs[1, 6:12])

    # Row 3: e (full width, compact)
    ax_e = fig.add_subplot(gs[2, :])

    print("\n[3/5] Drawing panels...")
    print("  Panel a: Strategy comparison (HERO)...")
    panel_A_strategy(ax_a, abl, regime)

    print("  Panel b: Delta vs K trajectories...")
    panel_B_delta_k(ax_b, abl_k, regime)  # use regime_map so DI colours match Results DI

    print("  Panel c: DI vs Delta scatter...")
    panel_C_di_scatter(ax_c, di_delta, regime)

    print("  Panel d: Unsupervised clustering harm (ΔARI)...")
    panel_D_unsupervised_delta_ari(ax_d, unsup, regime)

    print("  Panel e: Regime-bracketed diverging bars...")
    panel_E_regime_bars(ax_e, abl, regime)

    # --- Shared DI colorbar ---
    cax = fig.add_axes([0.30, 0.009, 0.40, 0.008])
    sm = ScalarMappable(cmap=DI_CMAP, norm=DI_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Decoupling Index (DI)", fontsize=9, labelpad=3,
                   color="black")
    cbar.ax.tick_params(labelsize=8, colors=DARK_SEAGREEN)
    cbar.ax.xaxis.set_major_locator(mticker.FixedLocator([0.65, 0.80, 0.95, 1.0, 1.05, 1.10]))
    cbar.ax.text(0.0, 1.8, "Coupled", transform=cbar.ax.transAxes,
                 fontsize=8, ha="left", color=COUPLED_GREEN, fontweight="bold")
    cbar.ax.text(1.0, 1.8, "Anti-aligned", transform=cbar.ax.transAxes,
                 fontsize=8, ha="right", color=ANTI_YLGREEN, fontweight="bold")

    # --- Save ---
    print("\n[4/5] Saving...")
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_path, dpi=300, facecolor="white", edgecolor="none")
    print(f"  PNG: {png_path}")
    fig.savefig(pdf_path, facecolor="white", edgecolor="none")
    print(f"  PDF: {pdf_path}")

    # --- Inventory ---
    support_dir = png_path.parent / "figure_4"
    support_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "figure": "Figure 4: Downstream Consequences of Variance Filtering",
        "style": "High-impact journal | White background | Green-family palette",
        "source_panels": {
            "a": "panel_A_strategy",
            "b": "panel_B_delta_k",
            "c": "panel_C_di_scatter",
            "d": "panel_D_unsupervised_delta_ari",
            "e": "panel_E_regime_bars",
        },
        "data_sources": {
            "ablation_by_regime": str(outputs_dir / SEC4 / "ablation_by_regime.csv"),
            "ablation_across_k": str(outputs_dir / SEC4 / "ablation_across_k.csv"),
            "di_vs_delta": str(outputs_dir / SEC4 / "di_vs_delta_scatter_k10.csv"),
            "unsupervised_table": str(outputs_dir / SEC4 / "unsupervised_clustering_table.csv"),
            "regime_map": str(outputs_dir / SEC1 / "regime_map.csv"),
            "regime_consensus": str(outputs_dir / "04_importance" / "aggregated" / "regime_consensus.csv"),
        },
        "panel_mapping": {
            "a": "Strategy comparison at K=10% -- variance filtering is a coin flip",
            "b": "Delta vs K trajectories -- harm amplifies at tighter thresholds (pp scale)",
            "c": "DI vs Delta scatter -- DI predicts harm magnitude (continuous gradient)",
            "d": "Unsupervised clustering harm -- ΔARI(TopVar - Random) ordered by DI",
            "e": "Regime-stratified summary -- Δ(TopVar - Random) (pp) at K=10%",
        },
    }
    inv_path = support_dir / "figure_4_inventory.json"
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
    parser = argparse.ArgumentParser(
        description="Figure 4: Downstream Consequences of Variance Filtering")
    parser.add_argument("--outputs-dir", type=str,
                        default=r"C:\Users\ms\Desktop\var-pre\outputs")
    parser.add_argument("--output", type=str,
                        default=r"C:\Users\ms\Desktop\var-pre\outputs\figures\figure_4.png")
    args = parser.parse_args()

    create_figure(Path(args.outputs_dir), Path(args.output))