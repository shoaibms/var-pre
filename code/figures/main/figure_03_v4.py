#!/usr/bin/env python3
r"""
Figure 3
========
"Mechanistic basis: signal fraction and geometric misalignment"

Panels:
  a = Paired dots: rho(P,Rj) vs rho(P,V)           rj_correlations_by_regime.csv
  b = Correlation fingerprint heatmap                shap_vs_variance_correlations.csv
  c = Diverging horizontal bars: etaES enrichment    signal_enrichment_by_regime.csv
  d = Butterfly chart: between vs within decomp      variance_decomposition_exemplars.csv
  e = Raincloud permutation collapse                 permutation_collapse.csv + ablation

Colors: from colourlist.py

Usage:
  python figure_03_v4.py
  python figure_03_v4.py --outputs-dir C:/Users/ms/Desktop/var-pre/outputs
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# =====================================================================
# IMPORT COLOURS FROM colourlist.py
# =====================================================================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colourlist import (
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY, SPINE_COLOR, GRID_COLOR, BG_WHITE,
    SIGNAL_GREEN, VARIANCE_COLOR, BETWEEN_COLOR, WITHIN_COLOR,
    GREY, GREY_LIGHTER, GREY_LIGHT,
    TEAL, DARK_TURQUOISE, STEEL_BLUE, CORNFLOWER,
    DS_MARKERS, DS_SHORT, DS_DISPLAY, DS_LABEL_COLORS, FONT,
    bugreen, ylgreen, greens,
)

# =====================================================================
# DESIGN SYSTEM
# =====================================================================

BG_DARK       = BG_WHITE
BG_PANEL      = BG_WHITE
BG_CARD       = BG_WHITE
WHITE_GLOW    = "#FFFFFF"
GREY_MID      = "#6E7681"

DEEP_GREEN    = COUPLED_GREEN
FOREST_GREEN  = MID_BLUEGREEN
MEDIUM_GREEN  = LIGHT_BLUEGREEN
LIGHT_GREEN   = LIGHT_BLUEGREEN
PALE_GREEN    = NEUTRAL_GREEN
MINT          = greens[2]
SOFT_MINT     = greens[1]
DARK_SEAGREEN = TEXT_SECONDARY

TEAL_BRIGHT   = DARK_TURQUOISE
ROYAL_BLUE    = CORNFLOWER          # vibrant blue (was faded #31688e)

YLGREEN       = ANTI_YLGREEN
YLGREEN_LIGHT = LIGHT_YLGREEN
LIME_GREEN    = VARIANCE_COLOR

COLOR_RJ       = SIGNAL_GREEN
COLOR_VAR      = VARIANCE_COLOR
COLOR_BETWEEN  = BETWEEN_COLOR
COLOR_WITHIN   = WITHIN_COLOR

LIGHT_GREY     = GREY_LIGHTER

# DI colormap
DI_CMAP = LinearSegmentedColormap.from_list("di_green", [
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN,
    NEUTRAL_GREEN, LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
], N=256)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)

# Glow effects
GLOW = [pe.withStroke(linewidth=3, foreground=SIGNAL_GREEN, alpha=0.15)]


# =====================================================================
# HELPERS
# =====================================================================

def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def draw_glow_line(ax, x0, y0, x1, y1, color, lw=2.0, alpha=0.9, glow_layers=3):
    for i in range(glow_layers, 0, -1):
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw + i * 1.5,
                alpha=alpha * 0.08 / i, solid_capstyle="round", zorder=1)
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=alpha,
            solid_capstyle="round", zorder=2)


def draw_glow_scatter(ax, x, y, color, s=70, marker="o", zorder=5):
    ax.scatter(x, y, color=color, s=s * 2.5, alpha=0.08, marker=marker,
               zorder=zorder - 2, edgecolors="none")
    ax.scatter(x, y, color=color, s=s * 1.5, alpha=0.15, marker=marker,
               zorder=zorder - 1, edgecolors="none")
    ax.scatter(x, y, color=color, s=s, alpha=0.95, marker=marker, zorder=zorder,
               edgecolors=WHITE_GLOW, linewidths=0.4)


def panel_label(ax, letter, subtitle="", x=-0.02, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=16, fontweight="black", color=TEXT_PRIMARY,
            va="bottom", ha="left",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])
    if subtitle:
        ax.text(x + 0.04, y - 0.005, subtitle, transform=ax.transAxes,
                fontsize=9.5, fontweight="medium", color=TEXT_SECONDARY,
                va="bottom", ha="left", style="italic")


# =====================================================================
# STYLE
# =====================================================================

def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"],
        "font.size": FONT["base"],
        "axes.titlesize": FONT["title"],
        "axes.titleweight": "bold",
        "axes.labelsize": FONT["label"],
        "axes.labelweight": "medium",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_SECONDARY,
        "figure.facecolor": BG_DARK,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"],
        "legend.frameon": False,
        "legend.labelcolor": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.20,
        "savefig.facecolor": BG_DARK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
    })


# =====================================================================
# DATA LOADING
# =====================================================================

SEC1 = Path("results/main_results/section_1_paradox_discovery")
SEC2 = Path("results/main_results/section_2_regime_characterisation")
SEC3 = Path("results/main_results/section_3_mechanism")


def load_regime_map(base: Path) -> pd.DataFrame:
    """Load regime map -- try regime_map.csv first, then regime_consensus.csv."""
    p1 = base / SEC1 / "regime_map.csv"
    p2 = base / "04_importance" / "aggregated" / "regime_consensus.csv"
    if p1.exists():
        df = pd.read_csv(p1)
        renames = {}
        for old, new in [("consensus_regime", "regime"), ("spearman_rho_consensus", "rho")]:
            if old in df.columns and new not in df.columns:
                renames[old] = new
        if renames:
            df = df.rename(columns=renames)
        di_col = _find_col(df, ["DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean", "DI"])
        if di_col and di_col != "DI":
            df["DI"] = df[di_col]
        print(f"  regime_map:           {len(df)} views, cols: {list(df.columns)[:10]}")
        return df
    elif p2.exists():
        df = pd.read_csv(p2)
        di_col = _find_col(df, ["DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean", "DI"])
        if di_col and di_col != "DI":
            df["DI"] = df[di_col]
        print(f"  regime_consensus:     {len(df)} views, cols: {list(df.columns)[:10]}")
        return df
    else:
        raise FileNotFoundError(f"No regime file found at {p1} or {p2}")


def load_rj_correlations(base: Path) -> pd.DataFrame:
    p = base / SEC3 / "rj_correlations_by_regime.csv"
    df = pd.read_csv(p)
    print(f"  rj_correlations:      {len(df)} rows, cols={list(df.columns)[:8]}...")
    return df


def load_shap_vs_var_corr(base: Path) -> pd.DataFrame:
    p = base / SEC3 / "shap_vs_variance_correlations.csv"
    df = pd.read_csv(p)
    print(f"  shap_vs_var_corr:     {len(df)} rows")
    return df


def load_signal_enrichment(base: Path) -> pd.DataFrame:
    p = base / SEC3 / "signal_enrichment_by_regime.csv"
    df = pd.read_csv(p)
    print(f"  signal_enrichment:    {len(df)} rows, cols={list(df.columns)[:10]}...")
    return df


def load_var_decomp(base: Path) -> pd.DataFrame:
    p = base / SEC3 / "variance_decomposition_exemplars.csv"
    df = pd.read_csv(p)
    print(f"  var_decomp:           {len(df)} rows")
    return df


def load_perm_collapse(base: Path) -> pd.DataFrame:
    p = base / SEC3 / "permutation_collapse.csv"
    df = pd.read_csv(p)
    print(f"  perm_collapse:        {len(df)} rows, cols={list(df.columns)}")
    return df


def load_cross_model_ablation(base: Path) -> pd.DataFrame:
    p = base / SEC2 / "cross_model_ablation_comparison.csv"
    df = pd.read_csv(p)
    print(f"  cross_model_ablation: {len(df)} rows (for observed SHAP advantage)")
    return df


# =====================================================================
# PANEL A: Paired Dots - rho(P,Rj) vs rho(P,V)
# =====================================================================

def panel_a_paired_dots(ax, rj_df: pd.DataFrame, regime_df: pd.DataFrame):
    """Paired dots showing Rj always beats V as predictor of importance."""
    print("    Panel a: Paired dots rho(P,Rj) vs rho(P,V)")

    di_col = _find_col(regime_df, ["DI", "DI_10pct_consensus"])
    di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df[di_col]
    ))

    df = rj_df.copy()
    # Build view_id if not present
    if 'view_id' not in df.columns and 'dataset' in df.columns and 'view' in df.columns:
        df['view_id'] = df['dataset'] + ':' + df['view']
    df['di'] = df['view_id'].map(di_map)
    df = df.sort_values('di', ascending=True).reset_index(drop=True)

    pr_col = _find_col(df, ['corr_P_R', 'corr_P_Rj', 'rho_P_Rj'])
    pv_col = _find_col(df, ['corr_P_V', 'corr_P_totalvar', 'rho_P_V'])
    if not pr_col or not pv_col:
        ax.text(0.5, 0.5, f"Missing columns\n{list(df.columns[:10])}",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        return

    y_pos = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        di = row['di'] if pd.notna(row['di']) else 1.0
        color = DI_CMAP(DI_NORM(di))
        marker = DS_MARKERS.get(row['dataset'], 'o')

        rho_pv = row[pv_col]
        rho_pr = row[pr_col]

        # Gradient connecting line
        n_seg = 30
        x_pts = np.linspace(rho_pv, rho_pr, n_seg + 1)
        y_pts = np.full(n_seg + 1, i)
        points = np.array([x_pts, y_pts]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        seg_colors = []
        for j in range(n_seg):
            alpha = 0.25 + 0.55 * (j / n_seg)
            c = list(mcolors.to_rgba(color))
            c[3] = alpha
            seg_colors.append(c)

        lc = LineCollection(segments, colors=seg_colors, linewidths=2.5, zorder=2)
        ax.add_collection(lc)

        # V dot (hollow)
        ax.scatter(rho_pv, i, marker='o', s=50, c=[LIGHT_GREY],
                   edgecolors=color, linewidths=1.2, zorder=3, label='_')
        # Rj dot (filled)
        ax.scatter(rho_pr, i, marker=marker, s=70, c=[color],
                   edgecolors='white', linewidths=1.0, zorder=4, label='_')

        # Direction arrow
        mid_x = (rho_pv + rho_pr) / 2
        if rho_pr > rho_pv:
            ax.annotate('', xy=(rho_pr - 0.02, i), xytext=(mid_x, i),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
                        zorder=2)

    # View labels
    labels = [f"{DS_SHORT.get(r['dataset'], r['dataset'])}:{r['view']}" for _, r in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)

    ax.axvline(0, color=GREY, linewidth=1, alpha=0.3, linestyle='--')
    ax.set_xlabel('Spearman $\\rho_s$ with SHAP importance (P)', fontsize=FONT['label'] - 1,
                  color=TEXT_SECONDARY)
    ax.set_ylim(-0.8, len(df) - 0.2)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)

    # Legend
    ax.scatter([], [], marker='o', s=40, c=[LIGHT_GREY], edgecolors=GREY,
               linewidths=1.0, label='$\\rho$(P, V)')
    ax.scatter([], [], marker='o', s=50, c=[SIGNAL_GREEN],
               edgecolors='white', linewidths=1.0, label='$\\rho$(P, $\\eta^2$)')
    leg = ax.legend(loc='upper right', fontsize=8, handletextpad=0.3)
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)

    # Summary
    mean_pr = df[pr_col].mean()
    mean_pv = df[pv_col].mean()
    n_wins = (df[pr_col] > df[pv_col]).sum()
    ax.text(0.03, 0.03,
            f'$\\rho$(P,$\\eta^2$) > $\\rho$(P,V) in {n_wins}/{len(df)} views\n'
            f'Mean $\\rho$(P,$\\eta^2$) = {mean_pr:.2f}\n'
            f'Mean $\\rho$(P,V) = {mean_pv:.2f}',
            transform=ax.transAxes, fontsize=7.5, ha='left', va='bottom',
            color=TEXT_PRIMARY, family="monospace",
            bbox=dict(boxstyle='round,pad=0.4', fc=BG_CARD, ec=SIGNAL_GREEN, alpha=0.85,
                      linewidth=0.8))

    panel_label(ax, "a")


# =====================================================================
# PANEL B: Correlation Fingerprint Heatmap
# =====================================================================

def panel_b_corr_fingerprint(ax, corr_df: pd.DataFrame, regime_df: pd.DataFrame):
    """Heatmap of correlation metrics across views, ordered by DI."""
    print("    Panel b: Correlation fingerprint heatmap")

    di_col = _find_col(regime_df, ["DI", "DI_10pct_consensus"])
    di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df[di_col]
    ))

    df = corr_df.copy()
    if 'view_id' not in df.columns and 'dataset' in df.columns and 'view' in df.columns:
        df['view_id'] = df['dataset'] + ':' + df['view']
    df['di'] = df['view_id'].map(di_map)
    df = df.sort_values('di', ascending=True).reset_index(drop=True)

    # Metrics to display
    metrics = ['corr_P_R', 'corr_P_V', 'corr_P_between', 'corr_V_between', 'corr_V_within']
    metric_labels = ['$\\rho$(P,$\\eta^2$)', '$\\rho$(P,V)', '$\\rho$(P,$V_{between}$)',
                     '$\\rho$(V,$V_{between}$)', '$\\rho$(V,$V_{within}$)']

    n_views = len(df)
    n_metrics = len(metrics)
    matrix = np.full((n_views, n_metrics), np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        for j, m in enumerate(metrics):
            if m in df.columns:
                matrix[i, j] = row[m]

    # Diverging green colormap
    green_div = LinearSegmentedColormap.from_list('green_div', [
        YLGREEN, SOFT_MINT, DEEP_GREEN
    ], N=256)

    im = ax.imshow(matrix, cmap=green_div, vmin=-0.5, vmax=1.0,
                   aspect='auto', interpolation='nearest')

    # Annotate cells
    for i in range(n_views):
        for j in range(n_metrics):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > 0.6 else TEXT_SECONDARY
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=text_color, fontweight='medium')

    # Labels
    short_labels = []
    for _, row in df.iterrows():
        ds_short = DS_SHORT.get(row['dataset'], row['dataset'])
        short_labels.append(f"{ds_short}:{row['view']}")

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_labels, fontsize=7, rotation=35, ha='right')
    ax.set_yticks(range(n_views))
    ax.set_yticklabels(short_labels, fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    panel_label(ax, "b", x=-0.05, y=1.08)


# =====================================================================
# PANEL C: Diverging bars -- etaES enrichment
# =====================================================================

def panel_c_enrichment(ax, sig_enrich: pd.DataFrame, regime_df: pd.DataFrame):
    """Diverging bars showing signal enrichment ratio centered on 1.0."""
    print("    Panel c: Diverging bars etaES enrichment")

    di_col = _find_col(regime_df, ["DI", "DI_10pct_consensus"])
    df = sig_enrich.copy()

    if "DI" not in df.columns:
        df = df.merge(regime_df[["dataset", "view", di_col]].rename(columns={di_col: "DI"}),
                       on=["dataset", "view"], how="left")

    eta_col = _find_col(df, [
        "eta_es", "eta_es_mean", "eta_ES", "etaES", "signal_enrichment_ratio",
        "eta_enrichment", "enrichment_ratio",
    ])
    vsa_col = _find_col(df, ["vsa", "vsa_mean", "VSA", "variance_signal_alignment"])

    print(f"      eta_col found: {eta_col}, vsa_col found: {vsa_col}")

    if not eta_col:
        ax.text(0.5, 0.5, f"No eta_ES column found\nAvailable: {list(df.columns)[:12]}",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color=TEXT_SECONDARY)
        return

    df["eta"] = pd.to_numeric(df[eta_col], errors="coerce")

    # Filter to K=10
    k_col = _find_col(df, ["k_pct", "K_pct", "K"])
    if k_col:
        k_vals = df[k_col].unique()
        if len(k_vals) > 1:
            target = 10 if 10 in k_vals else k_vals[0]
            df = df[df[k_col] == target].copy()

    if "dataset" in df.columns and "view" in df.columns:
        df = df.groupby(["dataset", "view"], as_index=False).first()

    df = df.dropna(subset=["eta"]).sort_values("eta", ascending=True).reset_index(drop=True)
    n = len(df)
    if n == 0:
        ax.text(0.5, 0.5, "No valid eta values", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_SECONDARY)
        return

    y = np.arange(n)

    # Diverging bars from eta=1
    for i, (_, row) in enumerate(df.iterrows()):
        eta = row["eta"]
        di_val = row.get("DI", 1.0)
        di_val = di_val if np.isfinite(di_val) else 1.0

        if eta <= 1.0:
            bar_color = FOREST_GREEN
            bar_alpha = 0.7
        else:
            bar_color = YLGREEN
            intensity = min((eta - 1.0) / 4.0, 1.0)
            bar_alpha = 0.5 + 0.4 * intensity

        bar_width = eta - 1.0
        ax.barh(i, bar_width, left=1.0, color=bar_color, height=0.65,
                alpha=bar_alpha, edgecolor="none", zorder=2)

        draw_glow_scatter(ax, eta, i, bar_color, s=30, marker="|")

        offset = 0.08 if eta >= 1.0 else -0.08
        ha = "left" if eta >= 1.0 else "right"
        ax.text(eta + offset, i, f"{eta:.2f}", va="center", ha=ha,
                fontsize=7.5, color=bar_color, fontweight="bold")

    # Reference line at eta=1
    ax.axvline(1.0, color=TEXT_TERTIARY, linewidth=1.2, linestyle="-", alpha=0.5, zorder=3)

    # Zone annotations
    ax.text(0.55, n - 0.5, "Signal IN\nhigh-var tail", fontsize=8,
            color=FOREST_GREEN, style="italic", alpha=0.7, va="top", ha="center")
    max_eta = df["eta"].max()
    if max_eta > 1.5:
        ax.text(min(max_eta * 0.75, 5), 0.5, "Signal OUTSIDE\nhigh-var tail",
                fontsize=8, color=YLGREEN, style="italic", alpha=0.7, va="bottom")

    # Y labels
    for i, (_, row) in enumerate(df.iterrows()):
        ds = row["dataset"]
        vw = row["view"]
        label = f"{DS_SHORT.get(ds, ds)}/{vw}"
        ax.text(-0.03, i, label, va="center", ha="right", fontsize=8,
                color="black", fontweight="medium",
                transform=ax.get_yaxis_transform())

    ax.set_yticks([])
    ax.set_ylim(-0.7, n + 0.5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.15, color=GRID_COLOR)

    # Summary
    eta_mean = df["eta"].mean()
    summary = f"Mean ηES = {eta_mean:.2f} | Range: {df['eta'].min():.2f} -- {df['eta'].max():.2f}"
    if vsa_col and vsa_col in df.columns:
        vsa_mean = pd.to_numeric(df[vsa_col], errors="coerce").mean()
        if pd.notna(vsa_mean):
            summary += f" | VSA = {vsa_mean:.3f}"
    ax.text(0.5, -0.26, summary, transform=ax.transAxes, fontsize=8,
            ha="center", color=TEXT_TERTIARY, style="italic")

    ax.set_xlabel("Signal enrichment ratio (ηES)", fontsize=FONT["label"] - 1, labelpad=4,
                  color=TEXT_SECONDARY)
    ax.tick_params(axis="x", colors="black")
    panel_label(ax, "c")


# =====================================================================
# PANEL D: Butterfly chart -- Between vs Within decomp
# =====================================================================

def panel_d_butterfly(ax, decomp: pd.DataFrame, regime_df: pd.DataFrame):
    """Butterfly chart: between-class extends left, within-class extends right."""
    print("    Panel d: Butterfly chart between vs within")

    if decomp.empty:
        ax.text(0.5, 0.5, "No decomposition exemplars available",
                transform=ax.transAxes, ha="center", va="center", color=TEXT_SECONDARY)
        return

    feat_col = _find_col(decomp, ["feature", "gene", "feature_id"])
    total_col = _find_col(decomp, ["var_total", "total_var", "v_total"])
    between_col = _find_col(decomp, ["var_between", "between_var", "v_between"])
    within_col = _find_col(decomp, ["var_within", "within_var", "v_within"])
    ds_col = _find_col(decomp, ["dataset"])
    vw_col = _find_col(decomp, ["view"])

    if not feat_col or not total_col:
        ax.text(0.5, 0.5, f"Missing columns\n{list(decomp.columns)[:8]}...",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color=TEXT_SECONDARY)
        return

    df = decomp.copy()

    # Compute from R_j if needed
    if not between_col and "R_j" in df.columns and total_col:
        df["_Rj"] = pd.to_numeric(df["R_j"], errors="coerce")
        df["_total"] = pd.to_numeric(df[total_col], errors="coerce")
        df["_between"] = df["_total"] * df["_Rj"] / (1 + df["_Rj"])
        df["_within"] = df["_total"] - df["_between"]
        between_col = "_between"
        within_col = "_within"
    elif not between_col and "eta_sq" in df.columns and total_col:
        df["_between"] = df["eta_sq"] * df[total_col]
        df["_within"] = (1 - df["eta_sq"]) * df[total_col]
        between_col = "_between"
        within_col = "_within"

    if not between_col or not within_col:
        ax.text(0.5, 0.5, "No between/within decomposition",
                transform=ax.transAxes, ha="center", va="center", color=TEXT_SECONDARY)
        return

    if len(df) > 15:
        df = df.head(15)

    total = pd.to_numeric(df[total_col], errors="coerce").values
    between = pd.to_numeric(df[between_col], errors="coerce").values
    within = pd.to_numeric(df[within_col], errors="coerce").values

    total_safe = np.where(total > 0, total, 1)
    frac_betw = between / total_safe
    frac_with = within / total_safe

    df = df.copy()
    df["_frac_betw"] = frac_betw
    df["_frac_with"] = frac_with
    df = df.sort_values("_frac_betw", ascending=True).reset_index(drop=True)
    frac_betw = df["_frac_betw"].values
    frac_with = df["_frac_with"].values

    n = len(df)
    y = np.arange(n)

    # Butterfly: between LEFT (negative), within RIGHT (positive)
    for i in range(n):
        ax.barh(i, -frac_betw[i], color=COLOR_BETWEEN, height=0.6,
                alpha=0.75, edgecolor="none", zorder=2)
        ax.barh(i, frac_with[i], color=COLOR_WITHIN, height=0.6,
                alpha=0.65, edgecolor="none", zorder=2)

    # Center line
    ax.axvline(0, color=TEXT_TERTIARY, linewidth=1.2, alpha=0.5, zorder=3)

    # Feature labels
    for i, (_, row) in enumerate(df.iterrows()):
        feat = str(row[feat_col])[:18]
        ds_vw = ""
        if ds_col and vw_col:
            ds_vw = f" ({DS_SHORT.get(row[ds_col], row[ds_col])}/{row[vw_col]})"
        ax.text(0, i, f" {feat}{ds_vw} ", va="center", ha="center",
                fontsize=7, color=TEXT_PRIMARY, fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG_DARK, alpha=0.8,
                          edgecolor="none"),
                zorder=6)

    # eta^2 annotation
    eta_col_name = _find_col(df, ["eta_sq", "eta2", "R_j", "r_snr"])
    if eta_col_name:
        for i, (_, row) in enumerate(df.iterrows()):
            eta = row.get(eta_col_name)
            if pd.notna(eta):
                eta_val = float(eta)
                if eta_col_name in ["R_j", "r_snr"]:
                    eta_display = eta_val / (1 + eta_val) if eta_val > 0 else 0
                    label = f"η²={eta_display:.2f}"
                else:
                    eta_display = eta_val
                    label = f"η²={eta_val:.2f}"
                color = SIGNAL_GREEN if eta_display > 0.3 else (TEXT_SECONDARY if eta_display > 0.1 else YLGREEN)
                ax.text(1.02, i, label, va="center", ha="left",
                        fontsize=7, color=color, fontweight="bold",
                        transform=ax.get_yaxis_transform())

    ax.set_xlabel("Signal fraction (η²)", fontsize=FONT["label"] - 1, color=TEXT_SECONDARY)
    ax.set_yticks([])
    ax.set_ylim(-0.6, n + 0.3)

    # Direction labels
    ax.text(-0.5, n + 0.3, "Between-class\n(signal)", fontsize=8.5,
            color=COLOR_BETWEEN, fontweight="bold", ha="center", va="bottom")
    ax.text(0.5, n + 0.3, "Within-class\n(noise)", fontsize=8.5,
            color=COLOR_WITHIN, fontweight="bold", ha="center", va="bottom")

    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.15, color=GRID_COLOR)

    panel_label(ax, "d")


# =====================================================================
# PANEL E: Raincloud Permutation Collapse
# =====================================================================

def panel_e_raincloud(ax, perm_df: pd.DataFrame, ablation_df: pd.DataFrame,
                      regime_df: pd.DataFrame):
    """
    Raincloud plot showing permutation null distributions vs observed SHAP advantage.

    For each view (from permutation_collapse.csv):
    - Half-violin "cloud" = reconstructed null distribution
    - Jittered "rain" dots = simulated from null
    - Observed SHAP advantage = annotated diamond
    """
    print("    Panel e: Raincloud permutation collapse")

    di_col = _find_col(regime_df, ["DI", "DI_10pct_consensus"])
    di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df[di_col]
    ))

    # Use balanced_accuracy metric
    perm = perm_df[perm_df['metric'] == 'balanced_accuracy'].copy()
    print(f"      Permutation views (balanced_accuracy): {list(perm['view_id'])}")

    if len(perm) == 0:
        perm = perm_df.drop_duplicates(subset='view_id', keep='first').copy()
        print(f"      FALLBACK: using first available metric, {len(perm)} views")

    # ---- FIX: Deduplicate MLO:methylation and CCLE:mRNA ----
    # Keep only the first occurrence of each view_id
    before_dedup = len(perm)
    perm = perm.drop_duplicates(subset='view_id', keep='first').copy()
    after_dedup = len(perm)
    if before_dedup != after_dedup:
        print(f"      DEDUP: removed {before_dedup - after_dedup} duplicate view_id entries "
              f"(e.g. MLO:methylation, CCLE:mRNA)")

    # Get observed SHAP advantage from ablation data
    obs_map = dict(zip(
        ablation_df['view_id'],
        ablation_df['xgb_delta_shap_var']
    ))

    perm['di'] = perm['view_id'].map(di_map)
    perm['observed'] = perm['view_id'].map(obs_map)
    perm = perm.sort_values('di', ascending=True).reset_index(drop=True)

    n_views = len(perm)
    spacing = 1.0
    cloud_height = 0.35
    rain_spread = 0.12
    rng = np.random.default_rng(42)

    for i, (_, row) in enumerate(perm.iterrows()):
        y_center = i * spacing
        di = row['di'] if pd.notna(row['di']) else 1.0
        color = DI_CMAP(DI_NORM(di))
        vid = row['view_id']

        null_mean = row['delta_shap_var_mean']
        null_q05 = row['delta_shap_var_q05']
        null_q95 = row['delta_shap_var_q95']
        obs_val = row['observed']

        # Reconstruct null std from quantiles
        q_range = null_q95 - null_q05
        if q_range > 0:
            null_std = q_range / (2 * 1.645)
        else:
            null_std = 0.005

        # --- HALF-VIOLIN (cloud) ---
        x_grid = np.linspace(null_mean - 4 * null_std, null_mean + 4 * null_std, 200)
        density = stats.norm.pdf(x_grid, loc=null_mean, scale=null_std)
        if density.max() > 0:
            density_norm = density / density.max() * cloud_height
        else:
            density_norm = density

        y_cloud = y_center + density_norm

        # Gradient fill strips
        n_strips = 40
        strip_idx = np.linspace(0, len(x_grid) - 1, n_strips + 1, dtype=int)
        for j in range(n_strips):
            lo, hi = strip_idx[j], strip_idx[j + 1]
            if hi <= lo:
                continue
            x_s = x_grid[lo:hi + 1]
            y_s = y_cloud[lo:hi + 1]
            ax.fill_between(x_s, y_center, y_s, color=color, alpha=0.20,
                            linewidth=0, zorder=2)

        # Cloud outline
        ax.plot(x_grid, y_cloud, color=color, linewidth=1.2, alpha=0.7, zorder=3)
        # Baseline
        ax.plot([x_grid[0], x_grid[-1]], [y_center, y_center],
                color=LIGHT_GREY, linewidth=0.4, alpha=0.5, zorder=1)

        # --- RAIN (jittered dots below baseline) ---
        n_rain = 25
        rain_x = rng.normal(loc=null_mean, scale=null_std, size=n_rain)
        rain_y = y_center - rng.uniform(0.04, rain_spread, size=n_rain)
        ax.scatter(rain_x, rain_y, s=8, c=[color], alpha=0.45,
                   edgecolors='none', zorder=3)

        # --- MINI BOXPLOT ---
        bp_y = y_center - 0.02
        bp_height = 0.03
        iqr_lo = null_mean - 0.675 * null_std
        iqr_hi = null_mean + 0.675 * null_std
        rect = mpatches.FancyBboxPatch(
            (iqr_lo, bp_y - bp_height / 2), iqr_hi - iqr_lo, bp_height,
            boxstyle='round,pad=0.001', fc=color, alpha=0.4, ec=color, lw=0.8,
            zorder=4)
        ax.add_patch(rect)
        ax.plot([null_mean, null_mean], [bp_y - bp_height / 2, bp_y + bp_height / 2],
                color='white', linewidth=1.5, zorder=5)
        ax.plot([null_q05, iqr_lo], [bp_y, bp_y], color=color, linewidth=0.8, zorder=4)
        ax.plot([iqr_hi, null_q95], [bp_y, bp_y], color=color, linewidth=0.8, zorder=4)

        # --- OBSERVED VALUE (diamond) ---
        if pd.notna(obs_val):
            obs_pp = obs_val
            ax.scatter(obs_pp, y_center + cloud_height * 0.15, marker='D', s=80,
                       c=[color], edgecolors='white', linewidths=1.5, zorder=6)

            ax.annotate('', xy=(obs_pp, y_center + cloud_height * 0.15),
                        xytext=(null_mean, y_center + cloud_height * 0.15),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                                        linestyle='--'),
                        zorder=5)

            # Keep raw units (Results reports raw Δ values); no pp scaling
            side = 'left' if obs_pp > null_mean else 'right'
            ha = side
            dx = 0.003 if obs_pp > null_mean else -0.003
            ax.text(obs_pp + dx, y_center + cloud_height * 0.15 + 0.12,
                    f'{obs_pp:+.4f}', fontsize=6.5, ha=ha, va='bottom',
                    color=color, fontweight='bold')

    # Set proper xlim
    all_obs = perm['observed'].dropna().values
    all_q05 = perm['delta_shap_var_q05'].values
    all_q95 = perm['delta_shap_var_q95'].values
    x_lo = min(min(all_q05), min(all_obs) if len(all_obs) > 0 else 0) - 0.02
    x_hi = max(max(all_q95), max(all_obs) if len(all_obs) > 0 else 0) + 0.02
    ax.set_xlim(x_lo, x_hi)

    # View labels (after xlim is set)
    for i, (_, row) in enumerate(perm.iterrows()):
        vid = row['view_id']
        parts = vid.split(':')
        label = f"{DS_SHORT.get(parts[0], parts[0])}:{parts[1]}"
        di = row['di'] if pd.notna(row['di']) else 1.0
        color = DI_CMAP(DI_NORM(di))
        ax.text(x_lo - 0.003, i * spacing, label, fontsize=8,
                ha='right', va='center', color=color, fontweight='bold')

    # Reference line at 0
    ax.axvline(0, color=GREY, linewidth=1.2, alpha=0.5, linestyle='-', zorder=1)
    ax.text(0.001, (n_views - 1) * spacing + cloud_height + 0.35,
            'No advantage', fontsize=7, ha='left', va='bottom',
            color=GREY, style='italic')

    # Legend
    ax.scatter([], [], marker='D', s=60, c=[SIGNAL_GREEN],
               edgecolors='white', linewidths=1.0, label='Observed (real labels)')
    cloud_patch = mpatches.Patch(color=SIGNAL_GREEN, alpha=0.3,
                                 label='Null distribution\n(shuffled labels)')
    leg = ax.legend(handles=[ax.collections[-1], cloud_patch],
                    loc='upper right', bbox_to_anchor=(0.90, 1.00),
                    fontsize=7.5, handletextpad=0.3)
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)

    ax.set_xlabel('$\\Delta$(SHAP $-$ Var)', fontsize=FONT['label'] - 1, color=TEXT_SECONDARY)
    ax.tick_params(axis="x", colors="black")
    ax.set_yticks([])
    ax.set_ylim(-0.5, (n_views - 1) * spacing + cloud_height + 0.6)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(GRID_COLOR)

    # Summary annotation
    ax.text(0.97, 0.03,
            'Under shuffled labels,\nSHAP advantage collapses to 0\n'
            '(confirming real class structure)',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
            color=SIGNAL_GREEN, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc=BG_CARD, ec=GRID_COLOR, alpha=0.9))

    panel_label(ax, "e")


# =====================================================================
# MAIN FIGURE ASSEMBLY
# =====================================================================

def create_figure(outputs_dir: Path, output_path: Path):
    apply_style()

    print("=" * 70)
    print("CREATING FIGURE 3")
    print("=" * 70)

    # Load all data
    print("\n[1] Loading data...")
    regime_df = load_regime_map(outputs_dir)
    rj_df = load_rj_correlations(outputs_dir)
    corr_df = load_shap_vs_var_corr(outputs_dir)
    enrich_df = load_signal_enrichment(outputs_dir)
    decomp_df = load_var_decomp(outputs_dir)
    perm_df = load_perm_collapse(outputs_dir)
    ablation_df = load_cross_model_ablation(outputs_dir)

    # Create figure
    print("\n[2] Building layout...")
    fig = plt.figure(figsize=(12, 11))
    fig.patch.set_facecolor(BG_DARK)

    # Asymmetric 12-column GridSpec
    # Row 1: a=paired dots (7 cols) + b=correlation fingerprint (5 cols)
    # Row 2: c=etaES diverging bars (5 cols) + d=butterfly decomposition (7 cols)
    # Row 3: e=raincloud (centered)
    gs = GridSpec(3, 12, figure=fig,
                  height_ratios=[1.3, 1.15, 1.0],
                  hspace=0.42, wspace=0.9,
                  left=0.07, right=0.96, top=0.94, bottom=0.10)

    # Row 1
    ax_a = fig.add_subplot(gs[0, 0:7])
    ax_b = fig.add_subplot(gs[0, 7:12])

    # Row 2
    ax_c = fig.add_subplot(gs[1, 0:5])
    ax_d = fig.add_subplot(gs[1, 5:12])

    # Row 3
    ax_e = fig.add_subplot(gs[2, 1:11])

    # Fine layout adjustments per panel
    # Slightly shrink a/b to reduce row-1 crowding, widen c/e per request.
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    pos_e = ax_e.get_position()
    ax_a.set_position([pos_a.x0, pos_a.y0, pos_a.width * 0.94, pos_a.height])
    ax_b.set_position([pos_b.x0 + 0.015, pos_b.y0, pos_b.width * 0.90, pos_b.height])
    ax_c.set_position([pos_c.x0, pos_c.y0, pos_c.width * 1.03, pos_c.height])
    e_new_w = pos_e.width * 1.20
    e_new_x = pos_e.x0 - (e_new_w - pos_e.width) / 2
    ax_e.set_position([e_new_x, pos_e.y0, e_new_w, pos_e.height])

    # Set panel backgrounds
    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
        ax.set_facecolor(BG_PANEL)

    # Render panels
    print("\n[3] Rendering panels...")
    panel_a_paired_dots(ax_a, rj_df, regime_df)
    panel_b_corr_fingerprint(ax_b, corr_df, regime_df)
    panel_c_enrichment(ax_c, enrich_df, regime_df)
    panel_d_butterfly(ax_d, decomp_df, regime_df)
    panel_e_raincloud(ax_e, perm_df, ablation_df, regime_df)

    # DI colorbar
    print("\n[4] Adding colorbar...")
    cax = fig.add_axes([0.30, 0.030, 0.40, 0.008])
    cax.set_facecolor(BG_DARK)
    sm = ScalarMappable(cmap=DI_CMAP, norm=DI_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Decoupling Index (DI)", fontsize=9.5, labelpad=3, color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    cbar.ax.xaxis.set_major_locator(mticker.FixedLocator([0.65, 0.80, 0.95, 1.0, 1.05, 1.10]))
    cbar.outline.set_edgecolor(GRID_COLOR)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.text(0.12, 1.8, "Coupled", transform=cbar.ax.transAxes,
                 fontsize=8, ha="center", color=DEEP_GREEN, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])
    cbar.ax.text(0.88, 1.8, "Anti-aligned", transform=cbar.ax.transAxes,
                 fontsize=8, ha="center", color=YLGREEN, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])

    # Save
    print("\n[5] Saving...")
    png_path = output_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, facecolor=BG_DARK, edgecolor="none")
    print(f"  Saved: {png_path}")

    # Supporting inventory
    support_dir = png_path.parent / "figure_3"
    support_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "figure": str(png_path),
        "design": "Light canvas, colourlist.py palette, magazine quality",
        "panels": {
            "a": "Paired dots: rho(P,Rj) vs rho(P,V)",
            "b": "Correlation fingerprint heatmap",
            "c": "Diverging bars: etaES enrichment centered on 1.0",
            "d": "Butterfly chart: between vs within decomposition",
            "e": "Raincloud permutation collapse",
        },
        "color_source": "colourlist.py (green family + accents)",
    }
    inv_path = support_dir / "figure_3_v4_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"  Inventory: {inv_path}")

    plt.close(fig)

    print("\n" + "=" * 70)
    print("Figure 3 complete!")
    print("=" * 70)
    return png_path


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 3: Mechanistic basis")
    parser.add_argument("--outputs-dir", type=str,
                        default=r"C:\Users\ms\Desktop\var-pre\outputs")
    parser.add_argument("--output", type=str,
                        default=r"C:\Users\ms\Desktop\var-pre\outputs\figures\figure_3.png")
    args = parser.parse_args()

    create_figure(Path(args.outputs_dir), Path(args.output))
    print("\nDone!")