#!/usr/bin/env python3
r"""
Figure 2 -- Reproducibility & View-Specificity
=====================================================
"The variance-importance relationship is reproducible and view-specific"

5 panels in asymmetric 2-row layout:
  a = XGB vs RF scatter (hero)             [cross_model_ablation_comparison.csv]
  b = Cross-K regime stability heatmap     [cross_k_regime_stability.csv]
  c = SHAP advantage lollipop              [cross_model_ablation_comparison.csv]
  d = Dot-matrix model agreement           [cross_model_regime_agreement.csv + regime_consensus.csv]
  e = Modality inconsistency dumbbell      [regime_consensus.csv]

Usage:
  python figure_02_v6.py
  python figure_02_v6.py --outputs-dir C:/Users/ms/Desktop/var-pre/outputs
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT COLOURS FROM colourlist.py
# ============================================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colourlist import (
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    TEXT_PRIMARY, TEXT_SECONDARY, SPINE_COLOR, GRID_COLOR,
    GREY, GREY_LIGHTER, GREY_LIGHT,
    REGIME_COUPLED, REGIME_MIXED, REGIME_ANTI_ALIGNED,
    DS_MARKERS, DS_SHORT, DS_DISPLAY, FONT,
    bugreen, ylgreen, greens,
)

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

DARK_SEAGREEN = TEXT_SECONDARY
LIGHT_GREY = GREY_LIGHTER

DI_CMAP = LinearSegmentedColormap.from_list('di_green', [
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN,
    NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
], N=256)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)

REGIME_COLORS = {
    'COUPLED': REGIME_COUPLED,
    'MIXED': REGIME_MIXED,
    'ANTI_ALIGNED': REGIME_ANTI_ALIGNED,
}
REGIME_CODE = {'COUPLED': 0, 'MIXED': 1, 'ANTI_ALIGNED': 2}


def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


# ============================================================================
# STYLE
# ============================================================================

def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FONT['base'],
        'axes.titlesize': FONT['title'],
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT['label'],
        'axes.labelweight': 'medium',
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': FONT['tick'],
        'ytick.labelsize': FONT['tick'],
        'legend.fontsize': FONT['legend'],
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
    })


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cross_model_ablation(outputs_dir: Path) -> pd.DataFrame:
    p = outputs_dir / "results" / "main_results" / "section_2_regime_characterisation" / "cross_model_ablation_comparison.csv"
    df = pd.read_csv(p)
    print(f"  [+] cross_model_ablation: {len(df)} rows, cols={list(df.columns)[:8]}...")
    return df

def load_cross_k_stability(outputs_dir: Path) -> pd.DataFrame:
    p = outputs_dir / "results" / "main_results" / "section_2_regime_characterisation" / "cross_k_regime_stability.csv"
    df = pd.read_csv(p)
    print(f"  [+] cross_k_stability: {len(df)} rows")
    return df

def load_regime_agreement(outputs_dir: Path) -> pd.DataFrame:
    p = outputs_dir / "results" / "main_results" / "section_2_regime_characterisation" / "cross_model_regime_agreement.csv"
    df = pd.read_csv(p)
    print(f"  [+] regime_agreement: {len(df)} rows, cols={list(df.columns)}")
    return df

def load_regime_consensus(outputs_dir: Path) -> pd.DataFrame:
    """
    Return a regime_df that:
      - preserves model-specific DI columns used by Panel D: DI_10pct_xgb_bal, DI_10pct_rf
      - uses Results DI for sorting/annotation: DI_10pct_uncertainty_xgb_bal (from regime_map.csv)
    """
    p_cons = (outputs_dir / "results" / "main_results" / "section_2_regime_characterisation"
              / "source_tables" / "regime_consensus.csv")
    p_map  = outputs_dir / "results" / "main_results" / "section_1_paradox_discovery" / "regime_map.csv"
    if not p_cons.exists():
        raise FileNotFoundError(f"MISSING: {p_cons}")
    if not p_map.exists():
        raise FileNotFoundError(f"MISSING: {p_map}")

    cons = pd.read_csv(p_cons)
    m = pd.read_csv(p_map)

    # Bring in Results DI + regime/rho (from regime_map)
    keep = [c for c in ["dataset", "view", "regime", "rho",
                        "DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus"]
            if c in m.columns]
    m = m[keep].copy()

    df = cons.merge(m, on=["dataset", "view"], how="left", suffixes=("", "_map"))

    # Require Results DI to exist after merge
    if "DI_10pct_uncertainty_xgb_bal" not in df.columns or df["DI_10pct_uncertainty_xgb_bal"].isna().all():
        raise ValueError(
            "regime_map DI_10pct_uncertainty_xgb_bal missing after merge. "
            "Check regime_map.csv columns and dataset/view keys."
        )

    # Preserve original consensus DI (optional)
    if "DI_10pct_consensus" in df.columns:
        df["DI_10pct_consensus_raw"] = df["DI_10pct_consensus"]

    # Canonical DI for this figure = Results DI (uncertainty-xgb)
    df["DI"] = df["DI_10pct_uncertainty_xgb_bal"]

    # Legacy alias: some panels may reference DI_10pct_consensus; point it to Results DI
    df["DI_10pct_consensus"] = df["DI"]

    # Legacy alias for regime name used in some panels
    if "consensus_regime" not in df.columns and "regime" in df.columns:
        df["consensus_regime"] = df["regime"]

    # Hard fail if panel D essentials are missing
    for req in ["DI_10pct_xgb_bal", "DI_10pct_rf"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column for Panel D: {req}. cols={list(df.columns)}")

    print(
        f"  [+] regime_consensus+map: {len(df)} rows | DI source=DI_10pct_uncertainty_xgb_bal "
        f"| has DI_10pct_xgb_bal={'DI_10pct_xgb_bal' in df.columns} "
        f"| has DI_10pct_rf={'DI_10pct_rf' in df.columns}"
    )
    return df




# ============================================================================
# PANEL A: XGB vs RF Scatter (Hero)
# ============================================================================

def panel_a_xgb_rf_scatter(ax, ablation_df: pd.DataFrame, regime_df: pd.DataFrame):
    """Hero scatter: Delta(Var-Random) XGB vs RF per view."""
    print("    Panel a: XGB vs RF scatter")
    
    x = ablation_df['xgb_delta_var_random'].values * 100  # to pp
    y = ablation_df['rf_delta_var_random'].values * 100

    # Get DI for coloring
    di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df['DI_10pct_consensus']
    ))

    lim = max(abs(x).max(), abs(y).max()) * 1.15
    
    # Quadrant shading
    ax.axhspan(0, lim, xmin=0.5, alpha=0.10, color="#00CED1", zorder=0)
    ax.axhspan(-lim, 0, xmax=0.5, alpha=0.10, color="#00FF7F", zorder=0)
    
    # Reference lines
    ax.axhline(0, color=GREY, linewidth=1, alpha=0.4, linestyle='-')
    ax.axvline(0, color=GREY, linewidth=1, alpha=0.4, linestyle='-')
    ax.plot([-lim, lim], [-lim, lim], color=GREY, linewidth=1, alpha=0.3, linestyle=':')

    for i, (_, row) in enumerate(ablation_df.iterrows()):
        key = f"{row['dataset']}:{row['view']}"
        di = di_map.get(key, 1.0)
        color = DI_CMAP(DI_NORM(di))
        marker = DS_MARKERS.get(row['dataset'], 'o')
        ax.scatter(x[i], y[i], marker=marker, s=100, c=[color],
                   edgecolors='white', linewidths=1.2, zorder=4, alpha=0.9)

        # Label extremes
        if abs(x[i]) > 4 or abs(y[i]) > 4 or not row.get('direction_agree', True):
            label = row.get('view', '')
            dx = 0.5 if x[i] < 0 else -0.5
            ax.annotate(label, (x[i], y[i]), xytext=(x[i] + dx, y[i] + 0.5),
                       fontsize=7, color=DARK_SEAGREEN, fontweight='bold',
                       arrowprops=dict(arrowstyle='-', color=LIGHT_GREY, lw=0.5))

    # Correlation annotation
    r_val = np.corrcoef(x, y)[0, 1]
    rho_s, p_s = stats.spearmanr(x, y)
    n_agree = ablation_df['direction_agree'].sum()
    n_total = len(ablation_df)
    ax.text(0.03, 0.97,
            f'r = {r_val:.2f}, Spearman $\\rho$ = {rho_s:.2f} (p = {p_s:.4f});\nDirection: {n_agree}/{n_total}',
            transform=ax.transAxes, fontsize=8, ha='left', va='top', color=DARK_SEAGREEN,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=LIGHT_GREY, alpha=0.9))

    ax.set_xlabel('$\\Delta$(TopVar $-$ Random), XGBoost (pp)', fontsize=FONT['label'])
    ax.set_ylabel('$\\Delta$(TopVar $-$ Random), Random Forest (pp)', fontsize=FONT['label'])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.spines['left'].set_color(DARK_SEAGREEN)
    ax.spines['bottom'].set_color(DARK_SEAGREEN)

    # Quadrant labels
    ax.text(0.95, 0.95, 'Both\nhelpful', transform=ax.transAxes, fontsize=7,
            ha='right', va='top', color=COUPLED_GREEN, alpha=0.4, style='italic')
    ax.text(0.05, 0.05, 'Both\nharmful', transform=ax.transAxes, fontsize=7,
            ha='left', va='bottom', color=ANTI_YLGREEN, alpha=0.5, style='italic')

    ax.text(-0.02, 1.03, 'a', transform=ax.transAxes, fontsize=FONT['panel'],
            fontweight='bold', va='bottom', ha='right', color='black')


# ============================================================================
# PANEL B: Cross-K Regime Stability Heatmap
# ============================================================================

def panel_b_cross_k_heatmap(ax, stability_df: pd.DataFrame, regime_df: pd.DataFrame):
    """Heatmap: DI at K=1,5,10,20% per view, sorted by consensus DI."""
    print("    Panel b: Cross-K heatmap")
    
    # Get consensus DI for sorting
    di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df['DI_10pct_consensus']
    ))
    
    k_values = sorted(stability_df['K_pct'].unique())
    views = stability_df.groupby('view_id').first().reset_index()
    views['sort_di'] = views['view_id'].map(di_map)
    views = views.sort_values('sort_di', ascending=True)
    view_order = views['view_id'].tolist()
    
    # Build matrix
    n_views = len(view_order)
    n_k = len(k_values)
    matrix = np.full((n_views, n_k), np.nan)
    
    for _, row in stability_df.iterrows():
        vid = row['view_id']
        if vid in view_order:
            i = view_order.index(vid)
            j = k_values.index(row['K_pct'])
            matrix[i, j] = row['DI']
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap=DI_CMAP, norm=DI_NORM, aspect='auto',
                   interpolation='nearest')

    # Keep right-side stability markers/legend *inside* the axes so savefig(bbox='tight')
    # doesn't expand the whole figure width with extra whitespace.
    extra_x = 0.9   # space for '=' / '~' markers (right side)
    extra_left = 1.25  # internal left margin so y tick labels don't spill into panel a
    ax.set_xlim(-0.5 - extra_left, (n_k - 0.5) + extra_x)
    # imshow defaults to origin='upper' so y-axis is inverted; extend upward for legend line.
    ax.set_ylim(n_views - 0.5, -1.4)
    
    # Annotate cells
    for i in range(n_views):
        for j in range(n_k):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.75 or val > 1.04 else DARK_SEAGREEN
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7.5, color=text_color, fontweight='bold')
    
    # Stability indicator (right side)
    for i, vid in enumerate(view_order):
        sub = stability_df[stability_df['view_id'] == vid]
        if len(sub) > 0:
            stable = sub['cross_k_stable'].iloc[0]
            symbol = '=' if stable else '~'
            color = COUPLED_GREEN if stable else ANTI_YLGREEN
            ax.text(n_k + 0.1, i, symbol, fontsize=9, ha='left', va='center',
                    color=color, fontweight='bold')
    
    # Labels
    short_labels = []
    for vid in view_order:
        parts = vid.split(':')
        ds_short = DS_SHORT.get(parts[0], parts[0])
        short_labels.append(f"{ds_short}:{parts[1]}")
    
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f'K={k}%' for k in k_values], fontsize=8)
    ax.set_yticks(range(n_views))
    ax.set_yticklabels(short_labels, fontsize=7)
    # Prevent overlap with panel a: put y tick labels inside this axes
    for t in ax.get_yticklabels():
        t.set_ha('left')
    ax.tick_params(axis='y', pad=-2)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=0)
    
    # Stable legend
    ax.text(n_k + 0.1, -1, '= stable\n~ shifts', fontsize=7.5, ha='left',
            va='bottom', color=DARK_SEAGREEN)
    
    ax.text(0.02, 1.03, 'b', transform=ax.transAxes, fontsize=FONT['panel'],
            fontweight='bold', va='bottom', ha='right', color='black')


# ============================================================================
# PANEL C: SHAP Advantage Lollipop
# ============================================================================

def panel_c_shap_advantage(ax, ablation_df: pd.DataFrame, regime_df: pd.DataFrame):
    """Horizontal lollipop: SHAP advantage over Variance per view."""
    print("    Panel c: SHAP advantage lollipop")
    
    di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df['DI_10pct_consensus']
    ))
    
    df = ablation_df.copy()
    # Results reports XGBoost-only mean SHAP advantage (delta_shap_minus_var).
    xgb_col = _find_col(df, [
        "xgb_delta_shap_minus_var", "xgb_delta_shap_var",
        "delta_shap_minus_var_xgb", "delta_shap_minus_var"
    ])
    if xgb_col is None:
        raise ValueError(f"No XGB SHAP-vs-Var delta column found. cols={list(df.columns)}")

    vals = pd.to_numeric(df[xgb_col], errors="coerce")
    # Handle either fraction units (0.082) or pp units (8.2)
    scale = 100.0 if np.nanmax(np.abs(vals)) <= 1.0 else 1.0

    df["shap_adv_mean"] = vals * scale
    df['sort_key'] = df.apply(lambda r: di_map.get(f"{r['dataset']}:{r['view']}", 1.0), axis=1)
    df = df.sort_values('sort_key', ascending=True).reset_index(drop=True)
    
    y_pos = np.arange(len(df))
    
    for i, (_, row) in enumerate(df.iterrows()):
        key = f"{row['dataset']}:{row['view']}"
        di = di_map.get(key, 1.0)
        color = DI_CMAP(DI_NORM(di))
        adv = row['shap_adv_mean']
        
        ax.plot([0, adv], [i, i], color=color, linewidth=2.5,
                solid_capstyle='round', alpha=0.7, zorder=2)
        marker = DS_MARKERS.get(row['dataset'], 'o')
        ax.scatter(adv, i, marker=marker, s=70, c=[color],
                   edgecolors='white', linewidths=1.0, zorder=4)
    
    ax.axvline(0, color=GREY, linewidth=1, alpha=0.4)
    
    labels = [f"{DS_SHORT.get(r['dataset'], r['dataset'])}:{r['view']}" for _, r in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('$\\Delta$(TopSHAP $-$ TopVar) in pp (XGBoost)', fontsize=FONT['label'] - 1)
    ax.set_ylim(-0.8, len(df) - 0.2)
    ax.spines['left'].set_color(DARK_SEAGREEN)
    ax.spines['bottom'].set_color(DARK_SEAGREEN)
    
    # Summary
    mean_adv = df['shap_adv_mean'].mean()
    n_pos = (df['shap_adv_mean'] > 0).sum()
    ax.text(0.97, 0.03, f'Mean: {mean_adv:.1f} pp\n{n_pos}/{len(df)} TopSHAP > TopVar',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
            color=DARK_SEAGREEN,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=LIGHT_GREY, alpha=0.9))
    
    ax.text(-0.15, 1.03, 'c', transform=ax.transAxes, fontsize=FONT['panel'],
            fontweight='bold', va='bottom', ha='right', color='black')


# ============================================================================
# PANEL D: Dot-Matrix Model Agreement
# ============================================================================

def panel_d_dot_matrix(ax, agreement_df: pd.DataFrame, regime_df: pd.DataFrame):
    """
    14x2 dot matrix: views x models (XGB, RF).
    Each dot = DI-colored circle with regime label inside.
    Bridge between dots: solid = agree, dashed = disagree.
    Sorted by consensus DI.
    """
    print("    Panel d: Dot-matrix model agreement")
    
    # Get per-model DI from regime_consensus
    xgb_di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df['DI_10pct_xgb_bal']
    ))
    rf_di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df['DI_10pct_rf']
    ))
    cons_di_map = dict(zip(
        regime_df['dataset'] + ':' + regime_df['view'],
        regime_df['DI_10pct_consensus']
    ))
    
    df = agreement_df.copy()
    df['cons_di'] = df['view_id'].map(cons_di_map)
    df = df.sort_values('cons_di', ascending=True).reset_index(drop=True)
    
    n = len(df)
    x_xgb = 0.3   # x-position for XGB column
    x_rf = 0.7    # x-position for RF column
    dot_size = 320
    
    # Column headers
    ax.text(x_xgb, n + 0.3, 'XGBoost', fontsize=9, ha='center', va='bottom',
            color=DARK_SEAGREEN, fontweight='bold')
    ax.text(x_rf, n + 0.3, 'Random\nForest', fontsize=9, ha='center', va='bottom',
            color=DARK_SEAGREEN, fontweight='bold')
    
    for i, (_, row) in enumerate(df.iterrows()):
        vid = row['view_id']
        xgb_regime = row['regime_xgb_bal']
        rf_regime = row['regime_rf']
        agree = row['models_agree']
        
        xgb_di = xgb_di_map.get(vid, 1.0)
        rf_di = rf_di_map.get(vid, 1.0)
        
        xgb_color = DI_CMAP(DI_NORM(xgb_di))
        rf_color = DI_CMAP(DI_NORM(rf_di))
        
        # Bridge connecting the two dots
        if agree:
            # Solid bridge = agreement
            bridge_x = np.linspace(x_xgb + 0.04, x_rf - 0.04, 30)
            bridge_di = np.linspace(xgb_di, rf_di, 30)
            points = np.array([bridge_x, np.full(30, i)]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = [DI_CMAP(DI_NORM(d)) for d in bridge_di[:-1]]
            lc = LineCollection(segments, colors=colors, linewidths=3.0,
                               alpha=0.5, zorder=2)
            ax.add_collection(lc)
        else:
            # Dashed bridge with gap = disagreement
            gap = 0.5
            mid = (x_xgb + x_rf) / 2
            # Left segment
            ax.plot([x_xgb + 0.04, mid - 0.04], [i, i],
                    color=GREY, linewidth=1.5, alpha=0.4, linestyle='--', zorder=2)
            # Right segment
            ax.plot([mid + 0.04, x_rf - 0.04], [i, i],
                    color=GREY, linewidth=1.5, alpha=0.4, linestyle='--', zorder=2)
            # Gap indicator
            ax.scatter(mid, i, marker='x', s=30, c=GREY, alpha=0.5, zorder=3, linewidths=1.5)
        
        # XGB dot
        ax.scatter(x_xgb, i, s=dot_size, c=[xgb_color],
                   edgecolors='white', linewidths=1.5, zorder=4, alpha=0.9)
        # Regime initial inside dot
        regime_initial = xgb_regime[0]  # C, M, or A
        text_color = 'white' if xgb_di < 0.80 else DARK_SEAGREEN
        ax.text(x_xgb, i, regime_initial, fontsize=8, ha='center', va='center',
                color=text_color, fontweight='bold', zorder=5)
        
        # RF dot
        ax.scatter(x_rf, i, s=dot_size, c=[rf_color],
                   edgecolors='white', linewidths=1.5, zorder=4, alpha=0.9)
        regime_initial = rf_regime[0]
        text_color = 'white' if rf_di < 0.80 else DARK_SEAGREEN
        ax.text(x_rf, i, regime_initial, fontsize=8, ha='center', va='center',
                color=text_color, fontweight='bold', zorder=5)
        
        # View label (left)
        parts = vid.split(':')
        label = f"{DS_SHORT.get(parts[0], parts[0])}:{parts[1]}"
        ax.text(-0.02, i, label, fontsize=7.5, ha='right', va='center',
                color=DARK_SEAGREEN, fontweight='medium')
        
        # Agreement indicator (right)
        if agree:
            ax.text(1.02, i, '=', fontsize=10, ha='left', va='center',
                    color=COUPLED_GREEN, fontweight='bold')
        else:
            ax.text(1.02, i, '=/=', fontsize=8, ha='left', va='center',
                    color=GREY, fontweight='bold')
    
    # Summary
    n_agree = df['models_agree'].sum()
    ax.text(0.5, -1.5, f'{n_agree}/{n} views agree',
            fontsize=8.5, ha='center', va='top', color=DARK_SEAGREEN,
            fontweight='bold')
    
    # Legend
    legend_text = 'C = Coupled   M = Mixed   A = Anti-aligned'
    ax.text(0.5, -2.2, legend_text, fontsize=8, ha='center', va='top',
            color=GREY, style='italic')
    ax.text(0.5, -2.65,
            'Right annotations indicate model agreement (“=”) or disagreement (“=/=”);',
            fontsize=8, ha='center', va='top', color=GREY, style='italic')
    
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-2.8, n + 0.8)
    ax.axis('off')
    
    ax.text(-0.05, 1.02, 'd', transform=ax.transAxes, fontsize=FONT['panel'],
            fontweight='bold', va='bottom', ha='right', color='black')

# ============================================================================
# PANEL E: Modality Inconsistency Dumbbell
# ============================================================================

def panel_e_modality_dumbbell(ax, regime_df: pd.DataFrame):
    """
    Range-bar + scatter: bar width IS the inconsistency.
    - Gradient-filled horizontal bar per modality spans min→max DI
    - Dataset markers overlaid on the bar (no inline text labels)
    - Spread (Δ) annotated to the right of each bar
    - Sorted by spread descending (widest bar = most inconsistent at top)
    """
    print("    Panel e: Modality inconsistency range-bar")

    di_col = 'DI_10pct_consensus'
    EPS = 0.01  # tolerance around DI=1 for consistency classification

    df = regime_df.copy()

    # Find modalities present in 2+ datasets
    modality_datasets = df.groupby('view')['dataset'].nunique()
    shared = sorted(modality_datasets[modality_datasets >= 2].index.tolist())

    if not shared:
        ax.text(0.5, 0.5, 'No shared modalities', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color=GREY)
        return

    # Compute spread per modality and sort descending (widest bar at top)
    modality_spread = {}
    for modality in shared:
        sub = df[df['view'] == modality]
        di_vals = sub[di_col].values
        modality_spread[modality] = di_vals.max() - di_vals.min()
    shared_sorted = sorted(shared, key=lambda m: modality_spread[m], reverse=True)

    # Compute consistency per modality with epsilon tolerance
    n_consistent = 0
    modality_consistent = {}
    for modality in shared_sorted:
        sub = df[df['view'] == modality]
        di_vals = sub[di_col].values
        consistent = (all(di >= 1 + EPS for di in di_vals) or
                      all(di <= 1 - EPS for di in di_vals))
        modality_consistent[modality] = consistent
        if consistent:
            n_consistent += 1

    # Layout
    n_mod = len(shared_sorted)
    bar_height = 0.55
    y_step = 1.3

    for idx, modality in enumerate(shared_sorted):
        y = idx * y_step
        sub = df[df['view'] == modality].sort_values(di_col)
        di_min, di_max = sub[di_col].min(), sub[di_col].max()
        span = di_max - di_min

        # --- Gradient range bar via imshow ---
        # Ensure a minimum visual width so ultra-narrow bars are still visible
        bar_min = di_min - 0.005
        bar_max = di_max + 0.005
        n_pixels = 256
        gradient = np.linspace(bar_min, bar_max, n_pixels).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=DI_CMAP, norm=DI_NORM,
                  extent=[bar_min, bar_max, y - bar_height / 2, y + bar_height / 2],
                  zorder=1, alpha=0.55, interpolation='bilinear')

        # Subtle rounded-look outline
        rect = mpatches.FancyBboxPatch(
            (bar_min, y - bar_height / 2), bar_max - bar_min, bar_height,
            boxstyle=mpatches.BoxStyle.Round(pad=0.01),
            fill=False, edgecolor=DARK_SEAGREEN, linewidth=1.0, alpha=0.45, zorder=2)
        ax.add_patch(rect)

        # --- Dataset markers on top of bar ---
        n_ds = len(sub)
        # Vertical jitter within bar so markers don't stack
        jitter = np.linspace(-bar_height * 0.28, bar_height * 0.28, n_ds) if n_ds > 1 else [0.0]
        for j, (_, row) in enumerate(sub.iterrows()):
            di = row[di_col]
            marker = DS_MARKERS.get(row['dataset'], 'o')
            color = DI_CMAP(DI_NORM(di))
            ax.scatter(di, y + float(jitter[j]), marker=marker, s=100,
                       c=[color], edgecolors='white', linewidths=1.3, zorder=4)

        # --- Modality label (left of bar) ---
        ax.text(bar_min - 0.02, y, modality, ha='right', va='center',
                fontsize=10, fontweight='bold', color=DARK_SEAGREEN)

        # --- Spread annotation (right of bar) ---
        consist_sym = '=' if modality_consistent[modality] else chr(8800)  # ≠
        consist_clr = COUPLED_GREEN if modality_consistent[modality] else GREY
        ax.text(bar_max + 0.015, y + 0.05,
                f'{chr(916)}={span:.2f}',
                ha='left', va='center', fontsize=8.5, color=GREY, style='italic')
        ax.text(bar_max + 0.015, y - 0.20,
                consist_sym, ha='left', va='center', fontsize=10,
                color=consist_clr, fontweight='bold')

    # DI = 1 reference line
    ax.axvline(1.0, color=GREY, linewidth=1.0, linestyle='-', alpha=0.4, zorder=0)
    # thin label at the top
    ax.text(1.0, (n_mod - 1) * y_step + bar_height / 2 + 0.15, 'DI=1',
            ha='center', va='bottom', fontsize=7, color=GREY, alpha=0.7)

    # Consistency summary box
    ax.text(0.03, 0.97,
            f'{n_consistent}/{len(shared_sorted)} modalities\nconsistent across datasets',
            transform=ax.transAxes, fontsize=8.5, ha='left', va='top',
            fontweight='bold', color=MID_YLGREEN,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=LIGHT_GREY, alpha=0.9))

    # Dataset legend (replaces all inline labels → zero overlap)
    handles = [
        plt.Line2D([0], [0], marker=m, color='none', markerfacecolor=GREY,
                   markeredgecolor='white', markersize=7,
                   label=DS_DISPLAY.get(ds, ds))
        for ds, m in DS_MARKERS.items()
    ]
    # LHS-bottom legend placement for panel E
    ax.legend(
        handles=handles,
        loc='lower left',
        bbox_to_anchor=(0.02, 0.02),
        fontsize=9,
        handletextpad=0.3,
        borderpad=0.5,
        labelspacing=0.25,
    )

    # Axis styling
    ax.set_xlabel('DI at K = 10%', fontsize=FONT['label'] - 1)
    ax.set_xlim(0.55, 1.12)
    ax.set_ylim(-0.7, (n_mod - 1) * y_step + bar_height / 2 + 0.7)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(DARK_SEAGREEN)

    ax.text(-0.10, 1.03, 'e', transform=ax.transAxes, fontsize=FONT['panel'],
            fontweight='bold', va='bottom', ha='right', color='black')


# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================

def create_figure(outputs_dir: Path, output_path: Path):
    apply_style()
    
    print("=" * 70)
    print("CREATING FIGURE 2")
    print("Reproducibility & View-Specificity (5 panels)")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    ablation_df = load_cross_model_ablation(outputs_dir)
    stability_df = load_cross_k_stability(outputs_dir)
    agreement_df = load_regime_agreement(outputs_dir)
    regime_df = load_regime_consensus(outputs_dir)
    
    # Create figure -- 2-row layout
    print("\n[2] Building layout...")
    fig = plt.figure(figsize=(11, 9))
    
    # Asymmetric 12-column GridSpec, 2 rows
    # Row 1: a=hero scatter (7 cols) + b=cross-K heatmap (5 cols)
    # Row 2: c=SHAP advantage (4 cols) + d=dot-matrix (4 cols) + e=modality (4 cols)
    gs = GridSpec(2, 12, figure=fig,
                  height_ratios=[1.2, 1.0],
                  hspace=0.28, wspace=0.8,
                  left=0.08, right=0.95, top=0.97, bottom=0.075)
    
    # Row 1
    ax_a = fig.add_subplot(gs[0, 0:7])
    ax_b = fig.add_subplot(gs[0, 7:12])
    
    # Row 2
    ax_c = fig.add_subplot(gs[1, 0:4])
    ax_d = fig.add_subplot(gs[1, 4:8])
    ax_e = fig.add_subplot(gs[1, 8:12])

    # Small layout nudges: reduce the gap between panels a and b.
    # Expand panel a by ~4% (relative to its GridSpec position).
    pos_a = ax_a.get_position()
    new_w_a = pos_a.width * 1.04
    ax_a.set_position([pos_a.x0, pos_a.y0, new_w_a, pos_a.height])

    pos_b = ax_b.get_position()
    # Expand panel b by ~3% while keeping its right edge fixed.
    new_w_b = pos_b.width * 1.03
    ax_b.set_position([pos_b.x0 + (pos_b.width - new_w_b), pos_b.y0, new_w_b, pos_b.height])

    # Small layout nudge: shrink panel c slightly to prevent overlap into panel d
    pos_c = ax_c.get_position()
    ax_c.set_position([pos_c.x0, pos_c.y0, pos_c.width * 0.97, pos_c.height])

    # Small layout nudge: shrink panel e slightly to prevent overlap into panel d
    # Keep the right edge fixed so the gap opens between d and e.
    pos_e = ax_e.get_position()
    new_w_e = pos_e.width * 0.95
    ax_e.set_position([pos_e.x0 + (pos_e.width - new_w_e), pos_e.y0, new_w_e, pos_e.height])
    
    # Render panels
    print("\n[3] Rendering panels...")
    panel_a_xgb_rf_scatter(ax_a, ablation_df, regime_df)
    panel_b_cross_k_heatmap(ax_b, stability_df, regime_df)
    panel_c_shap_advantage(ax_c, ablation_df, regime_df)
    panel_d_dot_matrix(ax_d, agreement_df, regime_df)
    panel_e_modality_dumbbell(ax_e, regime_df)
    
    # DI colorbar at bottom
    # Keep it lower to open a clearer gap from panel x-axis labels.
    cax = fig.add_axes([0.25, 0.0005, 0.50, 0.011])
    sm = ScalarMappable(cmap=DI_CMAP, norm=DI_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    # Put label above (ticks below) to avoid overlap in tight bottom margin
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.set_label('Decoupling Index (DI)', fontsize=9, labelpad=6, color=DARK_SEAGREEN)
    cbar.ax.tick_params(labelsize=8, colors=DARK_SEAGREEN, pad=0.5)
    cbar.ax.xaxis.set_ticks([0.65, 0.80, 0.95, 1.00, 1.05])
    # Put end-member captions clearly above the bar/ticks.
    cbar.ax.text(0.0, 1.35, 'Coupled', transform=cbar.ax.transAxes,
                 fontsize=7.5, ha='left', va='bottom',
                 color=COUPLED_GREEN, fontweight='bold', fontstyle='normal')
    cbar.ax.text(1.0, 1.35, 'Anti-aligned', transform=cbar.ax.transAxes,
                 fontsize=7.5, ha='right', va='bottom',
                 color=ANTI_YLGREEN, fontweight='bold', fontstyle='normal')
    
    # Save
    print("\n[4] Saving...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    print("\n" + "=" * 70)
    print("Figure 2 complete (5 panels)")
    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 2: Reproducibility & View-Specificity")
    parser.add_argument("--outputs-dir", type=str,
                        default=r"C:\Users\ms\Desktop\var-pre\outputs")
    parser.add_argument("--output", type=str,
                        default=r"C:\Users\ms\Desktop\var-pre\outputs\figures\figure_2.png")
    args = parser.parse_args()
    
    create_figure(Path(args.outputs_dir), Path(args.output))
    print("\nDone!")