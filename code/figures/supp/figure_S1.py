#!/usr/bin/env python3
"""
Figure S1: SHAP Robustness & DI Uncertainty (4 panels)
======================================================

Panel a: Within-model SHAP stability (Jaccard across CV folds)
Panel b: Cross-model SHAP agreement (observed vs null Jaccard)
Panel c: Bootstrap DI distributions (ridgeline)
Panel d: DI +/- CI forest plot

Layout: Asymmetric 2x2 GridSpec, green-family palette, DI-sorted.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\ms\Desktop\var-pre")
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
FIGURE_OUT   = OUTPUTS_DIR / "figures"
SUPPORT_DIR  = PROJECT_ROOT / "code" / "figures" / "supp" / "figure_S1"

# Import colourlist from main figures
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
from colourlist import (
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    TEXT_PRIMARY, TEXT_SECONDARY, SPINE_COLOR, GRID_COLOR, BG_WHITE,
    GREY, GREY_LIGHTER, GREY_LIGHT,
    DS_MARKERS, DS_SHORT, DS_DISPLAY, FONT,
    bugreen, ylgreen, greens,
)

# =============================================================================
# COLOUR PALETTE
# =============================================================================

REGIME_COLORS = {
    'ANTI_ALIGNED': MID_YLGREEN,
    'MIXED':        NEUTRAL_GREEN,
    'DECOUPLED':    LIGHT_BLUEGREEN,
    'COUPLED':      MID_BLUEGREEN,
}

COLORS = {
    **REGIME_COLORS,
    'text':        TEXT_PRIMARY,
    'spine':       SPINE_COLOR,
    'grid':        GRID_COLOR,
    'null_band':   GRID_COLOR,
    'random_line': LIGHT_BLUEGREEN,
    'observed':    SPINE_COLOR,
    'null':        GRID_COLOR,
    'ridge_fill':  LIGHT_BLUEGREEN,
    'ci_bar':      MID_BLUEGREEN,
    'diamond':     TEXT_PRIMARY,
    'ref_line':    LIGHT_BLUEGREEN,
}

# DI colormap (coupled -> anti-aligned)
DI_CMAP = LinearSegmentedColormap.from_list(
    'green_di', [
        COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN,
        NEUTRAL_GREEN, LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    ], N=256
)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)

DISPLAY_NAMES = {
    'mlomics':  'MLOmics',
    'ibdmdb':   'IBDMDB',
    'ccle':     'CCLE',
    'tcga_gbm': 'TCGA-GBM',
}


def view_label(dataset: str, view: str) -> str:
    return f"{DISPLAY_NAMES.get(dataset, dataset)}/{view}"


# =============================================================================
# STYLE
# =============================================================================

def apply_style():
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          9,
        'text.color':         'black',
        'axes.labelcolor':    'black',
        'axes.titlecolor':    'black',
        'xtick.color':        'black',
        'ytick.color':        'black',
        'legend.labelcolor':  'black',
        'axes.titlesize':     11,
        'axes.titleweight':   'bold',
        'axes.labelsize':     10,
        'axes.labelweight':   'medium',
        'axes.linewidth':     0.8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'legend.fontsize':    8,
        'legend.frameon':     False,
        'figure.dpi':         150,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.1,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    })


# =============================================================================
# DATA LOADING — NO FALLBACKS
# =============================================================================

def load_regime_consensus() -> pd.DataFrame:
    """Load regime map for DI-based ordering and regime labels."""
    path = (OUTPUTS_DIR / "results" / "main_results" / "section_1_paradox_discovery" / "regime_map.csv")
    assert path.exists(), f"MISSING: {path}"
    df = pd.read_csv(path)

    di_src = None
    for c in ["DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean", "DI"]:
        if c in df.columns:
            di_src = c
            break
    if di_src is None:
        raise ValueError(f"No DI column found in regime_map.csv. cols={list(df.columns)}")

    df["DI"] = df[di_src]
    df["DI_10pct_consensus"] = df["DI"]  # legacy alias (now Results DI)
    if "consensus_regime" not in df.columns and "regime" in df.columns:
        df["consensus_regime"] = df["regime"]

    print(f"  regime_map: {len(df)} views")
    return df


def load_shap_stability() -> pd.DataFrame:
    p = (
        OUTPUTS_DIR
        / "results"
        / "main_results"
        / "section_2_regime_characterisation"
        / "source_tables"
        / "shap_stability_summary.csv"
    )
    assert p.exists(), f"MISSING: {p}"
    df = pd.read_csv(p)
    print(f"  shap_stability: {len(df)} rows")
    return df


def load_shap_agreement() -> pd.DataFrame:
    p = (
        OUTPUTS_DIR
        / "results"
        / "main_results"
        / "section_2_regime_characterisation"
        / "source_tables"
        / "shap_agreement_summary.csv"
    )
    assert p.exists(), f"MISSING: {p}"
    df = pd.read_csv(p)
    print(f"  shap_agreement: {len(df)} rows")
    return df


def load_di_per_repeat() -> pd.DataFrame:
    """Load all di_per_repeat__*.csv files and concatenate."""
    uncertainty_dir = OUTPUTS_DIR / "results" / "main_results" / "section_2_regime_characterisation" / "source_tables" / "uncertainty"
    assert uncertainty_dir.exists(), f"MISSING: {uncertainty_dir}"

    files = sorted(uncertainty_dir.glob("di_per_repeat__*.csv"))
    assert len(files) > 0, f"No di_per_repeat__*.csv files found in {uncertainty_dir}"

    parts = []
    for f in files:
        df = pd.read_csv(f)
        # Extract dataset/view from filename if not in columns
        if 'dataset' not in df.columns or 'view' not in df.columns:
            stem = f.stem  # e.g. di_per_repeat__mlomics__methylation
            tokens = stem.replace("di_per_repeat__", "").split("__")
            if len(tokens) >= 2:
                df['dataset'] = tokens[0]
                df['view'] = tokens[1]
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    n_views = combined.groupby(['dataset', 'view']).ngroups
    print(f"  di_per_repeat: {len(combined)} rows across {n_views} views")
    return combined


# =============================================================================
# HELPERS
# =============================================================================

def get_di_for_sorting(regime_df: pd.DataFrame) -> pd.DataFrame:
    """Get DI_10pct_consensus per view for sorting, return sorted df."""
    df = regime_df[['dataset', 'view', 'DI_10pct_consensus', 'consensus_regime']].copy()
    df['view_label'] = df.apply(lambda r: view_label(r['dataset'], r['view']), axis=1)
    df = df.sort_values('DI_10pct_consensus', ascending=True).reset_index(drop=True)
    return df


def di_to_color(di_val: float) -> str:
    """Map DI value to color using the DI colormap."""
    rgba = DI_CMAP(DI_NORM(di_val))
    return rgba


# =============================================================================
# PANEL A: Within-model SHAP stability — Horizontal Gradient Bar + CI
# =============================================================================

def plot_panel_a(ax, stability_df: pd.DataFrame, sort_df: pd.DataFrame):
    """
    Horizontal gradient bar chart: Jaccard stability per view at K=10%,
    sorted by DI. Bars filled with DI-gradient colour, CI whiskers if available.
    Visually distinct from the airy dumbbell in panel b.
    """
    # Filter to K=10% and preferred model
    k_col = next((c for c in ['k_pct', 'K_pct'] if c in stability_df.columns), None)
    assert k_col is not None, f"No k_pct column in stability. Cols: {list(stability_df.columns)}"

    df = stability_df[stability_df[k_col].astype(int) == 10].copy()
    assert len(df) > 0, "No K=10% rows in shap_stability_summary"

    # Prefer xgb_bal model if available
    if 'model' in df.columns:
        models = df['model'].unique()
        # Use xgb_bal if available
        if 'xgb_bal' in models:
            df = df[df['model'] == 'xgb_bal'].copy()
        else:
            df = df[df['model'] == models[0]].copy()

    # Find Jaccard column
    j_col = next((c for c in ['jaccard_mean', 'mean_jaccard', 'jaccard',
                               'pairwise_jaccard_mean'] if c in df.columns), None)
    assert j_col is not None, f"No Jaccard column in stability. Cols: {list(df.columns)}"

    # Find CI columns if available
    ci_lo_col = next((c for c in ['ci_lo', 'jaccard_ci_lo', 'lo'] if c in df.columns), None)
    ci_hi_col = next((c for c in ['ci_hi', 'jaccard_ci_hi', 'hi'] if c in df.columns), None)
    has_ci = ci_lo_col is not None and ci_hi_col is not None

    # Merge with sort order
    df = df.merge(sort_df[['dataset', 'view', 'DI_10pct_consensus', 'consensus_regime', 'view_label']],
                  on=['dataset', 'view'], how='inner')
    df = df.sort_values('DI_10pct_consensus', ascending=True).reset_index(drop=True)

    assert len(df) > 0, "No views matched between stability and regime_consensus"

    y_pos = np.arange(len(df))
    bar_height = 0.65

    for i, (_, row) in enumerate(df.iterrows()):
        di = row['DI_10pct_consensus']
        jaccard = row[j_col]
        color = di_to_color(di)

        # Solid filled bar — the visual anchor
        ax.barh(i, jaccard, height=bar_height, color=color, alpha=0.75,
                edgecolor='white', linewidth=0.6, zorder=3)

        # CI whiskers if available
        if has_ci:
            lo, hi = row[ci_lo_col], row[ci_hi_col]
            ax.plot([lo, hi], [i, i], color=COLORS['text'], linewidth=1.2,
                    alpha=0.5, zorder=4, solid_capstyle='round')
            # Whisker caps
            cap_h = bar_height * 0.35
            ax.plot([lo, lo], [i - cap_h / 2, i + cap_h / 2],
                    color=COLORS['text'], linewidth=0.9, alpha=0.5, zorder=4)
            ax.plot([hi, hi], [i - cap_h / 2, i + cap_h / 2],
                    color=COLORS['text'], linewidth=0.9, alpha=0.5, zorder=4)

        # Value annotation — inside bar if wide enough, else outside
        if jaccard > 0.25:
            ax.text(jaccard - 0.02, i, f'{jaccard:.2f}', va='center', ha='right',
                    fontsize=7, color='black', fontweight='bold', zorder=5)
        else:
            ax.text(jaccard + 0.01, i, f'{jaccard:.2f}', va='center', ha='left',
                    fontsize=7, color='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['view_label'], fontsize=7)
    ax.set_xlabel('Pairwise Jaccard (CV folds, K = 10%)', fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.5, len(df) - 0.5)

    # Subtle grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linewidth=0.5, alpha=0.3, color=COLORS['grid'])

    ax.set_title('a', loc='left', pad=10, fontsize=12,
                 fontweight='bold', color='black')

    ax.spines['left'].set_color(COLORS['spine'])
    ax.spines['bottom'].set_color(COLORS['spine'])


# =============================================================================
# PANEL B: Cross-model SHAP agreement — Paired Dumbbell
# =============================================================================

def plot_panel_b(ax, agreement_df: pd.DataFrame, sort_df: pd.DataFrame):
    """
    Paired dumbbell chart: observed vs null Jaccard (XGB vs RF), per view.
    Connected dots with ratio annotated.
    """
    # Filter to K=10%
    k_col = next((c for c in ['k_pct', 'K_pct'] if c in agreement_df.columns), None)
    assert k_col is not None, f"No k_pct column. Cols: {list(agreement_df.columns)}"

    df = agreement_df[agreement_df[k_col].astype(int) == 10].copy()
    assert len(df) > 0, "No K=10% rows in shap_agreement_summary"

    # Filter to per_repeat_shap method if method column exists
    if 'method' in df.columns:
        methods = df['method'].unique()
        # Use per_repeat_shap if available
        if 'per_repeat_shap' in methods:
            df = df[df['method'] == 'per_repeat_shap'].copy()
        elif 'per_model_rank' in methods:
            df = df[df['method'] == 'per_model_rank'].copy()

    # Find observed and null columns
    obs_col = next((c for c in ['jaccard_mean', 'jaccard', 'mean_jaccard']
                    if c in df.columns), None)
    null_col = next((c for c in ['jaccard_null', 'null_jaccard']
                     if c in df.columns), None)
    assert obs_col is not None, f"No observed Jaccard col. Cols: {list(df.columns)}"
    assert null_col is not None, f"No null Jaccard col. Cols: {list(df.columns)}"

    # Merge with sort order
    df = df.merge(sort_df[['dataset', 'view', 'DI_10pct_consensus', 'consensus_regime', 'view_label']],
                  on=['dataset', 'view'], how='inner')
    df = df.sort_values('DI_10pct_consensus', ascending=True).reset_index(drop=True)

    assert len(df) > 0, "No views matched between agreement and regime_consensus"

    y_pos = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        obs = row[obs_col]
        null = row[null_col]
        di = row['DI_10pct_consensus']
        color = di_to_color(di)

        # Connecting line
        ax.plot([null, obs], [i, i], color=color, linewidth=1.2, alpha=0.6,
                solid_capstyle='round')

        # Null dot (open circle)
        ax.scatter(null, i, s=50, facecolors='none', edgecolors=COLORS['null'],
                   linewidth=1.0, zorder=4, marker='o')

        # Observed dot (filled)
        ax.scatter(obs, i, s=60, color=color, edgecolor='white',
                   linewidth=0.8, zorder=5)

        # Ratio badge
        if null > 0:
            ratio = obs / null
            badge_x = max(obs, null) + 0.01
            ax.text(badge_x, i, f'{ratio:.1f}×', fontsize=6.5,
                    va='center', ha='left', color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor=color, alpha=0.8, linewidth=0.5))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['view_label'], fontsize=7)
    ax.set_xlabel('Jaccard (XGBoost vs Random Forest, K = 10%)', fontsize=9)
    ax.set_ylim(-0.5, len(df) - 0.5)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['observed'],
                   markeredgecolor='white', markersize=8, label='Observed'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor=COLORS['null'], markersize=8,
                   markeredgewidth=1.0, label='Null expectation'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0),
              fontsize=7, borderaxespad=0.0)

    # Subtle grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linewidth=0.5, alpha=0.3, color=COLORS['grid'])

    ax.set_title('b', loc='left', pad=10, fontsize=12,
                 fontweight='bold', color='black')

    ax.spines['left'].set_color(COLORS['spine'])
    ax.spines['bottom'].set_color(COLORS['spine'])


# =============================================================================
# PANEL C: Bootstrap DI distributions — Ridgeline (Joy Plot)
# =============================================================================

def plot_panel_c(ax, di_repeat_df: pd.DataFrame, sort_df: pd.DataFrame):
    """
    Ridgeline (joy plot) of DI distributions from CV repeats at K=10%.
    Each view gets a KDE ridge, sorted by median DI, colored by DI gradient.
    """
    # Filter to K=10%
    df = di_repeat_df[di_repeat_df['k_pct'] == 10].copy()
    assert len(df) > 0, "No K=10% rows in di_per_repeat data"

    # Merge with sort order for DI sorting
    df = df.merge(sort_df[['dataset', 'view', 'DI_10pct_consensus', 'view_label']],
                  on=['dataset', 'view'], how='inner')

    # Get unique views sorted by DI
    view_order = (df.groupby(['dataset', 'view', 'view_label', 'DI_10pct_consensus'])
                  .size().reset_index(name='n')
                  .sort_values('DI_10pct_consensus', ascending=True))

    n_views = len(view_order)
    assert n_views > 0, "No views with DI data for ridgeline"

    # Ridgeline parameters
    ridge_height = 0.7      # vertical spacing between ridges
    ridge_scale  = 2.5      # KDE amplitude scaling
    di_grid = np.linspace(0.45, 1.20, 300)

    for i, (_, vrow) in enumerate(view_order.iterrows()):
        ds, vw = vrow['dataset'], vrow['view']
        label = vrow['view_label']
        di_consensus = vrow['DI_10pct_consensus']

        mask = (df['dataset'] == ds) & (df['view'] == vw)
        di_vals = df.loc[mask, 'DI'].values

        if len(di_vals) < 3:
            continue

        # KDE
        try:
            kde = stats.gaussian_kde(di_vals, bw_method='silverman')
            density = kde(di_grid)
            density = density / density.max() * ridge_scale * ridge_height
        except Exception:
            continue

        baseline = i * ridge_height
        color = di_to_color(di_consensus)

        # Fill
        ax.fill_between(di_grid, baseline, baseline + density,
                         color=color, alpha=0.45, zorder=n_views - i + 1)
        # Outline
        ax.plot(di_grid, baseline + density, color=color,
                linewidth=1.2, alpha=0.9, zorder=n_views - i + 2)
        # Baseline
        ax.plot(di_grid, np.full_like(di_grid, baseline),
                color=color, linewidth=0.5, alpha=0.3)

        # View label on left
        ax.text(0.44, baseline + ridge_height * 0.15, label,
                fontsize=6.5, ha='right', va='bottom', color='black',
                fontweight='normal')

        # Median tick mark
        med = np.median(di_vals)
        ax.plot([med, med], [baseline, baseline + ridge_height * 0.3],
                color=COLORS['text'], linewidth=1.0, alpha=0.7,
                zorder=n_views - i + 3)

    # DI = 1 reference line
    ax.axvline(x=1.0, color=COLORS['ref_line'], linestyle='--',
               linewidth=0.9, alpha=0.7, zorder=0)
    # Place label in axes coords to avoid ridge overlap
    ax.text(0.98, 0.98, 'DI = 1\n(random)',
            transform=ax.transAxes,
            fontsize=7, ha='right', va='top', color='black',
            style='italic')

    # Anti-aligned zone
    ax.axvspan(1.0, 1.20, alpha=0.10, color=LIGHT_YLGREEN, zorder=0)

    ax.set_xlim(0.45, 1.20)
    ax.set_ylim(-0.2, n_views * ridge_height + 0.3)
    ax.set_xlabel('Decoupling Index (DI)', fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel('')

    ax.set_title('c', loc='left', pad=10, fontsize=12,
                 fontweight='bold', color='black')

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['spine'])


# =============================================================================
# PANEL D: DI ± CI — Forest Plot (floating bar + diamond)
# =============================================================================

def plot_panel_d(ax, di_repeat_df: pd.DataFrame, sort_df: pd.DataFrame):
    """
    Forest plot: DI point estimate + 95% percentile interval per view at K=10%.
    Diamond centre point, DI=1 reference line with danger zone shading.
    """
    # Filter to K=10%
    df = di_repeat_df[di_repeat_df['k_pct'] == 10].copy()
    assert len(df) > 0, "No K=10% rows in di_per_repeat data"

    # Aggregate per view
    agg_rows = []
    for (ds, vw), g in df.groupby(['dataset', 'view']):
        di_vals = g['DI'].values
        n = len(di_vals)
        agg_rows.append({
            'dataset': ds, 'view': vw,
            'DI_mean': np.mean(di_vals),
            'DI_median': np.median(di_vals),
            'DI_lo': np.percentile(di_vals, 2.5) if n > 1 else np.mean(di_vals),
            'DI_hi': np.percentile(di_vals, 97.5) if n > 1 else np.mean(di_vals),
            'n_repeats': n,
        })
    agg = pd.DataFrame(agg_rows)

    # Merge with sort order
    agg = agg.merge(sort_df[['dataset', 'view', 'DI_10pct_consensus',
                              'consensus_regime', 'view_label']],
                    on=['dataset', 'view'], how='inner')
    agg = agg.sort_values('DI_10pct_consensus', ascending=True).reset_index(drop=True)

    assert len(agg) > 0, "No views matched for forest plot"

    y_pos = np.arange(len(agg))

    for i, (_, row) in enumerate(agg.iterrows()):
        di = row['DI_10pct_consensus']
        lo, hi = row['DI_lo'], row['DI_hi']
        mid = row['DI_mean']
        color = di_to_color(di)
        regime = row['consensus_regime']

        # Confidence interval bar
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.2, alpha=0.6,
                solid_capstyle='round', zorder=3)

        # Diamond centre point
        ax.scatter(mid, i, s=60, marker='D', color=color,
                   edgecolor='white', linewidth=0.8, zorder=5)

        # CI text: nudge away from marker to avoid overlap in dense rows
        ci_text = f'{mid:.2f}'
        ax.text(max(hi, mid) + 0.028, i + 0.10, ci_text,
                fontsize=6.5, va='center', ha='left',
                color='black', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.75, boxstyle='round,pad=0.12'))

    # DI = 1 reference
    ax.axvline(x=1.0, color=COLORS['ref_line'], linestyle='--',
               linewidth=0.9, alpha=0.7, label='DI = 1 (random)')

    # Anti-aligned zone shading
    ax.axvspan(1.0, 1.15, alpha=0.10, color=LIGHT_YLGREEN,
               label='Anti-aligned zone')

    # Coupled zone shading
    ax.axvspan(0.50, 0.85, alpha=0.12, color=LIGHT_BLUEGREEN,
               label='Coupled zone')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg['view_label'], fontsize=6.5)
    ax.set_xlabel('DI at K = 10%', fontsize=9)
    ax.set_xlim(0.50, 1.15)
    ax.set_ylim(-0.5, len(agg) - 0.5)

    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=6.5, borderaxespad=0.0)

    # Subtle grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linewidth=0.5, alpha=0.3, color=COLORS['grid'])

    ax.set_title('d', loc='left', pad=10, fontsize=12,
                 fontweight='bold', color='black')

    ax.spines['left'].set_color(COLORS['spine'])
    ax.spines['bottom'].set_color(COLORS['spine'])


# =============================================================================
# MAIN FIGURE ASSEMBLY
# =============================================================================

def create_figure():
    apply_style()

    print("Figure S1: SHAP Robustness & DI Uncertainty")

    # ── Load data ──
    print("Loading data...")
    regime_df   = load_regime_consensus()
    stability   = load_shap_stability()
    agreement   = load_shap_agreement()
    di_repeat   = load_di_per_repeat()

    # ── Sort order ──
    sort_df = get_di_for_sorting(regime_df)

    # ── Create figure ──
    fig = plt.figure(figsize=(10, 9))

    # 12-column asymmetric grid — matches main figure design philosophy
    # Row 0: Panel a (5 cols, compact bars) | Panel b (7 cols, airy dumbbells)
    # Row 1: Panel c (8 cols, HERO ridgeline) | Panel d (4 cols, tight forest)
    gs = GridSpec(2, 12, figure=fig,
                  height_ratios=[1, 1.3],
                  hspace=0.35, wspace=1.0)

    ax_a = fig.add_subplot(gs[0, 0:5])     # narrower — solid bars are compact
    ax_b = fig.add_subplot(gs[0, 6:12])    # wider — ratio badges need space
    ax_c = fig.add_subplot(gs[1, 0:8])     # HERO — ridgeline gets most width
    ax_d = fig.add_subplot(gs[1, 9:12])    # tight — forest plot is information-dense

    # Give panel d's long y-labels a bit more room by shrinking panel c slightly
    pos_c = ax_c.get_position()
    ax_c.set_position([pos_c.x0, pos_c.y0, pos_c.width * 0.95, pos_c.height])

    # ── Plot panels ──
    plot_panel_a(ax_a, stability, sort_df)
    plot_panel_b(ax_b, agreement, sort_df)
    plot_panel_c(ax_c, di_repeat, sort_df)
    plot_panel_d(ax_d, di_repeat, sort_df)

    # ── Shared DI colorbar at bottom ──
    cax = fig.add_axes([0.25, 0.02, 0.50, 0.012])
    sm = ScalarMappable(cmap=DI_CMAP, norm=DI_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Decoupling Index (DI)', fontsize=9, labelpad=3,
                   color='black')
    cbar.ax.tick_params(labelsize=7, colors='black')
    cbar.ax.xaxis.set_ticks([0.60, 0.70, 0.80, 0.90, 1.00, 1.10])
    cbar.ax.text(0.0, -2.5, 'Coupled', transform=cbar.ax.transAxes,
                 fontsize=7, ha='left', color='black', style='italic')
    cbar.ax.text(1.0, -2.5, 'Anti-aligned', transform=cbar.ax.transAxes,
                 fontsize=7, ha='right', color='black', style='italic')

    # ── Save ──
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = FIGURE_OUT / "figure_s1.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved: {out_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    create_figure()