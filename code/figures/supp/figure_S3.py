#!/usr/bin/env python3
"""
Figure S3: Full Ablation & Permutation Robustness (4 panels)
=============================================================

Panel a: Ablation heatmap Delta(TopVar-Random) (pp) -- diverging green, views x K
Panel b: Ablation heatmap Delta(TopSHAP-TopVar) (pp) -- sequential green
Panel c: Permutation null Delta(TopSHAP-TopVar): CI bars + observed overlay
Panel d: Seed robustness (n=100 seeds): beeswarm + box
"""

import sys
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\ms\Desktop\var-pre")
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
RESULTS_DIR  = OUTPUTS_DIR / "results"
FIGURE_OUT   = OUTPUTS_DIR / "figures"

# Exact file paths
P_ABLATION_XGB = (RESULTS_DIR / "main_results" / "section_4_consequences"
                  / "source_tables" / "ablation_master_summary__xgb.csv")
P_PERM_SUMMARY = (RESULTS_DIR / "main_results" / "section_3_mechanism"
                  / "source_tables" / "perm_summary.csv")
P_REGIME       = (RESULTS_DIR / "main_results" / "section_1_paradox_discovery"
                  / "regime_map.csv")
P_SEED_LONG    = (RESULTS_DIR / "main_results" / "section_3_mechanism"
                  / "source_tables" / "label_perm_long.csv.gz")

# Import shared palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
from colourlist import (
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    TEXT_PRIMARY, TEXT_SECONDARY, SPINE_COLOR, GRID_COLOR, BG_WHITE,
    GREY, GREY_LIGHTER, GREY_LIGHT,
    DS_DISPLAY,
)

# =============================================================================
# COLOUR PALETTE
# =============================================================================

COLORS = {
    'text':       TEXT_PRIMARY,
    'spine':      SPINE_COLOR,
    'grid':       GRID_COLOR,
    'null_fill':  GRID_COLOR,
    'null_edge':  LIGHT_BLUEGREEN,
    'observed':   TEXT_PRIMARY,
}

DI_CMAP = LinearSegmentedColormap.from_list('di_green', [
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN,
    NEUTRAL_GREEN, LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
], N=256)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)

# Diverging green cmap for panel a (Δ can be negative or positive)
DIVGREEN_CMAP = LinearSegmentedColormap.from_list('divgreen', [
    ANTI_YLGREEN, MID_YLGREEN, '#E0F7FA', '#ffffff',
    '#E0F7FA', LIGHT_BLUEGREEN, COUPLED_GREEN,
], N=256)

# Sequential green cmap for panel b (SHAP advantage)
SEQGREEN_CMAP = LinearSegmentedColormap.from_list('seqgreen', [
    '#ffffff', '#E0F7FA', '#80CBC4', '#4DB6AC', COUPLED_GREEN,
], N=256)

def view_label(dataset: str, view: str) -> str:
    return f"{DS_DISPLAY.get(dataset, dataset)}/{view}"


# =============================================================================
# STYLE
# =============================================================================

def apply_style():
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          9,
        'axes.titlesize':     11,
        'axes.titleweight':   'bold',
        'axes.labelsize':     10,
        'axes.labelweight':   'medium',
        'axes.linewidth':     1.0,
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
# DATA LOADING
# =============================================================================

def load_regime() -> pd.DataFrame:
    assert P_REGIME.exists(), f"MISSING: {P_REGIME}"
    df = pd.read_csv(P_REGIME)

    di_src = None
    for c in ["DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean", "DI"]:
        if c in df.columns:
            di_src = c
            break
    if di_src is None:
        raise ValueError(f"No DI column found in regime_map.csv. cols={list(df.columns)}")

    df["DI"] = df[di_src]
    df["DI_10pct_consensus"] = df["DI"]  # legacy alias expected downstream
    if "consensus_regime" not in df.columns and "regime" in df.columns:
        df["consensus_regime"] = df["regime"]

    print(f"  regime_map: {len(df)} views")
    return df


def load_ablation_xgb() -> pd.DataFrame:
    assert P_ABLATION_XGB.exists(), f"MISSING: {P_ABLATION_XGB}"
    df = pd.read_csv(P_ABLATION_XGB)
    print(f"  ablation_xgb: {len(df)} rows")
    return df


def load_perm_summary() -> pd.DataFrame:
    assert P_PERM_SUMMARY.exists(), f"MISSING: {P_PERM_SUMMARY}"
    df = pd.read_csv(P_PERM_SUMMARY)
    print(f"  perm_summary: {len(df)} rows")
    return df


def load_seed_long() -> pd.DataFrame:
    assert P_SEED_LONG.exists(), f"MISSING: {P_SEED_LONG}"
    df = pd.read_csv(P_SEED_LONG)
    print(f"  seed_long: {len(df)} rows")
    return df


# =============================================================================
# HELPERS
# =============================================================================

def get_di_sort(regime_df: pd.DataFrame) -> pd.DataFrame:
    df = regime_df[['dataset', 'view', 'DI_10pct_consensus', 'consensus_regime']].copy()
    df['view_label'] = df.apply(lambda r: view_label(r['dataset'], r['view']), axis=1)
    df = df.sort_values('DI_10pct_consensus', ascending=True).reset_index(drop=True)
    return df


def build_heatmap_matrix(abl: pd.DataFrame, sort_df: pd.DataFrame,
                          value_col: str,
                          k_only: int | None = None) -> Tuple[np.ndarray, list, list]:
    """Build views × K matrix. Uses EXACT column names from ablation CSV.
       If k_only is set, restrict to that single K (e.g., 10)."""
    sub = abl[abl['metric'] == 'balanced_accuracy'].copy()
    sub = sub.merge(sort_df[['dataset', 'view', 'view_label']],
                     on=['dataset', 'view'], how='inner')

    k_avail = sorted(sub['K_pct'].dropna().astype(int).unique())
    if k_only is not None:
        k_only = int(k_only)
        assert k_only in k_avail, f"Requested K={k_only} not in available K={k_avail}"
        k_use = [k_only]
    else:
        k_use = [k for k in [1, 5, 10, 20] if k in k_avail]
        assert len(k_use) > 0, f"No K values in {k_avail}"

    view_order = sort_df['view_label'].tolist()
    view_order = [v for v in view_order if v in sub['view_label'].values]

    matrix = np.full((len(view_order), len(k_use)), np.nan)
    for i, vl in enumerate(view_order):
        for j, k in enumerate(k_use):
            mask = (sub['view_label'] == vl) & (sub['K_pct'].astype(int) == k)
            vals = sub.loc[mask, value_col]
            if len(vals) > 0:
                matrix[i, j] = float(vals.iloc[0])

    col_labels = [f'K={k}%' for k in k_use]
    return matrix, view_order, col_labels


# =============================================================================
# PANEL A: Δ(TopVar−Random) Heatmap — diverging green, centred at 0
# =============================================================================

def plot_panel_a(ax, abl: pd.DataFrame, sort_df: pd.DataFrame):
    matrix, row_labels, col_labels = build_heatmap_matrix(
        abl, sort_df, 'delta_var_random_mean', k_only=10
    )
    matrix = matrix * 100  # to pp

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(matrix, aspect='auto', cmap=DIVGREEN_CMAP, norm=norm,
                    interpolation='nearest')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if abs(val) > vmax * 0.55 else COLORS['text']
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=6.5, color=text_color, fontweight='medium')

    _add_di_sidebar(ax, row_labels, sort_df)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    for tick in ax.get_yticklabels():
        tick.set_horizontalalignment('right')
    # Extra padding keeps labels clear of the heatmap + DI sidebar.
    ax.tick_params(axis='y', pad=60)
    ax.set_title('a', loc='left', pad=10, fontsize=10,
                 fontweight='bold', color=COLORS['text'])
    _add_white_grid(ax, matrix.shape)
    return im


# =============================================================================
# PANEL B: Δ(TopSHAP−TopVar) Heatmap — sequential green
# =============================================================================

def plot_panel_b(ax, abl: pd.DataFrame, sort_df: pd.DataFrame):
    matrix, row_labels, col_labels = build_heatmap_matrix(
        abl, sort_df, 'delta_shap_var_mean', k_only=10
    )
    matrix = matrix * 100

    vmin = min(0, np.nanmin(matrix)) if not np.all(np.isnan(matrix)) else 0
    vmax = max(np.nanmax(matrix), 0.1) if not np.all(np.isnan(matrix)) else 1

    im = ax.imshow(matrix, aspect='auto', cmap=SEQGREEN_CMAP,
                    vmin=vmin, vmax=vmax, interpolation='nearest')

    ds_map = dict(zip(sort_df['view_label'], sort_df['dataset']))
    row_datasets = [ds_map.get(vl, '') for vl in row_labels]
    _add_dataset_bands(ax, row_datasets, alpha=0.12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if val > vmax * 0.50 else COLORS['text']
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=6.5, color=text_color, fontweight='medium')

    _add_di_sidebar(ax, row_labels, sort_df)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6.5)
    for tick in ax.get_yticklabels():
        tick.set_horizontalalignment('right')
    # Extra padding keeps labels clear of the heatmap + DI sidebar.
    ax.tick_params(axis='y', pad=55)
    ax.set_title('b', loc='left', pad=10, fontsize=10,
                 fontweight='bold', color=COLORS['text'])
    _add_white_grid(ax, matrix.shape)
    return im


# =============================================================================
# SHARED HEATMAP HELPERS
# =============================================================================

def _add_di_sidebar(ax, row_labels, sort_df):
    for i, vl in enumerate(row_labels):
        mask = sort_df['view_label'] == vl
        if mask.any():
            di = sort_df.loc[mask, 'DI_10pct_consensus'].iloc[0]
            di_color = DI_CMAP(DI_NORM(di))
            rect = mpatches.FancyBboxPatch(
                (-0.7, i - 0.4), 0.45, 0.8,
                boxstyle='round,pad=0.05', facecolor=di_color,
                edgecolor='none', clip_on=False, zorder=10
            )
            ax.add_patch(rect)


def _add_white_grid(ax, shape):
    ax.set_xticks(np.arange(shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.spines[:].set_visible(False)


def _add_dataset_bands(ax, datasets, alpha=0.12):
    """Add subtle horizontal background bands for contiguous dataset blocks."""
    if len(datasets) == 0:
        return

    start = 0
    current = datasets[0]
    for i in range(1, len(datasets) + 1):
        is_boundary = (i == len(datasets)) or (datasets[i] != current)
        if is_boundary:
            # Warmer tint than the v7 panel-c blue-green band.
            ax.axhspan(start - 0.5, i - 0.5, color=LIGHT_YLGREEN, alpha=alpha, zorder=0)
            if i < len(datasets):
                start = i
                current = datasets[i]


# =============================================================================
# PANEL C: Permutation Null — Forest CI + Observed Overlay
# =============================================================================

def plot_panel_c(ax, perm: pd.DataFrame, abl: pd.DataFrame, sort_df: pd.DataFrame):
    """Forest-style: permutation null CI band (q05-q95) vs observed delta."""
    pm = perm[perm['metric'] == 'balanced_accuracy'].copy()
    pm = pm.merge(sort_df[['dataset', 'view', 'DI_10pct_consensus', 'view_label']],
                   on=['dataset', 'view'], how='inner')
    pm = pm.sort_values('DI_10pct_consensus').reset_index(drop=True)

    # Observed from ablation at K = 10%
    obs = abl[(abl['metric'] == 'balanced_accuracy') & (abl['K_pct'] == 10)].copy()
    obs = obs.merge(sort_df[['dataset', 'view', 'view_label']],
                     on=['dataset', 'view'], how='inner')

    n_views = len(pm)
    assert n_views > 0, "No views in perm data after filtering"

    _add_dataset_bands(ax, pm['dataset'].tolist(), alpha=0.07)

    for i, (_, row) in enumerate(pm.iterrows()):
        di = row['DI_10pct_consensus']
        di_color = DI_CMAP(DI_NORM(di))
        label = row['view_label']

        # Null CI band — raw units (no pp scaling)
        q05 = row['delta_shap_var_q05']
        q95 = row['delta_shap_var_q95']
        null_mean = row['delta_shap_var_mean']

        # CI shaded bar
        ax.barh(i, q95 - q05, left=q05, height=0.55,
                color=COLORS['null_fill'], edgecolor=COLORS['null_edge'],
                linewidth=0.8, alpha=0.7, zorder=2)

        # Null mean tick
        ax.plot([null_mean, null_mean], [i - 0.25, i + 0.25],
                color=COLORS['null_edge'], linewidth=2, zorder=3)

        # Observed Δ from ablation
        obs_mask = obs['view_label'] == label
        if obs_mask.any():
            obs_val = float(obs.loc[obs_mask, 'delta_shap_var_mean'].iloc[0])
            ax.scatter(obs_val, i, s=90, marker='D', color=di_color,
                       edgecolor='white', linewidth=1.3, zorder=10)
            ax.text(obs_val + 0.002, i + 0.28, f'{obs_val:+.4f}',
                    fontsize=6.5, color=di_color, fontweight='bold', va='bottom')

    ax.axvline(x=0, color=GREY, linestyle='--', linewidth=1.2, alpha=0.5)

    ax.set_yticks(range(n_views))
    ax.set_yticklabels(pm['view_label'], fontsize=7)
    ax.set_xlabel('Δ(TopSHAP − TopVar)', fontsize=9)
    ax.set_ylim(-0.5, n_views - 0.5)

    legend_elements = [
        mpatches.Patch(facecolor=COLORS['null_fill'], edgecolor=COLORS['null_edge'],
                       alpha=0.7, label='Null 5–95% CI'),
        plt.Line2D([0], [0], color=COLORS['null_edge'], linewidth=2,
                   label='Null mean'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=TEXT_SECONDARY,
                   markeredgecolor='white', markersize=8, label='Observed (K = 10%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linewidth=0.5, alpha=0.3, color=COLORS['grid'])
    ax.set_title('c',
                 loc='left', pad=10, fontsize=10,
                 fontweight='bold', color=COLORS['text'])
    ax.spines['left'].set_color(TEXT_SECONDARY)
    ax.spines['bottom'].set_color(TEXT_SECONDARY)


# =============================================================================
# PANEL D: Seed Robustness — Beeswarm + Box (n=100 seeds, 3 views)
# =============================================================================

def plot_panel_d(ax, seed_long: pd.DataFrame, sort_df: pd.DataFrame):
    """Beeswarm + box: per-seed Delta(TopSHAP-TopVar) for seed robustness."""
    df = seed_long.copy()

    # Filter: K = 10%, balanced_accuracy
    df = df[(df['k_pct'] == 10) & (df['metric'] == 'balanced_accuracy')].copy()

    # Average across folds+repeats within each (dataset, view, perm_seed, strategy)
    agg = df.groupby(['dataset', 'view', 'perm_seed', 'strategy'],
                      as_index=False)['value'].mean()

    # Pivot: one column per strategy
    pivot = agg.pivot_table(index=['dataset', 'view', 'perm_seed'],
                             columns='strategy', values='value').reset_index()

    strategies = [c for c in pivot.columns if c not in ['dataset', 'view', 'perm_seed']]
    print(f"    Panel d: {len(strategies)} strategies")

    # Compute delta (topSHAP − topVar)
    shap_col = next((c for c in ['shap_topk', 'topshap'] if c in pivot.columns), None)
    var_col  = next((c for c in ['var_topk', 'topvar'] if c in pivot.columns), None)
    assert shap_col is not None, f"No shap column in {strategies}"
    assert var_col is not None, f"No var column in {strategies}"

    pivot['delta'] = (pivot[shap_col] - pivot[var_col]) # raw units

    # Merge DI
    pivot = pivot.merge(sort_df[['dataset', 'view', 'DI_10pct_consensus', 'view_label']],
                         on=['dataset', 'view'], how='inner')

    # Get unique views sorted by DI
    views = (pivot.drop_duplicates(subset=['dataset', 'view'])
             .sort_values('DI_10pct_consensus')
             .reset_index(drop=True))
    n_views = len(views)

    for i, (_, vrow) in enumerate(views.iterrows()):
        ds, vw = vrow['dataset'], vrow['view']
        di = vrow['DI_10pct_consensus']
        di_color = DI_CMAP(DI_NORM(di))

        vals = pivot.loc[(pivot['dataset'] == ds) & (pivot['view'] == vw),
                         'delta'].dropna().values
        n_seeds = len(vals)
        if n_seeds == 0:
            continue

        print(f"    {vrow['view_label']}: {n_seeds} seeds")

        # Box (background)
        ax.boxplot([vals], positions=[i], widths=0.45, vert=True,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=di_color, alpha=0.2,
                                  edgecolor=di_color, linewidth=1),
                   medianprops=dict(color=di_color, linewidth=2),
                   whiskerprops=dict(color=di_color, linewidth=1),
                   capprops=dict(color=di_color, linewidth=1))

        # Beeswarm jitter
        rng = np.random.default_rng(42 + i)
        jitter_x = i + rng.uniform(-0.15, 0.15, size=n_seeds)
        ax.scatter(jitter_x, vals, s=12, color=di_color, alpha=0.5,
                   edgecolors='white', linewidth=0.3, zorder=5, rasterized=True)

        # Median diamond
        ax.scatter(i, np.median(vals), s=70, marker='D', color=di_color,
                   edgecolor='white', linewidth=1.2, zorder=10)

        # n label
        y_bottom = vals.min() - (vals.max() - vals.min()) * 0.08
        ax.text(i, y_bottom, f'n={n_seeds}', fontsize=6.5, ha='center',
                va='top', color=COLORS['text'])

    ax.set_xticks(range(n_views))
    ax.set_xticklabels(views['view_label'], fontsize=6.5, rotation=25, ha='right')
    ax.set_ylabel('Δ(TopSHAP − TopVar)', fontsize=9)
    ax.axhline(y=0, color=GREY, linestyle='--', linewidth=1.2, alpha=0.5)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.3, color=COLORS['grid'])

    ax.set_title('d', loc='left', pad=10, fontsize=10,
                 fontweight='bold', color=COLORS['text'])
    ax.spines['left'].set_color(TEXT_SECONDARY)
    ax.spines['bottom'].set_color(TEXT_SECONDARY)


# =============================================================================
# MAIN FIGURE ASSEMBLY
# =============================================================================

def create_figure():
    apply_style()

    print("=" * 70)
    print("FIGURE S3: Full Ablation & Permutation Robustness")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    regime_df = load_regime()
    sort_df   = get_di_sort(regime_df)
    abl       = load_ablation_xgb()
    perm      = load_perm_summary()
    seed_long = load_seed_long()

    print("\n[2/5] Building asymmetric layout...")
    fig = plt.figure(figsize=(11, 9.5))

    gs = GridSpec(2, 12, figure=fig,
                  height_ratios=[1, 1.3],
                  hspace=0.40, wspace=0.8)

    ax_a = fig.add_subplot(gs[0, 0:5])
    ax_b = fig.add_subplot(gs[0, 6:11])
    ax_c = fig.add_subplot(gs[1, 0:7])
    ax_d = fig.add_subplot(gs[1, 8:12])

    # Slightly narrow top heatmaps to increase center gap and avoid overlap.
    shrink = 0.90
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    new_w_a = pos_a.width * shrink
    new_w_b = pos_b.width * shrink
    ax_a.set_position([pos_a.x0, pos_a.y0, new_w_a, pos_a.height])  # keep left edge
    ax_b.set_position([pos_b.x1 - new_w_b, pos_b.y0, new_w_b, pos_b.height])  # keep right edge

    # Shift bottom row left to compensate for top-row label padding
    # extending the tight-bbox left boundary.
    offset = 0.05
    pos_c = ax_c.get_position()
    pos_d = ax_d.get_position()
    ax_c.set_position([pos_c.x0 - offset, pos_c.y0, pos_c.width, pos_c.height])
    ax_d.set_position([pos_d.x0 - offset, pos_d.y0, pos_d.width, pos_d.height])

    print("\n[3/5] Plotting panels...")

    print("  Panel a: Δ(TopVar−Random) heatmap...")
    im_a = plot_panel_a(ax_a, abl, sort_df)

    print("  Panel b: Δ(TopSHAP−TopVar) heatmap...")
    im_b = plot_panel_b(ax_b, abl, sort_df)

    print("  Panel c: Permutation null forest...")
    plot_panel_c(ax_c, perm, abl, sort_df)

    print("  Panel d: Seed robustness beeswarm...")
    plot_panel_d(ax_d, seed_long, sort_df)

    print("\n[4/5] Adding colorbars...")
    cax_a = fig.add_axes([0.08, 0.56, 0.30, 0.012])
    cb_a = fig.colorbar(im_a, cax=cax_a, orientation='horizontal')
    cb_a.set_label('Δ(TopVar−Random) (pp)', fontsize=7, color=COLORS['text'])
    cb_a.ax.tick_params(labelsize=6, colors=COLORS['text'])

    cax_b = fig.add_axes([0.55, 0.56, 0.30, 0.012])
    cb_b = fig.colorbar(im_b, cax=cax_b, orientation='horizontal')
    cb_b.set_label('Δ(TopSHAP−TopVar) (pp)', fontsize=7, color=COLORS['text'])
    cb_b.ax.tick_params(labelsize=6, colors=COLORS['text'])

    cax_di = fig.add_axes([0.20, 0.02, 0.60, 0.012])
    sm = ScalarMappable(cmap=DI_CMAP, norm=DI_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax_di, orientation='horizontal')
    cbar.set_label('DI (row sidebar colour)', fontsize=8, labelpad=3,
                   color=COLORS['text'])
    cbar.ax.tick_params(labelsize=7, colors=COLORS['text'])
    cbar.ax.xaxis.set_ticks([0.60, 0.70, 0.80, 0.90, 1.00, 1.10])
    cbar.ax.text(0.0, 1.6, 'Coupled', transform=cbar.ax.transAxes,
                 fontsize=7, ha='left', va='bottom',
                 color=COUPLED_GREEN, style='italic')
    cbar.ax.text(1.0, 1.6, 'Anti-aligned', transform=cbar.ax.transAxes,
                 fontsize=7, ha='right', va='bottom',
                 color=ANTI_YLGREEN, style='italic')

    print("\n[5/5] Saving...")
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)

    out_path = FIGURE_OUT / "figure_s3.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n{'=' * 70}")
    print(f"DONE: {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    create_figure()