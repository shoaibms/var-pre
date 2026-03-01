#!/usr/bin/env python3
"""
Figure S2: Complete Rank-Rank Atlas (14 panels)
================================================

14 scatter panels in a grid, sorted by DI ascending.
Each panel: variance-rank vs prediction-rank percentile,
with Q4 features highlighted, DI-colored top-bar frame,
and annotation box (view label, DI, rho, Q4%).

Layout: Asymmetric 3 rows x 15 columns GridSpec.
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
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\ms\Desktop\var-pre")
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
FIGURE_OUT   = OUTPUTS_DIR / "figures"
SUPPORT_DIR  = PROJECT_ROOT / "code" / "figures" / "supp" / "figure_S2"

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
    'q4_accent':   '#5a7a2e',
    'bulk':        GRID_COLOR,
    'overlap':     GRID_COLOR,
    'variance':    SPINE_COLOR,
    'prediction':  LIGHT_BLUEGREEN,
    'frame_bg':    '#E0F7FA',
}

# DI colormap (coupled -> anti-aligned)
DI_CMAP = LinearSegmentedColormap.from_list('di_green', [
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN,
    NEUTRAL_GREEN, LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
], N=256)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)

# Hex density colormaps
HEX_CMAP_COUPLED = LinearSegmentedColormap.from_list(
    'hex_coupled', ['#ffffff', '#E0F7FA', '#80CBC4', '#4DB6AC', COUPLED_GREEN], N=256)
HEX_CMAP_ANTI = LinearSegmentedColormap.from_list(
    'hex_anti', ['#ffffff', '#F9FBE7', '#F0F4C3', '#DCE775', '#9ACD32', '#558B2F'], N=256)
HEX_CMAP_MID = LinearSegmentedColormap.from_list(
    'hex_mid', ['#ffffff', '#E0F7FA', '#B2DFDB', '#4DB6AC', MID_BLUEGREEN], N=256)

DISPLAY_NAMES = {
    'mlomics':  'MLOmics',
    'ibdmdb':   'IBDMDB',
    'ccle':     'CCLE',
    'tcga_gbm': 'TCGA-GBM',
}

# Views that appear in main-text figures
MAIN_TEXT_VIEWS = {
    ('mlomics', 'methylation'),
    ('ibdmdb', 'MGX'),
    ('mlomics', 'CNV'),
    ('ccle', 'mRNA'),
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
        'axes.titlesize':     11,
        'axes.titleweight':   'bold',
        'axes.labelsize':     10,
        'axes.labelweight':   'medium',
        'axes.linewidth':     0.8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
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

def load_regime_map() -> pd.DataFrame:
    """Load regime map for DI-based ordering and regime labels."""
    path = OUTPUTS_DIR / "results" / "main_results" / "section_1_paradox_discovery" / "regime_map.csv"
    assert path.exists(), f"MISSING: {path}"
    df = pd.read_csv(path)

    di_col = next((c for c in ["DI_10pct_uncertainty_xgb_bal", "DI_10pct_consensus", "DI_mean", "DI"] if c in df.columns), None)
    if di_col is None:
        raise ValueError(f"No DI column found in regime_map.csv. cols={list(df.columns)}")
    df["DI"] = df[di_col]

    print(f"  regime_map: {len(df)} views")
    return df


def load_vp_joined(dataset: str, view: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / "results" / "main_results" / "section_1_paradox_discovery" / "source_tables" / "joined_vp" / f"vp_joined__{dataset}__{view}.csv.gz"
    assert path.exists(), f"MISSING: {path}"
    return pd.read_csv(path)


# =============================================================================
# SINGLE PANEL PLOTTER
# =============================================================================

def _pick_hex_cmap(di: float):
    """Choose hex colormap based on DI regime."""
    if di < 0.85:
        return HEX_CMAP_COUPLED
    elif di >= 1.0:
        return HEX_CMAP_ANTI
    else:
        return HEX_CMAP_MID


def plot_one_scatter(ax, dataset: str, view: str, di: float, rho: float,
                     regime: str, is_main_text: bool):
    """Hex-density + KDE contour rank-rank plot for one view."""
    # Load data
    df = load_vp_joined(dataset, view)
    n = len(df)
    k_pct = 10

    # Determine rank percentiles — detect available columns
    v_pct_col = next((c for c in ['v_rank_pct', 'var_rank_pct'] if c in df.columns), None)
    p_pct_col = next((c for c in ['p_consensus_rank_pct', 'pred_rank_pct'] if c in df.columns), None)

    if v_pct_col and p_pct_col:
        x = df[v_pct_col].values * 100
        y_raw = df[p_pct_col].values
        y = y_raw * 100 if y_raw.max() <= 1.5 else y_raw
    else:
        v_rank_col = next((c for c in ['v_rank', 'var_rank', 'variance_rank']
                           if c in df.columns), None)
        p_rank_col = next((c for c in ['p_consensus_rank_int', 'p_rank', 'pred_rank',
                                        'importance_rank'] if c in df.columns), None)
        assert v_rank_col is not None, f"No v_rank column in {dataset}/{view}. Cols: {list(df.columns)[:15]}"
        assert p_rank_col is not None, f"No p_rank column in {dataset}/{view}. Cols: {list(df.columns)[:15]}"
        x = df[v_rank_col].values / n * 100
        y = df[p_rank_col].values / n * 100

    # Hex-density plot
    hex_cmap = _pick_hex_cmap(di)
    ax.hexbin(x, y, gridsize=25, cmap=hex_cmap, mincnt=1,
              linewidths=0.08, edgecolors='white', alpha=0.85, zorder=1)

    # KDE contour overlay
    try:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), min(2500, len(x)), replace=False)
        xs, ys = x[idx], y[idx]
        kde = stats.gaussian_kde(np.vstack([xs, ys]), bw_method=0.18)
        xg = np.linspace(0, 100, 60)
        yg = np.linspace(0, 100, 60)
        Xg, Yg = np.meshgrid(xg, yg)
        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        Z = gaussian_filter(Z, sigma=1.5)
        ax.contour(Xg, Yg, Z, levels=4, colors=TEXT_SECONDARY,
                   linewidths=0.5, alpha=0.45, zorder=2)
    except Exception as e:
        pass  # KDE contour failed

    # Reference lines
    ax.axhline(y=k_pct, color=TEXT_SECONDARY, linewidth=0.7, alpha=0.35,
               linestyle='-', zorder=3)
    ax.axvline(x=k_pct, color=TEXT_SECONDARY, linewidth=0.7, alpha=0.35,
               linestyle='-', zorder=3)
    ax.plot([0, 100], [0, 100], color=GREY, linestyle=':',
            linewidth=0.8, alpha=0.35, zorder=3)

    # Q2 badge (top-left of panel)
    ax.text(0.07, 0.86, 'Q2', transform=ax.transAxes,
            fontsize=5.5, ha='center', va='center',
            color='white', fontweight='bold', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.12', fc=COUPLED_GREEN,
                      ec='none', alpha=0.5))

    # Axes setup
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])

    # DI-coloured top-bar frame
    di_color = DI_CMAP(DI_NORM(di))
    bar_rect = mpatches.FancyBboxPatch(
        (0, 0), 1, 0.045, boxstyle="square,pad=0",
        transform=ax.transAxes, facecolor=di_color,
        edgecolor='none', clip_on=False, zorder=10
    )
    ax.add_patch(bar_rect)

    # View title — always black
    label = view_label(dataset, view)
    star = ' ★' if is_main_text else ''
    ax.set_title(f'{label}{star}', fontsize=7, fontweight='bold',
                 color='black', pad=6)

    # Annotation box (top-left inside plot)
    box_text = f'DI={di:.2f}\nρ={rho:.2f}\nn={n:,}'
    ax.text(0.03, 0.97, box_text, transform=ax.transAxes, fontsize=5,
            ha='left', va='top', color=TEXT_SECONDARY,
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec=GREY_LIGHTER, alpha=0.9, linewidth=0.5))

    # Spines
    ax.spines['left'].set_color(TEXT_SECONDARY)
    ax.spines['bottom'].set_color(TEXT_SECONDARY)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)


# =============================================================================
# LEGEND PANEL
# =============================================================================

def plot_legend_panel(ax):
    """Shared legend + colorbar in an empty cell."""
    ax.axis('off')

    y_start = 0.90
    items = [
        ('hex',     'Hex density (feature count)'),
        ('contour', 'KDE contour lines'),
        ('q1',      'Q2 badge (top-V ∩ top-P)'),
        ('diag',    'Perfect coupling (diagonal)'),
        ('thresh',  'K = 10% threshold'),
    ]

    for i, (kind, label) in enumerate(items):
        y = y_start - i * 0.11
        if kind == 'hex':
            # Small coloured square
            rect = mpatches.FancyBboxPatch(
                (0.03, y - 0.025), 0.08, 0.05,
                boxstyle='round,pad=0.01', transform=ax.transAxes,
                facecolor='#80CBC4', edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)
        elif kind == 'contour':
            ax.plot([0.03, 0.11], [y, y], color=TEXT_SECONDARY,
                    linewidth=1.0, alpha=0.6, transform=ax.transAxes)
        elif kind == 'q1':
            ax.text(0.07, y, 'Q2', fontsize=5.5, ha='center', va='center',
                    transform=ax.transAxes, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', fc=COUPLED_GREEN,
                              ec='none', alpha=0.6))
        elif kind == 'diag':
            ax.plot([0.03, 0.11], [y, y], color=GREY, linestyle=':',
                    linewidth=1.0, alpha=0.5, transform=ax.transAxes)
        elif kind == 'thresh':
            ax.plot([0.03, 0.11], [y, y], color=TEXT_SECONDARY,
                    linewidth=0.7, alpha=0.4, transform=ax.transAxes)

        ax.text(0.16, y, label, fontsize=6.5, va='center', ha='left',
                transform=ax.transAxes, color=COLORS['text'])

    # Star badge explanation
    y = y_start - 5 * 0.11 - 0.06
    ax.text(0.07, y, '★', fontsize=10, va='center', ha='center',
            transform=ax.transAxes, color=ANTI_YLGREEN)
    ax.text(0.16, y, 'Shown in main figures', fontsize=6.5, va='center',
            ha='left', transform=ax.transAxes, color=COLORS['text'],
            style='italic')

    # Hex cmap explanation
    y -= 0.12
    ax.text(0.07, y, 'Hex cmap:', fontsize=6, va='center', ha='center',
            transform=ax.transAxes, color=COLORS['text'], fontweight='bold')
    y -= 0.08
    for cname, cmap, lbl in [('coupled', HEX_CMAP_COUPLED, 'Coupled'),
                              ('mid', HEX_CMAP_MID, 'Mixed'),
                              ('anti', HEX_CMAP_ANTI, 'Anti-aligned')]:
        gradient = np.linspace(0, 1, 50).reshape(1, -1)
        extent = [0.03, 0.25, y - 0.02, y + 0.02]
        ax.imshow(gradient, aspect='auto', cmap=cmap,
                  extent=extent, transform=ax.transAxes, zorder=5)
        ax.text(0.27, y, lbl, fontsize=5.5, va='center', ha='left',
                transform=ax.transAxes, color=COLORS['text'])
        y -= 0.07

    # Sort direction
    y -= 0.06
    ax.annotate('', xy=(0.85, y), xytext=(0.05, y),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=TEXT_SECONDARY, lw=1.2))
    ax.text(0.45, y - 0.05, 'DI: coupled → anti-aligned',
            fontsize=6, ha='center', va='top', transform=ax.transAxes,
            color=COLORS['text'], style='italic')


# =============================================================================
# MAIN FIGURE ASSEMBLY
# =============================================================================

def create_figure():
    apply_style()

    print("Figure S2: Complete Rank-Rank Atlas")

    # ── Load data ──
    print("Loading data...")
    regime_df = load_regime_map()

    regime_df = regime_df.sort_values("DI", ascending=True).reset_index(drop=True)

    rho_col = next((c for c in ["rho", "spearman_rho_consensus", "rho_consensus", "spearman_rho"] if c in regime_df.columns), None)
    assert rho_col is not None, f"No rho column. Cols: {list(regime_df.columns)}"

    n_views = len(regime_df)
    assert n_views > 0, "No views found in regime_consensus"

    # ── Layout ──

    # Distribute views across rows: 5 + 5 + remaining
    row_sizes = [5, 5, n_views - 10] if n_views > 10 else [n_views]

    # 15-column grid → 5 panels × 3 cols each for rows 0-1,
    # row 2: 4 panels × 3 cols + legend × 3 cols
    n_cols_grid = 15
    fig = plt.figure(figsize=(11, 9))

    gs = GridSpec(3, n_cols_grid, figure=fig,
                  height_ratios=[1, 1, 1],
                  hspace=0.32, wspace=0.35)

    # ── Plot panels ──

    view_idx = 0
    for row in range(3):
        if row < 2:
            # 5 panels per row, each 3 cols wide
            n_in_row = min(5, n_views - view_idx)
            for col in range(n_in_row):
                ax = fig.add_subplot(gs[row, col * 3:(col + 1) * 3])
                r = regime_df.iloc[view_idx]

                is_main = (r['dataset'], r['view']) in MAIN_TEXT_VIEWS

                plot_one_scatter(
                    ax, r['dataset'], r['view'],
                    di=float(r["DI"]),
                    rho=float(r[rho_col]),
                    regime=str(r["regime"]) if "regime" in r.index else "",
                    is_main_text=is_main,
                )

                # Axis labels only on edges
                if row == 2 or (row == 1 and n_views <= 10):
                    ax.set_xlabel('Variance rank (percentile)', fontsize=7)
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel('SHAP-importance rank (percentile)', fontsize=7)
                else:
                    ax.set_yticklabels([])

                view_idx += 1

        else:
            # Row 2: remaining panels + legend in last cell
            n_in_row = n_views - view_idx
            n_panel_slots = n_in_row

            for col in range(n_panel_slots):
                ax = fig.add_subplot(gs[row, col * 3:(col + 1) * 3])
                r = regime_df.iloc[view_idx]

                is_main = (r['dataset'], r['view']) in MAIN_TEXT_VIEWS

                plot_one_scatter(
                    ax, r['dataset'], r['view'],
                    di=float(r["DI"]),
                    rho=float(r[rho_col]),
                    regime=str(r["regime"]) if "regime" in r.index else "",
                    is_main_text=is_main,
                )

                ax.set_xlabel('Variance rank (percentile)', fontsize=7)
                if col == 0:
                    ax.set_ylabel('SHAP-importance rank (percentile)', fontsize=7)
                else:
                    ax.set_yticklabels([])

                view_idx += 1

            # Legend in remaining space
            legend_start = n_panel_slots * 3
            if legend_start < n_cols_grid:
                ax_legend = fig.add_subplot(gs[row, legend_start:n_cols_grid])
                plot_legend_panel(ax_legend)

    # ── DI colorbar at bottom ──
    cax = fig.add_axes([0.20, 0.02, 0.60, 0.012])
    sm = ScalarMappable(cmap=DI_CMAP, norm=DI_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Decoupling Index (DI) — panel top-bar color',
                   fontsize=8, labelpad=3, color=COLORS['text'])
    cbar.ax.tick_params(labelsize=7, colors=COLORS['text'])
    cbar.ax.xaxis.set_ticks([0.60, 0.70, 0.80, 0.90, 1.00, 1.10])
    cbar.ax.text(0.0, 1.8, 'Coupled', transform=cbar.ax.transAxes,
                 fontsize=7, ha='left', color=COUPLED_GREEN, fontweight='bold')
    cbar.ax.text(1.0, 1.8, 'Anti-aligned', transform=cbar.ax.transAxes,
                 fontsize=7, ha='right', color=ANTI_YLGREEN, fontweight='bold')

    # ── Save ──
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = FIGURE_OUT / "figure_s2.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved: {out_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    create_figure()