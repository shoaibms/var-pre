#!/usr/bin/env python3
"""
Figure S4: VAD Metric Selection & Simulation (4 panels)
========================================================

Panel a: All 6 candidate metrics vs ablation harm, per model (dot plot with rho)
Panel b: VAD metric intercorrelation heatmap (6x6) -- PCLA/VSA complementarity
Panel c: Simulation parameter sweep: DI vs within/between variance ratio
Panel d: Simulation DI by scenario (strip + box, zone bands)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\ms\Desktop\var-pre")
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
RESULTS_DIR  = OUTPUTS_DIR / "results"
FIGURE_OUT   = OUTPUTS_DIR / "figures"

P_VAD_VALIDATION = (RESULTS_DIR / "main_results" / "section_5_diagnostic"
                    / "vad_validation_by_model.csv")
P_VAD_OVERVIEW   = (RESULTS_DIR / "main_results" / "section_5_diagnostic"
                    / "vad_metric_overview.csv")
# Results-only: ST11 contains param_sweep_1d rows tagged by table_source
P_PARAM_SWEEP    = RESULTS_DIR / "sup_table" / "ST11_simulation_summary.csv"
P_SIMULATION     = (RESULTS_DIR / "main_results" / "section_5_diagnostic"
                    / "simulation_validation.csv")
P_REGIME         = (RESULTS_DIR / "main_results" / "section_1_paradox_discovery"
                    / "regime_map.csv")

# Import shared palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
from colourlist import (
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN, NEUTRAL_GREEN,
    LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
    TEXT_PRIMARY, TEXT_SECONDARY, SPINE_COLOR, GRID_COLOR, BG_WHITE,
    GREY, GREY_LIGHTER, GREY_LIGHT,
)

# =============================================================================
# COLOUR PALETTE
# =============================================================================

COLORS = {
    'text':     TEXT_PRIMARY,
    'spine':    SPINE_COLOR,
    'grid':     GRID_COLOR,
    'sig':      COUPLED_GREEN,
    'nonsig':   '#b0b0b0',
    'xgb':      COUPLED_GREEN,
    'rf':       '#4DB6AC',
}

DI_CMAP = LinearSegmentedColormap.from_list('di_green', [
    COUPLED_GREEN, MID_BLUEGREEN, LIGHT_BLUEGREEN,
    NEUTRAL_GREEN, LIGHT_YLGREEN, MID_YLGREEN, ANTI_YLGREEN,
], N=256)
DI_NORM = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.10)

# Metric intercorrelation colourmap (green diverging)
CORR_CMAP = LinearSegmentedColormap.from_list('corr_green', [
    ANTI_YLGREEN, NEUTRAL_GREEN, '#ffffff', NEUTRAL_GREEN, COUPLED_GREEN,
], N=256)

# Scenario colours
SCENARIO_COLORS = {
    'coupled':      COUPLED_GREEN,
    'decoupled':    MID_YLGREEN,
    'anti_aligned': ANTI_YLGREEN,
}

# Metric display order (matching paper)
METRIC_ORDER = ['PCLA', 'VSA', 'ηES', "α'", 'SAS', 'F-DI']
METRIC_COLS  = ['pcla_mean', 'vsa_mean', 'eta_es_mean',
                'alpha_prime_mean', 'sas_mean', 'f_di_mean']


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

def load_all():
    """Load all data files."""
    for name, path in [('vad_validation', P_VAD_VALIDATION),
                        ('vad_overview', P_VAD_OVERVIEW),
                        ('param_sweep', P_PARAM_SWEEP),
                        ('simulation', P_SIMULATION),
                        ('regime', P_REGIME)]:
        assert path.exists(), f"MISSING: {path}"

    val = pd.read_csv(P_VAD_VALIDATION)
    print(f"  vad_validation: {len(val)} rows")

    ovw = pd.read_csv(P_VAD_OVERVIEW)
    print(f"  vad_overview: {len(ovw)} rows")

    sweep_all = pd.read_csv(P_PARAM_SWEEP)

    # ST11 contains multiple sources; keep only the 1D sweep rows.
    if "table_source" in sweep_all.columns:
        sweep = sweep_all.loc[sweep_all["table_source"] == "param_sweep_1d"].copy()
    else:
        # Fallback: raw sweep file without table_source column.
        sweep = sweep_all

    assert len(sweep) > 0, f"No param_sweep_1d rows found in: {P_PARAM_SWEEP}"

    required = {"wb_ratio", "DI_mean", "DI_std", "regime"}
    missing = required - set(sweep.columns)
    assert not missing, f"param_sweep missing columns {sorted(missing)} in: {P_PARAM_SWEEP}"
    print(f"  param_sweep: {len(sweep)} rows")

    sim = pd.read_csv(P_SIMULATION)
    print(f"  simulation: {len(sim)} rows")

    regime = pd.read_csv(P_REGIME)
    print(f"  regime: {len(regime)} views")

    return val, ovw, sweep, sim, regime


# =============================================================================
# PANEL A: Metric Validation — Dot plot with ρ, per model
# =============================================================================

def plot_panel_a(ax, val: pd.DataFrame):
    """Six candidate metrics x 2 models. Lollipop with significance highlighting."""
    # Order metrics consistently
    metric_positions = {m: i for i, m in enumerate(METRIC_ORDER)}

    models = ['xgb_bal', 'rf']
    model_labels = {'xgb_bal': 'XGBoost', 'rf': 'Random Forest'}
    model_colors = {'xgb_bal': COLORS['xgb'], 'rf': COLORS['rf']}
    model_markers = {'xgb_bal': 'o', 'rf': 's'}
    model_offsets = {'xgb_bal': 0.15, 'rf': -0.15}

    y_positions = np.arange(len(METRIC_ORDER))

    for mdl in models:
        sub = val[val['model'] == mdl].copy()
        color = model_colors[mdl]
        marker = model_markers[mdl]
        offset = model_offsets[mdl]

        for _, row in sub.iterrows():
            label = row['metric_label']
            if label not in metric_positions:
                continue
            y = metric_positions[label] + offset
            rho = row['rho']
            sig = row['significant_005']

            # Colour: green if significant, grey if not
            dot_color = color if sig else COLORS['nonsig']
            dot_alpha = 1.0 if sig else 0.5

            # Lollipop stem
            ax.plot([0, rho], [y, y], color=dot_color, linewidth=1.5,
                    alpha=dot_alpha * 0.6, zorder=2)

            # Dot
            ax.scatter(rho, y, s=80, marker=marker, color=dot_color,
                       edgecolor='white', linewidth=0.8, alpha=dot_alpha, zorder=5)

            # ρ and p annotation
            p_str = f'p={row["p_value"]:.3f}'
            if row['p_value'] < 0.05:
                p_str += ' *'
            text_x = rho + 0.03 if rho >= 0 else rho - 0.03
            ha = 'left' if rho >= 0 else 'right'
            ax.text(text_x, y, f'ρ={rho:.2f}\n{p_str}',
                    fontsize=6, color=dot_color, ha=ha, va='center',
                    fontweight='bold' if sig else 'normal')

    # Zero reference
    ax.axvline(x=0, color=GREY, linestyle='-', linewidth=0.8, alpha=0.4)

    # Y axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(METRIC_ORDER, fontsize=8)
    ax.set_xlabel('Spearman ρ (metric vs ablation harm)', fontsize=9)
    ax.set_ylim(-0.5, len(METRIC_ORDER) - 0.5)
    ax.invert_yaxis()

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['xgb'],
                   markeredgecolor='white', markersize=8, label='XGBoost'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['rf'],
                   markeredgecolor='white', markersize=8, label='Random Forest'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['nonsig'],
                   markeredgecolor='white', markersize=8, alpha=0.5,
                   label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linewidth=0.5, alpha=0.3, color=COLORS['grid'])

    ax.set_xlim(-0.6, 0.65)

    ax.set_title('a',
                 loc='left', pad=8, fontsize=10,
                 fontweight='bold', color=COLORS['text'])

    ax.spines['left'].set_color(TEXT_SECONDARY)
    ax.spines['bottom'].set_color(TEXT_SECONDARY)


# =============================================================================
# PANEL B: Metric Intercorrelation Heatmap (6×6)
# =============================================================================

def plot_panel_b(ax, ovw: pd.DataFrame):
    """Pairwise Spearman rho across views for 6 candidate metrics (6x6 heatmap)."""
    # Extract metric matrix (14 views × 6 metrics)
    metric_data = ovw[METRIC_COLS].copy()

    n_metrics = len(METRIC_COLS)
    corr_matrix = np.full((n_metrics, n_metrics), np.nan)

    for i in range(n_metrics):
        for j in range(n_metrics):
            x = metric_data[METRIC_COLS[i]].dropna()
            y = metric_data[METRIC_COLS[j]].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) >= 3:
                rho, _ = stats.spearmanr(x[idx], y[idx])
                corr_matrix[i, j] = rho

    pcla_vsa_r = corr_matrix[0, 1]
    print(f"    PCLA-VSA intercorrelation: rho={pcla_vsa_r:.3f}")

    # Heatmap
    im = ax.imshow(corr_matrix, cmap=CORR_CMAP, vmin=-1, vmax=1,
                    aspect='auto', interpolation='nearest')

    # Cell annotations
    for i in range(n_metrics):
        for j in range(n_metrics):
            val = corr_matrix[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if abs(val) > 0.6 else COLORS['text']
            weight = 'bold' if (i == 0 and j == 1) or (i == 1 and j == 0) else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=text_color, fontweight=weight)

    # Highlight PCLA-VSA cell with border
    for (r, c) in [(0, 1), (1, 0)]:
        rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=2.5,
                                   edgecolor=COUPLED_GREEN, facecolor='none',
                                   zorder=10)
        ax.add_patch(rect)

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(METRIC_ORDER, fontsize=7, rotation=35, ha='right')
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(METRIC_ORDER, fontsize=7)

    # White grid
    ax.set_xticks(np.arange(n_metrics + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_metrics + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.spines[:].set_visible(False)

    ax.set_title('b',
                 loc='left', pad=8, fontsize=10,
                 fontweight='bold', color=COLORS['text'])

    # PCLA-VSA annotation outside
    ax.annotate(f'PCLA–VSA: ρ={pcla_vsa_r:.2f}',
                xy=(1, 0), xytext=(2.5, -1.2),
                fontsize=7, color=COUPLED_GREEN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COUPLED_GREEN, lw=1.2))

    return im


# =============================================================================
# PANEL C: Simulation Parameter Sweep — DI vs wb_ratio
# =============================================================================

def plot_panel_c(ax, sweep: pd.DataFrame):
    """1D parameter sweep: within/between variance ratio vs DI."""
    df = sweep.sort_values('wb_ratio').copy()

    x = df['wb_ratio'].values
    y = df['DI_mean'].values
    yerr = df['DI_std'].values

    # Main line
    ax.plot(x, y, color=COUPLED_GREEN, linewidth=2.2, zorder=5)

    # ±1 SD shading
    ax.fill_between(x, y - yerr, y + yerr, color=LIGHT_BLUEGREEN,
                     alpha=0.3, zorder=2)

    # Dots coloured by regime
    for _, row in df.iterrows():
        regime = row['regime'].lower()
        color = SCENARIO_COLORS.get(regime, GREY)
        ax.scatter(row['wb_ratio'], row['DI_mean'], s=40, color=color,
                   edgecolor='white', linewidth=0.6, zorder=6)

    # DI=1 reference line (regime boundary)
    ax.axhline(y=1.0, color=GREY, linestyle='--', linewidth=1.2, alpha=0.5,
               label='DI = 1 (regime boundary)')

    # Zone bands
    ax.axhspan(0, 0.85, color=LIGHT_BLUEGREEN, alpha=0.10, zorder=0)
    ax.axhspan(0.85, 1.0, color=NEUTRAL_GREEN, alpha=0.15, zorder=0)
    ax.axhspan(1.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 1.0 else 1.2,
               color=LIGHT_YLGREEN, alpha=0.10, zorder=0)

    # Zone labels
    ax.text(0.02, 0.15, 'Coupled', transform=ax.transAxes,
            fontsize=7, color=COUPLED_GREEN, style='italic', alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlabel('Within-class / between-class variance ratio', fontsize=9)
    ax.set_ylabel('DI (K = 10%)', fontsize=9)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.3, color=COLORS['grid'])
    ax.xaxis.grid(True, linewidth=0.5, alpha=0.3, color=COLORS['grid'])

    ax.legend(fontsize=7, loc='upper left')

    ax.set_title('c',
                 loc='left', pad=8, fontsize=10,
                 fontweight='bold', color=COLORS['text'])

    ax.spines['left'].set_color(TEXT_SECONDARY)
    ax.spines['bottom'].set_color(TEXT_SECONDARY)


# =============================================================================
# PANEL D: Simulation DI by Scenario — Strip + Box
# =============================================================================

def plot_panel_d(ax, sim: pd.DataFrame):
    """Three scenarios x 2 seeds each. Strip + box with zone bands."""
    scenarios = ['coupled', 'decoupled', 'anti_aligned']
    scenario_labels = {
        'coupled': 'Coupled',
        'decoupled': 'Decoupled',
        'anti_aligned': 'Anti-aligned',
    }

    # Zone bands (horizontal)
    ax.axhspan(0, 0.85, color=LIGHT_BLUEGREEN, alpha=0.10, zorder=0)
    ax.axhspan(0.85, 1.0, color=NEUTRAL_GREEN, alpha=0.15, zorder=0)
    ax.axhspan(1.0, 1.15, color=LIGHT_YLGREEN, alpha=0.10, zorder=0)

    # DI=1 reference
    ax.axhline(y=1.0, color=GREY, linestyle='--', linewidth=1.2, alpha=0.5)

    for i, scen in enumerate(scenarios):
        sub = sim[sim['scenario'] == scen]
        vals = sub['DI_mean'].values
        color = SCENARIO_COLORS[scen]

        if len(vals) > 1:
            # Box
            ax.boxplot([vals], positions=[i], widths=0.4, vert=True,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=color, alpha=0.25,
                                      edgecolor=color, linewidth=1),
                       medianprops=dict(color=color, linewidth=2),
                       whiskerprops=dict(color=color, linewidth=1),
                       capprops=dict(color=color, linewidth=1))

        # Individual seed dots (jittered)
        rng = np.random.default_rng(42 + i)
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(i + jitter, vals, s=70, color=color,
                   edgecolor='white', linewidth=1.0, zorder=8, marker='D')

        # Seed labels (offset + subtle halo so symbol is readable over marker)
        for j, (_, row) in enumerate(sub.iterrows()):
            correct = row['classification_correct']
            mark = r'$\checkmark$' if correct else r'$\times$'
            ax.text(i + jitter[j] + 0.17, vals[j] + 0.004, f'{mark}',
                    fontsize=8.5, color=color, fontweight='bold',
                    ha='center', va='center', zorder=12,
                    bbox=dict(boxstyle='circle,pad=0.12',
                              facecolor='white', edgecolor=color,
                              linewidth=0.6, alpha=0.7))

        # Mean annotation
        mean_di = np.mean(vals)
        ax.text(i, vals.max() + 0.015, f'DI={mean_di:.2f}',
                fontsize=7, ha='center', va='bottom', color=color,
                fontweight='bold')

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([scenario_labels[s] for s in scenarios],
                        fontsize=8, fontweight='medium')
    ax.set_ylabel('DI (K = 10%)', fontsize=9)
    ax.set_ylim(0.35, 1.15)

    # Zone labels on right
    ax.text(1.03, 0.15, 'Coupled', transform=ax.transAxes,
            fontsize=7, color=COUPLED_GREEN, style='italic',
            rotation=90, va='center')
    ax.text(1.03, 0.75, 'Anti-\naligned', transform=ax.transAxes,
            fontsize=7, color=ANTI_YLGREEN, style='italic',
            rotation=90, va='center')

    # Legend for classification marker symbols
    ax.text(0.02, 0.97,
            r'$\checkmark$ = DI classification correct' '\n'
            r'$\times$ = misclassified',
            transform=ax.transAxes, fontsize=6.5, va='top',
            color=COLORS['text'], style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=NEUTRAL_GREEN,
                      edgecolor=GREY_LIGHTER, alpha=0.8))

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.3, color=COLORS['grid'])

    ax.set_title('d',
                 loc='left', pad=8, fontsize=10,
                 fontweight='bold', color=COLORS['text'])

    ax.spines['left'].set_color(TEXT_SECONDARY)
    ax.spines['bottom'].set_color(TEXT_SECONDARY)


# =============================================================================
# MAIN FIGURE ASSEMBLY
# =============================================================================

def create_figure():
    apply_style()

    print("=" * 70)
    print("FIGURE S4: VAD Metric Selection & Simulation")
    print("=" * 70)

    # ── Load ──
    print("\n[1/5] Loading data...")
    val, ovw, sweep, sim, regime = load_all()

    # ── Layout: asymmetric 12-col grid ──
    print("\n[2/5] Building layout...")
    fig = plt.figure(figsize=(11, 9))

    # Row 0: Panel a (7 cols, hero) | Panel b (5 cols, heatmap)
    # Row 1: Panel c (6 cols, sweep) | Panel d (6 cols, scenario)
    gs = GridSpec(2, 12, figure=fig,
                  height_ratios=[1, 1],
                  hspace=0.32, wspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0:7])
    ax_b = fig.add_subplot(gs[0, 8:12])
    ax_c = fig.add_subplot(gs[1, 0:6])
    ax_d = fig.add_subplot(gs[1, 7:12])

    # ── Plot ──
    print("\n[3/5] Plotting panels...")

    print("  Panel a: Metric validation dot plot...")
    plot_panel_a(ax_a, val)

    print("  Panel b: Metric intercorrelation heatmap...")
    im_b = plot_panel_b(ax_b, ovw)

    print("  Panel c: Parameter sweep...")
    plot_panel_c(ax_c, sweep)

    print("  Panel d: Simulation by scenario...")
    plot_panel_d(ax_d, sim)

    # ── Colorbar for panel b ──
    print("\n[4/5] Adding colorbar...")
    cax_b = fig.add_axes([0.72, 0.485, 0.18, 0.0105])
    cb = fig.colorbar(im_b, cax=cax_b, orientation='horizontal')
    cb.set_label('Spearman ρ', fontsize=7, color=COLORS['text'])
    cb.ax.tick_params(labelsize=6, colors=COLORS['text'])

    # ── Save ──
    print("\n[5/5] Saving...")
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)

    out_path = FIGURE_OUT / "figure_s4.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n{'=' * 70}")
    print(f"DONE: {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    create_figure()