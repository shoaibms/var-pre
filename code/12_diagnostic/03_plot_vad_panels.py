#!/usr/bin/env python3
"""
PHASE 12 — Figure panels for VAD (diagnostic + validation)

Reads:
  - <outputs_dir>/<validation_dir>/vad_vs_ablation.csv

Writes:
  - vad_scatter_etaES_vs_delta.png
  - vad_phase_etaES_vs_PCLA.png
  - vad_phase_etaES_vs_SAS.png

Usage:
  python .\\code\\compute\12_diagnostic\03_plot_vad_panels.py `
    --outputs-dir "<path-to-outputs>" `
    --validation-dirname "12_diagnostic_validation"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--validation-dirname", default="12_diagnostic_validation")
    ap.add_argument("--out-dirname", default="12_diagnostic_figures")
    ap.add_argument("--material-harm", type=float, default=0.01, help="Harm threshold for horizontal guide line")
    return ap.parse_args()


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    val_dir = outputs_dir / args.validation_dirname
    out_dir = outputs_dir / args.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = val_dir / "vad_vs_ablation.csv"
    if not inp.exists():
        raise FileNotFoundError(f"Missing: {inp}. Run 02_validate_against_ablation.py first.")
    df = pd.read_csv(inp)

    # --- 1) etaES vs delta (validation scatter) ---
    fig = plt.figure(figsize=(6.5, 4.8))
    ax = fig.add_subplot(111)

    # plot per dataset (matplotlib cycles colours automatically)
    for ds, g in df.groupby("dataset"):
        ax.scatter(g["eta_es_mean"], g["delta_var_minus_random_mean"], label=str(ds), alpha=0.9)

    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axhline(-float(args.material_harm), linestyle=":", linewidth=1.0)

    ax.set_xlabel("ηES (TopVar enrichment, k=10%)")
    ax.set_ylabel("Δ(Var − Random) (ablation)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title("VAD predicts harm of variance filtering")

    _save(fig, out_dir / "vad_scatter_etaES_vs_delta.png")

    # --- 2) phase diagram: etaES vs PCLA ---
    fig = plt.figure(figsize=(6.5, 4.8))
    ax = fig.add_subplot(111)
    for ds, g in df.groupby("dataset"):
        ax.scatter(g["eta_es_mean"], g["pcla_mean"], label=str(ds), alpha=0.9)

    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.axhline(0.2, linestyle="--", linewidth=1.0)  # conservative "alignment" guide

    ax.set_xlabel("ηES (k=10%)")
    ax.set_ylabel("PCLA (PCA λ-weighted η²)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title("VAD Phase Diagram (univariate + multivariate)")

    _save(fig, out_dir / "vad_phase_etaES_vs_PCLA.png")

    # --- 3) phase diagram: etaES vs SAS ---
    fig = plt.figure(figsize=(6.5, 4.8))
    ax = fig.add_subplot(111)
    for ds, g in df.groupby("dataset"):
        ax.scatter(g["eta_es_mean"], g["sas_mean"], label=str(ds), alpha=0.9)

    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xlabel("ηES (k=10%)")
    ax.set_ylabel("SAS (Spearman(λ_k, η²(PC_k)))")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title("Spectral alignment complements ηES")

    _save(fig, out_dir / "vad_phase_etaES_vs_SAS.png")

    print("[OK] VAD panels saved to:", out_dir)


if __name__ == "__main__":
    main()
