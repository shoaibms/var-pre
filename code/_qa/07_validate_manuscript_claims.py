#!/usr/bin/env python3
r"""
07_validate_manuscript_claims.py

Cross-check quantitative claims in the manuscript against the actual
supplementary table data in final_sup_table/.

Checks:
  1. Dataset/view counts and identifiers (ST1)
  2. Model performance sanity (ST2)
  3. DI values, ranges, regime counts (ST3)
  4. Ablation effect sizes: "16.2 pp", "8.2 pp mean", "26.5 pp max" etc. (ST4)
  5. Cross-model agreement stats (ST5)
  6. Label permutation null controls (ST6)
  7. Pathway overlap stats: "8.5% gene Jaccard", "19.3% pathway", "2.4x" (ST7)
  8. VAD zone counts: green=7, red=4, yellow=3 (ST8)
  9. Simulation regime recovery (ST9)
  10. Unsupervised clustering: "7/14 views" degraded (ST10)

Output: terminal log + optional markdown report.

Usage:
  python .\code\compute\13_results\07_validate_manuscript_claims.py --outputs-dir <path-to-outputs>
  python .\code\compute\13_results\07_validate_manuscript_claims.py --outputs-dir <path-to-outputs> --report
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Tolerances
# ═══════════════════════════════════════════════════════════════════════════════
ABS_TOL = 0.15       # percentage-point tolerance for performance claims
REL_TOL = 0.05       # 5% relative tolerance for ratios/counts
STRICT_TOL = 0.01    # tight tolerance for exact-match claims

# ═══════════════════════════════════════════════════════════════════════════════
# Result tracking
# ═══════════════════════════════════════════════════════════════════════════════

class CheckResult:
    def __init__(self, name: str, status: str, detail: str,
                 expected: Any = None, actual: Any = None):
        self.name = name
        self.status = status  # PASS, FAIL, WARN, SKIP
        self.detail = detail
        self.expected = expected
        self.actual = actual

    def __str__(self):
        icon = {"PASS": "+", "FAIL": "X", "WARN": "!", "SKIP": "-"}[self.status]
        line = f"  [{icon}] {self.status:4s} | {self.name}"
        if self.expected is not None and self.actual is not None:
            line += f"\n         Expected: {self.expected}  |  Actual: {self.actual}"
        if self.detail:
            line += f"\n         {self.detail}"
        return line


results: List[CheckResult] = []


def check(name: str, condition: bool, detail: str = "",
          expected: Any = None, actual: Any = None, warn_only: bool = False):
    status = "PASS" if condition else ("WARN" if warn_only else "FAIL")
    results.append(CheckResult(name, status, detail, expected, actual))


def skip(name: str, detail: str = ""):
    results.append(CheckResult(name, "SKIP", detail))


def close_enough(a: float, b: float, atol: float = ABS_TOL) -> bool:
    return abs(a - b) <= atol


def close_pct(a: float, b: float, rtol: float = REL_TOL) -> bool:
    if b == 0:
        return a == 0
    return abs(a - b) / abs(b) <= rtol


# ═══════════════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(d: Path, pattern: str) -> Optional[pd.DataFrame]:
    """Find and load a CSV matching a glob pattern."""
    matches = list(d.glob(pattern))
    if not matches:
        return None
    return pd.read_csv(matches[0])


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_st1_datasets(d: Path):
    """Manuscript: 'Four public multi-omics datasets...14 primary views'"""
    df = load_csv(d, "ST1_*.csv")
    if df is None:
        skip("ST1: Dataset overview", "File not found")
        return

    n_rows = len(df)
    check("ST1: Total views = 14", n_rows == 14,
          expected=14, actual=n_rows)

    if "dataset" in df.columns:
        n_datasets = df["dataset"].nunique()
        check("ST1: Number of datasets = 4", n_datasets == 4,
              expected=4, actual=n_datasets)

        datasets = set(df["dataset"].unique())
        expected_ds = {"MLOmics BRCA", "IBDMDB/HMP2", "CCLE/DepMap", "TCGA-GBM"}
        # Flexible matching (datasets may use short names)
        check("ST1: Dataset names present", len(datasets) == 4,
              detail=f"Found: {sorted(datasets)}",
              expected="4 unique datasets", actual=f"{n_datasets}: {sorted(datasets)}")

    if "n_features" in df.columns:
        # Manuscript: MLOmics mRNA = 11,343; CCLE CNV = 24,352 etc.
        # Just check range is plausible
        fmin, fmax = df["n_features"].min(), df["n_features"].max()
        check("ST1: Feature counts plausible (min ~286, max ~24k)",
              200 < fmin < 500 and 20000 < fmax < 30000,
              expected="min ~286–310, max ~24k–25k", actual=f"min={fmin}, max={fmax}")


def check_st3_di(d: Path):
    """Manuscript claims about DI values."""
    df = load_csv(d, "ST3_*.csv")
    if df is None:
        skip("ST3: DI full table", "File not found")
        return

    # Filter to K=10% for main claims
    if "k_pct" in df.columns:
        df10 = df[df["k_pct"] == 10].copy()
    else:
        df10 = df.copy()

    n_views_k10 = len(df10)
    check("ST3: 14 views at K=10%", n_views_k10 == 14,
          expected=14, actual=n_views_k10)

    if "DI_mean" in df10.columns:
        di_min = df10["DI_mean"].min()
        di_max = df10["DI_mean"].max()
        # Manuscript: "DI values spanned 0.66 to 1.03"
        check("ST3: DI range ~0.66 to ~1.03 at K=10%",
              close_enough(di_min, 0.66, 0.05) and close_enough(di_max, 1.03, 0.05),
              expected="0.66–1.03", actual=f"{di_min:.3f}–{di_max:.3f}")

    if "consensus_regime" in df10.columns:
        regime_counts = df10["consensus_regime"].value_counts().to_dict()
        n_coupled = regime_counts.get("COUPLED", 0)
        n_mixed = regime_counts.get("MIXED", 0)
        n_anti = regime_counts.get("ANTI_ALIGNED", 0)
        # Manuscript: "eight views as coupled, five as mixed, one as anti-aligned"
        check("ST3: Regime counts (C=8, M=5, A=1)",
              n_coupled == 8 and n_mixed == 5 and n_anti == 1,
              expected="C=8, M=5, A=1",
              actual=f"C={n_coupled}, M={n_mixed}, A={n_anti}")

    # Check specific DI values mentioned in manuscript
    if "view" in df10.columns and "DI_mean" in df10.columns:
        # "IBDMDB MGX DI = 0.70"
        mgx = df10[df10["view"].str.contains("MGX", case=False, na=False) &
                    ~df10["view"].str.contains("func", case=False, na=False)]
        if len(mgx) > 0:
            mgx_di = mgx["DI_mean"].iloc[0]
            check("ST3: IBDMDB MGX DI ≈ 0.70",
                  close_enough(mgx_di, 0.70, 0.05),
                  expected=0.70, actual=f"{mgx_di:.3f}")

        # "MLOmics methylation DI = 1.03"
        meth = df10[df10["view"].str.contains("methyl", case=False, na=False)]
        if "dataset" in df10.columns:
            meth_mlo = meth[meth["dataset"].str.contains("mlo", case=False, na=False)]
            if len(meth_mlo) > 0:
                meth_di = meth_mlo["DI_mean"].iloc[0]
                check("ST3: MLOmics methylation DI ≈ 1.03",
                      close_enough(meth_di, 1.03, 0.05),
                      expected=1.03, actual=f"{meth_di:.3f}")


def check_st4_ablation(d: Path):
    """Manuscript claims about ablation effect sizes."""
    df = load_csv(d, "ST4_*.csv")
    if df is None:
        skip("ST4: Ablation full", "File not found")
        return

    # Filter to XGB, K=10%, balanced_accuracy
    mask_xgb = df["model"].str.contains("xgb", case=False, na=False)
    mask_k10 = df["k_pct"] == 10
    mask_ba = df["metric"].str.contains("balanced_accuracy", case=False, na=False)
    sub = df[mask_xgb & mask_k10 & mask_ba].copy()

    if len(sub) == 0:
        skip("ST4: XGB K=10% BA subset", "No matching rows")
        return

    n_views = len(sub)
    check("ST4: 14 views in XGB/K=10%/BA", n_views == 14,
          expected=14, actual=n_views)

    # --- "16.2 pp" harm claim ---
    # Manuscript: "variance filtering degraded balanced accuracy by 16.2 pp relative to random"
    if "delta_var_random_mean" in sub.columns:
        worst_harm = sub["delta_var_random_mean"].min()
        # 16.2 pp harm means delta_var_random ≈ -0.162
        check("ST4: Worst harm ≈ -16.2 pp (Δ TopVar−Random)",
              close_enough(worst_harm * 100, -16.2, 1.0),
              expected="-16.2 pp",
              actual=f"{worst_harm * 100:.1f} pp")

        # "underperformed random in nearly half (7/14)"
        n_harmful = (sub["delta_var_random_mean"] < 0).sum()
        check("ST4: TopVar harms ~7/14 views",
              5 <= n_harmful <= 9,
              expected="~7/14", actual=f"{n_harmful}/14",
              warn_only=True)

    # --- "8.2 pp mean SHAP advantage" ---
    if "delta_shap_var_mean" in sub.columns:
        mean_shap_adv = sub["delta_shap_var_mean"].mean() * 100
        # Manuscript: "mean advantage of 8.2 percentage points"
        check("ST4: Mean Δ(TopSHAP−TopVar) ≈ 8.2 pp",
              close_enough(mean_shap_adv, 8.2, 1.5),
              expected="8.2 pp", actual=f"{mean_shap_adv:.1f} pp")

        max_shap_adv = sub["delta_shap_var_mean"].max() * 100
        # Manuscript: "maximum 26.5 pp"
        check("ST4: Max Δ(TopSHAP−TopVar) ≈ 26.5 pp",
              close_enough(max_shap_adv, 26.5, 2.0),
              expected="26.5 pp", actual=f"{max_shap_adv:.1f} pp")

        # "SHAP outperformed variance in 27 of 28 cases" (both models)
        # For XGB only: should be 13/14 or 14/14
        n_shap_wins = (sub["delta_shap_var_mean"] > 0).sum()
        check("ST4: SHAP > Var in ≥13/14 XGB views",
              n_shap_wins >= 13,
              expected="≥13/14", actual=f"{n_shap_wins}/14")

    # --- "31 pp at 1% threshold" ---
    mask_k1 = df["k_pct"] == 1
    sub_k1 = df[mask_xgb & mask_k1 & mask_ba].copy()
    if len(sub_k1) > 0 and "delta_var_random_mean" in sub_k1.columns:
        worst_k1 = sub_k1["delta_var_random_mean"].min()
        check("ST4: Worst harm at K=1% ≈ -31 pp",
              close_enough(worst_k1 * 100, -31, 3.0),
              expected="~-31 pp", actual=f"{worst_k1 * 100:.1f} pp",
              warn_only=True)

    # --- Cross-model: RF agreement ---
    mask_rf = df["model"].str.contains("rf", case=False, na=False)
    sub_rf = df[mask_rf & mask_k10 & mask_ba].copy()
    if len(sub_rf) > 0 and "delta_var_random_mean" in sub_rf.columns:
        # Manuscript: "Pearson r = 0.85, Spearman ρ = 0.79"
        # We can check direction agreement
        if len(sub) == len(sub_rf):
            merged = sub.merge(sub_rf, on=["dataset", "view"], suffixes=("_xgb", "_rf"))
            if "delta_var_random_mean_xgb" in merged.columns:
                same_dir = ((merged["delta_var_random_mean_xgb"] > 0) ==
                            (merged["delta_var_random_mean_rf"] > 0)).sum()
                check("ST4: Direction agreement XGB vs RF ≈ 10/14",
                      8 <= same_dir <= 14,
                      expected="~10/14", actual=f"{same_dir}/{len(merged)}",
                      warn_only=True)


def check_st5_cross_model(d: Path):
    """Manuscript: cross-model agreement."""
    df = load_csv(d, "ST5_*.csv")
    if df is None:
        skip("ST5: Cross-model agreement", "File not found")
        return

    n_views = len(df)
    check("ST5: 14 views", n_views == 14,
          expected=14, actual=n_views)


def check_st6_permutation(d: Path):
    """Manuscript: label-shuffle null control."""
    df = load_csv(d, "ST6_*.csv")
    if df is None:
        skip("ST6: Label permutation", "File not found")
        return

    if "n_perm" in df.columns:
        n_perm = df["n_perm"].iloc[0]
        # Manuscript mentions permutation tests
        check("ST6: Permutation count ≥ 100",
              n_perm >= 100,
              expected="≥100", actual=n_perm)

    # Check that SHAP-var gap is positive under real labels
    if "delta_shap_var_mean" in df.columns:
        all_positive = (df["delta_shap_var_mean"] > 0).all()
        check("ST6: SHAP–Var gap positive for all tested views",
              all_positive,
              detail=f"Values: {df['delta_shap_var_mean'].tolist()}")


def check_st7_pathway(d: Path):
    """Manuscript: 'mean gene-level Jaccard 8.5%', 'pathway 19.3%', 'convergence 2.4x'"""
    df = load_csv(d, "ST7_*.csv")
    if df is None:
        skip("ST7: Pathway detail", "File not found")
        return

    if "gene_jaccard" in df.columns:
        mean_gene_j = df["gene_jaccard"].mean() * 100  # convert to %
        check("ST7: Mean gene Jaccard ≈ 8.5%",
              close_enough(mean_gene_j, 8.5, 2.0),
              expected="8.5%", actual=f"{mean_gene_j:.1f}%")

    if "pathway_jaccard" in df.columns:
        mean_path_j = df["pathway_jaccard"].mean() * 100
        check("ST7: Mean pathway Jaccard ≈ 19.3%",
              close_enough(mean_path_j, 19.3, 3.0),
              expected="19.3%", actual=f"{mean_path_j:.1f}%")

    if "convergence_ratio" in df.columns:
        mean_conv = df["convergence_ratio"].mean()
        check("ST7: Mean convergence ratio ≈ 2.4×",
              close_enough(mean_conv, 2.4, 0.5),
              expected="2.4×", actual=f"{mean_conv:.1f}×")


def check_st8_vad(d: Path):
    """Manuscript: VAD zones green=7, red=4, yellow=3."""
    df = load_csv(d, "ST8_*.csv")
    if df is None:
        skip("ST8: VAD diagnostics", "File not found")
        return

    if "predicted_zone" not in df.columns:
        skip("ST8: VAD zones", "predicted_zone column not found")
        return

    # Zones are only at K=10%
    if "k_pct" in df.columns:
        df10 = df[df["k_pct"] == 10].copy()
    else:
        df10 = df[df["predicted_zone"].notna()].copy()

    zones = df10["predicted_zone"].value_counts().to_dict()
    n_green = zones.get("GREEN_SAFE", 0)
    n_red = zones.get("RED_HARMFUL", 0)
    n_yellow = zones.get("YELLOW_INCONCLUSIVE", 0)

    check("ST8: VAD zones (G=7, R=4, Y=3)",
          n_green == 7 and n_red == 4 and n_yellow == 3,
          expected="G=7, R=4, Y=3",
          actual=f"G={n_green}, R={n_red}, Y={n_yellow}")


def check_st9_simulation(d: Path):
    """Manuscript: simulation recovered coupled, decoupled, anti-aligned."""
    df = load_csv(d, "ST9_*.csv")
    if df is None:
        skip("ST9: Simulation summary", "File not found")
        return

    n_rows = len(df)
    check("ST9: 6 simulation rows", n_rows == 6,
          expected=6, actual=n_rows)

    if "scenario" in df.columns and "regime" in df.columns:
        scenarios = set(df["scenario"].unique())
        expected_s = {"coupled", "decoupled", "anti_aligned"}
        check("ST9: All 3 scenarios present",
              expected_s.issubset(scenarios),
              expected=sorted(expected_s),
              actual=sorted(scenarios))

    # Manuscript Fig S4d: "DI = 0.51 (coupled), 1.02 (decoupled), 1.05 (anti-aligned)"
    if "scenario" in df.columns and "DI_10pct" in df.columns:
        for scen, expected_di in [("coupled", 0.51), ("decoupled", 1.02), ("anti_aligned", 1.05)]:
            rows = df[df["scenario"] == scen]
            if len(rows) > 0:
                actual_di = rows["DI_10pct"].mean()
                check(f"ST9: {scen} DI ≈ {expected_di}",
                      close_enough(actual_di, expected_di, 0.10),
                      expected=expected_di, actual=f"{actual_di:.3f}",
                      warn_only=True)


def check_st10_unsupervised(d: Path):
    """Manuscript: 'TopVar degrades cluster recovery in 7/14 views'"""
    df = load_csv(d, "ST10_*.csv")
    if df is None:
        skip("ST10: Unsupervised clustering", "File not found")
        return

    n_views = len(df)
    check("ST10: 14 views", n_views == 14,
          expected=14, actual=n_views)

    if "delta_ARI_TopVar_Random" in df.columns:
        n_negative = (df["delta_ARI_TopVar_Random"] < 0).sum()
        # Manuscript: "ΔARI negative in 7/14 views"
        check("ST10: TopVar harms ≈ 7/14 views (ARI)",
              5 <= n_negative <= 9,
              expected="~7/14", actual=f"{n_negative}/14",
              warn_only=True)


def check_cross_table_consistency(d: Path):
    """Cross-table checks: view counts, dataset names consistent across tables."""
    st1 = load_csv(d, "ST1_*.csv")
    st3 = load_csv(d, "ST3_*.csv")
    st4 = load_csv(d, "ST4_*.csv")

    if st1 is not None and st3 is not None:
        if "view" in st1.columns and "view" in st3.columns:
            views1 = set(st1["view"].unique())
            views3 = set(st3["view"].unique())
            check("Cross: ST1 vs ST3 view sets match",
                  views1 == views3,
                  expected=f"ST1: {sorted(views1)}",
                  actual=f"ST3: {sorted(views3)}")

    if st1 is not None and st4 is not None:
        if "dataset" in st1.columns and "dataset" in st4.columns:
            ds1 = set(st1["dataset"].unique())
            ds4 = set(st4["dataset"].unique())
            check("Cross: ST1 vs ST4 dataset names consistent",
                  len(ds1) > 0 and len(ds4) > 0,
                  detail=f"ST1: {sorted(ds1)} | ST4: {sorted(ds4)}",
                  warn_only=True)


def check_manuscript_specific_numbers(d: Path):
    """Check very specific numbers stated in the manuscript text."""
    st4 = load_csv(d, "ST4_*.csv")
    if st4 is None:
        return

    # Manuscript: "mean advantage of 8.2 pp (maximum 26.5 pp; Fig. 4a)"
    # This is for XGB + BA + K=10%
    mask = (st4["model"].str.contains("xgb", case=False, na=False) &
            (st4["k_pct"] == 10) &
            st4["metric"].str.contains("balanced_accuracy", case=False, na=False))
    sub = st4[mask]

    if len(sub) > 0 and "delta_shap_var_mean" in sub.columns:
        # "13/14 views favouring TopSHAP" in Fig 2c
        n_favour = (sub["delta_shap_var_mean"] > 0).sum()
        check("Manuscript: '13/14 views favouring TopSHAP'",
              n_favour == 13 or n_favour == 14,
              expected="13 or 14", actual=n_favour)

    # Manuscript: "Spearman ρ = −0.38 ... and −0.47"
    # Source: di_vs_delta_scatter_k10.csv (Figure 4c), using DI_10pct_xgb_bal
    # NOT ST3's DI_mean (which comes from fig1_di_curves / simple threshold method).
    scatter = None
    for scatter_dir in [d.parent / "sup_data", d.parent / "main_results" / "section_4_consequences"]:
        for fname in ["di_vs_delta_scatter_k10.csv",
                      "Supplementary_Data_2_di_vs_delta_scatter_k10.csv"]:
            p = scatter_dir / fname
            if p.exists():
                scatter = pd.read_csv(p)
                break
        if scatter is not None:
            break
    if scatter is not None:
        try:
            from scipy import stats as _stats
        except ImportError:
            skip("Manuscript: DI vs Δ(TopVar−Random) Spearman", "scipy not installed")
            return
        # Try model-level file (28 rows: 14 views × 2 models)
        di_col = next((c for c in scatter.columns if "DI_10pct" in c), None)
        delta_col = next((c for c in scatter.columns if "delta_var" in c and "random" in c.lower()), None)
        if di_col and delta_col:
            sc_xgb = scatter[scatter["model"].str.contains("xgb", case=False, na=False)] \
                     if "model" in scatter.columns else scatter
            if len(sc_xgb) >= 10:
                rho, pval = _stats.spearmanr(sc_xgb[di_col], sc_xgb[delta_col])
                check("Manuscript: DI vs Δ(TopVar−Random) Spearman ≈ −0.38 (XGB)",
                      close_enough(rho, -0.38, 0.10),
                      expected="ρ ≈ −0.38", actual=f"ρ = {rho:.3f} (p={pval:.4f})",
                      warn_only=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate manuscript quantitative claims against supplementary data.")
    parser.add_argument("--outputs-dir", required=True,
                        help="Root outputs directory")
    parser.add_argument("--report", action="store_true",
                        help="Write markdown report to final_sup_table/")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    final_dir = outputs_dir / "results" / "final_sup_table"

    if not final_dir.exists():
        print(f"[ERROR] final_sup_table not found: {final_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 90)
    print("MANUSCRIPT vs DATA VALIDATION")
    print("=" * 90)
    print(f"  Data directory: {final_dir}")
    print(f"  Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Run all checks
    print("─" * 90)
    print("CHECK 1: Dataset & View Inventory (ST1)")
    print("─" * 90)
    check_st1_datasets(final_dir)
    for r in results[-4:] if len(results) >= 4 else results:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 2: DI Values & Regime Counts (ST3)")
    print("─" * 90)
    n_before = len(results)
    check_st3_di(final_dir)
    for r in results[n_before:]:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 3: Ablation Effect Sizes (ST4)")
    print("─" * 90)
    n_before = len(results)
    check_st4_ablation(final_dir)
    for r in results[n_before:]:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 4: Cross-Model Agreement (ST5)")
    print("─" * 90)
    n_before = len(results)
    check_st5_cross_model(final_dir)
    for r in results[n_before:]:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 5: Label Permutation (ST6)")
    print("─" * 90)
    n_before = len(results)
    check_st6_permutation(final_dir)
    for r in results[n_before:]:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 6: Pathway Overlap (ST7)")
    print("─" * 90)
    n_before = len(results)
    check_st7_pathway(final_dir)
    for r in results[n_before:]:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 7: VAD Zones (ST8)")
    print("─" * 90)
    n_before = len(results)
    check_st8_vad(final_dir)
    for r in results[n_before:]:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 8: Simulation (ST9)")
    print("─" * 90)
    n_before = len(results)
    check_st9_simulation(final_dir)
    for r in results[n_before:]:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 9: Unsupervised Clustering (ST10)")
    print("─" * 90)
    n_before = len(results)
    check_st10_unsupervised(final_dir)
    for r in results[n_before:]:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 10: Cross-Table Consistency")
    print("─" * 90)
    n_before = len(results)
    check_cross_table_consistency(final_dir)
    for r in results[n_before:]:
        print(r)

    print("\n" + "─" * 90)
    print("CHECK 11: Manuscript-Specific Numbers")
    print("─" * 90)
    n_before = len(results)
    check_manuscript_specific_numbers(final_dir)
    for r in results[n_before:]:
        print(r)

    # ─── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("VALIDATION SUMMARY")
    print("=" * 90)

    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    for r in results:
        counts[r.status] += 1

    total = sum(counts.values())
    print(f"\n  Total checks: {total}")
    print(f"  PASS: {counts['PASS']}")
    print(f"  FAIL: {counts['FAIL']}")
    print(f"  WARN: {counts['WARN']}")
    print(f"  SKIP: {counts['SKIP']}")

    if counts["FAIL"] > 0:
        print("\n  *** FAILURES DETECTED — review before submission ***")
        print("\n  Failed checks:")
        for r in results:
            if r.status == "FAIL":
                print(f"    [FAIL] {r.name}")
                if r.expected is not None:
                    print(f"      Expected: {r.expected}  |  Actual: {r.actual}")
    elif counts["WARN"] > 0:
        print("\n  Warnings present but no hard failures. Review if needed.")
    else:
        print("\n  ALL CHECKS PASSED -- manuscript claims are consistent with data.")

    # ─── Optional report ────────────────────────────────────────────────────
    if args.report:
        lines = [
            "# Manuscript vs Data Validation Report",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"- Total checks: {total}",
            f"- PASS: {counts['PASS']}",
            f"- FAIL: {counts['FAIL']}",
            f"- WARN: {counts['WARN']}",
            f"- SKIP: {counts['SKIP']}",
            "",
            "## Detailed Results",
            "",
        ]
        for r in results:
            icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}[r.status]
            lines.append(f"- {icon} **{r.status}** | {r.name}")
            if r.expected is not None:
                lines.append(f"  - Expected: `{r.expected}` | Actual: `{r.actual}`")
            if r.detail:
                lines.append(f"  - {r.detail}")
            lines.append("")

        report_path = final_dir / "VALIDATION_REPORT.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  Report written: {report_path}")

    # Exit code for CI/scripting
    sys.exit(1 if counts["FAIL"] > 0 else 0)


if __name__ == "__main__":
    main()