#!/usr/bin/env python
"""
SAF feature-table QC for hero views.

Run BEFORE the SAF ablation script (03_feature_subset_ablation_saf.py).

Usage:
    python 14_supplementary_analyses/02_saf_feature_table_qc.py --outputs-dir outputs

Checks:
  1. Hero-view feature_eta_sq files exist
  2. Required columns present (feature, var_total, var_between, var_within, eta_sq)
  3. No duplicated feature IDs
  4. var_between >= 0 for all rows
  5. Descending sort on var_between behaves correctly
  6. Degeneracy guard: fraction of features with var_between < 1e-8

Outputs:
  - Console GO / NO-GO verdict per view
  - outputs/14_supplementary_analyses/saf_qc/saf_feature_table_qc.csv  (machine-readable QC log)

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ── Hero views (must match HERO_VIEWS in 01_feature_subset_ablation.py) ──
HERO_VIEWS = [
    ("mlomics", "methylation"),
    ("ibdmdb", "MGX"),
    ("ccle", "mRNA"),
]

REQUIRED_COLS = {"feature", "var_total", "var_between", "var_within", "eta_sq"}
DEGENERACY_EPS = 1e-8
DEGENERACY_FRAC_FAIL = 0.50  # NO-GO if >50% features are near-zero on var_between


def resolve_feature_table(outputs_dir: Path, dataset: str, view: str) -> Optional[Path]:
    """Try exact path first, then glob fallback."""
    per_view_dir = outputs_dir / "12_diagnostic" / "per_view"
    exact = per_view_dir / f"feature_eta_sq__{dataset}__{view}.csv.gz"
    if exact.exists():
        return exact

    candidates = list(per_view_dir.glob(f"feature_eta_sq__{dataset}__{view}*"))
    if candidates:
        return sorted(candidates)[0]

    return None


def qc_one_view(outputs_dir: Path, dataset: str, view: str) -> dict:
    """Run all QC checks on a single hero view. Returns one QC row."""
    row = {
        "dataset": dataset,
        "view": view,
        "file_exists": False,
        "file_path": "",
        "n_features_total": 0,
        "cols_ok": False,
        "missing_cols": "",
        "no_dup_features": False,
        "n_dup_features": 0,
        "var_between_nonneg": False,
        "n_negative_var_between": 0,
        "sort_ok": False,
        "n_features_var_between_lt_eps": 0,
        "frac_features_var_between_lt_eps": 0.0,
        "saf_qc_flag": "FAIL",
        "saf_qc_note": "",
    }

    path = resolve_feature_table(outputs_dir, dataset, view)
    if path is None:
        row["saf_qc_note"] = "File not found"
        return row

    row["file_exists"] = True
    row["file_path"] = str(path)

    try:
        df = pd.read_csv(path, compression="gzip" if str(path).endswith(".gz") else None)
    except Exception as e:
        row["saf_qc_note"] = f"Read error: {e}"
        return row

    row["n_features_total"] = int(len(df))

    present = set(df.columns)
    missing = REQUIRED_COLS - present
    row["cols_ok"] = len(missing) == 0
    row["missing_cols"] = ", ".join(sorted(missing)) if missing else ""
    if missing:
        row["saf_qc_note"] = f"Missing columns: {row['missing_cols']}"
        return row

    if len(df) == 0:
        row["saf_qc_note"] = "Empty feature table"
        return row

    # Duplicate features
    n_dup = int(df["feature"].astype(str).duplicated().sum())
    row["no_dup_features"] = (n_dup == 0)
    row["n_dup_features"] = n_dup

    # var_between numeric / non-negative checks
    vb = pd.to_numeric(df["var_between"], errors="coerce")
    n_nonfinite = int(vb.isna().sum())
    n_neg = int((vb < 0).sum())
    row["var_between_nonneg"] = (n_nonfinite == 0 and n_neg == 0)
    row["n_negative_var_between"] = n_neg

    # Sort check: after sorting descending, series must be monotone decreasing
    sorted_vb = vb.sort_values(ascending=False, kind="mergesort", na_position="last").reset_index(drop=True)
    row["sort_ok"] = bool(sorted_vb.notna().all() and sorted_vb.is_monotonic_decreasing)

    # Degeneracy guard
    n_near_zero = int((vb.abs() < DEGENERACY_EPS).sum()) if n_nonfinite < len(vb) else 0
    frac_near_zero = (n_near_zero / len(df)) if len(df) > 0 else 0.0
    row["n_features_var_between_lt_eps"] = n_near_zero
    row["frac_features_var_between_lt_eps"] = round(float(frac_near_zero), 6)

    # Final verdict: strict PASS / FAIL only
    issues = []
    if n_dup > 0:
        issues.append(f"{n_dup} duplicate feature IDs")
    if n_nonfinite > 0:
        issues.append(f"{n_nonfinite} non-finite var_between values")
    if n_neg > 0:
        issues.append(f"{n_neg} negative var_between values")
    if not row["sort_ok"]:
        issues.append("descending sort check failed")
    if frac_near_zero > DEGENERACY_FRAC_FAIL:
        issues.append(f"{frac_near_zero:.1%} features near-zero on var_between")

    if issues:
        row["saf_qc_flag"] = "FAIL"
        row["saf_qc_note"] = "; ".join(issues)
    else:
        row["saf_qc_flag"] = "PASS"
        row["saf_qc_note"] = "All checks passed"

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="SAF Phase 1 — feature-table QC")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Root outputs directory (default: outputs)",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"[ERROR] outputs-dir not found: {outputs_dir}")
        sys.exit(1)

    print("=" * 60)
    print("SAF Phase 1 — Feature-table QC for hero views")
    print("=" * 60)

    rows: List[dict] = []
    all_pass = True

    for dataset, view in HERO_VIEWS:
        print(f"\n── {dataset} / {view} ──")
        row = qc_one_view(outputs_dir, dataset, view)
        rows.append(row)

        flag = row["saf_qc_flag"]
        if flag != "PASS":
            all_pass = False

        if not row["file_exists"]:
            print("  [FAIL] File not found")
            print(f"         Expected: outputs/12_diagnostic/per_view/feature_eta_sq__{dataset}__{view}.csv.gz")
        else:
            print(f"  File:     {row['file_path']}")
            print(f"  Features: {row['n_features_total']}")
            print(f"  Columns:  {'OK' if row['cols_ok'] else 'MISSING: ' + row['missing_cols']}")
            print(f"  Dups:     {row['n_dup_features']}")
            print(f"  Neg Vb:   {row['n_negative_var_between']}")
            print(f"  Near-0:   {row['n_features_var_between_lt_eps']} ({row['frac_features_var_between_lt_eps']:.2%})")
            print(f"  Sort OK:  {row['sort_ok']}")
        print(f"  ──> {flag}: {row['saf_qc_note']}")

    qc_dir = outputs_dir / "14_supplementary_analyses" / "saf_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_path = qc_dir / "saf_feature_table_qc.csv"
    pd.DataFrame(rows).to_csv(qc_path, index=False)
    print(f"\nQC log saved: {qc_path}")

    print("\n" + "=" * 60)
    if all_pass:
        print("VERDICT: GO — all 3 hero views pass QC.")
        print("Next step: apply SAF patch to 01_feature_subset_ablation.py")
    else:
        failed = [f"{r['dataset']}/{r['view']}" for r in rows if r["saf_qc_flag"] != "PASS"]
        print(f"VERDICT: NO-GO — {len(failed)} view(s) failed: {', '.join(failed)}")
        print("Fix issues above before proceeding.")
        missing = [r for r in rows if not r["file_exists"]]
        if missing:
            print("\nTo regenerate missing feature tables, rerun Phase 12 with --write-feature-tables:")
            print("  python code/compute/12_diagnostic/01_compute_vad.py --outputs-dir outputs --write-feature-tables")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()