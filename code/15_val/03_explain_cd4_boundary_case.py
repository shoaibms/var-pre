#!/usr/bin/env python3
r"""
03_explain_cd4_boundary_case.py

Optional GSE138746 CD4 boundary-case explanation.

This script consolidates feature-level diagnosis, annotation, class contrast,
and robustness checks originally developed for the CD4 boundary-case analysis:

    archive\11_diagnose_cd4_vad_boundary.py
    archive\12_annotate_cd4_hidden_genes.py
    archive\13_cd4_hidden_gene_class_contrast.py
    archive\14_cd4_hidden_gene_robustness.py

The core diagnosis step is dispatched to the validated backend diagnosis script.
The annotation, class-contrast, and robustness steps are implemented here with
the same default paths and outputs as the original one-off scripts.

Default run tag:
    manuscript_audit_v3_batch1_valid4

Example
-------
From the project root:

    python code\compute\15_val\03_explain_cd4_boundary_case.py `
      --run-tag manuscript_audit_v3_batch1_valid4 `
      --all

Outputs
-------
    outputs\15_val\_manuscript_audit\<run_tag>\cd4_vad_boundary_diagnosis\
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from _val_utils import find_project_root, legacy_script_path, run_python_script


DEFAULT_RUN_TAG = "manuscript_audit_v3_batch1_valid4"
DEFAULT_BUNDLE = Path(
    "data/raw/val/gse138746_ra_antitnf_rnaseq/"
    "gse138746__cd4_rnaseq__eular_good_vs_none__bundle.npz"
)


def diagnosis_dir(project_root: Path, run_tag: str) -> Path:
    return (
        project_root
        / "outputs"
        / "15_val"
        / "_manuscript_audit"
        / run_tag
        / "cd4_vad_boundary_diagnosis"
    )


def safe_load_npz(path: Path):
    try:
        return np.load(path, allow_pickle=False)
    except ValueError:
        return np.load(path, allow_pickle=True)


def run_diagnose(project_root: Path, run_tag: str, bundle: Path, k_values: str, top_hidden: int) -> int:
    """Run the validated feature-level diagnosis backend script."""
    script = legacy_script_path("11_diagnose_cd4_vad_boundary.py")
    args = [
        "--project-root", str(project_root),
        "--run-tag", run_tag,
        "--bundle", str(bundle),
        "--k-values", str(k_values),
        "--top-hidden", str(top_hidden),
    ]
    return run_python_script(script, args, cwd=project_root)


def annotate_hidden_genes(project_root: Path, run_tag: str) -> Path:
    """Annotate hidden-signal Ensembl genes with mygene, preserving the old output name."""
    try:
        import mygene
    except Exception as e:
        raise RuntimeError(
            "The annotation step requires the 'mygene' package. "
            "Install it in the active venv with: pip install mygene"
        ) from e

    out_dir = diagnosis_dir(project_root, run_tag)
    inp = out_dir / "cd4_hidden_signal_candidates.csv"
    out = out_dir / "cd4_hidden_signal_candidates_annotated.csv"

    if not inp.exists():
        raise FileNotFoundError(f"Missing input for annotation: {inp}")

    df = pd.read_csv(inp)
    ids = (
        df["feature"]
        .astype(str)
        .str.replace(r"\.\d+$", "", regex=True)
        .dropna()
        .unique()
        .tolist()
    )

    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        ids,
        scopes="ensembl.gene",
        fields="symbol,name,entrezgene,type_of_gene,summary",
        species="human",
        as_dataframe=False,
        verbose=False,
    )

    ann_rows = []
    for r in res:
        ann_rows.append(
            {
                "feature_clean": r.get("query"),
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "entrezgene": r.get("entrezgene"),
                "type_of_gene": r.get("type_of_gene"),
                "summary": r.get("summary"),
                "notfound": r.get("notfound", False),
            }
        )

    ann = pd.DataFrame(ann_rows).drop_duplicates("feature_clean")
    df["feature_clean"] = df["feature"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    merged = df.merge(ann, on="feature_clean", how="left")
    merged.to_csv(out, index=False)

    cols = [
        "k_pct",
        "feature",
        "symbol",
        "name",
        "variance",
        "between_variance",
        "eta2",
        "variance_rank_desc",
        "eta2_rank_desc",
        "type_of_gene",
        "notfound",
    ]
    cols = [c for c in cols if c in merged.columns]
    print(merged[cols].head(60).to_string(index=False))
    print("\nWROTE:", out)
    return out


def class_contrast(project_root: Path, run_tag: str, bundle: Path, top_n: int = 100) -> Path:
    """Compute class-wise contrasts for annotated hidden-signal candidates."""
    out_dir = diagnosis_dir(project_root, run_tag)
    hidden = out_dir / "cd4_hidden_signal_candidates_annotated.csv"
    out = out_dir / "cd4_hidden_gene_class_contrast.csv"

    if not hidden.exists():
        raise FileNotFoundError(f"Missing annotated hidden-gene file: {hidden}")

    z = safe_load_npz(bundle)
    X = z["X"].astype(float)
    y = z["y"].astype(int)
    features = (
        z["feature_names"].astype(str)
        if "feature_names" in z.files
        else np.array([f"f{i}" for i in range(X.shape[1])])
    )

    df = pd.read_csv(hidden).drop_duplicates("feature").head(int(top_n))
    idx = pd.Series(np.arange(len(features)), index=features)

    rows = []
    for _, r in df.iterrows():
        f = str(r["feature"])
        if f not in idx:
            continue
        j = int(idx[f])
        x0 = X[y == 0, j]
        x1 = X[y == 1, j]
        m0, m1 = np.nanmean(x0), np.nanmean(x1)
        s0, s1 = np.nanstd(x0, ddof=1), np.nanstd(x1, ddof=1)
        pooled = np.sqrt(((len(x0) - 1) * s0 * s0 + (len(x1) - 1) * s1 * s1) / max(1, len(x0) + len(x1) - 2))
        d = (m1 - m0) / pooled if pooled > 0 else np.nan

        rows.append(
            {
                "feature": f,
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "eta2": r.get("eta2"),
                "variance_rank_desc": r.get("variance_rank_desc"),
                "eta2_rank_desc": r.get("eta2_rank_desc"),
                "mean_none": m0,
                "mean_good": m1,
                "diff_good_minus_none": m1 - m0,
                "cohens_d_good_minus_none": d,
                "sd_none": s0,
                "sd_good": s1,
                "min_none": np.nanmin(x0),
                "max_none": np.nanmax(x0),
                "min_good": np.nanmin(x1),
                "max_good": np.nanmax(x1),
            }
        )

    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)

    cols = [
        "symbol",
        "feature",
        "eta2",
        "variance_rank_desc",
        "eta2_rank_desc",
        "mean_none",
        "mean_good",
        "diff_good_minus_none",
        "cohens_d_good_minus_none",
        "sd_none",
        "sd_good",
    ]
    cols = [c for c in cols if c in res.columns]
    print(res[cols].head(40).to_string(index=False))
    print("\nWROTE:", out)
    return out


def robustness(project_root: Path, run_tag: str, bundle: Path, top_n: int = 100) -> Path:
    """Compute non-parametric and outlier-sensitivity summaries for hidden genes."""
    try:
        from scipy.stats import mannwhitneyu
    except Exception as e:
        raise RuntimeError("The robustness step requires scipy.") from e

    out_dir = diagnosis_dir(project_root, run_tag)
    inp = out_dir / "cd4_hidden_gene_class_contrast.csv"
    out = out_dir / "cd4_hidden_gene_robustness.csv"

    if not inp.exists():
        raise FileNotFoundError(f"Missing class-contrast file: {inp}")

    z = safe_load_npz(bundle)
    X = z["X"].astype(float)
    y = z["y"].astype(int)
    features = (
        z["feature_names"].astype(str)
        if "feature_names" in z.files
        else np.array([f"f{i}" for i in range(X.shape[1])])
    )
    feature_to_idx = {f: i for i, f in enumerate(features)}

    df = pd.read_csv(inp).head(int(top_n))
    rows = []
    for _, r in df.iterrows():
        f = str(r["feature"])
        if f not in feature_to_idx:
            continue
        j = feature_to_idx[f]
        x0 = X[y == 0, j]
        x1 = X[y == 1, j]

        med0 = np.nanmedian(x0)
        med1 = np.nanmedian(x1)
        q10_0, q90_0 = np.nanquantile(x0, [0.10, 0.90])
        q10_1, q90_1 = np.nanquantile(x1, [0.10, 0.90])

        try:
            u_p = mannwhitneyu(x1, x0, alternative="two-sided").pvalue
        except Exception:
            u_p = np.nan

        diff = float(r["diff_good_minus_none"])
        abs_dev = np.abs(np.concatenate([x0 - np.nanmean(x0), x1 - np.nanmean(x1)]))
        max_abs_dev = np.nanmax(abs_dev)
        pooled_sd = np.nanmean([np.nanstd(x0, ddof=1), np.nanstd(x1, ddof=1)])
        outlier_score = max_abs_dev / pooled_sd if pooled_sd > 0 else np.nan

        rows.append(
            {
                "symbol": r.get("symbol"),
                "feature": f,
                "eta2": r.get("eta2"),
                "variance_rank_desc": r.get("variance_rank_desc"),
                "eta2_rank_desc": r.get("eta2_rank_desc"),
                "cohens_d": r.get("cohens_d_good_minus_none"),
                "median_none": med0,
                "median_good": med1,
                "median_diff_good_minus_none": med1 - med0,
                "mean_diff_good_minus_none": diff,
                "q10_none": q10_0,
                "q90_none": q90_0,
                "q10_good": q10_1,
                "q90_good": q90_1,
                "mannwhitney_p": u_p,
                "outlier_score": outlier_score,
            }
        )

    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)

    cols = [
        "symbol",
        "feature",
        "eta2",
        "variance_rank_desc",
        "cohens_d",
        "median_diff_good_minus_none",
        "mean_diff_good_minus_none",
        "mannwhitney_p",
        "outlier_score",
    ]
    cols = [c for c in cols if c in res.columns]
    print(res[cols].head(40).to_string(index=False))
    print("\nWROTE:", out)
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Explain the optional GSE138746 CD4 VAD-boundary case.")
    ap.add_argument("--project-root", default="", help="Project root; defaults to auto-detected current tree.")
    ap.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--k-values", default="1,5,10,20")
    ap.add_argument("--top-hidden", type=int, default=200)
    ap.add_argument("--top-n", type=int, default=100, help="Top hidden candidates for contrast/robustness.")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--class-contrast", action="store_true")
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve() if args.project_root else find_project_root()
    bundle = Path(args.bundle)
    if not bundle.is_absolute():
        bundle = (project_root / bundle).resolve()

    if args.all:
        args.diagnose = args.annotate = args.class_contrast = args.robustness = True

    if not any([args.diagnose, args.annotate, args.class_contrast, args.robustness]):
        ap.print_help()
        print("\nNo step selected. Use --all or one of --diagnose/--annotate/--class-contrast/--robustness.")
        return 2

    print(f"Project root: {project_root}")
    print(f"Run tag     : {args.run_tag}")
    print(f"Bundle      : {bundle}")
    print(f"Output dir  : {diagnosis_dir(project_root, args.run_tag)}")

    if args.diagnose:
        rc = run_diagnose(project_root, args.run_tag, bundle, args.k_values, args.top_hidden)
        if rc != 0:
            return rc

    if args.annotate:
        annotate_hidden_genes(project_root, args.run_tag)

    if args.class_contrast:
        class_contrast(project_root, args.run_tag, bundle, top_n=args.top_n)

    if args.robustness:
        robustness(project_root, args.run_tag, bundle, top_n=args.top_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
