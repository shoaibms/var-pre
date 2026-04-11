#!/usr/bin/env python
"""
compile_supplementary_tables.py

Compile *all* Supplementary Tables into:
  outputs/results/sup_table/

Table index (ST1–ST15):
  ST1   Dataset overview
  ST2   Model performance (per-fold)
  ST3   DI full table (+ CI where available)
  ST4   Ablation full (XGB + RF)
  ST5   Cross-model agreement (derived from ST4)
  ST6   Permutation results (label permutation robustness)
  ST7   SHAP stability
  ST8   Pathway detail
  ST9   VAD diagnostic full (all K)
  ST10  VAD perm-null full
  ST11  Simulation parameters/results (+ param sweeps)
  ST12  Top-100 variance features per view          [MASTER_PLAN S4]
  ST13  Q4 feature lists (low-var, high-importance)  [MASTER_PLAN S6]
  ST14  Unsupervised clustering comparison (C4.1)   [REVIEWER RESPONSE]
  ST15  Context-vs-modality signal enrichment (C4.2) [REVIEWER RESPONSE]

Principle: COPY + STANDARDISE + SUMMARISE (no heavy recomputation).
Safe to run before permutation finishes (ST6 will be skipped with notes).

Usage:
  python .\\code\\compute\\13_results\\compile_supplementary_tables.py --outputs-dir <path-to-outputs>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===================================================================
# Utilities
# ===================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv_any(path: Path) -> pd.DataFrame:
    """Read .csv or .csv.gz; return empty DataFrame on failure."""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError, Exception):
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_copy(src: Path, dst: Path) -> bool:
    try:
        ensure_dir(dst.parent)
        dst.write_bytes(src.read_bytes())
        return True
    except Exception:
        return False


def find_first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


def coerce_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from *candidates* that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


DISPLAY_NAMES = {
    "mlomics": "MLOmics",
    "ibdmdb": "IBDMDB",
    "ccle": "CCLE",
    "tcga_gbm": "TCGA-GBM",
}


def view_label(dataset: str, view: str) -> str:
    return f"{DISPLAY_NAMES.get(dataset, dataset)}/{view}"


# ===================================================================
# ST1  Dataset overview
# ===================================================================

def st1_dataset_overview(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    ST1: Dataset overview from bundle manifests.
    Falls back to regime_consensus.csv (14 views) if bundle manifest is absent.
    Enriches from splits, baselines, and QC manifests where available.
    """
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}

    # --- Try bundle manifest (several possible schema variants) ---
    cand_manifest = [
        outputs_dir / "bundles" / "bundle_manifest.json",
        outputs_dir / "01_bundles" / "bundle_manifest.json",
        outputs_dir / "01_bundles" / "bundles_manifest.json",
        outputs_dir / "01_bundles" / "manifest.json",
    ]
    p = find_first_existing(cand_manifest)
    rows: List[Dict[str, Any]] = []
    if p:
        meta["sources"].append(str(p))
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                # Schema A: {"results": [{"name":..., "views":{...}, "n_samples":...}]}
                for entry in obj.get("results", []):
                    if entry.get("status") != "success":
                        continue
                    ds = entry.get("name")
                    for vname, n_feat in entry.get("views", {}).items():
                        rows.append({
                            "dataset": ds,
                            "view": vname,
                            "n_samples": entry.get("n_samples"),
                            "n_features": n_feat,
                        })
                # Schema B: {"bundles": [...]}
                if not rows:
                    for b in obj.get("bundles", []):
                        rows.append({
                            "dataset": b.get("dataset"),
                            "view": b.get("view"),
                            "n_samples": b.get("n_samples"),
                            "n_features": b.get("n_features"),
                            "label": b.get("label"),
                        })
                # Schema C: {"views": [...]}
                if not rows:
                    for v in obj.get("views", []):
                        rows.append({
                            "dataset": v.get("dataset"),
                            "view": v.get("view"),
                            "n_samples": v.get("n_samples"),
                            "n_features": v.get("n_features"),
                        })
                # Schema D: flat dict-of-dicts
                if not rows:
                    for k, v in obj.items():
                        if isinstance(v, dict) and ("dataset" in v or "view" in v):
                            rows.append({
                                "dataset": v.get("dataset"),
                                "view": v.get("view"),
                                "n_samples": v.get("n_samples"),
                                "n_features": v.get("n_features"),
                                "label": v.get("label"),
                            })
            if not rows:
                notes.append("ST1: bundle_manifest.json found but schema could not be flattened.")
        except Exception as e:
            notes.append(f"ST1: Failed to parse bundle manifest ({p.name}): {e}")

    # --- Enrich from splits manifest ---
    splits_path = find_first_existing([
        outputs_dir / "splits" / "splits_manifest.json",
        outputs_dir / "03_supervised" / "splits" / "splits_manifest.json",
    ])
    if splits_path:
        meta["sources"].append(str(splits_path))
        try:
            sm = json.loads(splits_path.read_text(encoding="utf-8"))
            for ds, info in sm.get("datasets", {}).items():
                for r in rows:
                    if r["dataset"] == ds:
                        r.setdefault("n_classes", info.get("n_classes"))
                        r.setdefault("n_splits", info.get("n_splits"))
                        r.setdefault("n_repeats", info.get("n_repeats"))
                        gm = info.get("group_meta", {})
                        r.setdefault("group_cv", gm.get("status", "none"))
        except Exception:
            pass

    # --- Enrich from baselines / tree_models manifest ---
    baselines_path = find_first_existing([
        outputs_dir / "metrics" / "baselines" / "baselines_manifest.json",
        outputs_dir / "metrics" / "tree_models" / "tree_models_manifest.json",
    ])
    if baselines_path:
        meta["sources"].append(str(baselines_path))
        try:
            bsm = json.loads(baselines_path.read_text(encoding="utf-8"))
            for ds, ds_info in bsm.get("datasets", {}).items():
                for r in rows:
                    if r["dataset"] == ds:
                        if "n_samples" in ds_info and r.get("n_samples") is None:
                            r["n_samples"] = ds_info["n_samples"]
                        if "n_classes" in ds_info and r.get("n_classes") is None:
                            r["n_classes"] = ds_info["n_classes"]
                        if "label_meta" in ds_info:
                            lm = ds_info["label_meta"]
                            r.setdefault("task", lm.get(
                                "label_description", lm.get("label_column", "")))
        except Exception:
            pass

    # --- Enrich from QC bundle integrity ---
    qc_path = find_first_existing([outputs_dir / "qc" / "bundle_integrity.json"])
    if qc_path:
        meta["sources"].append(str(qc_path))
        try:
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            for ds, ds_info in qc.get("datasets", {}).items():
                for vname, v_info in ds_info.get("views", {}).items():
                    for r in rows:
                        if r["dataset"] == ds and r["view"] == vname:
                            if r.get("n_features") is None:
                                r["n_features"] = v_info.get("n_features")
                            if r.get("n_samples") is None:
                                r["n_samples"] = v_info.get("n_samples")
        except Exception:
            pass

    # --- Build DataFrame (or fallback to regime_consensus) ---
    if rows:
        df = pd.DataFrame(rows)
    else:
        rc = find_first_existing([
            outputs_dir / "04_importance" / "aggregated" / "regime_consensus.csv",
        ])
        if rc:
            meta["sources"].append(str(rc))
            df = read_csv_any(rc)
            keep = [c for c in ["dataset", "view", "n_features"] if c in df.columns]
            df = df[keep].copy() if keep else df.copy()
            notes.append("ST1: Using regime_consensus.csv as fallback view inventory.")
        else:
            return pd.DataFrame(), notes + [
                "ST1: No bundle manifest or regime_consensus.csv found."
            ], meta

    # Standardise types and add label
    if "n_samples" in df.columns:
        df["n_samples"] = coerce_int(df["n_samples"])
    if "n_features" in df.columns:
        df["n_features"] = coerce_int(df["n_features"])
    if "dataset" in df.columns and "view" in df.columns:
        df["view_label"] = df.apply(
            lambda r: view_label(r["dataset"], r["view"]), axis=1)

    # Reorder columns
    lead = ["dataset", "view", "view_label", "n_samples", "n_features",
            "n_classes", "task", "n_splits", "n_repeats", "group_cv"]
    present = [c for c in lead if c in df.columns]
    extra = [c for c in df.columns if c not in present]
    df = df[present + extra]

    return df, notes, meta


# ===================================================================
# ST2  Model performance
# ===================================================================

def st2_model_performance(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """ST2: Model performance from 03_supervised/eval."""
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}

    candidates = [
        outputs_dir / "03_supervised" / "eval" / "model_performance__wide.csv.gz",
        outputs_dir / "03_supervised" / "eval_xgb_bal" / "model_performance__wide.csv.gz",
        outputs_dir / "03_supervised" / "eval_rf" / "model_performance__wide.csv.gz",
        outputs_dir / "03_supervised" / "eval" / "model_performance__long.csv.gz",
    ]
    p = find_first_existing(candidates)
    if not p:
        return pd.DataFrame(), [
            "ST2: No 03_supervised eval model_performance file found."
        ], meta

    meta["sources"].append(str(p))
    return read_csv_any(p), notes, meta


# ===================================================================
# ST3  DI full table
# ===================================================================

def st3_di_full(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    ST3: DI full table across K, from fig1_di_curves.csv.
    Optionally merges bootstrap CI from di_summary.csv.
    """
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}

    di_curves = find_first_existing([
        outputs_dir / "05_decoupling" / "fig1_di_curves.csv",
    ])
    if not di_curves:
        return pd.DataFrame(), ["ST3: fig1_di_curves.csv not found."], meta

    meta["sources"].append(str(di_curves))
    di = read_csv_any(di_curves)

    # Optional: merge bootstrap percentiles
    di_sum = find_first_existing([
        outputs_dir / "04_importance" / "uncertainty" / "di_summary.csv",
    ])
    if di_sum:
        meta["sources"].append(str(di_sum))
        s = read_csv_any(di_sum)
        # Normalise percentile column names
        renames = {}
        if "DI_pctl_2_5" in s.columns and "DI_pctl_2.5" not in s.columns:
            renames["DI_pctl_2_5"] = "DI_pctl_2.5"
        if "DI_pctl_97_5" in s.columns and "DI_pctl_97.5" not in s.columns:
            renames["DI_pctl_97_5"] = "DI_pctl_97.5"
        if renames:
            s = s.rename(columns=renames)

        ci_cols = [c for c in s.columns if c.startswith("DI_pctl")]
        if ci_cols:
            merge_keys = [k for k in ["dataset", "view", "model", "k_pct"]
                          if k in di.columns and k in s.columns]
            if merge_keys:
                keep = merge_keys + ci_cols
                keep = [c for c in keep if c in s.columns]
                di = di.merge(s[keep].drop_duplicates(subset=merge_keys),
                              on=merge_keys, how="left", suffixes=("", "_ci"))
            else:
                notes.append("ST3: di_summary.csv found but no common merge keys.")
        else:
            notes.append("ST3: di_summary.csv found but CI columns missing.")
    else:
        notes.append("ST3: di_summary.csv not found; ST3 will not include CI columns.")

    return di, notes, meta


# ===================================================================
# ST4  Ablation full
# ===================================================================

def st4_ablation_full(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """ST4: Ablation full merge (XGB + RF masters)."""
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}

    ax = find_first_existing([
        outputs_dir / "07_ablation" / "ablation_master_summary.csv",
    ])
    ar = find_first_existing([
        outputs_dir / "07_ablation_rf" / "ablation_master_summary.csv",
    ])

    if not ax and not ar:
        return pd.DataFrame(), ["ST4: ablation_master_summary.csv not found."], meta

    frames = []
    if ax:
        meta["sources"].append(str(ax))
        frames.append(read_csv_any(ax))
    if ar and (ar != ax):
        meta["sources"].append(str(ar))
        frames.append(read_csv_any(ar))

    ab = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Standardise K column
    if "K_pct" in ab.columns and "k_pct" not in ab.columns:
        ab = ab.rename(columns={"K_pct": "k_pct"})
    if "k_pct" in ab.columns:
        ab["k_pct"] = coerce_int(ab["k_pct"])

    return ab, notes, meta


# ===================================================================
# ST5  Cross-model agreement (derived from ST4)
# ===================================================================

def st5_cross_model_agreement(
    ablation_full: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    ST5: Cross-model agreement derived from ST4.
    Pivots XGB vs RF delta(Var-Random) at K=10, adds Spearman + sign flags.
    """
    notes: List[str] = []
    if ablation_full.empty:
        return pd.DataFrame(), [
            "ST5: ablation_full is empty; cannot compute cross-model agreement."
        ]

    df = ablation_full.copy()

    # Normalise delta column name
    for old_name in ["delta_var_random_mean", "delta_var_random_mean_pp"]:
        if old_name in df.columns and "delta_var_minus_random" not in df.columns:
            df = df.rename(columns={old_name: "delta_var_minus_random"})

    # Filter to K=10
    k_col = pick_first_col(df, ["k_pct", "K_pct"])
    if k_col:
        df10 = df[df[k_col] == 10].copy()
    else:
        df10 = df.copy()
        notes.append("ST5: No k_pct column; using all rows.")

    # Filter to balanced_accuracy if metric column exists
    if "metric" in df10.columns:
        dfm = df10[df10["metric"].astype(str).str.contains(
            "bal|acc", case=False, na=False)].copy()
        if dfm.empty:
            dfm = df10.copy()
    else:
        dfm = df10.copy()

    if "model" not in dfm.columns or "delta_var_minus_random" not in dfm.columns:
        return pd.DataFrame(), notes + [
            "ST5: Required columns missing (model, delta_var_minus_random)."
        ]

    dfm["delta_var_minus_random"] = coerce_float(dfm["delta_var_minus_random"])

    # Aggregate per (dataset, view, metric, model)
    g = dfm.groupby(
        ["dataset", "view", "metric", "model"], as_index=False
    )["delta_var_minus_random"].mean()

    piv = g.pivot_table(
        index=["dataset", "view", "metric"],
        columns="model",
        values="delta_var_minus_random",
        aggfunc="mean",
    ).reset_index()

    # Identify model columns
    cols = list(piv.columns)
    xgb_col = next((c for c in cols if str(c) in ("xgb_bal", "xgb")), None)
    rf_col = "rf" if "rf" in cols else None

    if xgb_col is not None:
        piv["delta_xgb"] = piv[xgb_col]
    if rf_col is not None:
        piv["delta_rf"] = piv[rf_col]

    if "delta_xgb" in piv.columns and "delta_rf" in piv.columns:
        piv["sign_xgb"] = np.sign(piv["delta_xgb"])
        piv["sign_rf"] = np.sign(piv["delta_rf"])
        piv["sign_agree"] = (
            (piv["sign_xgb"] == piv["sign_rf"])
            & piv["sign_xgb"].notna()
            & piv["sign_rf"].notna()
        )
        piv["both_hurt"] = (piv["delta_xgb"] < 0) & (piv["delta_rf"] < 0)
        piv["both_help"] = (piv["delta_xgb"] > 0) & (piv["delta_rf"] > 0)

        # Spearman rho summary rows per metric
        extra_rows = []
        for metric, sub in piv.groupby("metric"):
            valid = sub.dropna(subset=["delta_xgb", "delta_rf"])
            if len(valid) >= 3:
                rho = valid["delta_xgb"].corr(valid["delta_rf"], method="spearman")
                extra_rows.append({
                    "dataset": "__ALL__",
                    "view": "__ALL__",
                    "metric": metric,
                    "delta_xgb": np.nan,
                    "delta_rf": np.nan,
                    "sign_agree": np.nan,
                    "both_hurt": np.nan,
                    "both_help": np.nan,
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "n_views": int(len(valid)),
                })
        if extra_rows:
            piv["spearman_rho"] = np.nan
            piv["n_views"] = np.nan
            piv = pd.concat([piv, pd.DataFrame(extra_rows)], ignore_index=True)

    return piv, notes


# ===================================================================
# ST6  Label permutation robustness
# ===================================================================

def st6_label_permutation(
    outputs_dir: Path,
    max_bytes_long: int = 250_000_000,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    ST6: Label permutation results.
    Prefers compact summary file; falls back to summarising _long file.
    """
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}

    dirs = [
        outputs_dir / "06_robustness_100" / "label_perm",
        outputs_dir / "06_robustness" / "label_perm",
    ]

    summary_candidates = []
    long_candidates = []
    for d in dirs:
        summary_candidates += [
            d / "label_perm_summary.csv",
            d / "label_perm_summary.csv.gz",
            d / "label_perm_deltas_summary.csv",
        ]
        long_candidates += [
            d / "label_perm_long.csv.gz",
            d / "label_perm_long.csv",
        ]

    # Prefer pre-computed summary
    p_sum = find_first_existing(summary_candidates)
    if p_sum:
        meta["sources"].append(str(p_sum))
        return read_csv_any(p_sum), notes, meta

    # Try long file with size guard
    p_long = find_first_existing(long_candidates)
    if not p_long:
        return pd.DataFrame(), [
            "ST6: No label permutation outputs found (summary or long)."
        ], meta

    meta["sources"].append(str(p_long))
    try:
        size = p_long.stat().st_size
        if size > max_bytes_long:
            return pd.DataFrame(), notes + [
                f"ST6: label_perm_long is large ({size:,} bytes) and no summary exists; "
                "skipping heavy summarisation.",
                "Tip: generate a compact label_perm_summary.csv then rerun.",
            ], meta
    except Exception:
        pass

    df = read_csv_any(p_long)
    if df.empty:
        return df, notes + ["ST6: label_perm_long exists but appears empty."], meta

    # Check required schema
    req = {"dataset", "view", "model", "metric", "strategy", "value"}
    present = req & set(df.columns)
    if present != req:
        return pd.DataFrame(), notes + [
            f"ST6: label_perm_long schema unexpected (missing {req - present}); "
            "cannot summarise safely."
        ], meta

    df["value"] = coerce_float(df["value"])
    strat_vals = sorted(df["strategy"].astype(str).unique())
    obs_keys = [s for s in strat_vals if s.lower() in ("real", "observed", "true")]
    perm_keys = [s for s in strat_vals if "perm" in s.lower()]
    if not obs_keys:
        obs_keys = [s for s in strat_vals if s not in perm_keys]
    if not perm_keys:
        return pd.DataFrame(), notes + [
            "ST6: No permutation strategy detected in label_perm_long."
        ], meta

    grp = ["dataset", "view", "model", "metric"]
    obs = df[df["strategy"].isin(obs_keys)]
    perm = df[df["strategy"].isin(perm_keys)]

    obs_g = obs.groupby(grp, as_index=False)["value"].mean().rename(
        columns={"value": "observed_mean"})
    perm_s = perm.groupby(grp)["value"].agg(
        perm_mean="mean", perm_std="std", n_perm="count"
    ).reset_index()

    out = obs_g.merge(perm_s, on=grp, how="left")

    # One-sided p-value: Pr(perm >= observed)
    merged = perm.merge(obs_g, on=grp, how="left")
    merged["ge"] = merged["value"] >= merged["observed_mean"]
    pvals = merged.groupby(grp, as_index=False)["ge"].agg(
        count_ge="sum", n_perm_total="count")
    pvals["p_perm_ge"] = (pvals["count_ge"] + 1.0) / (pvals["n_perm_total"] + 1.0)

    out = out.merge(pvals[grp + ["p_perm_ge"]], on=grp, how="left")
    return out, notes + [
        "ST6: Summarised from label_perm_long (no compact summary found)."
    ], meta


# ===================================================================
# ST7  SHAP stability
# ===================================================================

def st7_shap_stability(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    ST7: SHAP stability from dedicated summary or regime_consensus columns.
    """
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}

    # Try dedicated stability summary first
    stab = find_first_existing([
        outputs_dir / "06_robustness" / "stability" / "shap_stability_summary.csv",
        outputs_dir / "05_decoupling" / "shap_stability_summary.csv",
    ])
    if stab:
        meta["sources"].append(str(stab))
        return read_csv_any(stab), notes, meta

    # Fallback: extract from regime_consensus
    rc = find_first_existing([
        outputs_dir / "04_importance" / "aggregated" / "regime_consensus.csv",
    ])
    if not rc:
        return pd.DataFrame(), [
            "ST7: No stability summary or regime_consensus.csv found."
        ], meta

    meta["sources"].append(str(rc))
    df = read_csv_any(rc)
    keep = [c for c in df.columns
            if c in ("dataset", "view") or "stability" in c.lower()]
    if len(keep) <= 2:
        return pd.DataFrame(), notes + [
            "ST7: stability columns missing in regime_consensus.csv."
        ], meta

    return df[keep].copy(), notes, meta


# ===================================================================
# ST8  Pathway detail
# ===================================================================

def st8_pathway_detail(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """ST8: Pathway convergence summary or per-view enrichment results."""
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}

    p = find_first_existing([
        outputs_dir / "08_biology_multi_k" / "convergence_summary_multi_k.csv",
        outputs_dir / "08_biology" / "convergence_summary_multi_k.csv",
        outputs_dir / "05_decoupling" / "fig5_pathway_convergence.csv",
    ])
    if p:
        meta["sources"].append(str(p))
        return read_csv_any(p), notes, meta

    # Fallback: concatenate per-view enrichment CSVs
    for dirname in ["08_biology_k10", "08_biology"]:
        enr_dir = outputs_dir / dirname / "pathway_enrichment"
        if enr_dir.exists():
            parts = []
            for fp in sorted(enr_dir.rglob("*.csv*")):
                if fp.is_file():
                    meta["sources"].append(str(fp))
                    try:
                        df = read_csv_any(fp)
                        if not df.empty:
                            df["source_file"] = fp.name
                            parts.append(df)
                    except Exception:
                        pass
            if parts:
                result = pd.concat(parts, ignore_index=True).drop_duplicates()
                if "dataset" in result.columns and "view" in result.columns:
                    if "view_label" not in result.columns:
                        result["view_label"] = result.apply(
                            lambda r: view_label(r["dataset"], r["view"]), axis=1)
                return result, notes + [
                    "ST8: Convergence summary not found; concatenated per-view enrichment files."
                ], meta

    return pd.DataFrame(), [
        "ST8: No pathway convergence summary or enrichment files found."
    ], meta


# ===================================================================
# ST9  VAD diagnostic full
# ===================================================================

def st9_vad_full(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}
    p = find_first_existing([outputs_dir / "12_diagnostic" / "vad_summary.csv"])
    if not p:
        return pd.DataFrame(), ["ST9: vad_summary.csv not found."], meta
    meta["sources"].append(str(p))
    return read_csv_any(p), notes, meta


# ===================================================================
# ST10  VAD perm-null
# ===================================================================

def st10_vad_permnull(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}
    p = find_first_existing([
        outputs_dir / "12_diagnostic_permnull" / "permnull_summary.csv",
    ])
    if not p:
        return pd.DataFrame(), ["ST10: permnull_summary.csv not found."], meta
    meta["sources"].append(str(p))
    return read_csv_any(p), notes, meta


# ===================================================================
# ST11  Simulation (+ parameter sweeps)
# ===================================================================

def st11_simulation(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    ST11: Simulation validation.
    Merges fig6_simulation.csv with 1D/2D parameter sweep CSVs if present.
    """
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}
    parts: List[pd.DataFrame] = []

    # Core simulation summary
    p = find_first_existing([
        outputs_dir / "05_decoupling" / "fig6_simulation.csv",
        outputs_dir / "09_simulation" / "simulation_summary.csv",
    ])
    if p:
        meta["sources"].append(str(p))
        df = read_csv_any(p)
        if not df.empty:
            df["table_source"] = "regime_validation"
            parts.append(df)

    # 1D parameter sweep
    sweep1d = find_first_existing([
        outputs_dir / "09_simulation" / "param_sweeps" / "param_sweep_1d.csv",
        outputs_dir / "09_simulation" / "param_sweep_1d.csv",
    ])
    if sweep1d:
        meta["sources"].append(str(sweep1d))
        df = read_csv_any(sweep1d)
        if not df.empty:
            df["table_source"] = "param_sweep_1d"
            parts.append(df)

    # 2D parameter sweep
    sweep2d = find_first_existing([
        outputs_dir / "09_simulation" / "param_sweeps" / "param_sweep_2d.csv",
        outputs_dir / "09_simulation" / "param_sweep_2d.csv",
    ])
    if sweep2d:
        meta["sources"].append(str(sweep2d))
        df = read_csv_any(sweep2d)
        if not df.empty:
            df["table_source"] = "param_sweep_2d"
            parts.append(df)

    if not parts:
        return pd.DataFrame(), [
            "ST11: No simulation summary or sweep files found."
        ], meta

    result = pd.concat(parts, ignore_index=True)
    return result, notes, meta


# ===================================================================
# ST12  Top-N variance features per view  [MASTER_PLAN S4]
# ===================================================================

def _load_top_n_features(
    outputs_dir: Path,
    feature_type: str,
    top_n: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Shared helper for ST12 (variance).
    Glob-scans per-view canonical score files and returns top-N per view.
    """
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}
    all_parts: List[pd.DataFrame] = []

    if feature_type == "variance":
        score_dirs = [outputs_dir / "02_unsupervised" / "variance_scores"]
        patterns = ["variance_scores__*__*.csv*"]
        score_col_candidates = ["score", "marginal_variance", "variance"]
        rank_col_candidates = ["rank", "marginal_rank"]
        st_num = "ST12"
    else:  # shap / importance
        # Prefer P_consensus (model-aggregated), fall back to per_model
        score_dirs = [
            outputs_dir / "04_importance" / "aggregated",
            outputs_dir / "04_importance" / "per_model",
        ]
        patterns = [
            "importance__*__*__P_consensus.csv*",
            "importance__*__*__xgb_bal.csv*",  # fallback: xgb_bal per-model
        ]
        score_col_candidates = [
            "importance", "mean_abs_shap", "shap_importance",
            "P_consensus", "mean_importance",
        ]
        rank_col_candidates = ["rank"]
        st_num = "ST12_shap"  # unused; kept for potential future reactivation

    # Collect matching files across candidate dirs/patterns
    matched_files: List[Path] = []
    for sd, pat in zip(score_dirs, patterns):
        if sd.exists():
            hits = sorted(sd.glob(pat))
            if hits:
                matched_files = hits
                notes.append(f"{st_num}: Using {sd.name}/{pat} ({len(hits)} files).")
                break
    if not matched_files:
        return pd.DataFrame(), [
            f"{st_num}: No {feature_type} score files found in "
            f"{', '.join(str(d) for d in score_dirs)}."
        ], meta

    for fp in matched_files:
        meta["sources"].append(str(fp))
        try:
            df = read_csv_any(fp)
        except Exception:
            continue
        if df.empty:
            continue

        # Parse dataset/view from filename  (e.g. variance_scores__mlomics__mRNA.csv.gz)
        stem = fp.stem.replace(".csv", "")
        parts = stem.split("__")
        if feature_type == "variance" and len(parts) >= 3:
            dataset, view = parts[1], parts[2]
        elif feature_type == "shap" and len(parts) >= 4:
            dataset, view = parts[1], parts[2]
        else:
            continue

        score_col = pick_first_col(df, score_col_candidates)
        feat_col = pick_first_col(df, ["feature", "feature_name", "gene", "name"])

        if score_col is None or feat_col is None:
            continue

        # Sort descending, take top N
        df = df.sort_values(score_col, ascending=False).head(top_n).copy()

        out = pd.DataFrame({
            "dataset": dataset,
            "view": view,
            "view_label": view_label(dataset, view),
            "rank": range(1, len(df) + 1),
            "feature": df[feat_col].values,
            "score": df[score_col].values,
        })
        all_parts.append(out)

    if not all_parts:
        return pd.DataFrame(), notes + [
            f"{st_num}: No {feature_type} score files found in searched directories."
        ], meta

    result = pd.concat(all_parts, ignore_index=True)
    return result, notes, meta


def st12_top_variance_features(
    outputs_dir: Path, top_n: int = 100,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """ST12: Top-N variance features per view."""
    return _load_top_n_features(outputs_dir, "variance", top_n)


# ===================================================================
# ST13  Q4 feature lists  [MASTER_PLAN S6]
# ===================================================================

def st13_q4_features(
    outputs_dir: Path,
    k_pct: int = 10,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    ST13: Q4 (low-variance, high-importance) feature lists.

    Primary source: vp_joined files (per-view variance × importance tables),
    located at outputs/04_importance/joined_vp/vp_joined__{dataset}__{view}.csv(.gz)

    Q4 definition:
      - Low variance:       v_rank_pct > 0.50     (bottom half by variance)
      - High importance:    p_rank_pct <= k_pct/100  (top K% by importance)
    Uses consensus importance ranks when available.

    ST13 lists Q4 features under the low-variance × top-K% importance definition;
    some views may contribute zero features under this stricter criterion (see
    ST13_q4_features_view_summary.csv).

    Returns a tidy table with one row per Q4 feature.
    """
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": [], "k_pct": k_pct}
    parts: List[pd.DataFrame] = []
    view_rows: List[Dict[str, Any]] = []

    vp_dir = outputs_dir / "04_importance" / "joined_vp"
    if not vp_dir.exists():
        return pd.DataFrame(), [f"ST13: missing directory: {vp_dir}"], meta

    vp_files = sorted(vp_dir.glob("vp_joined__*__*.csv*"))
    if not vp_files:
        return pd.DataFrame(), [f"ST13: no vp_joined files found in: {vp_dir}"], meta

    notes.append(f"ST13: Extracting Q4 from {len(vp_files)} vp_joined files (K={k_pct}%).")

    for fp in vp_files:
        # Parse dataset/view from filename: vp_joined__DATASET__VIEW.csv(.gz)
        stem = fp.stem  # strips only last suffix (.gz); leaves .csv
        if stem.endswith(".csv"):
            stem = stem[:-4]
        toks = stem.split("__")
        if len(toks) < 3:
            continue
        dataset, view = toks[1], toks[2]

        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df.empty:
            continue

        meta["sources"].append(str(fp))  # counts processed views even if n_q4 == 0

        # Required columns
        if "feature" not in df.columns:
            continue

        # Variance rank + pct (prefer existing)
        if "v_rank" in df.columns:
            v_rank = pd.to_numeric(df["v_rank"], errors="coerce")
        elif "var_rank" in df.columns:
            v_rank = pd.to_numeric(df["var_rank"], errors="coerce")
        else:
            continue

        if "v_rank_pct" in df.columns:
            v_pct = pd.to_numeric(df["v_rank_pct"], errors="coerce")
        elif "var_rank_pct" in df.columns:
            v_pct = pd.to_numeric(df["var_rank_pct"], errors="coerce")
        else:
            n = float(len(df))
            v_pct = v_rank / max(1.0, n)

        # Importance rank + pct (prefer consensus)
        pred_rank_col = None
        for c in ["p_consensus_rank_int", "p_consensus_rank", "p_xgb_bal_rank", "p_rf_rank", "pred_rank", "shap_rank"]:
            if c in df.columns:
                pred_rank_col = c
                break
        if pred_rank_col is None:
            continue
        p_rank = pd.to_numeric(df[pred_rank_col], errors="coerce")

        if "p_consensus_rank_pct" in df.columns:
            p_pct = pd.to_numeric(df["p_consensus_rank_pct"], errors="coerce")
        elif "p_rank_pct" in df.columns:
            p_pct = pd.to_numeric(df["p_rank_pct"], errors="coerce")
        else:
            n = float(len(df))
            p_pct = p_rank / max(1.0, n)

        # --- Q4 selection (robust) ---
        # Low variance half:
        if "v_rank_pct" in df.columns:
            v_pct = pd.to_numeric(df["v_rank_pct"], errors="coerce")
            lowV = v_pct > 0.50
        elif "var_rank_pct" in df.columns:
            v_pct = pd.to_numeric(df["var_rank_pct"], errors="coerce")
            lowV = v_pct > 0.50
        elif "v_rank" in df.columns:
            v_rank = pd.to_numeric(df["v_rank"], errors="coerce")
            lowV = (v_rank / len(df)) > 0.50
        elif "var_rank" in df.columns:
            v_rank = pd.to_numeric(df["var_rank"], errors="coerce")
            lowV = (v_rank / len(df)) > 0.50
        else:
            raise ValueError("ST13: missing variance rank information (need v_rank_pct or v_rank).")

        # High prediction: prefer pipeline boolean if present (avoids pct scaling bugs)
        if "in_topP" in df.columns:
            topP = df["in_topP"].fillna(0).astype(int).astype(bool)
        else:
            # fallback to percentiles / ranks
            pred_pct_col = None
            for c in ["p_consensus_rank_pct", "p_xgb_bal_rank_pct", "p_rf_rank_pct", "pred_rank_pct", "p_rank_pct"]:
                if c in df.columns:
                    pred_pct_col = c
                    break
            if pred_pct_col is None:
                raise ValueError("ST13: missing prediction percentile info (need p_*_rank_pct or pred_rank_pct).")

            pred_pct = pd.to_numeric(df[pred_pct_col], errors="coerce")

            # normalize 0–100 -> 0–1 if needed
            mx = pred_pct.max(skipna=True)
            if pd.notna(mx) and mx > 1.0 + 1e-9:
                pred_pct = pred_pct / 100.0

            topP = pred_pct <= (k_pct / 100.0)

        q4_mask = lowV & topP
        q4 = df.loc[q4_mask].copy()
        q4["quadrant"] = "Q4"
        # --- end Q4 selection ---

        # Per-view summary (write even if n_q4 == 0)
        n_total = int(len(df))
        n_q4 = int(q4_mask.sum())
        view_label = f"{dataset}:{view}"
        view_rows.append({
            "dataset": dataset,
            "view": view,
            "view_label": view_label,
            "n_features_total": n_total,
            "n_q4": n_q4,
            "q4_pct": (100.0 * n_q4 / n_total) if n_total else 0.0,
        })

        if not q4_mask.any():
            continue

        out = pd.DataFrame({
            "dataset": dataset,
            "view": view,
            "view_label": f"{dataset}:{view}",
            "feature": df.loc[q4_mask, "feature"].astype(str).values,
            "var_rank": v_rank.loc[q4_mask].values,
            "var_rank_pct": v_pct.loc[q4_mask].round(4).values,
            "pred_rank": p_rank.loc[q4_mask].values,
            "pred_rank_pct": p_pct.loc[q4_mask].round(4).values,
            "n_features_total": int(len(df)),
            "quadrant": "Q4",
        }).sort_values("pred_rank").reset_index(drop=True)

        parts.append(out)

    # Write per-view Q4 summary
    summary_path = outputs_dir / "results" / "sup_table" / "ST13_q4_features_view_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if view_rows:
        summary_df = pd.DataFrame(view_rows).sort_values(["dataset", "view"])
    else:
        summary_df = pd.DataFrame(columns=[
            "dataset", "view", "view_label", "n_features_total", "n_q4", "q4_pct"
        ])
    summary_df.to_csv(summary_path, index=False)

    if not parts:
        return pd.DataFrame(), ["ST13: Q4 extraction found zero rows across vp_joined files."], meta

    result = pd.concat(parts, ignore_index=True).drop_duplicates()
    notes.append(f"ST13: Extracted {len(result)} Q4 rows across {result[['dataset','view']].drop_duplicates().shape[0]} views.")
    return result, notes, meta


# ===================================================================
# ST14  Unsupervised clustering comparison  [C4.1 REVIEWER RESPONSE]
# ===================================================================

def st14_unsupervised_clustering(
    outputs_dir: Path,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    ST14: Unsupervised clustering comparison across feature subsets.
    One row per view with ARI, NMI, PC_between for TopVar/Random/TopSHAP/All
    plus deltas.
    Source: outputs/02_unsupervised/clustering_comparison/
    """
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}

    clust_path = find_first_existing([
        outputs_dir / "02_unsupervised" / "clustering_comparison" / "clustering_comparison.csv",
    ])
    pc_path = find_first_existing([
        outputs_dir / "02_unsupervised" / "clustering_comparison" / "pc_class_signal.csv",
    ])

    if clust_path is None:
        return pd.DataFrame(), [
            "ST14: clustering_comparison.csv not found. "
            "Run 03_clustering_comparison.py first."
        ], meta

    meta["sources"].append(str(clust_path))
    clust = read_csv_any(clust_path)
    if clust.empty:
        return pd.DataFrame(), ["ST14: clustering_comparison.csv is empty."], meta

    # Pivot: one row per (dataset, view) with columns per strategy
    rows = []
    for (ds, vw), grp in clust.groupby(["dataset", "view"]):
        row = {"dataset": ds, "view": vw, "view_label": view_label(ds, vw)}
        for _, r in grp.iterrows():
            strat = r["strategy"]
            row[f"ARI_{strat}"] = r.get("ARI_mean")
            row[f"NMI_{strat}"] = r.get("NMI_mean")
        # Deltas
        ari_tv = row.get("ARI_TopVar")
        ari_rnd = row.get("ARI_Random")
        ari_ts = row.get("ARI_TopSHAP")
        if ari_tv is not None and ari_rnd is not None:
            row["delta_ARI_TopVar_Random"] = ari_tv - ari_rnd
        if ari_ts is not None and ari_rnd is not None:
            row["delta_ARI_TopSHAP_Random"] = ari_ts - ari_rnd
        rows.append(row)

    # Merge PC signal if available
    if pc_path and pc_path.exists():
        meta["sources"].append(str(pc_path))
        pc = read_csv_any(pc_path)
        if not pc.empty:
            for row in rows:
                ds, vw = row["dataset"], row["view"]
                for strat in ["TopVar", "TopSHAP", "Random", "All"]:
                    match = pc[(pc["dataset"] == ds) & (pc["view"] == vw)
                              & (pc["strategy"] == strat)]
                    if not match.empty:
                        row[f"PC_between_{strat}"] = match.iloc[0].get(
                            "pc_between_frac_mean")
            # PC delta
            for row in rows:
                pc_tv = row.get("PC_between_TopVar")
                pc_rnd = row.get("PC_between_Random")
                if pc_tv is not None and pc_rnd is not None:
                    row["delta_PC_between_TopVar_Random"] = pc_tv - pc_rnd

    result = pd.DataFrame(rows)

    # Reorder columns
    col_order = [
        "dataset", "view", "view_label",
        "ARI_All", "ARI_TopVar", "ARI_Random", "ARI_TopSHAP",
        "delta_ARI_TopVar_Random", "delta_ARI_TopSHAP_Random",
        "NMI_All", "NMI_TopVar", "NMI_Random", "NMI_TopSHAP",
        "PC_between_All", "PC_between_TopVar", "PC_between_Random",
        "PC_between_TopSHAP", "delta_PC_between_TopVar_Random",
    ]
    col_order = [c for c in col_order if c in result.columns]
    result = result[col_order]

    return result, notes, meta


# ===================================================================
# ST15  Context-vs-modality signal enrichment (C4.2)
# ===================================================================

def st15_context_vs_modality_signal_enrichment(
    outputs_dir: Path,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    ST15: C4.2 context-vs-modality summary.
    Source of truth:
      outputs/results/main_results/section_3_mechanism/signal_enrichment_by_regime.csv
    One row per dataset×view×k_pct with eta_topv_mean and eta_es_mean (plus DI/regime fields where present).
    """
    notes: List[str] = []
    meta: Dict[str, Any] = {"sources": []}

    p = find_first_existing([
        # Primary (your confirmed path)
        outputs_dir / "results" / "main_results" / "section_3_mechanism" / "signal_enrichment_by_regime.csv",
        outputs_dir / "results" / "main_results" / "section_3_mechanism" / "signal_enrichment_by_regime.csv.gz",

        # Reasonable fallbacks (in case of refactors)
        outputs_dir / "results" / "main_results" / "section_4_mechanism" / "signal_enrichment_by_regime.csv",
        outputs_dir / "results" / "main_results" / "section_4_mechanism" / "signal_enrichment_by_regime.csv.gz",
        outputs_dir / "05_decoupling" / "signal_enrichment_by_regime.csv",
        outputs_dir / "05_decoupling" / "signal_enrichment_by_regime.csv.gz",
    ])

    if not p:
        return pd.DataFrame(), [
            "ST15: signal_enrichment_by_regime.csv not found under results/main_results/section_3_mechanism/ "
            "or fallbacks."
        ], meta

    meta["sources"].append(str(p))
    df = read_csv_any(p)
    if df.empty:
        return pd.DataFrame(), ["ST15: signal_enrichment_by_regime.csv is empty."], meta

    # Standardise types / filter to K=10 where possible
    if "k_pct" in df.columns:
        df["k_pct"] = coerce_int(df["k_pct"])
        before = len(df)
        df = df[df["k_pct"] == 10].copy()
        if len(df) == 0:
            notes.append("ST15: k_pct column present but no rows at k_pct=10; returning all rows.")
            df = read_csv_any(p)
        else:
            notes.append(f"ST15: filtered to k_pct=10 ({len(df)}/{before} rows).")
    else:
        notes.append("ST15: no k_pct column; returning all rows.")

    # Add view_label for readability if dataset/view present
    if "dataset" in df.columns and "view" in df.columns and "view_label" not in df.columns:
        df["view_label"] = df.apply(lambda r: view_label(str(r["dataset"]), str(r["view"])), axis=1)

    # Preferred column order (keep only those present; append any remaining)
    preferred = [
        "dataset", "view", "view_label", "k_pct",
        "eta_topv_mean", "eta_es_mean",
        "f_di_mean", "consensus_regime", "predicted_zone",
        "n_rows",
    ]
    present = [c for c in preferred if c in df.columns]
    extra = [c for c in df.columns if c not in present]
    df = df[present + extra].copy()

    return df, notes, meta


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compile all Supplementary Tables (ST1-ST15)")
    ap.add_argument(
        "--outputs-dir", required=True,
        help="Path to outputs/")
    ap.add_argument(
        "--top-n-features", type=int, default=100,
        help="Number of top features for ST12 table.")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    top_n = args.top_n_features

    out_root = outputs_dir / "results" / "sup_table"
    src_dir = ensure_dir(out_root / "source_tables")
    ensure_dir(out_root)

    manifest: Dict[str, Any] = {
        "created_at": now_stamp(),
        "outputs_dir": str(outputs_dir),
        "tables": [],
        "notes": [],
        "missing": [],
    }

    def _record(
        name: str,
        out_path: Optional[Path],
        sources: List[str],
        notes: List[str],
        n_rows: Optional[int] = None,
    ):
        entry = {
            "name": name,
            "output": str(out_path) if out_path else None,
            "n_rows": n_rows,
            "sources": sources,
            "notes": notes,
        }
        if out_path and out_path.exists():
            entry["sha256"] = sha256_file(out_path)
            entry["bytes"] = out_path.stat().st_size
        manifest["tables"].append(entry)
        for n in notes:
            manifest["notes"].append(f"{name}: {n}")
        if out_path is None or not out_path.exists():
            manifest["missing"].append(name)

    def _build_csv(name: str, filename: str, builder_fn):
        """Run builder, write CSV, record in manifest."""
        print(f"  [{name}] Building {filename}...")
        df, notes, meta = builder_fn()
        p = out_root / filename
        if not df.empty:
            write_csv(df, p)
            _record(name, p, meta.get("sources", []), notes, n_rows=int(len(df)))
            print(f"    {len(df)} rows, {len(meta.get('sources', []))} sources")
        else:
            _record(name, None, meta.get("sources", []), notes)
            print(f"    EMPTY - see notes")

    # ==================================================================
    print("=" * 70)
    print("Compiling Supplementary Tables (ST1-ST15)")
    print(f"Output: {out_root}")
    print("=" * 70)

    # ---- ST1  Dataset overview ----
    _build_csv("ST1", "ST1_dataset_overview.csv",
               lambda: st1_dataset_overview(outputs_dir))

    # ---- ST2  Model performance ----
    _build_csv("ST2", "ST2_model_performance.csv",
               lambda: st2_model_performance(outputs_dir))

    # ---- ST3  DI full table ----
    _build_csv("ST3", "ST3_DI_full_table.csv",
               lambda: st3_di_full(outputs_dir))

    # ---- ST4  Ablation full (special: we need df for ST5) ----
    print(f"  [ST4] Building ST4_ablation_full.csv...")
    ab, ab_notes, ab_meta = st4_ablation_full(outputs_dir)
    p4 = out_root / "ST4_ablation_full.csv"
    if not ab.empty:
        write_csv(ab, p4)
        _record("ST4", p4, ab_meta.get("sources", []), ab_notes, n_rows=int(len(ab)))
        print(f"    {len(ab)} rows, {len(ab_meta.get('sources', []))} sources")
    else:
        _record("ST4", None, ab_meta.get("sources", []), ab_notes)
        print(f"    EMPTY - see notes")

    # ---- ST5  Cross-model agreement (derived from ST4) ----
    print(f"  [ST5] Building ST5_cross_model_agreement.csv...")
    df5, notes5 = st5_cross_model_agreement(ab)
    p5 = out_root / "ST5_cross_model_agreement.csv"
    if not df5.empty:
        write_csv(df5, p5)
        _record("ST5", p5, [], notes5, n_rows=int(len(df5)))
        print(f"    {len(df5)} rows")
    else:
        _record("ST5", None, [], notes5)
        print(f"    EMPTY - see notes")

    # ---- ST6  Label permutation ----
    _build_csv("ST6", "ST6_label_permutation_summary.csv",
               lambda: st6_label_permutation(outputs_dir))

    # ---- ST7  SHAP stability ----
    _build_csv("ST7", "ST7_shap_stability.csv",
               lambda: st7_shap_stability(outputs_dir))

    # ---- ST8  Pathway detail ----
    _build_csv("ST8", "ST8_pathway_detail.csv",
               lambda: st8_pathway_detail(outputs_dir))

    # ---- ST9  VAD diagnostic full ----
    _build_csv("ST9", "ST9_vad_summary_allK.csv",
               lambda: st9_vad_full(outputs_dir))

    # ---- ST10  VAD perm-null ----
    _build_csv("ST10", "ST10_vad_permnull_full.csv",
               lambda: st10_vad_permnull(outputs_dir))

    # ---- ST11  Simulation ----
    _build_csv("ST11", "ST11_simulation_summary.csv",
               lambda: st11_simulation(outputs_dir))

    # ---- ST12  Top variance features  [MASTER_PLAN S4] ----
    _build_csv("ST12", "ST12_top_variance_features.csv",
               lambda: st12_top_variance_features(outputs_dir, top_n))

    # ---- ST13  Q4 feature lists  [MASTER_PLAN S6] ----
    _build_csv("ST13", "ST13_q4_features.csv",
               lambda: st13_q4_features(outputs_dir))

    # ---- ST14  Unsupervised clustering comparison  [C4.1] ----
    _build_csv("ST14", "ST14_unsupervised_clustering.csv",
               lambda: st14_unsupervised_clustering(outputs_dir))

    # ---- ST15  Context-vs-modality signal enrichment  [C4.2] ----
    _build_csv("ST15", "ST15_context_vs_modality_signal_enrichment.csv",
               lambda: st15_context_vs_modality_signal_enrichment(outputs_dir))

    # ==================================================================
    # Provenance: copy key source tables
    # ==================================================================
    print(f"\n  Copying source tables for provenance...")
    copy_list = [
        ("regime_consensus.csv",
         outputs_dir / "04_importance" / "aggregated" / "regime_consensus.csv"),
        ("fig1_di_curves.csv",
         outputs_dir / "05_decoupling" / "fig1_di_curves.csv"),
        ("fig6_simulation.csv",
         outputs_dir / "05_decoupling" / "fig6_simulation.csv"),
        ("vad_summary.csv",
         outputs_dir / "12_diagnostic" / "vad_summary.csv"),
        ("permnull_summary.csv",
         outputs_dir / "12_diagnostic_permnull" / "permnull_summary.csv"),
        ("ablation_master_xgb.csv",
         outputs_dir / "07_ablation" / "ablation_master_summary.csv"),
        ("ablation_master_rf.csv",
         outputs_dir / "07_ablation_rf" / "ablation_master_summary.csv"),
        ("clustering_comparison.csv",
         outputs_dir / "02_unsupervised" / "clustering_comparison" / "clustering_comparison.csv"),
        ("pc_class_signal.csv",
         outputs_dir / "02_unsupervised" / "clustering_comparison" / "pc_class_signal.csv"),
        ("signal_enrichment_by_regime.csv",
         outputs_dir / "results" / "main_results" / "section_3_mechanism" / "signal_enrichment_by_regime.csv"),
    ]
    copied = []
    for name, src in copy_list:
        if src.exists():
            if safe_copy(src, src_dir / name):
                copied.append(name)
                print(f"    OK: {name}")
    manifest["copied_source_tables"] = copied

    # ==================================================================
    # Manifest + summary markdown
    # ==================================================================
    man_path = out_root / "MANIFEST_SUP_TABLE.json"
    write_json(manifest, man_path)

    # Summary markdown
    table_descriptions = {
        "ST1": "Dataset characteristics (N, features per view, task)",
        "ST2": "Model performance (per-fold, per-repeat)",
        "ST3": "DI at all K values (J, delta-J, DI) per view + CI",
        "ST4": "Ablation results (full: all K, all models, all metrics)",
        "ST5": "Cross-model agreement (XGB vs RF, Spearman, sign flags)",
        "ST6": "Label permutation robustness",
        "ST7": "SHAP stability (repeat-level rank overlap)",
        "ST8": "Pathway enrichment (full results / convergence)",
        "ST9": "VAD diagnostic metrics (all views x all K)",
        "ST10": "VAD perm-null full",
        "ST11": "Simulation validation (regime DI + parameter sweeps)",
        "ST12": "Top 100 variance features per view",
        "ST13": "Q4 feature lists (low-variance, high-importance)",
        "ST14": "Unsupervised clustering comparison (ARI, NMI, PC_between at K=10%)",
        "ST15": "Context-vs-modality signal enrichment at K=10% (eta_topv_mean, eta_es_mean, optional DI/regime)",
    }

    lines = [
        "# Supplementary Tables Compilation",
        "",
        f"- Created: {manifest['created_at']}",
        f"- outputs_dir: `{outputs_dir}`",
        "",
        "## Tables",
        "",
    ]
    total_rows = 0
    for t in manifest["tables"]:
        desc = table_descriptions.get(t["name"], "")
        n = t.get("n_rows")
        outp = t["output"] if t["output"] else "(missing)"
        status = "OK" if t["output"] else "MISSING"
        row_str = f"{n} rows" if n is not None else "n/a"
        lines.append(f"- **{t['name']}** ({desc}) -> `{outp}`  [{status}, {row_str}]")
        if n:
            total_rows += n
        if t.get("notes"):
            for note in t["notes"]:
                lines.append(f"  - _{note}_")

    lines += ["", f"**Total data rows: {total_rows:,}**"]

    if manifest["missing"]:
        lines += ["", "## Missing / deferred", ""]
        for m in manifest["missing"]:
            lines.append(f"- {m}")

    lines += ["", "## Provenance copies (`source_tables/`)", ""]
    for c in manifest.get("copied_source_tables", []):
        lines.append(f"- `{c}`")

    (out_root / "SUP_TABLE_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    # ==================================================================
    # Final console summary
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("Table Summary:")
    for t in manifest["tables"]:
        status = "OK" if t["output"] else "EMPTY"
        n = t.get("n_rows", 0) or 0
        name = t["name"]
        print(f"  {name:5s}  {status:5s}  {n:>7,d} rows")
    print(f"  {'':5s}  {'':5s}  {total_rows:>7,d} total rows")

    if manifest["missing"]:
        print(f"\nMissing tables: {', '.join(manifest['missing'])}")

    print(f"\n{'=' * 70}")
    print(f"DONE  -  Supplementary tables compiled to: {out_root}")
    print(f"Manifest: {man_path}")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())