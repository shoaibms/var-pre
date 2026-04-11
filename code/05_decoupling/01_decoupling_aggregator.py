#!/usr/bin/env python3
"""
05_decoupling_aggregator.py

Consolidates phase outputs into figure-ready CSVs.

Handles:
- Prefer ablation_master_summary.csv (avoid per-view JSON schema drift)
- Robust to DI percentile naming drift: DI_pctl_2_5 vs DI_pctl_2.5
- Robust to biology Q4 naming drift: Q4_pct vs Q4_fraction
- Allow label permutation results in alternate dir

Usage:
  python 05_decoupling_aggregator.py --outputs-dir <OUTPUTS_DIR>
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def view_key(ds: str, vw: str) -> str:
    return f"{ds}/{vw}"


def read_csv_any(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_csv_any(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.stat().st_size == 0:
        raise ValueError(f"empty CSV: {p}")
    return pd.read_csv(p)


def coalesce_cols(df: pd.DataFrame, candidates: List[str], out: str) -> pd.DataFrame:
    for c in candidates:
        if c in df.columns:
            df[out] = df[c]
            return df
    return df


def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_fig5_pathway_convergence_fallback(outputs_dir: Path | str, k_primary: int = 10) -> pd.DataFrame:
    """
    Fallback source for Fig5 when biology_summary.csv is missing:
      outputs/08_biology_multi_k/convergence_summary_multi_k.csv

    Returns a 14-row table with:
      dataset, view, gene_jaccard, pathway_jaccard, convergence_ratio
    """
    p = Path(outputs_dir) / "08_biology_multi_k" / "convergence_summary_multi_k.csv"
    if (not p.exists()) or (p.stat().st_size == 0):
        return pd.DataFrame(columns=["dataset", "view", "gene_jaccard", "pathway_jaccard", "convergence_ratio"])

    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["dataset", "view", "gene_jaccard", "pathway_jaccard", "convergence_ratio"])

    # Filter to k=10 if present
    if "k_pct" in df.columns:
        df = df[df["k_pct"] == k_primary].copy()
    elif "k" in df.columns:
        df = df[df["k"] == k_primary].copy()

    # Required keys
    if "dataset" not in df.columns or "view" not in df.columns:
        return pd.DataFrame(columns=["dataset", "view", "gene_jaccard", "pathway_jaccard", "convergence_ratio"])

    # Column name normalization (cover common schemas)
    gene_col = _pick_first_col(df, [
        "gene_jaccard", "obs_gene_jaccard", "gene_jaccard_mean", "obs_gene_jaccard_mean"
    ])
    path_col = _pick_first_col(df, [
        "pathway_jaccard", "obs_pathway_jaccard", "pathway_jaccard_mean", "obs_pathway_jaccard_mean"
    ])
    cr_col = _pick_first_col(df, [
        "convergence_ratio", "obs_convergence_ratio", "convergence_ratio_mean", "obs_convergence_ratio_mean"
    ])

    if gene_col is None or path_col is None or cr_col is None:
        # Keep it explicit: empty table → downstream "0 rows"
        return pd.DataFrame(columns=["dataset", "view", "gene_jaccard", "pathway_jaccard", "convergence_ratio"])

    out = df[["dataset", "view", gene_col, path_col, cr_col]].copy()
    out = out.rename(columns={
        gene_col: "gene_jaccard",
        path_col: "pathway_jaccard",
        cr_col: "convergence_ratio",
    })

    # If multiple rows remain per view (shouldn’t after k filter), keep one deterministically
    out = out.drop_duplicates(subset=["dataset", "view"], keep="first").reset_index(drop=True)
    return out


# -----------------------
# Loaders
# -----------------------

def load_di_summary(outputs_dir: Path) -> pd.DataFrame:
    path = outputs_dir / "04_importance" / "uncertainty" / "di_summary.csv"
    if not path.exists():
        print(f"  WARNING: DI summary not found: {path}")
        return pd.DataFrame()
    df = read_csv_any(path)
    df = coalesce_cols(df, ["DI_pctl_2_5", "DI_pctl_2.5"], "DI_pctl_2_5")
    df = coalesce_cols(df, ["DI_pctl_97_5", "DI_pctl_97.5"], "DI_pctl_97_5")
    return df


def load_ablation_master(outputs_dir: Path) -> pd.DataFrame:
    master_path = outputs_dir / "07_ablation" / "ablation_master_summary.csv"
    if not master_path.exists():
        print(f"  WARNING: Ablation master not found: {master_path}")
        return pd.DataFrame()
    df = read_csv_any(master_path)
    print(f"  Loaded ablation master: {len(df)} rows from {master_path}")
    return df


def load_ablation_rf(outputs_dir: Path) -> pd.DataFrame:
    candidates = [
        outputs_dir / "per_view__hero_rf_20260205" / "ablation_master_summary.csv",
        outputs_dir / "07_ablation_rf" / "ablation_master_summary.csv",
    ]
    for p in candidates:
        if p.exists():
            df = read_csv_any(p)
            df["model"] = "rf"
            print(f"  Loaded RF ablation: {len(df)} rows from {p}")
            return df
    print("  INFO: RF ablation not found (optional).")
    return pd.DataFrame()


def load_label_perm(outputs_dir: Path, label_perm_dirname: Optional[str]) -> pd.DataFrame:
    base = (outputs_dir / label_perm_dirname) if label_perm_dirname else (outputs_dir / "06_robustness" / "label_perm")

    # Prefer deltas summary if present
    for fname in ["label_perm_deltas_summary.csv", "label_perm_summary.csv", "label_perm_means_by_strategy.csv"]:
        p = base / fname
        if p.exists():
            df = read_csv_any(p)
            df.attrs["source_path"] = str(p)
            print(f"  Loaded label permutation: {len(df)} rows from {p}")
            return df

    print(f"  WARNING: label permutation summaries not found under: {base}")
    return pd.DataFrame()


def load_stability(outputs_dir: Path) -> pd.DataFrame:
    p = outputs_dir / "06_robustness" / "stability" / "shap_stability_summary.csv"
    if not p.exists():
        print(f"  INFO: stability summary not found: {p}")
        return pd.DataFrame()
    return read_csv_any(p)


def load_agreement(outputs_dir: Path) -> pd.DataFrame:
    p = outputs_dir / "06_robustness" / "agreement" / "shap_agreement_summary.csv"
    if not p.exists():
        print(f"  INFO: agreement summary not found: {p}")
        return pd.DataFrame()
    return read_csv_any(p)


def load_biology(outputs_dir: Path) -> pd.DataFrame:
    # Legacy + current layouts (don’t silently “succeed” with empty)
    candidates = [
        outputs_dir / "08_biology" / "biology_summary.csv",
        outputs_dir / "08_biology_k10" / "biology_summary.csv",
        outputs_dir / "08_biology_k10" / "biology_summary.csv.gz",
        outputs_dir / "08_biology_multi_k" / "biology_summary.csv",
        outputs_dir / "08_biology_multi_k" / "biology_summary.csv.gz",
    ]
    for p in candidates:
        try:
            return _read_csv_any(p)
        except Exception:
            continue

    # Fallback: use multi-k convergence summary (filter to k=10 if present)
    p = outputs_dir / "08_biology_multi_k" / "convergence_summary_multi_k.csv"
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p)

        # Try to filter to k=10 / k_pct=10 if columns exist
        if "k_pct" in df.columns:
            df = df[df["k_pct"] == 10].copy()
        elif "k" in df.columns:
            df = df[df["k"] == 10].copy()

        # Normalise column names to what fig5 builder expects
        rename = {}
        if "obs_gene_jaccard" in df.columns and "gene_jaccard" not in df.columns:
            rename["obs_gene_jaccard"] = "gene_jaccard"
        if "obs_pathway_jaccard" in df.columns and "pathway_jaccard" not in df.columns:
            rename["obs_pathway_jaccard"] = "pathway_jaccard"
        if "obs_convergence_ratio" in df.columns and "convergence_ratio" not in df.columns:
            rename["obs_convergence_ratio"] = "convergence_ratio"
        df = df.rename(columns=rename)

        return df

    print("[WARN] biology_summary not found (or empty) in any expected location; continuing with VP-based fallbacks.")
    return pd.DataFrame()


def load_simulation(outputs_dir: Path) -> pd.DataFrame:
    p = outputs_dir / "09_simulation" / "decoupling_results" / "sim_di_summary.csv"
    if not p.exists():
        print(f"  WARNING: simulation summary not found: {p}")
        return pd.DataFrame()
    return read_csv_any(p)


def load_vp_exemplars(outputs_dir: Path, exemplars: List[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
    vp_dir = outputs_dir / "04_importance" / "joined_vp"
    if not vp_dir.exists():
        print(f"  WARNING: joined_vp dir not found: {vp_dir}")
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for ds, vw in exemplars:
        p = vp_dir / f"vp_joined__{ds}__{vw}.csv.gz"
        if p.exists():
            out[view_key(ds, vw)] = read_csv_any(p)
    return out


# -----------------------
# Builders
# -----------------------

def build_fig1_di_curves(di_summary: pd.DataFrame) -> pd.DataFrame:
    if di_summary.empty:
        return pd.DataFrame()
    df = di_summary.copy()
    df = coalesce_cols(df, ["DI_pctl_2_5", "DI_pctl_2.5"], "DI_pctl_2_5")
    df = coalesce_cols(df, ["DI_pctl_97_5", "DI_pctl_97.5"], "DI_pctl_97_5")
    cols = [c for c in [
        "dataset", "view", "model", "k_pct",
        "DI_mean", "DI_pctl_2_5", "DI_pctl_97_5",
        "consensus_regime", "regime_confidence"
    ] if c in df.columns]
    return df[cols].copy()


def build_fig3_ablation(ablation: pd.DataFrame) -> pd.DataFrame:
    if ablation.empty:
        return pd.DataFrame()
    keep = [c for c in [
        "dataset","view","model","metric","K_pct","regime","regime_confidence","DI_mean",
        "perf_all_mean","perf_var_mean","perf_random_mean","perf_shap_mean",
        "perf_var_ci_lo","perf_var_ci_hi","perf_random_ci_lo","perf_random_ci_hi","perf_shap_ci_lo","perf_shap_ci_hi",
        "delta_shap_var_mean","delta_shap_var_ci_lo","delta_shap_var_ci_hi",
        "delta_var_random_mean","delta_var_random_ci_lo","delta_var_random_ci_hi",
        "jaccard_var_shap","n_features_total","n_features_at_K","runtime_seconds"
    ] if c in ablation.columns]
    return ablation[keep].copy()


def build_fig3_ablation_rf(ablation_rf: pd.DataFrame, di_summary: pd.DataFrame) -> pd.DataFrame:
    if ablation_rf.empty:
        return pd.DataFrame()
    df = ablation_rf.copy()
    if not di_summary.empty and "consensus_regime" in di_summary.columns:
        di10 = di_summary[di_summary["k_pct"] == 10][["dataset","view","consensus_regime","DI_mean"]].drop_duplicates()
        df = df.merge(di10, on=["dataset","view"], how="left")
        if "regime" in df.columns:
            df["regime"] = df["regime"].where(df["regime"].astype(str) != "UNKNOWN", df["consensus_regime"])
    keep = [c for c in [
        "dataset","view","model","metric","K_pct","regime","consensus_regime","DI_mean",
        "perf_var_mean","perf_random_mean","delta_var_random_mean"
    ] if c in df.columns]
    return df[keep].copy()


def _parse_vp_joined_name(p: Path) -> tuple[str, str]:
    # vp_joined__{dataset}__{view}.csv or .csv.gz
    name = p.name
    if not name.startswith("vp_joined__"):
        raise ValueError(f"unexpected vp_joined filename: {name}")
    parts = name.split("__", 2)
    if len(parts) != 3:
        raise ValueError(f"unexpected vp_joined filename: {name}")
    dataset = parts[1]
    view = parts[2]
    view = re.sub(r"\.csv(\.gz)?$", "", view)
    return dataset, view


def _rank_to_pct(rank_like: pd.Series) -> pd.Series:
    x = pd.to_numeric(rank_like, errors="coerce").astype(float)
    # Drop all-NA early
    if x.notna().sum() == 0:
        raise ValueError("rank column is all-NA")
    mn, mx = float(x.min()), float(x.max())
    if mx == mn:
        # Degenerate: everything same rank → median
        return pd.Series(np.full(len(x), 50.0), index=x.index)
    return (x - mn) / (mx - mn) * 100.0


def _get_axis_pct(df: pd.DataFrame, axis: str) -> pd.Series:
    """
    Return 0..100 'rank percentile' for:
      - axis='var' : variance rank (0 best/highest variance, 100 worst/lowest variance)
      - axis='pred': importance rank (0 best/highest importance, 100 worst/lowest importance)
    Accepts either *_pct/percentile columns, or rank columns, or (last resort) raw score columns.
    """
    cols = list(df.columns)

    if axis == "var":
        # Accept: var_rank_pct, variance_rank_pct, V_rank_pct, V_rank, V
        pct_pats  = [
            r"(?:^|_)(?:var|variance|v)_rank_(?:pct|percentile|pctl|pctile)$",
            r"(?:var|variance|v).*rank.*(?:pct|percentile|pctl|pctile)",
            r"^(?:V|v)_(?:rank_)?(?:pct|percentile|pctl|pctile)$",
        ]
        rank_pats = [
            r"(?:^|_)(?:var|variance|v)_rank$",
            r"(?:var|variance|v).*rank$",
            r"^(?:V|v)_rank$",
        ]
        val_pats  = [
            r"^(?:V|v)$",
            r"^(?:variance|var)$",
            r"(?:^|_)(?:variance|var)(?:$|_)",
            r"^(?:sd|std|sigma2|sigma)$",
        ]
        ascending = False

    elif axis == "pred":
        # Accept: pred_rank_pct, shap_rank_pct, importance_rank_pct, P_rank_pct, P_rank, P
        pct_pats  = [
            r"(?:^|_)(?:pred|p|shap|importance|imp)_rank_(?:pct|percentile|pctl|pctile)$",
            r"(?:pred|p|shap|importance|imp).*rank.*(?:pct|percentile|pctl|pctile)",
            r"^(?:P|p)_(?:rank_)?(?:pct|percentile|pctl|pctile)$",
        ]
        rank_pats = [
            r"(?:^|_)(?:pred|p|shap|importance|imp)_rank$",
            r"(?:pred|p|shap|importance|imp).*rank$",
            r"^(?:P|p)_rank$",
        ]
        val_pats  = [
            r"^(?:P|p)$",
            r"^(?:pred|shap|importance|imp)$",
            r"(?:^|_)(?:pred|shap|importance|imp)(?:$|_)",
            r"^(?:mean_abs_shap|abs_shap_mean)$",
        ]
        ascending = False
    else:
        raise ValueError(axis)

    def find_first(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            for c in cols:
                if re.search(pat, c, flags=re.IGNORECASE):
                    return c
        return None

    # 1) If we already have a pct/percentile col, standardize it to 0..100
    c_pct = find_first(pct_pats)
    if c_pct is not None:
        s = pd.to_numeric(df[c_pct], errors="coerce").astype(float)
        if s.notna().sum() == 0:
            raise ValueError(f"{axis} pct col all-NA: {c_pct}")
        # allow 0..1 or 0..100
        if float(s.max()) <= 1.5:
            s = s * 100.0
        return s

    # 2) Otherwise compute pct from a rank column
    c_rank = find_first(rank_pats)
    if c_rank is not None:
        return _rank_to_pct(df[c_rank])

    # 3) Last resort: compute ranks from a raw score column
    c_val = find_first(val_pats)
    if c_val is not None:
        v = pd.to_numeric(df[c_val], errors="coerce").astype(float)
        if v.notna().sum() == 0:
            raise ValueError(f"{axis} value col all-NA: {c_val}")
        # Higher score = better → rank ascending=False
        r = v.rank(method="average", ascending=ascending)
        return _rank_to_pct(r)

    raise ValueError(
        f"Could not infer {axis} axis columns (need rank_pct, rank, or raw score). "
        f"Columns: {list(df.columns)[:40]}"
    )


def build_fig4_hidden_biomarkers_from_vp(outputs_dir: Path) -> pd.DataFrame:
    cols_out = ["dataset", "view", "Q4_count", "Q4_pct"]
    vp_dir = outputs_dir / "04_importance" / "joined_vp"
    if not vp_dir.exists():
        print(f"[WARN] vp_joined directory not found: {vp_dir}")
        return pd.DataFrame(columns=cols_out)

    rows = []
    for i, p in enumerate(sorted(vp_dir.glob("vp_joined__*.csv*"))):
        try:
            df = _read_csv_any(p)
        except Exception as e:
            print(f"[WARN] skipping vp_joined (unreadable): {p} ({e})")
            continue

        dataset, view = _parse_vp_joined_name(p)

        try:
            var_pct = _get_axis_pct(df, "var")
            pred_pct = _get_axis_pct(df, "pred")
        except Exception as e:
            print(f"[WARN] skipping vp_joined (cannot infer ranks): {p} ({e})")
            continue

        if i == 0:
            print(f"[INFO] using columns for {p.name}: var_axis inferred OK, pred_axis inferred OK")

        # Q4: low-variance (bottom half) AND high-importance (top half)
        q4 = (var_pct >= 50.0) & (pred_pct <= 50.0)
        q4_count = int(q4.sum())
        n = int(len(df))
        q4_pct = (100.0 * q4_count / n) if n > 0 else float("nan")

        rows.append({"dataset": dataset, "view": view, "Q4_count": q4_count, "Q4_pct": q4_pct})

    out = pd.DataFrame(rows, columns=cols_out)
    return out


def build_fig4_hidden_biomarkers(outputs_dir: Path, bio: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "view", "Q4_count", "Q4_pct"]
    # Prefer biology_summary if it actually contains the fields
    if set(cols).issubset(bio.columns) and len(bio) > 0:
        return bio[cols].copy()
    # Fallback: compute from vp_joined ranks (canonical + always available)
    return build_fig4_hidden_biomarkers_from_vp(outputs_dir)


def build_fig5_pathway_convergence(biology: pd.DataFrame) -> pd.DataFrame:
    if biology.empty:
        return pd.DataFrame()
    cols = [c for c in ["dataset","view","gene_jaccard","pathway_jaccard","convergence_ratio"] if c in biology.columns]
    return biology[cols].copy() if len(cols) >= 3 else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--out-dirname", default="10_figures/data")
    ap.add_argument("--label-perm-dirname", default=None,
                    help=r'Override permutation dir, e.g. "06_robustness_100\label_perm"')
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / args.out_dirname
    ensure_dir(out_dir)

    print(f"[{now_iso()}] Aggregating for figures...")

    di = load_di_summary(outputs_dir)
    abl = load_ablation_master(outputs_dir)
    abl_rf = load_ablation_rf(outputs_dir)
    lp = load_label_perm(outputs_dir, args.label_perm_dirname)
    stab = load_stability(outputs_dir)
    agr = load_agreement(outputs_dir)
    bio = load_biology(outputs_dir)
    sim = load_simulation(outputs_dir)

    # VP exemplars (hero)
    vp = load_vp_exemplars(outputs_dir, [("mlomics","methylation"),("ibdmdb","MGX"),("ccle","mRNA")])

    fig1_curves = build_fig1_di_curves(di)
    fig3 = build_fig3_ablation(abl)
    fig3rf = build_fig3_ablation_rf(abl_rf, di)
    fig4 = build_fig4_hidden_biomarkers(outputs_dir, bio)
    fig5 = build_fig5_pathway_convergence(bio)

    # Fallback: pull from 08_biology_multi_k if biology_summary is missing
    if fig5 is None or fig5.empty:
        fig5 = load_fig5_pathway_convergence_fallback(outputs_dir, k_primary=10)

    def save(df: pd.DataFrame, name: str):
        out_path = out_dir / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  wrote {name}: {len(df)} rows")

    save(fig1_curves, "fig1_di_curves.csv")
    save(fig3, "fig3_ablation.csv")
    save(fig3rf, "fig3_ablation_rf.csv")
    save(fig4, "fig4_hidden_biomarkers.csv")
    save(fig5, "fig5_pathway_convergence.csv")
    save(sim if not sim.empty else pd.DataFrame(), "fig6_simulation.csv")
    save(lp if not lp.empty else pd.DataFrame(), "perm_summary.csv")
    save(stab if not stab.empty else pd.DataFrame(), "shap_stability_summary.csv")
    save(agr if not agr.empty else pd.DataFrame(), "shap_agreement_summary.csv")

    manifest = {
        "timestamp": now_iso(),
        "outputs_dir": str(outputs_dir),
        "out_dir": str(out_dir),
        "label_perm_source": lp.attrs.get("source_path") if isinstance(lp, pd.DataFrame) else None
    }
    (out_dir / "MANIFEST__05_decoupling_aggregator.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{now_iso()}] Done.")


if __name__ == "__main__":
    main()
