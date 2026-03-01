#!/usr/bin/env python3
"""
audit_summary.py

One-shot audit that produces:
  - outputs/10_audit/AUDIT_REPORT.md
  - outputs/10_audit/view_audit_table.csv
  - outputs/10_audit/hero_rankings.csv

Design goals:
  - Runs safely while long jobs are IN PROGRESS (RF ablation, label permutation).
  - Prefers master summaries when available; otherwise builds partial summaries from per_view JSON/CSV.
  - Focus: "what is the strongest story + which hero views to show", not figure rendering.

Usage (PowerShell):
  python .\code\compute\10_audit\audit_summary.py `
    --outputs-dir "<path-to-outputs>" `
    --label-perm-dirname "06_robustness_100\\label_perm" `
    --rf-ablation-dirname "07_ablation_rf" `
    --primary-k 10 `
    --primary-metric balanced_accuracy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd


# -------------------------
# small utilities
# -------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def read_csv_any(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    if p.suffix == ".gz":
        return pd.read_csv(p, compression="gzip")
    return pd.read_csv(p)

def pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coalesce_cols(df: pd.DataFrame, candidates: List[str], out: str) -> pd.DataFrame:
    if df.empty:
        return df
    if out in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df[out] = df[c]
            return df
    df[out] = np.nan
    return df

def view_key(dataset: str, view: str) -> str:
    return f"{dataset}:{view}"

def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str) and x.strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def sign_eps(x: float, eps: float = 0.005) -> int:
    """Return -1, 0, +1 with a deadzone around 0 to avoid noise flips."""
    x = safe_float(x)
    if np.isnan(x):
        return 0
    if abs(x) < eps:
        return 0
    return 1 if x > 0 else -1


# -------------------------
# loaders
# -------------------------

def load_di_summary(outputs_dir: Path) -> pd.DataFrame:
    p = outputs_dir / "04_importance" / "uncertainty" / "di_summary.csv"
    df = read_csv_any(p)
    if df.empty:
        return df
    df = coalesce_cols(df, ["DI_pctl_2_5", "DI_pctl_2.5"], "DI_pctl_2_5")
    df = coalesce_cols(df, ["DI_pctl_97_5", "DI_pctl_97.5"], "DI_pctl_97_5")
    return df

def build_master_summary_from_ablation_json(per_view_json: List[dict]) -> pd.DataFrame:
    """
    Mirrors the structure used by 07_ablation/01_feature_subset_ablation.py for ablation_master_summary.csv.
    Works on partial lists too (in-progress run).
    """
    rows = []
    for s in per_view_json:
        dataset = s["dataset"]
        view = s["view"]
        model = s["model"]
        metrics = s.get("metrics", [s.get("metric", "balanced_accuracy")])
        reg = s.get("regime_info", {}) or {}
        regime = reg.get("regime", "UNKNOWN")
        regime_conf = reg.get("regime_confidence", np.nan)
        DI_mean = reg.get("DI_mean", np.nan)
        DI_lo = reg.get("DI_pctl_2.5", reg.get("DI_pctl_2_5", np.nan))
        DI_hi = reg.get("DI_pctl_97.5", reg.get("DI_pctl_97_5", np.nan))
        regime_k = reg.get("regime_k_pct", np.nan)
        regime_source = reg.get("regime_source", "none")
        runtime_sec = s.get("runtime_seconds", np.nan)
        n_repeats = s.get("n_repeats", np.nan)
        n_folds = s.get("n_folds", np.nan)

        rbm = s.get("results_by_metric", {})
        for m in metrics:
            if m not in rbm:
                continue
            for K_str, block in rbm[m].items():
                K = int(K_str)

                def unpack(key: str):
                    b = (block or {}).get(key, {}) or {}
                    ci = b.get("ci95", [np.nan, np.nan]) or [np.nan, np.nan]
                    return b.get("mean", np.nan), ci[0], ci[1]

                pa, pa_lo, pa_hi = unpack("perf_all")
                pv, pv_lo, pv_hi = unpack("perf_var")
                ps, ps_lo, ps_hi = unpack("perf_shap")
                pr, pr_lo, pr_hi = unpack("perf_random")
                dsv, dsv_lo, dsv_hi = unpack("delta_shap_minus_var")
                dvr, dvr_lo, dvr_hi = unpack("delta_var_minus_random")

                rows.append(dict(
                    dataset=dataset, view=view, model=model,
                    metric=m, K_pct=K,
                    regime=regime, regime_confidence=regime_conf,
                    DI_mean=DI_mean, DI_pctl_2_5=DI_lo, DI_pctl_97_5=DI_hi,
                    regime_k_pct=regime_k, regime_source=regime_source,
                    perf_all_mean=pa, perf_all_ci_lo=pa_lo, perf_all_ci_hi=pa_hi,
                    perf_var_mean=pv, perf_var_ci_lo=pv_lo, perf_var_ci_hi=pv_hi,
                    perf_shap_mean=ps, perf_shap_ci_lo=ps_lo, perf_shap_ci_hi=ps_hi,
                    perf_random_mean=pr, perf_random_ci_lo=pr_lo, perf_random_ci_hi=pr_hi,
                    delta_shap_var_mean=dsv, delta_shap_var_ci_lo=dsv_lo, delta_shap_var_ci_hi=dsv_hi,
                    delta_var_random_mean=dvr, delta_var_random_ci_lo=dvr_lo, delta_var_random_ci_hi=dvr_hi,
                    jaccard_var_shap=(block or {}).get("jaccard_var_shap", np.nan),
                    runtime_seconds=runtime_sec, n_repeats=n_repeats, n_folds=n_folds
                ))

    return pd.DataFrame(rows)

def load_ablation(outputs_dir: Path, dirname: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Returns: (master_like_df, status)
      status contains: { "views_completed": int, "views_expected": Optional[int] }
    """
    out_root = outputs_dir / dirname
    master_path = out_root / "ablation_master_summary.csv"
    if master_path.exists():
        df = read_csv_any(master_path)
        # estimate completed views from unique dataset/view
        views_completed = int(df.dropna(subset=["dataset", "view"]).drop_duplicates(["dataset", "view"]).shape[0]) if not df.empty else 0
        return df, {"views_completed": views_completed, "views_expected": -1, "source": "master"}

    # partial (in-progress): read per_view JSON summaries
    per_view_dir = out_root / "per_view"
    json_files = sorted(per_view_dir.glob("ablation_summary__*__*__*.json"))
    per_view = []
    for jf in json_files:
        try:
            per_view.append(read_json(jf))
        except Exception:
            continue

    df = build_master_summary_from_ablation_json(per_view) if per_view else pd.DataFrame()
    views_completed = len({view_key(s["dataset"], s["view"]) for s in per_view}) if per_view else 0
    return df, {"views_completed": views_completed, "views_expected": -1, "source": "per_view_json"}

def load_label_perm(outputs_dir: Path, label_perm_dirname: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    base = (outputs_dir / label_perm_dirname) if label_perm_dirname else (outputs_dir / "06_robustness" / "label_perm")

    # Prefer summary if present (created at end of run)
    summ = base / "label_perm_summary.csv"
    if summ.exists():
        df = read_csv_any(summ)
        return df, {"source": str(summ)}

    # Otherwise, attempt to assemble from per_view files if any exist
    per_view_dir = base / "per_view"
    files = sorted(per_view_dir.glob("label_perm__*.csv.gz"))
    if not files:
        return pd.DataFrame(), {"source": f"missing (checked {summ} and {per_view_dir})"}

    dfs = []
    for f in files:
        try:
            dfs.append(read_csv_any(f))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(), {"source": f"failed_read (files={len(files)})"}

    long_df = pd.concat(dfs, ignore_index=True)

    # compute lightweight summary matching the in-script intent:
    # mean across folds/repeats per perm_seed, then delta stats across perm_seeds
    need = {"dataset", "view", "model", "perm_seed", "metric", "k_pct", "strategy", "value"}
    if not need.issubset(set(long_df.columns)):
        return pd.DataFrame(), {"source": f"per_view_unexpected_schema ({files[0].name})"}

    g1 = (long_df
          .groupby(["dataset", "view", "model", "perm_seed", "metric", "k_pct", "strategy"], as_index=False)["value"]
          .mean())
    pivot = (g1.pivot_table(
        index=["dataset", "view", "model", "perm_seed", "metric", "k_pct"],
        columns="strategy", values="value", aggfunc="first"
    ).reset_index())

    # some runs may not have all strategies depending on config; guard
    def col(name):  # strategy col
        return pivot[name] if name in pivot.columns else np.nan

    pivot["delta_var_minus_random"] = col("var") - col("random")
    pivot["delta_shap_minus_var"] = col("shap") - col("var")

    def q(x, p):
        x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
        return float(np.quantile(x, p)) if x.size else float("nan")

    summ_rows = []
    for (ds, vw, mdl, met, k), sub in pivot.groupby(["dataset", "view", "model", "metric", "k_pct"], dropna=False):
        dvr = sub["delta_var_minus_random"]
        dsv = sub["delta_shap_minus_var"]
        summ_rows.append({
            "dataset": ds, "view": vw, "model": mdl, "metric": met, "k_pct": k,
            "n_perm_seeds": int(sub["perm_seed"].nunique()),
            "delta_var_random_mean": float(pd.to_numeric(dvr, errors="coerce").mean()),
            "delta_var_random_q05": q(dvr, 0.05),
            "delta_var_random_q95": q(dvr, 0.95),
            "delta_shap_var_mean": float(pd.to_numeric(dsv, errors="coerce").mean()),
            "delta_shap_var_q05": q(dsv, 0.05),
            "delta_shap_var_q95": q(dsv, 0.95),
        })

    df = pd.DataFrame(summ_rows)
    return df, {"source": f"computed_from_per_view ({len(files)} files)"}

def load_multi_k_biology(outputs_dir: Path, path_override: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if path_override:
        p = Path(path_override)
        df = read_csv_any(p)
        return df, str(p)
    p = outputs_dir / "08_biology_multi_k" / "convergence_summary_multi_k.csv"
    df = read_csv_any(p)
    return df, str(p)

def resolve_phase11_run_dir(outputs_dir: Path, phase11_dirname: str, phase11_run: str) -> Optional[Path]:
    """
    Resolves the Phase 11 run directory.

    Expected structure:
      outputs/11_diagnostic_validation/
        latest.txt  (contains e.g. "run__YYYYMMDD_HHMMSS" or a relative/absolute path)
        runs/run__YYYYMMDD_HHMMSS/...

    Returns a Path to the run directory, or None if not found.
    """
    root = outputs_dir / phase11_dirname
    if not root.exists():
        return None

    runs_root = root / "runs"

    def try_resolve(token: str) -> Optional[Path]:
        token = (token or "").strip().strip('"').strip("'")
        if not token:
            return None

        p = Path(token)

        # absolute path
        if p.is_absolute() and p.exists():
            return p

        # interpret relative tokens against a few likely bases
        for base in [runs_root, root, outputs_dir]:
            cand = base / token
            if cand.exists():
                return cand
        return None

    # off
    if phase11_run.lower() in ("off", "none", "false"):
        return None

    # auto/latest: prefer latest.txt
    if phase11_run.lower() in ("auto", "latest"):
        latest_txt = root / "latest.txt"
        if latest_txt.exists():
            token = latest_txt.read_text(encoding="utf-8", errors="ignore").strip()
            rd = try_resolve(token)
            if rd:
                return rd

        # fallback: most recent run__* in runs/
        if runs_root.exists():
            run_dirs = sorted(
                [p for p in runs_root.glob("run__*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return run_dirs[0] if run_dirs else None
        return None

    # explicit run id or path
    return try_resolve(phase11_run)


def _find_first_csv_with_columns(search_dirs: List[Path], required_cols: List[str]) -> Optional[Path]:
    """
    Recursively search for the first CSV that contains required columns.
    Reads only header + a few rows to detect schema.
    """
    for d in search_dirs:
        if not d or not d.exists():
            continue
        for p in sorted(d.rglob("*.csv")):
            try:
                df = pd.read_csv(p, nrows=5)
                cols = set(df.columns)
                if all(c in cols for c in required_cols):
                    return p
            except Exception:
                continue
    return None


def load_phase11_enrichment(run_dir: Path) -> Dict[str, object]:
    """
    Loads Phase 11 decision assets and (if present) a per-view risk table.

    Returns dict with:
      - run_dir (Path)
      - decision_rule (dict or None)
      - phase11_summary (str or None)
      - per_view_df (DataFrame with dataset, view, risk_zone, confidence_tier, recommended_action)
      - stats (dict) : prospective/lodo/null summaries if found
    """
    out: Dict[str, object] = {
        "run_dir": run_dir,
        "decision_rule": None,
        "phase11_summary": None,
        "per_view_df": pd.DataFrame(),
        "stats": {},
    }

    if not run_dir or not run_dir.exists():
        return out

    # 1) Decision rule JSON (search a few likely locations)
    rule_candidates = [
        run_dir / "decision_assets" / "DECISION_RULE.json",
        run_dir / "08_decision_assets" / "DECISION_RULE.json",  # legacy
    ]
    rule_path = next((p for p in rule_candidates if p.exists()), None)
    if rule_path is None:
        # fallback: search
        for p in run_dir.rglob("DECISION_RULE.json"):
            rule_path = p
            break
    if rule_path and rule_path.exists():
        try:
            out["decision_rule"] = read_json(rule_path)
        except Exception:
            out["decision_rule"] = None

    # 2) Human summary (optional)
    summary_candidates = [
        run_dir / "PHASE11_SUMMARY.md",
        run_dir / "decision_assets" / "PHASE11_SUMMARY.md",
    ]
    summary_path = next((p for p in summary_candidates if p.exists()), None)
    if summary_path and summary_path.exists():
        out["phase11_summary"] = summary_path.read_text(encoding="utf-8", errors="ignore")

    # 3) Per-view risk table (schema-flexible)
    # Prefer dataset+view tables. We search decision_assets/ and tables/
    search_dirs = [
        run_dir / "decision_assets",
        run_dir / "tables",
        run_dir / "05_inconclusive",
        run_dir / "08_decision_assets",
    ]

    # try a strict schema first
    p = _find_first_csv_with_columns(search_dirs, required_cols=["dataset", "view"])
    if p is not None:
        df = pd.read_csv(p)

        # map possible column names -> canonical
        colmap = {}

        # risk zone
        for cand in ["risk_zone", "zone", "risk", "regime", "label"]:
            if cand in df.columns:
                colmap[cand] = "risk_zone"
                break

        # confidence
        for cand in ["confidence_tier", "confidence", "tier"]:
            if cand in df.columns:
                colmap[cand] = "confidence_tier"
                break

        # action
        for cand in ["recommended_action", "action", "recommendation"]:
            if cand in df.columns:
                colmap[cand] = "recommended_action"
                break

        df = df.rename(columns=colmap)

        keep = ["dataset", "view"]
        for c in ["risk_zone", "confidence_tier", "recommended_action"]:
            if c in df.columns:
                keep.append(c)

        out["per_view_df"] = df[keep].drop_duplicates(["dataset", "view"])

    # 4) Validation stats (optional JSONs)
    # prospective
    for name in ["prospective_accuracy.json", "prospective_summary.json"]:
        for p in run_dir.rglob(name):
            try:
                out["stats"]["prospective"] = read_json(p)
                break
            except Exception:
                continue
        if "prospective" in out["stats"]:
            break

    # lodo
    for name in ["lodo_accuracy.json", "lodo_summary.json"]:
        for p in run_dir.rglob(name):
            try:
                out["stats"]["lodo"] = read_json(p)
                break
            except Exception:
                continue
        if "lodo" in out["stats"]:
            break

    # null baseline
    for name in ["null_report.json", "null_summary.json"]:
        for p in run_dir.rglob(name):
            try:
                out["stats"]["null"] = read_json(p)
                break
            except Exception:
                continue
        if "null" in out["stats"]:
            break

    return out


# -------------------------
# hero ranking logic
# -------------------------

def rank_heroes(ab_xgb: pd.DataFrame,
               ab_rf: pd.DataFrame,
               mkbio: pd.DataFrame,
               primary_k: int,
               primary_metric: str,
               lp_summ: Optional[pd.DataFrame] = None,
               di_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Returns ranked list with categories:
      - ANTI_ALIGNED (variance worse than random; SHAP rescues)
      - COUPLED (variance ~ SHAP; both above random)
      - MIXED (intermediate / inconsistent)
    Uses primary_k + primary_metric.
    """
    def prep(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
        if df.empty:
            return df
        d = df.copy()
        d = d.rename(columns={"K_pct": "k_pct"})
        d = d[(d["k_pct"] == primary_k) & (d["metric"] == primary_metric)].copy()
        d["model"] = model_label
        # keep only the fields we need
        keep = ["dataset", "view", "model", "regime", "regime_confidence", "DI_mean",
                "perf_var_mean", "perf_shap_mean", "perf_random_mean",
                "delta_var_random_mean", "delta_shap_var_mean"]
        for c in keep:
            if c not in d.columns:
                d[c] = np.nan
        return d[keep]

    x = prep(ab_xgb, "xgb")
    r = prep(ab_rf, "rf")

    # merge per view
    base = x.rename(columns={c: f"{c}__xgb" for c in x.columns if c not in ("dataset", "view")})
    if not r.empty:
        r2 = r.rename(columns={c: f"{c}__rf" for c in r.columns if c not in ("dataset", "view")})
        base = base.merge(r2, on=["dataset", "view"], how="outer")

    # --- attach DI summary (if available) ---
    if di_df is not None and not di_df.empty and {"dataset", "view"}.issubset(di_df.columns):
        di_col = pick_first_col(di_df, ["DI_mean", "DI", "di_mean", "di"])
        if di_col is None:
            # fallback: any column starting with 'DI' and containing 'mean'
            for c in di_df.columns:
                if str(c).lower().startswith("di") and "mean" in str(c).lower():
                    di_col = c
                    break
        if di_col is not None:
            di_sub = di_df[["dataset", "view", di_col]].rename(columns={di_col: "DI_mean__xgb"})
            base = base.merge(di_sub, on=["dataset", "view"], how="left", suffixes=("", "__di"))
            if "DI_mean__xgb__di" in base.columns:
                base["DI_mean__xgb"] = base["DI_mean__xgb__di"].combine_first(base.get("DI_mean__xgb"))
                base = base.drop(columns=["DI_mean__xgb__di"])

    # --- RF vs XGB direction agreement (with epsilon deadzone) ---
    if "delta_var_random_mean__rf" in base.columns:
        eps = 0.005  # tune if needed; 0.005 BA is a safe default
        base["dvr_sign__xgb"] = base["delta_var_random_mean__xgb"].apply(lambda v: sign_eps(v, eps))
        base["dvr_sign__rf"] = base["delta_var_random_mean__rf"].apply(lambda v: sign_eps(v, eps))

        # agree only if both non-zero and equal
        base["xgb_rf_sign_agree"] = (
            (base["dvr_sign__xgb"] != 0) &
            (base["dvr_sign__rf"] != 0) &
            (base["dvr_sign__xgb"] == base["dvr_sign__rf"])
        )

        # optional: a strength score (how big the deltas are jointly)
        base["xgb_rf_agree_strength"] = (
            np.nan_to_num(base["delta_var_random_mean__xgb"].abs()) +
            np.nan_to_num(base["delta_var_random_mean__rf"].abs())
        )
    else:
        base["xgb_rf_sign_agree"] = False
        base["xgb_rf_agree_strength"] = np.nan

    # attach multi-k biology at k=10 (if present)
    if not mkbio.empty and {"dataset", "view", "k_pct"}.issubset(mkbio.columns):
        mk10 = mkbio[mkbio["k_pct"] == primary_k].copy()
        for c in ["gene_jaccard", "pathway_jaccard", "convergence_ratio"]:
            if c not in mk10.columns:
                mk10[c] = np.nan
        mk10 = mk10[["dataset", "view", "gene_jaccard", "pathway_jaccard", "convergence_ratio"]]
        base = base.merge(mk10, on=["dataset", "view"], how="left")

    # --- attach permutation summary (if available) ---
    if lp_summ is not None and not lp_summ.empty:
        lp2 = lp_summ.copy()
        if "k_pct" in lp2.columns:
            lp2 = lp2[lp2["k_pct"] == primary_k]
        if "metric" in lp2.columns:
            lp2 = lp2[lp2["metric"] == primary_metric]

        # expected columns from your audit builder / label_perm_summary
        keep = [c for c in [
            "dataset", "view", "n_perm_seeds",
            "delta_var_random_q05", "delta_var_random_q95",
            "delta_shap_var_q05", "delta_shap_var_q95"
        ] if c in lp2.columns]

        lp2 = lp2[keep].drop_duplicates(["dataset", "view"])
        base = base.merge(lp2, on=["dataset", "view"], how="left")

        # real effect outside perm interval => validated
        base["perm_validated_dvr"] = (
            (base["delta_var_random_mean__xgb"] < base["delta_var_random_q05"]) |
            (base["delta_var_random_mean__xgb"] > base["delta_var_random_q95"])
        )
        base["perm_validated_dsv"] = (
            (base["delta_shap_var_mean__xgb"] < base["delta_shap_var_q05"]) |
            (base["delta_shap_var_mean__xgb"] > base["delta_shap_var_q95"])
        )
    else:
        base["perm_validated_dvr"] = False
        base["perm_validated_dsv"] = False
        base["n_perm_seeds"] = np.nan

    base["bio_available"] = pd.to_numeric(
        base.get("convergence_ratio", np.nan),
        errors="coerce"
    ).notna()

    # --- confidence tier ---
    # High: direction agrees (XGB/RF) AND permutation validates (either dvr or dsv) AND biology available
    # Medium: (direction agrees AND (perm validates OR bio available)) OR (perm validates AND bio available)
    # Low: otherwise / missing components
    base["confidence_tier"] = "LOW"

    high = (
        base["xgb_rf_sign_agree"] &
        (base["perm_validated_dvr"] | base["perm_validated_dsv"]) &
        base["bio_available"]
    )

    med = (
        (base["xgb_rf_sign_agree"] & ((base["perm_validated_dvr"] | base["perm_validated_dsv"]) | base["bio_available"])) |
        ((base["perm_validated_dvr"] | base["perm_validated_dsv"]) & base["bio_available"])
    )

    base.loc[med, "confidence_tier"] = "MEDIUM"
    base.loc[high, "confidence_tier"] = "HIGH"

    # scoring rules (simple + transparent)
    def score_anti(row) -> float:
        dvr = safe_float(row.get("delta_var_random_mean__xgb"))
        dsv = safe_float(row.get("delta_shap_var_mean__xgb"))
        di = safe_float(row.get("DI_mean__xgb"))
        # prefer: very negative variance-vs-random + strong rescue (shap-var)
        return (max(0.0, -dvr) * 2.0) + (max(0.0, dsv) * 1.5) + (np.nan_to_num(di) * 0.2)

    def score_coupled(row) -> float:
        dvr = safe_float(row.get("delta_var_random_mean__xgb"))
        dsv = safe_float(row.get("delta_shap_var_mean__xgb"))
        pv = safe_float(row.get("perf_var_mean__xgb"))
        pr = safe_float(row.get("perf_random_mean__xgb"))
        # prefer: variance above random, and shap≈var (small dsv)
        return (max(0.0, dvr) * 2.0) + (max(0.0, pv - pr) * 1.0) + (max(0.0, 0.05 - abs(dsv)) * 0.5)

    def score_bio(row) -> float:
        cr = safe_float(row.get("convergence_ratio"))
        pj = safe_float(row.get("pathway_jaccard"))
        return (np.nan_to_num(cr) * 1.0) + (np.nan_to_num(pj) * 2.0)

    base["score_anti"] = base.apply(score_anti, axis=1)
    base["score_coupled"] = base.apply(score_coupled, axis=1)
    base["score_bio"] = base.apply(score_bio, axis=1)

    # assign category heuristics
    dvr = pd.to_numeric(base.get("delta_var_random_mean__xgb", pd.Series(dtype=float)), errors="coerce")
    dsv = pd.to_numeric(base.get("delta_shap_var_mean__xgb", pd.Series(dtype=float)), errors="coerce")

    base["category"] = "MIXED"
    base.loc[(dvr < -0.02) & (dsv > 0.02), "category"] = "ANTI_ALIGNED"
    base.loc[(dvr > 0.02) & (dsv.abs() < 0.02), "category"] = "COUPLED"

    # category rank score
    base["rank_score"] = np.where(
        base["category"].eq("ANTI_ALIGNED"), base["score_anti"],
        np.where(base["category"].eq("COUPLED"), base["score_coupled"], (base["score_anti"] + base["score_coupled"]) / 2.0)
    )

    base = base.sort_values(["category", "rank_score"], ascending=[True, False])
    return base


# -------------------------
# report writer
# -------------------------

def make_report(outputs_dir: Path,
                out_dir: Path,
                di: pd.DataFrame,
                ab_xgb: pd.DataFrame,
                st_xgb: Dict[str, int],
                ab_rf: pd.DataFrame,
                rf_status: str,
                rf_ablation_dirname: str,
                lp: pd.DataFrame,
                lp_meta: Dict[str, str],
                mkbio: pd.DataFrame,
                mkbio_path: str,
                heroes: pd.DataFrame,
                primary_k: int,
                primary_metric: str,
                phase11: Optional[Dict[str, object]] = None) -> str:

    lines = []
    lines.append("# Audit Summary")
    lines.append(f"- Generated: {now_iso()}")
    lines.append(f"- outputs_dir: `{outputs_dir}`")
    lines.append(f"- primary_k: {primary_k}%")
    lines.append(f"- primary_metric: {primary_metric}")
    lines.append("")

    # status
    lines.append("## Run status (detected)")
    lines.append(f"- DI summary: {'OK' if not di.empty else 'MISSING'} (`04_importance/uncertainty/di_summary.csv`)")
    lines.append(f"- XGB ablation: {'OK' if not ab_xgb.empty else 'MISSING'} (`{st_xgb.get('source','07_ablation')}`; views_completed={st_xgb.get('views_completed',0)})")
    lines.append(f"- RF ablation: {rf_status}")
    lines.append(f"- Label permutation: {'OK' if not lp.empty else 'IN PROGRESS / MISSING'} (source: {lp_meta.get('source')})")
    lines.append(f"- Multi-K biology: {'OK' if not mkbio.empty else 'MISSING'} (source: {mkbio_path})")
    lines.append("")

    # Phase 11 enrichment
    lines.append("## Phase 11 diagnostic validation (enrichment)")
    if phase11 and phase11.get("run_dir"):
        lines.append(f"- run_dir: `{phase11['run_dir']}`")

        rule = phase11.get("decision_rule")
        if isinstance(rule, dict) and rule:
            # show a couple of common keys if present (won't error if absent)
            for k in ["primary_k", "primary_metric", "threshold_DI", "threshold_delta", "cost_ratio_false_safe"]:
                if k in rule:
                    lines.append(f"- {k}: {rule[k]}")

        stats = phase11.get("stats", {}) or {}
        if "prospective" in stats:
            # try a few common keys
            p = stats["prospective"]
            acc = p.get("accuracy", p.get("overall_accuracy", None))
            if acc is not None:
                lines.append(f"- prospective_accuracy: {acc}")
        if "lodo" in stats:
            l = stats["lodo"]
            acc = l.get("accuracy", l.get("overall_accuracy", None))
            if acc is not None:
                lines.append(f"- lodo_accuracy: {acc}")

        pv = phase11.get("per_view_df", pd.DataFrame())
        if isinstance(pv, pd.DataFrame) and not pv.empty:
            lines.append(f"- per_view_labels: OK (n={pv.drop_duplicates(['dataset','view']).shape[0]})")
        else:
            lines.append("- per_view_labels: NOT FOUND (Phase 11 may not have written a dataset/view table; audit continues without it)")
    else:
        lines.append("- Not available (no Phase 11 run detected or disabled).")
    lines.append("")

    # quick highlights
    lines.append("## Hero view recommendations (auto-ranked)")
    if heroes.empty:
        lines.append("- No hero rankings available (need at least XGB ablation master or per_view JSON).")
        return "\n".join(lines)

    def topn(cat: str, n: int = 3) -> pd.DataFrame:
        return heroes[heroes["category"].eq(cat)].head(n)

    for cat, title in [("ANTI_ALIGNED", "Anti-aligned (variance worse than random; SHAP rescues)"),
                       ("COUPLED", "Coupled (variance ≈ SHAP; both above random)"),
                       ("MIXED", "Mixed / transitional")]:
        t = topn(cat, 3)
        lines.append(f"### {title}")
        if t.empty:
            lines.append("- (none detected yet under current thresholds)")
            lines.append("")
            continue
        for _, row in t.iterrows():
            ds, vw = row["dataset"], row["view"]
            dvr = row.get("delta_var_random_mean__xgb", np.nan)
            dsv = row.get("delta_shap_var_mean__xgb", np.nan)
            pv  = row.get("perf_var_mean__xgb", np.nan)
            ps  = row.get("perf_shap_mean__xgb", np.nan)
            pr  = row.get("perf_random_mean__xgb", np.nan)
            cr  = row.get("convergence_ratio", np.nan)
            lines.append(
                f"- **{ds}:{vw}** | Δ(V−R)={dvr:.3f} | Δ(S−V)={dsv:.3f} | Var={pv:.3f} | SHAP={ps:.3f} | Rand={pr:.3f}"
                + (f" | BioConv={cr:.2f}×" if pd.notna(cr) else "")
            )
        lines.append("")

    # multi-k biology sanity
    if not mkbio.empty and {"k_pct", "convergence_ratio"}.issubset(mkbio.columns):
        mk_sub = mkbio.copy()
        mk_sub["convergence_ratio"] = pd.to_numeric(mk_sub["convergence_ratio"], errors="coerce")
        mk_sub = mk_sub.dropna(subset=["convergence_ratio"])
        if not mk_sub.empty:
            lines.append("## Multi-K biology robustness (sanity)")
            for k in sorted(mk_sub["k_pct"].unique()):
                s = mk_sub[mk_sub["k_pct"] == k]["convergence_ratio"]
                lines.append(f"- K={int(k)}%: mean convergence_ratio = {s.mean():.2f}× (n={s.shape[0]} gene-mappable views)")
            lines.append("")
            lines.append("Note: NaNs are expected for non-gene-mappable views (miRNA / IBDMDB / some methylation views depending on mapping).")
            lines.append("")

    # permutation quick check
    if not lp.empty:
        lines.append("## Label permutation (negative control) quick check")
        # show only one line per view for primary metric/k if available
        lp2 = lp.copy()
        if "k_pct" in lp2.columns:
            lp2 = lp2[lp2["k_pct"] == primary_k]
        if "metric" in lp2.columns:
            lp2 = lp2[lp2["metric"] == primary_metric]
        # take top few rows
        show_cols = [c for c in ["dataset", "view", "n_perm_seeds", "delta_var_random_mean", "delta_shap_var_mean"] if c in lp2.columns]
        lp3 = lp2[show_cols].drop_duplicates()
        lp3 = lp3.head(10)
        if not lp3.empty:
            lines.append(lp3.to_markdown(index=False))
        else:
            lines.append("- Permutation summary exists but did not match primary_k/metric filters (still OK).")
        lines.append("")

    # next actions
    lines.append("## Next actions (minimal)")
    if "OK" not in rf_status:
        lines.append(f"- When `{rf_ablation_dirname}` finishes: re-run this audit to lock cross-model (XGB vs RF) generality.")
    lines.append("- When `06_robustness_100\\label_perm` finishes: re-run your decoupling aggregator pointing to that dirname, then re-run this audit.")
    lines.append("")

    return "\n".join(lines)


# -------------------------
# main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--out-dirname", default="10_audit")
    ap.add_argument("--xgb-ablation-dirname", default="07_ablation")
    ap.add_argument("--rf-ablation-dirname", default="07_ablation_rf")
    ap.add_argument("--label-perm-dirname", default=None, help=r'e.g. "06_robustness_100\label_perm"')
    ap.add_argument("--multi-k-biology-path", default=None, help="Override path to convergence_summary_multi_k.csv")
    ap.add_argument("--primary-k", type=int, default=10)
    ap.add_argument("--primary-metric", default="balanced_accuracy")
    ap.add_argument("--phase11-dirname", default="11_diagnostic_validation",
                    help="Folder under outputs-dir that contains Phase 11 runs and latest.txt")
    ap.add_argument("--phase11-run", default="auto",
                    help="auto|latest|run__YYYYMMDD_HHMMSS|absolute_path|relative_path. auto uses latest.txt if present.")
    ap.add_argument("--no-phase11", action="store_true",
                    help="Disable Phase 11 enrichment even if outputs exist.")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / args.out_dirname
    ensure_dir(out_dir)

    print(f"[{now_iso()}] audit_summary starting")
    print(f"  outputs_dir: {outputs_dir}")
    print(f"  out_dir:     {out_dir}")

    di = load_di_summary(outputs_dir)

    ab_xgb, st_xgb = load_ablation(outputs_dir, args.xgb_ablation_dirname)
    st_xgb.setdefault("source", args.xgb_ablation_dirname)

    # --- RF ablation detection (NEW: accept master summary) ---
    rf_dir = outputs_dir / args.rf_ablation_dirname
    rf_master = rf_dir / "ablation_master_summary.csv"

    ab_rf = None
    rf_status = "IN PROGRESS / MISSING"

    if rf_master.exists():
        ab_rf = pd.read_csv(rf_master)

        # count completed views robustly
        views_completed = ab_rf[["dataset", "view"]].drop_duplicates().shape[0]

        rf_status = f"OK (`master`; views_completed={views_completed})"
    else:
        # fallback to legacy per_view_json logic (keep your existing code here)
        ab_rf, st_rf = load_ablation(outputs_dir, args.rf_ablation_dirname)
        st_rf.setdefault("source", args.rf_ablation_dirname)
        if not ab_rf.empty:
            rf_status = f"OK (`{st_rf.get('source', args.rf_ablation_dirname)}`; views_completed={st_rf.get('views_completed',0)})"

    if ab_rf is None:
        ab_rf = pd.DataFrame()

    lp, lp_meta = load_label_perm(outputs_dir, args.label_perm_dirname)

    mkbio, mkbio_path = load_multi_k_biology(outputs_dir, args.multi_k_biology_path)

    heroes = rank_heroes(
        ab_xgb=ab_xgb,
        ab_rf=ab_rf,
        mkbio=mkbio,
        primary_k=args.primary_k,
        primary_metric=args.primary_metric,
        lp_summ=lp,
        di_df=di
    )

    phase11 = None
    if not args.no_phase11:
        run_dir = resolve_phase11_run_dir(outputs_dir, args.phase11_dirname, args.phase11_run)
        if run_dir:
            phase11 = load_phase11_enrichment(run_dir)
            pv = phase11.get("per_view_df", pd.DataFrame())
            if isinstance(pv, pd.DataFrame) and not pv.empty:
                heroes = heroes.merge(pv, on=["dataset", "view"], how="left")
                heroes["phase11_run_dir"] = str(run_dir)
            else:
                heroes["phase11_run_dir"] = str(run_dir)

    # build a merged view-level audit table (mostly from hero table)
    view_table = heroes.copy()
    view_table.to_csv(out_dir / "view_audit_table.csv", index=False)
    heroes.to_csv(out_dir / "hero_rankings.csv", index=False)

    report = make_report(
        outputs_dir=outputs_dir,
        out_dir=out_dir,
        di=di,
        ab_xgb=ab_xgb,
        st_xgb=st_xgb,
        ab_rf=ab_rf,
        rf_status=rf_status,
        rf_ablation_dirname=args.rf_ablation_dirname,
        lp=lp,
        lp_meta=lp_meta,
        mkbio=mkbio,
        mkbio_path=mkbio_path,
        heroes=heroes,
        primary_k=args.primary_k,
        primary_metric=args.primary_metric,
        phase11=phase11
    )

    (out_dir / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    print(f"[{now_iso()}] wrote:")
    print(f"  {out_dir / 'AUDIT_REPORT.md'}")
    print(f"  {out_dir / 'view_audit_table.csv'}")
    print(f"  {out_dir / 'hero_rankings.csv'}")
    print("DONE")


if __name__ == "__main__":
    main()
