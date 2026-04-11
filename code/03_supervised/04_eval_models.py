#!/usr/bin/env python3
"""
PHASE 3 — 04_eval_models.py

Consolidate Phase-3 supervised outputs into analysis-ready summary tables.

Reads (produced by earlier Phase 3 scripts)
- Baselines:
    outputs/03_supervised/baselines/metrics/metrics__{dataset}__{view}.json
- Tree models (supports multiple model dirs):
    outputs/03_supervised/tree_models_xgb/metrics/metrics__{dataset}__{view}__{model}.json
    outputs/03_supervised/tree_models_xgb_bal/metrics/metrics__{dataset}__{view}__{model}.json
    outputs/03_supervised/tree_models_rf/metrics/metrics__{dataset}__{view}__{model}.json
- Importance (SHAP):
    outputs/03_supervised/{tree_dir}/importance/shap__{dataset}__{view}__{model}.npz
    outputs/03_supervised/{tree_dir}/importance/prediction_importance__{dataset}__{view}__{model}.csv.gz

Optional (Phase 2 unsupervised; for decoupling)
- outputs/02_unsupervised/variance_scores/variance_scores__{dataset}__{view}.csv.gz

Writes
- outputs/03_supervised/eval/model_performance__long.csv.gz
- outputs/03_supervised/eval/model_performance__wide.csv.gz
- outputs/03_supervised/eval/deltas__tree_vs_best_baseline.csv.gz
- outputs/03_supervised/eval/shap_stability__summary.csv.gz
- outputs/03_supervised/eval/decoupling__V_vs_P.csv.gz
- outputs/03_supervised/eval/eval_manifest.json

Design goals
- No pickles; gzip CSV + JSON only
- Robust to missing tasks (e.g., while jobs are still running)
- Deterministic and reviewer-proof
- Supports multiple tree model variants (xgb, xgb_bal, rf)

Author: variance-prediction paradox pipeline
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Registry (must match Phase 3 scripts)
# -----------------------------
VIEW_REGISTRY = {
    "mlomics": {"core_views": ["mRNA", "miRNA", "methylation", "CNV"], "sensitivity_views": [], "analysis_role": "primary"},
    "ibdmdb":  {"core_views": ["MGX", "MGX_func", "MPX", "MBX"],       "sensitivity_views": ["MGX_CLR"], "analysis_role": "primary"},
    "ccle":    {"core_views": ["mRNA", "CNV", "proteomics"],          "sensitivity_views": [], "analysis_role": "primary"},
    "tcga_gbm":{"core_views": ["mRNA", "methylation", "CNV"],         "sensitivity_views": ["methylation_Mval"], "analysis_role": "sensitivity"},
}

DEFAULT_K_PCTS = [1, 5, 10, 20]


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def ci95(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(values, 2.5))
    hi = float(np.percentile(values, 97.5))
    return lo, hi


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def resolve_views(dataset: str, which: str) -> List[str]:
    if dataset not in VIEW_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. Options: {', '.join(sorted(VIEW_REGISTRY))}")
    core = VIEW_REGISTRY[dataset]["core_views"]
    sens = VIEW_REGISTRY[dataset]["sensitivity_views"]
    if which == "core":
        return list(core)
    if which == "all":
        return list(core) + list(sens)
    if which == "sensitivity":
        return list(sens)
    raise ValueError("views must be one of: core, all, sensitivity")


def rank_desc(values: np.ndarray) -> np.ndarray:
    """
    Return 0-based rank positions (0 = largest value).
    Uses stable mergesort for determinism.
    """
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(values.size, dtype=np.int32)
    return ranks


def spearman_via_ranks(a: np.ndarray, b: np.ndarray) -> float:
    ra = rank_desc(a).astype(np.float64)
    rb = rank_desc(b).astype(np.float64)
    # Pearson corr of ranks
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum()) * np.sqrt((rb * rb).sum()))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def topk_set_from_scores(scores: np.ndarray, k_frac: float) -> set:
    n = scores.size
    k = max(1, int(math.ceil(k_frac * n)))
    idx = np.argsort(-scores, kind="mergesort")[:k]
    return set(idx.tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union > 0 else float("nan")


def j_rand(q: float) -> float:
    # random-set baseline for equal-sized selections
    # E[J] ≈ q/(2-q)
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    return float(q / (2.0 - q))


def decoupling_metrics(var_scores: np.ndarray, imp_scores: np.ndarray, k_pcts: List[int]) -> Dict[str, object]:
    """
    Compute J, ΔJ, DI across K%, plus AUC summary and Spearman correlation.
    var_scores, imp_scores must be aligned vectors over same features.
    """
    out: Dict[str, object] = {}
    out["spearman_var_vs_imp"] = spearman_via_ranks(var_scores, imp_scores)  # rank-rank spearman

    rows = []
    for K in k_pcts:
        q = K / 100.0
        Sv = topk_set_from_scores(var_scores, q)
        Sp = topk_set_from_scores(imp_scores, q)
        J = jaccard(Sv, Sp)
        Jr = j_rand(q)
        dJ = J - Jr
        # Normalised overlap and DI (per Section4)
        denom = (1.0 - Jr)
        J_tilde = (dJ / denom) if denom > 0 else float("nan")
        DI = 1.0 - J_tilde if not np.isnan(J_tilde) else float("nan")
        rows.append(
            {"K_pct": K, "q": q, "J": J, "J_rand": Jr, "deltaJ": dJ, "J_tilde": J_tilde, "DI": DI}
        )

    df = pd.DataFrame(rows).sort_values("K_pct")
    out["by_K"] = df.to_dict(orient="records")

    # AUC across K (trapezoid on q axis)
    if df.shape[0] >= 2:
        auc = float(np.trapezoid(df["DI"].values, df["q"].values))
    else:
        auc = float("nan")
    out["DI_AUC"] = auc
    return out


# -----------------------------
# Loaders for Phase 3 outputs
# -----------------------------
@dataclass
class ModelPerf:
    dataset: str
    view: str
    model: str
    family: str  # baseline | tree
    n_samples: int
    n_features: int
    n_classes: int
    n_repeats: int
    analysis_role: str
    metric_means: Dict[str, float]
    metric_cis: Dict[str, Tuple[float, float]]
    metric_per_repeat: Dict[str, List[float]]


def load_baseline_metrics_json(path: Path) -> List[ModelPerf]:
    """
    Baseline JSON contains multiple models in one file.
    Returns one ModelPerf per baseline model.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    ds = str(payload["dataset"])
    view = str(payload["view"])
    role = str(payload.get("analysis_role", VIEW_REGISTRY.get(ds, {}).get("analysis_role", "unknown")))
    n_samples = int(payload["n_samples"])
    n_features = int(payload["n_features"])
    n_classes = int(payload["n_classes"])
    n_repeats = int(payload["cv_config"]["n_repeats"])

    out: List[ModelPerf] = []
    for mname, mentry in payload["models"].items():
        summ = mentry["summary"]
        metric_means = {}
        metric_cis = {}
        metric_per_repeat = {}
        for metric, stats in summ.items():
            metric_means[metric] = safe_float(stats.get("mean"))
            ci = stats.get("ci95", [float("nan"), float("nan")])
            metric_cis[metric] = (safe_float(ci[0]), safe_float(ci[1]))
            metric_per_repeat[metric] = [safe_float(x) for x in stats.get("per_repeat", [])]

        out.append(
            ModelPerf(
                dataset=ds,
                view=view,
                model=str(mname),
                family="baseline",
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_repeats=n_repeats,
                analysis_role=role,
                metric_means=metric_means,
                metric_cis=metric_cis,
                metric_per_repeat=metric_per_repeat,
            )
        )
    return out


def load_tree_metrics_json(path: Path) -> ModelPerf:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ds = str(payload["dataset"])
    view = str(payload["view"])
    model = str(payload["model"])
    role = str(payload.get("analysis_role", VIEW_REGISTRY.get(ds, {}).get("analysis_role", "unknown")))
    n_samples = int(payload["n_samples"])
    n_features = int(payload["n_features"])
    n_classes = int(payload["n_classes"])
    n_repeats = int(payload["cv_config"]["n_repeats"])

    summ = payload["performance"]["summary"]
    metric_means = {}
    metric_cis = {}
    metric_per_repeat = {}
    for metric, stats in summ.items():
        metric_means[metric] = safe_float(stats.get("mean"))
        ci = stats.get("ci95", [float("nan"), float("nan")])
        metric_cis[metric] = (safe_float(ci[0]), safe_float(ci[1]))
        metric_per_repeat[metric] = [safe_float(x) for x in stats.get("per_repeat", [])]

    return ModelPerf(
        dataset=ds,
        view=view,
        model=model,
        family="tree",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_repeats=n_repeats,
        analysis_role=role,
        metric_means=metric_means,
        metric_cis=metric_cis,
        metric_per_repeat=metric_per_repeat,
    )


def load_shap_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(str(path), allow_pickle=False)
    feats = z["feature_names"].astype(str)
    per_rep = z["mean_abs_shap_per_repeat"].astype(np.float32)  # (n_repeats, n_features)
    return feats, per_rep


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 supervised outputs (baselines + tree models + SHAP stability).")
    parser.add_argument("--dataset", type=str, default="all", help=f"Dataset name or 'all'. Options: {', '.join(sorted(VIEW_REGISTRY))}")
    parser.add_argument("--views", type=str, default="core", help="Which views to evaluate: core | all | sensitivity")
    parser.add_argument("--primary-metric", type=str, default="balanced_accuracy",
                        help="Metric used to pick best baseline per dataset×view (default: balanced_accuracy).")
    parser.add_argument("--tree-model", type=str, default="xgb", help="Tree model name to compare for deltas/decoupling (default: xgb).")
    parser.add_argument("--k-pcts", type=str, default="1,5,10,20", help="Comma-separated K% values for overlap/DI (default: 1,5,10,20).")

    parser.add_argument("--repo-root", type=str, default=".", help="Repository root")
    parser.add_argument("--metrics-baselines-dir", type=str, default="outputs/03_supervised/baselines/metrics")
    parser.add_argument("--metrics-tree-dir", type=str, default="outputs/03_supervised/tree_models_xgb/metrics",
                        help="Tree model metrics dir. Use tree_models_xgb_bal or tree_models_rf for other variants.")
    parser.add_argument("--importance-tree-dir", type=str, default="outputs/03_supervised/tree_models_xgb/importance",
                        help="Tree model importance dir. Should match --metrics-tree-dir variant.")
    parser.add_argument("--variance-dir", type=str, default="outputs/02_unsupervised/variance_scores")

    parser.add_argument("--out-dir", type=str, default="outputs/03_supervised/eval", help="Output directory for evaluation tables")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metrics_baselines_dir = (repo_root / args.metrics_baselines_dir).resolve()
    metrics_tree_dir = (repo_root / args.metrics_tree_dir).resolve()
    importance_tree_dir = (repo_root / args.importance_tree_dir).resolve()
    variance_dir = (repo_root / args.variance_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    ensure_dir(out_dir)

    k_pcts = [int(x.strip()) for x in args.k_pcts.split(",") if x.strip()]
    primary_metric = args.primary_metric.strip()
    tree_model = args.tree_model.strip()

    datasets = sorted(VIEW_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    expected_pairs = [(ds, v) for ds in datasets for v in resolve_views(ds, args.views)]

    print("=" * 80)
    print("PHASE 3 — EVAL MODELS")
    print("=" * 80)
    print(f"Repo root:     {repo_root}")
    print(f"Datasets:      {datasets}")
    print(f"Views mode:    {args.views}")
    print(f"Primary metric:{primary_metric}")
    print(f"Tree model:    {tree_model}")
    print(f"K%:            {k_pcts}")
    print(f"Baselines dir: {metrics_baselines_dir}")
    print(f"Tree dir:      {metrics_tree_dir}")
    print(f"Importance dir:{importance_tree_dir}")
    print(f"Variance dir:  {variance_dir}")
    print(f"Out dir:       {out_dir}")
    print()

    # ---------
    # Load metrics
    # ---------
    perfs: List[ModelPerf] = []
    missing_baseline = []
    missing_tree = []

    for ds, view in expected_pairs:
        # baselines
        bpath = metrics_baselines_dir / f"metrics__{ds}__{view}.json"
        if bpath.exists():
            perfs.extend(load_baseline_metrics_json(bpath))
        else:
            missing_baseline.append(str(bpath))

        # tree
        tpath = metrics_tree_dir / f"metrics__{ds}__{view}__{tree_model}.json"
        if tpath.exists():
            perfs.append(load_tree_metrics_json(tpath))
        else:
            missing_tree.append(str(tpath))

    if len(perfs) == 0:
        print("No metrics found for the requested scope. Exiting.")
        return

    # ---------
    # Build long performance table
    # ---------
    all_metrics = sorted({m for p in perfs for m in p.metric_means.keys()})
    rows_long = []
    for p in perfs:
        row = {
            "dataset": p.dataset,
            "view": p.view,
            "family": p.family,
            "model": p.model,
            "analysis_role": p.analysis_role,
            "n_samples": p.n_samples,
            "n_features": p.n_features,
            "n_classes": p.n_classes,
            "n_repeats": p.n_repeats,
        }
        for met in all_metrics:
            row[f"{met}__mean"] = p.metric_means.get(met, float("nan"))
            ci = p.metric_cis.get(met, (float("nan"), float("nan")))
            row[f"{met}__ci95_lo"] = ci[0]
            row[f"{met}__ci95_hi"] = ci[1]
        rows_long.append(row)

    df_long = pd.DataFrame(rows_long).sort_values(["dataset", "view", "family", "model"])
    out_long = out_dir / "model_performance__long.csv.gz"
    df_long.to_csv(out_long, index=False, compression="gzip")

    # ---------
    # Wide table: tree model + best baseline (by primary_metric)
    # ---------
    wide_rows = []
    deltas_rows = []

    # Index perfs for quick lookup
    by_pair_family = {}
    for p in perfs:
        by_pair_family.setdefault((p.dataset, p.view, p.family), []).append(p)

    for ds, view in expected_pairs:
        baselines = by_pair_family.get((ds, view, "baseline"), [])
        trees = [p for p in by_pair_family.get((ds, view, "tree"), []) if p.model == tree_model]

        if not trees:
            continue
        tree_p = trees[0]

        # pick best baseline by primary_metric mean
        best_b = None
        best_val = -1e18
        for b in baselines:
            v = b.metric_means.get(primary_metric, float("nan"))
            if not np.isnan(v) and v > best_val:
                best_val = v
                best_b = b

        # wide summary row
        row = {
            "dataset": ds,
            "view": view,
            "analysis_role": tree_p.analysis_role,
            "n_samples": tree_p.n_samples,
            "n_features": tree_p.n_features,
            "n_classes": tree_p.n_classes,
            "tree_model": tree_model,
            "best_baseline": None if best_b is None else best_b.model,
            "primary_metric": primary_metric,
        }
        # tree metrics
        for met in all_metrics:
            row[f"tree__{met}__mean"] = tree_p.metric_means.get(met, float("nan"))
            ci = tree_p.metric_cis.get(met, (float("nan"), float("nan")))
            row[f"tree__{met}__ci95_lo"] = ci[0]
            row[f"tree__{met}__ci95_hi"] = ci[1]

        # baseline metrics
        if best_b is not None:
            for met in all_metrics:
                row[f"baseline__{met}__mean"] = best_b.metric_means.get(met, float("nan"))
                ci = best_b.metric_cis.get(met, (float("nan"), float("nan")))
                row[f"baseline__{met}__ci95_lo"] = ci[0]
                row[f"baseline__{met}__ci95_hi"] = ci[1]
        wide_rows.append(row)

        # paired deltas across repeats (for metrics available in both)
        if best_b is not None:
            common = sorted(set(tree_p.metric_per_repeat.keys()) & set(best_b.metric_per_repeat.keys()))
            for met in common:
                t = tree_p.metric_per_repeat.get(met, [])
                b = best_b.metric_per_repeat.get(met, [])
                m = min(len(t), len(b))
                if m == 0:
                    continue
                d = [float(t[i] - b[i]) for i in range(m)]
                lo, hi = ci95(d)
                deltas_rows.append({
                    "dataset": ds,
                    "view": view,
                    "tree_model": tree_model,
                    "best_baseline": best_b.model,
                    "metric": met,
                    "n_repeats_used": m,
                    "delta_mean": float(np.mean(d)),
                    "delta_ci95_lo": lo,
                    "delta_ci95_hi": hi,
                    "delta_per_repeat": json.dumps(d),
                })

    df_wide = pd.DataFrame(wide_rows).sort_values(["dataset", "view"])
    out_wide = out_dir / "model_performance__wide.csv.gz"
    df_wide.to_csv(out_wide, index=False, compression="gzip")

    df_deltas = pd.DataFrame(deltas_rows).sort_values(["dataset", "view", "metric"])
    out_deltas = out_dir / "deltas__tree_vs_best_baseline.csv.gz"
    df_deltas.to_csv(out_deltas, index=False, compression="gzip")

    # ---------
    # SHAP stability summary (only when shap__... exists)
    # ---------
    shap_rows = []
    for ds, view in expected_pairs:
        shap_path = importance_tree_dir / f"shap__{ds}__{view}__{tree_model}.npz"
        if not shap_path.exists():
            continue
        feats, per_rep = load_shap_npz(shap_path)
        n_rep, n_feat = per_rep.shape
        # pairwise spearman across repeats
        sp_vals = []
        jac_vals = {K: [] for K in k_pcts}
        for i in range(n_rep):
            for j in range(i + 1, n_rep):
                sp_vals.append(spearman_via_ranks(per_rep[i], per_rep[j]))
                for K in k_pcts:
                    q = K / 100.0
                    A = topk_set_from_scores(per_rep[i], q)
                    B = topk_set_from_scores(per_rep[j], q)
                    jac_vals[K].append(jaccard(A, B))

        shap_rows.append({
            "dataset": ds,
            "view": view,
            "tree_model": tree_model,
            "n_repeats": int(n_rep),
            "n_features": int(n_feat),
            "pairwise_spearman_mean": float(np.nanmean(sp_vals)) if sp_vals else float("nan"),
            "pairwise_spearman_min": float(np.nanmin(sp_vals)) if sp_vals else float("nan"),
            **{f"top{K}pct_jaccard_mean": float(np.nanmean(jac_vals[K])) if jac_vals[K] else float("nan") for K in k_pcts},
        })

    df_shap = pd.DataFrame(shap_rows).sort_values(["dataset", "view"])
    out_shap = out_dir / "shap_stability__summary.csv.gz"
    df_shap.to_csv(out_shap, index=False, compression="gzip")

    # ---------
    # Decoupling V vs P (only when both variance + importance exist)
    # ---------
    dec_rows = []
    for ds, view in expected_pairs:
        v_path = variance_dir / f"variance_scores__{ds}__{view}.csv.gz"
        p_path = importance_tree_dir / f"prediction_importance__{ds}__{view}__{tree_model}.csv.gz"
        if not v_path.exists() or not p_path.exists():
            continue
        dfV = pd.read_csv(v_path)
        dfP = pd.read_csv(p_path)

        # inner-join on feature ID
        m = dfV.merge(dfP, on="feature", how="inner", suffixes=("_V", "_P"))
        if m.shape[0] < 10:
            continue

        var_scores = m["score"].astype(float).values
        imp_scores = m["importance"].astype(float).values

        mets = decoupling_metrics(var_scores, imp_scores, k_pcts)
        row = {
            "dataset": ds,
            "view": view,
            "tree_model": tree_model,
            "n_features_joined": int(m.shape[0]),
            "spearman_var_vs_imp": float(mets["spearman_var_vs_imp"]),
            "DI_AUC": float(mets["DI_AUC"]),
            "by_K_json": json.dumps(mets["by_K"]),
        }
        # convenience columns for each K
        for rec in mets["by_K"]:
            K = int(rec["K_pct"])
            row[f"DI__{K}pct"] = safe_float(rec["DI"])
            row[f"J__{K}pct"] = safe_float(rec["J"])
            row[f"deltaJ__{K}pct"] = safe_float(rec["deltaJ"])
        dec_rows.append(row)

    df_dec = pd.DataFrame(dec_rows).sort_values(["dataset", "view"])
    out_dec = out_dir / "decoupling__V_vs_P.csv.gz"
    df_dec.to_csv(out_dec, index=False, compression="gzip")

    # ---------
    # Manifest
    # ---------
    manifest = {
        "created_at": now_iso(),
        "repo_root": str(repo_root),
        "datasets": datasets,
        "views_mode": args.views,
        "tree_model": tree_model,
        "primary_metric": primary_metric,
        "k_pcts": k_pcts,
        "inputs": {
            "metrics_baselines_dir": str(metrics_baselines_dir),
            "metrics_tree_dir": str(metrics_tree_dir),
            "importance_tree_dir": str(importance_tree_dir),
            "variance_dir": str(variance_dir),
        },
        "outputs": {
            "model_performance_long": str(out_long),
            "model_performance_wide": str(out_wide),
            "deltas_tree_vs_best_baseline": str(out_deltas),
            "shap_stability_summary": str(out_shap),
            "decoupling_V_vs_P": str(out_dec),
        },
        "missing": {
            "baseline_metrics_json": missing_baseline,
            "tree_metrics_json": missing_tree,
        },
        "counts": {
            "n_model_rows_long": int(df_long.shape[0]),
            "n_pairs_wide": int(df_wide.shape[0]),
            "n_delta_rows": int(df_deltas.shape[0]),
            "n_shap_rows": int(df_shap.shape[0]),
            "n_decoupling_rows": int(df_dec.shape[0]),
        },
        "notes": [
            "This script is robust to missing tasks; re-run after long jobs finish to fill gaps.",
            "Paired deltas assume both baseline and tree model used the same repeat order from splits__{dataset}.npz.",
        ],
        "script": "03_supervised/04_eval_models.py",
    }
    (out_dir / "eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"  {out_long}")
    print(f"  {out_wide}")
    print(f"  {out_deltas}")
    print(f"  {out_shap}")
    print(f"  {out_dec}")
    print(f"  {out_dir / 'eval_manifest.json'}")


if __name__ == "__main__":
    main()