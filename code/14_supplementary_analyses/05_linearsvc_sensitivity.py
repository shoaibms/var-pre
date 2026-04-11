#!/usr/bin/env python3
"""
LinearSVC hero-view sensitivity check.

Runs a minimal ablation (All, TopVar, TopSHAP, SAF, Random) with
LinearSVC on the 3 hero views and checks whether the regime classification
(based on delta_var_minus_random sign) agrees with XGBoost.

Uses the same bundles, splits, and rank tables as the main ablation.

Usage:
  python 14_supplementary_analyses/06_linearsvc_sensitivity.py --outputs-dir outputs

Outputs:
  outputs/14_supplementary_analyses/linearsvc_sensitivity/supp_linear_model_regime_check.csv
  outputs/14_supplementary_analyses/linearsvc_sensitivity/linearsvc_sensitivity_summary.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


HERO_VIEWS = [
    ("mlomics", "methylation"),
    ("ibdmdb", "MGX"),
    ("ccle", "mRNA"),
]

K_PCT = 10
N_RANDOM_DRAWS = 10
MAX_REPEATS = 3


# ── Data loading (mirrors ablation script) ──
def load_npz(path: Path):
    return np.load(path, allow_pickle=False)


def _prefer_paths(paths: List[Path]) -> List[Path]:
    def score(p: Path):
        s = str(p).lower()
        is_archive = int("_archive" in s or "archive_pre_migration" in s)
        is_01 = int("01_bundles" in s)
        return (is_archive, -is_01, len(p.parts))
    return sorted(paths, key=score)


def find_splits(outputs_dir: Path, dataset: str) -> Path:
    cands = list(outputs_dir.glob(f"**/splits__{dataset}.npz"))
    if not cands:
        raise FileNotFoundError(f"splits__{dataset}.npz not found")
    return _prefer_paths(cands)[0]


def find_bundle(outputs_dir: Path, dataset: str) -> Path:
    cands = list(outputs_dir.glob(f"**/{dataset}*bundle*normalized*.npz"))
    if not cands:
        raise FileNotFoundError(f"Bundle not found for {dataset}")
    return _prefer_paths(cands)[0]


def load_splits(path: Path):
    z = load_npz(path)
    info = json.loads(str(z["info"])) if "info" in z.files else {}
    return z["y"], z["sample_ids"].astype(str), z["fold_ids"].astype(np.int16), info


def load_bundle_view(path: Path, view: str):
    z = load_npz(path)
    x_key = f"X_{view}"
    if x_key not in z.files:
        alt = [k for k in z.files if k.lower() == x_key.lower()]
        x_key = alt[0] if alt else x_key
    X = z[x_key].astype(np.float32)
    sample_ids = z["sample_ids"].astype(str)
    for fk in [f"features_{view}", f"feature_names_{view}"]:
        if fk in z.files:
            return sample_ids, X, z[fk].astype(str)
    return sample_ids, X, np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)


def align_X(X, bundle_ids, splits_ids):
    if np.array_equal(bundle_ids, splits_ids):
        return X
    idx = {sid: i for i, sid in enumerate(bundle_ids)}
    return X[np.asarray([idx[sid] for sid in splits_ids], dtype=int), :]


# ── Rank table loading ──
def find_var_scores(outputs_dir: Path, dataset: str, view: str) -> Optional[Path]:
    p = outputs_dir / "03_variance" / "per_view" / f"variance_scores__{dataset}__{view}.csv"
    if p.exists():
        return p
    cands = list(outputs_dir.glob(f"**/variance_scores__{dataset}__{view}*"))
    return sorted(cands)[0] if cands else None


def find_shap_importance(outputs_dir: Path, dataset: str, view: str,
                         model: str = "xgb_bal") -> Optional[Path]:
    p = outputs_dir / "04_importance" / "per_model" / f"importance__{dataset}__{view}__{model}.csv.gz"
    if p.exists():
        return p
    cands = list(outputs_dir.glob(f"**/prediction_importance__{dataset}__{view}__*.csv.gz"))
    return sorted(cands)[0] if cands else None


def find_saf_table(outputs_dir: Path, dataset: str, view: str) -> Optional[Path]:
    p = outputs_dir / "12_diagnostic" / "per_view" / f"feature_eta_sq__{dataset}__{view}.csv.gz"
    return p if p.exists() else None


def topk_indices(rank_df: pd.DataFrame, feature_col: str,
                 rank_col: Optional[str], score_col: Optional[str],
                 bundle_feats: np.ndarray, n_select: int) -> np.ndarray:
    feat_to_idx = {f: i for i, f in enumerate(bundle_feats.astype(str))}
    df = rank_df.copy()
    if rank_col and rank_col in df.columns:
        df = df.sort_values(rank_col, ascending=True)
    elif score_col and score_col in df.columns:
        df = df.sort_values(score_col, ascending=False)
    else:
        raise ValueError(f"Missing rank/score columns")
    idxs = []
    for f in df[feature_col].astype(str).tolist():
        if f in feat_to_idx:
            idxs.append(feat_to_idx[f])
        if len(idxs) >= n_select:
            break
    return np.asarray(idxs, dtype=int)


# ── Stable seed (mirrors ablation) ──
def stable_seed(*parts) -> int:
    s = "|".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ── LinearSVC model ──
def make_linearsvc(seed: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                          random_state=seed, dual=True)),
    ])


def balanced_acc(y_true, y_pred):
    labels = np.unique(y_true)
    C = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C) / C.sum(axis=1)
    per_class = per_class[~np.isnan(per_class)]
    return float(np.mean(per_class)) if per_class.size else float("nan")


# ── Pull existing XGBoost regime ──
def get_xgb_regime(outputs_dir: Path, dataset: str, view: str) -> dict:
    """Pull DI and regime from existing ablation master or DI summary."""
    master_p = outputs_dir / "07_ablation" / "ablation_master_summary.csv"
    if master_p.exists():
        df = pd.read_csv(master_p)
        sub = df[(df["dataset"] == dataset) & (df["view"] == view) & (df["K_pct"] == K_PCT)]
        if not sub.empty:
            row = sub.iloc[0]
            return {
                "DI_mean": float(row.get("DI_mean", np.nan)),
                "regime": str(row.get("regime", "UNKNOWN")),
                "delta_var_random_xgb": float(row.get("delta_var_random_mean", np.nan)),
            }
    return {"DI_mean": np.nan, "regime": "UNKNOWN", "delta_var_random_xgb": np.nan}


# ── Main ──
def main():
    ap = argparse.ArgumentParser(description="LinearSVC hero-view sensitivity")
    ap.add_argument("--outputs-dir", type=str, required=True)
    ap.add_argument("--k-pct", type=int, default=K_PCT)
    ap.add_argument("--max-repeats", type=int, default=MAX_REPEATS)
    ap.add_argument("--n-random-draws", type=int, default=N_RANDOM_DRAWS)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / "14_supplementary_analyses" / "linearsvc_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    k_pct = args.k_pct

    print("=" * 60)
    print("LinearSVC Hero-View Sensitivity Check")
    print(f"K={k_pct}%  repeats={args.max_repeats}  random_draws={args.n_random_draws}")
    print("=" * 60)

    result_rows: List[dict] = []

    for dataset, view in HERO_VIEWS:
        print(f"\n── {dataset} / {view} ──")
        t_start = time.time()

        # Load data
        splits_path = find_splits(outputs_dir, dataset)
        bundle_path = find_bundle(outputs_dir, dataset)
        y, splits_ids, fold_ids, info = load_splits(splits_path)
        bundle_ids, X, feats = load_bundle_view(bundle_path, view)
        X = align_X(X, bundle_ids, splits_ids)
        n_samples, n_features = X.shape
        n_select = max(1, int(round((k_pct / 100.0) * n_features)))

        if args.max_repeats:
            fold_ids = fold_ids[:args.max_repeats, :]
        n_repeats = fold_ids.shape[0]

        print(f"  {n_samples} samples × {n_features} features, K={k_pct}% -> {n_select} selected")

        # Load rank tables
        var_path = find_var_scores(outputs_dir, dataset, view)
        shap_path = find_shap_importance(outputs_dir, dataset, view)
        saf_path = find_saf_table(outputs_dir, dataset, view)

        var_df = pd.read_csv(var_path) if var_path else None
        shap_df = pd.read_csv(shap_path) if shap_path else None
        saf_df = pd.read_csv(saf_path, compression="gzip") if saf_path else None

        # Compute feature indices
        strategies: Dict[str, Optional[np.ndarray]] = {"all": None}

        if var_df is not None:
            v_feat = "feature" if "feature" in var_df.columns else var_df.columns[0]
            v_rank = "v_rank" if "v_rank" in var_df.columns else ("rank" if "rank" in var_df.columns else None)
            v_score = "v_score" if "v_score" in var_df.columns else ("score" if "score" in var_df.columns else None)
            strategies["var_topk"] = topk_indices(var_df, v_feat, v_rank, v_score, feats, n_select)

        if shap_df is not None:
            p_feat = "feature" if "feature" in shap_df.columns else shap_df.columns[0]
            p_rank = "p_rank" if "p_rank" in shap_df.columns else ("rank" if "rank" in shap_df.columns else None)
            p_score = "p_score" if "p_score" in shap_df.columns else ("importance" if "importance" in shap_df.columns else None)
            strategies["shap_topk"] = topk_indices(shap_df, p_feat, p_rank, p_score, feats, n_select)

        if saf_df is not None:
            strategies["saf_vbetween_topk"] = topk_indices(saf_df, "feature", None, "var_between", feats, n_select)

        # Run CV ablation with LinearSVC
        perf_by_strategy: Dict[str, List[float]] = {s: [] for s in strategies}
        perf_by_strategy["random_mean"] = []

        for r in range(n_repeats):
            fold_row = fold_ids[r, :]
            for fold in np.unique(fold_row).astype(int):
                train_idx = np.where(fold_row != fold)[0]
                test_idx = np.where(fold_row == fold)[0]

                X_tr, y_tr = X[train_idx], y[train_idx]
                X_te, y_te = X[test_idx], y[test_idx]

                for strat, idx in strategies.items():
                    seed = stable_seed(dataset, view, "linearsvc", r, fold, k_pct, strat)
                    model = make_linearsvc(seed)
                    Xtr = X_tr if idx is None else X_tr[:, idx]
                    Xte = X_te if idx is None else X_te[:, idx]
                    try:
                        model.fit(Xtr, y_tr)
                        pred = model.predict(Xte)
                        perf_by_strategy[strat].append(balanced_acc(y_te, pred))
                    except Exception as e:
                        print(f"    [WARN] {strat} r={r} f={fold}: {e}")
                        perf_by_strategy[strat].append(float("nan"))

                # Random draws
                rand_perfs = []
                for d in range(args.n_random_draws):
                    rng = np.random.default_rng(stable_seed(dataset, view, "linearsvc", r, fold, k_pct, "RAND", d))
                    rand_idx = rng.choice(n_features, size=n_select, replace=False)
                    seed_r = stable_seed(dataset, view, "linearsvc", r, fold, k_pct, "RANDFIT", d)
                    model = make_linearsvc(seed_r)
                    try:
                        model.fit(X_tr[:, rand_idx], y_tr)
                        pred = model.predict(X_te[:, rand_idx])
                        rand_perfs.append(balanced_acc(y_te, pred))
                    except Exception:
                        rand_perfs.append(float("nan"))
                perf_by_strategy["random_mean"].append(float(np.nanmean(rand_perfs)))

        # Aggregate
        def mean_ci(vals):
            v = np.array([x for x in vals if np.isfinite(x)])
            if len(v) == 0:
                return np.nan, np.nan, np.nan
            m = float(np.mean(v))
            lo = float(np.percentile(v, 2.5)) if len(v) > 1 else m
            hi = float(np.percentile(v, 97.5)) if len(v) > 1 else m
            return m, lo, hi

        perf = {s: mean_ci(v) for s, v in perf_by_strategy.items()}

        var_mean = perf.get("var_topk", (np.nan,))[0]
        rand_mean = perf.get("random_mean", (np.nan,))[0]
        saf_mean = perf.get("saf_vbetween_topk", (np.nan,))[0]
        shap_mean = perf.get("shap_topk", (np.nan,))[0]
        delta_var_random = var_mean - rand_mean if np.isfinite(var_mean) and np.isfinite(rand_mean) else np.nan
        delta_saf_var = saf_mean - var_mean if np.isfinite(saf_mean) and np.isfinite(var_mean) else np.nan

        # Determine LinearSVC regime from sign of delta_var_random
        if np.isfinite(delta_var_random):
            if delta_var_random < -0.005:
                lsvc_regime = "ANTI_ALIGNED"
            elif delta_var_random > 0.005:
                lsvc_regime = "COUPLED"
            else:
                lsvc_regime = "MIXED"
        else:
            lsvc_regime = "UNKNOWN"

        # Pull XGBoost regime for comparison
        xgb_info = get_xgb_regime(outputs_dir, dataset, view)

        runtime = time.time() - t_start

        print(f"  LinearSVC results (K={k_pct}%):")
        for s in ["all", "var_topk", "shap_topk", "saf_vbetween_topk", "random_mean"]:
            if s in perf:
                m, lo, hi = perf[s]
                print(f"    {s:25s}  {m:.4f}  [{lo:.4f}, {hi:.4f}]")
        print(f"  delta_var_random (LinearSVC): {delta_var_random:+.4f}")
        print(f"  delta_saf_var    (LinearSVC): {delta_saf_var:+.4f}")
        print(f"  LinearSVC regime: {lsvc_regime}")
        print(f"  XGBoost regime:   {xgb_info['regime']}")
        print(f"  Regime agrees:    {lsvc_regime == xgb_info['regime']}")
        print(f"  Runtime: {runtime:.1f}s")

        row = {
            "dataset": dataset,
            "view": view,
            "k_pct": k_pct,
            "model_family": "linearsvc",
            "DI_mean": xgb_info["DI_mean"],
            "perf_all_mean": perf.get("all", (np.nan,))[0],
            "perf_var_mean": var_mean,
            "perf_shap_mean": shap_mean,
            "perf_saf_mean": saf_mean,
            "perf_random_mean": rand_mean,
            "delta_var_random": round(delta_var_random, 4) if np.isfinite(delta_var_random) else np.nan,
            "delta_saf_var": round(delta_saf_var, 4) if np.isfinite(delta_saf_var) else np.nan,
            "regime_linearsvc": lsvc_regime,
            "regime_xgb": xgb_info["regime"],
            "delta_var_random_xgb": xgb_info["delta_var_random_xgb"],
            "regime_agrees": lsvc_regime == xgb_info["regime"],
            "n_repeats": n_repeats,
            "n_folds": int(len(np.unique(fold_ids[0]))),
            "n_random_draws": args.n_random_draws,
            "runtime_seconds": round(runtime, 1),
        }
        result_rows.append(row)

    # Write outputs
    df = pd.DataFrame(result_rows)
    csv_path = out_dir / "supp_linear_model_regime_check.csv"
    df.to_csv(csv_path, index=False)

    n_agree = int(df["regime_agrees"].sum())
    n_total = len(df)
    summary = {
        "description": "LinearSVC hero-view sensitivity: does regime hold with a linear model?",
        "k_pct": k_pct,
        "n_views": n_total,
        "n_regime_agree": n_agree,
        "regime_agreement_rate": round(n_agree / n_total, 2) if n_total else 0,
        "views": df[["dataset", "view", "regime_linearsvc", "regime_xgb", "regime_agrees"]].to_dict("records"),
    }
    json_path = out_dir / "linearsvc_sensitivity_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    print(f"\n{'=' * 60}")
    print(f"Regime agreement: {n_agree}/{n_total}")
    if n_agree == n_total:
        print("VERDICT: LinearSVC confirms all XGBoost regime assignments.")
    else:
        disagreed = df[~df["regime_agrees"]]
        for _, r in disagreed.iterrows():
            print(f"  DISAGREE: {r['dataset']}/{r['view']} — XGB={r['regime_xgb']}, LSVC={r['regime_linearsvc']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
