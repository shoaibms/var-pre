#!/usr/bin/env python3
"""
04_rj_decomposition.py (FINAL)

Mechanistic diagnostic: Between/Within variance decomposition.

KEY POINTS:
- NaN-safe: uses np.nanmean/np.nanvar or fails fast with assertion
- Verified class-frequency weighting for between/within variance
- Merge sanity check with merge_rate reporting

R_j = Var(E[X_j|Y]) / (E[Var(X_j|Y)] + ε) = between / (within + ε)

Outputs:
    outputs/05_mechanistic/rj_scores__{dataset}__{view}.csv.gz
    outputs/05_mechanistic/rj_correlations.csv

Usage:
    python 04_rj_decomposition.py --outputs-dir outputs
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

VIEW_REGISTRY = {
    "mlomics": ["mRNA", "miRNA", "methylation", "CNV"],
    "ibdmdb": ["MGX", "MGX_func", "MPX", "MBX"],
    "ccle": ["mRNA", "CNV", "proteomics"],
    "tcga_gbm": ["mRNA", "methylation", "CNV"],
}
EPSILON = 1e-10
VAR_COLS = ["v_score", "v_marginal_score", "v_marginal", "v_total", "variance", "var"]
IMP_COLS = ["p_xgb_bal_score", "p_rf_score", "p_consensus_score", "p_consensus", "p_mean", "p_score"]


def find_bundle(outputs_dir: Path, dataset: str) -> Optional[Path]:
    for suffix in ["_bundle_normalized.npz", "_bundle.npz"]:
        p = outputs_dir / "bundles" / f"{dataset}{suffix}"
        if p.exists():
            return p
    for m in outputs_dir.rglob(f"*{dataset}*bundle*.npz"):
        return m
    return None


def find_vp_joined(outputs_dir: Path, dataset: str, view: str) -> Optional[Path]:
    for ext in [".csv.gz", ".csv"]:
        p = outputs_dir / "04_importance" / "joined_vp" / f"vp_joined__{dataset}__{view}{ext}"
        if p.exists():
            return p
    return None


def load_bundle(path: Path, view: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, features. Safe label encoding."""
    z = np.load(path, allow_pickle=False)
    
    y_raw = z["y"]
    if y_raw.dtype.kind not in ("i", "u"):
        y, _ = pd.factorize(y_raw)
    else:
        y = y_raw.astype(int)
    
    X = None
    for k in [f"X_{view}", f"x_{view}", f"data_{view}"]:
        if k in z.files:
            X = z[k].astype(np.float64)
            break
    if X is None:
        avail = [k for k in z.files if k.startswith("X_")]
        raise ValueError(f"View '{view}' not found. Available: {avail}")
    
    feat = None
    for k in [f"features_{view}", f"feature_names_{view}"]:
        if k in z.files:
            feat = z[k].astype(str)
            break
    if feat is None:
        feat = np.array([f"f{i}" for i in range(X.shape[1])])
    
    return X, y.astype(int), feat


def load_regime_consensus(outputs_dir: Path) -> Dict[Tuple[str, str], str]:
    p = outputs_dir / "04_importance" / "aggregated" / "regime_consensus.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {(r["dataset"], r["view"]): r.get("consensus_regime", "UNKNOWN") for _, r in df.iterrows()}


def compute_variance_decomposition(X: np.ndarray, y: np.ndarray, nan_policy: str = "ignore") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Law of total variance decomposition with class-frequency weighting.
    NaN-safe with nanmean/nanvar when nan_policy='ignore'.
    """
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    
    nan_count = int(np.isnan(X).sum())
    nan_features = int((np.isnan(X).sum(axis=0) > 0).sum())
    
    if nan_count > 0:
        if nan_policy == "fail":
            raise ValueError(f"NaN detected: {nan_count} values in {nan_features} features")
        mean_fn, var_fn = np.nanmean, lambda x, axis: np.nanvar(x, axis=axis, ddof=0)
    else:
        mean_fn, var_fn = np.mean, lambda x, axis: np.var(x, axis=axis, ddof=0)
    
    # Class statistics
    class_means = np.zeros((n_classes, n_features))
    class_vars = np.zeros((n_classes, n_features))
    class_sizes = np.zeros(n_classes, dtype=int)
    
    for i, c in enumerate(classes):
        mask = (y == c)
        class_sizes[i] = mask.sum()
        Xc = X[mask]
        class_means[i] = mean_fn(Xc, axis=0)
        class_vars[i] = var_fn(Xc, axis=0)
    
    # Class-frequency weights
    weights = class_sizes / class_sizes.sum()
    
    global_mean = mean_fn(X, axis=0)
    total_var = var_fn(X, axis=0)
    
    # Between-class: Var(E[X|Y]) = Σ_c w_c (μ_c - μ)²
    between_var = np.zeros(n_features)
    for i in range(n_classes):
        between_var += weights[i] * (class_means[i] - global_mean) ** 2
    
    # Within-class: E[Var(X|Y)] = Σ_c w_c σ²_c
    within_var = np.zeros(n_features)
    for i in range(n_classes):
        within_var += weights[i] * class_vars[i]
    
    R_j = between_var / (within_var + EPSILON)
    
    # Decomposition check
    max_diff = np.nanmax(np.abs(total_var - (between_var + within_var)))
    
    diag = {"n_samples": n_samples, "n_features": n_features, "n_classes": n_classes,
            "nan_count": nan_count, "nan_features": nan_features,
            "class_sizes": class_sizes.tolist(), "decomp_max_diff": float(max_diff)}
    
    return total_var, between_var, within_var, R_j, diag


def compute_correlations(rj_df: pd.DataFrame, vp_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute correlations with merge sanity check."""
    n_rj = len(rj_df)
    merged = rj_df.merge(vp_df, on="feature", how="inner", suffixes=("", "_vp"))
    n_merged = len(merged)
    merge_rate = n_merged / n_rj if n_rj > 0 else 0
    
    result = {"n_rj": n_rj, "n_vp": len(vp_df), "n_merged": n_merged, "merge_rate": merge_rate}
    
    if merge_rate < 0.95:
        result["merge_warning"] = f"Low merge rate ({merge_rate:.1%}): possible feature naming drift"
    
    if n_merged < 10:
        result["error"] = f"Only {n_merged} merged"
        return result
    
    R_j = merged["R_j"].values
    total_var = merged["total_var"].values
    
    # Find columns with precedence
    v_col = next((c for c in VAR_COLS if c in merged.columns), None)
    p_col = next((c for c in IMP_COLS if c in merged.columns), None)
    
    if not v_col:
        v_col = next((c for c in merged.columns if "var" in c.lower() and c != "total_var"), None)
    if not p_col:
        p_col = next((c for c in merged.columns if "shap" in c.lower() or "importance" in c.lower()), None)
    
    if not v_col or not p_col:
        result["error"] = f"Missing columns. v={v_col}, p={p_col}"
        return result
    
    V_j, P_j = merged[v_col].values, merged[p_col].values
    result["v_col"], result["p_col"] = v_col, p_col
    
    valid = np.isfinite(R_j) & np.isfinite(V_j) & np.isfinite(P_j)
    n_valid = valid.sum()
    result["n_valid"] = int(n_valid)
    
    if n_valid < 10:
        result["error"] = f"Only {n_valid} valid"
        return result
    
    R_j, V_j, P_j, total_var = R_j[valid], V_j[valid], P_j[valid], total_var[valid]
    between_var, within_var = merged["between_var"].values[valid], merged["within_var"].values[valid]
    
    # Key correlations
    result["corr_P_R"], result["pval_P_R"] = stats.spearmanr(P_j, R_j)
    result["corr_V_R"], result["pval_V_R"] = stats.spearmanr(V_j, R_j)
    result["corr_V_totalvar"], _ = stats.spearmanr(V_j, total_var)
    result["corr_P_V"], result["pval_P_V"] = stats.spearmanr(P_j, V_j)
    result["corr_P_between"], _ = stats.spearmanr(P_j, between_var)
    result["corr_V_between"], _ = stats.spearmanr(V_j, between_var)
    result["corr_V_within"], _ = stats.spearmanr(V_j, within_var)
    
    return result


def process_view(outputs_dir: Path, dataset: str, view: str, nan_policy: str):
    print(f"  {dataset}/{view}...", end=" ")
    
    bundle_path = find_bundle(outputs_dir, dataset)
    vp_path = find_vp_joined(outputs_dir, dataset, view)
    
    if not bundle_path or not vp_path:
        print("WARNING: files not found"); return None, None
    
    try:
        X, y, features = load_bundle(bundle_path, view)
        vp_df = pd.read_csv(vp_path)
    except Exception as e:
        print(f"FAILED load: {e}"); return None, None
    
    try:
        total_var, between_var, within_var, R_j, diag = compute_variance_decomposition(X, y, nan_policy)
    except Exception as e:
        print(f"FAILED decomp: {e}"); return None, None
    
    rj_df = pd.DataFrame({
        "feature": features, "total_var": total_var, "between_var": between_var,
        "within_var": within_var, "R_j": R_j, "R_j_log10": np.log10(R_j + EPSILON),
        "between_frac": between_var / (total_var + EPSILON),
    })
    rj_df["dataset"], rj_df["view"] = dataset, view
    
    corr = compute_correlations(rj_df, vp_df)
    corr["dataset"], corr["view"] = dataset, view
    
    if "error" not in corr:
        print(f"corr(P,R)={corr['corr_P_R']:.3f} corr(V,R)={corr['corr_V_R']:.3f} "
              f"merge={corr['merge_rate']:.0%}")
        if "merge_warning" in corr:
            print(f"    WARNING: {corr['merge_warning']}")
    else:
        print(f"WARNING: {corr['error']}")
    
    return rj_df, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--view", default=None)
    parser.add_argument("--nan-policy", default="ignore", choices=["fail", "ignore"])
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / "05_mechanistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("R_j BETWEEN/WITHIN DECOMPOSITION")
    print(f"NaN policy: {args.nan_policy}")
    print("=" * 70)
    
    tasks = ([(args.dataset, args.view)] if args.dataset and args.view else
             [(args.dataset, v) for v in VIEW_REGISTRY.get(args.dataset, [])] if args.dataset else
             [(d, v) for d, vs in VIEW_REGISTRY.items() for v in vs])
    
    regime_map = load_regime_consensus(outputs_dir)
    all_corr = []
    
    for ds, vw in tasks:
        rj_df, corr = process_view(outputs_dir, ds, vw, args.nan_policy)
        if rj_df is not None:
            rj_df.to_csv(out_dir / f"rj_scores__{ds}__{vw}.csv.gz", index=False, compression="gzip")
            if "error" not in corr:
                corr["regime"] = regime_map.get((ds, vw), "UNKNOWN")
                all_corr.append(corr)
    
    if all_corr:
        corr_df = pd.DataFrame(all_corr)
        cols = ["dataset", "view", "regime", "n_valid", "merge_rate",
                "corr_P_R", "pval_P_R", "corr_V_R", "pval_V_R", "corr_P_V",
                "corr_V_totalvar", "corr_P_between", "corr_V_between", "corr_V_within"]
        cols = [c for c in cols if c in corr_df.columns]
        corr_df[cols].to_csv(out_dir / "rj_correlations.csv", index=False)
        print(f"\nSaved: rj_correlations.csv")
        
        # Summary
        print("\nSUMMARY:")
        print(f"{'Dataset/View':<25} {'Regime':<12} {'corr(P,R)':>10} {'corr(V,R)':>10} {'merge%':>8}")
        print("-" * 70)
        for _, r in corr_df.iterrows():
            print(f"{r['dataset']}/{r['view']:<18} {r['regime']:<12} "
                  f"{r['corr_P_R']:>10.3f} {r['corr_V_R']:>10.3f} {r['merge_rate']*100:>7.0f}%")
        
        mean_PR, mean_VR = corr_df["corr_P_R"].mean(), corr_df["corr_V_R"].mean()
        print(f"\nHypothesis: corr(P,R)={mean_PR:.3f} vs corr(V,R)={mean_VR:.3f} → " +
              ("SUPPORTED" if mean_PR > mean_VR + 0.1 else "UNCLEAR"))
    
    print("\nDONE")


if __name__ == "__main__":
    main()
