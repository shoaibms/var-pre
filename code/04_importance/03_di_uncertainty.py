#!/usr/bin/env python3
"""
03_di_uncertainty.py (FINAL)

Compute DI uncertainty from CV repeats.

KEY POINTS:
- Uses mean_abs_shap_per_repeat from SHAP NPZ (actual CV repeats)
- Percentile intervals from repeat-CV variability (NOT bootstrap CI)
- Prints npz.files on failure for instant debugging
- Reports feature alignment: n_var, n_shap, n_common, common_fraction

Outputs:
    outputs/04_importance/uncertainty/di_per_repeat__{dataset}__{view}.csv
    outputs/04_importance/uncertainty/di_summary.csv

Usage:
    python 03_di_uncertainty.py --outputs-dir outputs
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

K_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
VIEW_REGISTRY = {
    "mlomics": ["mRNA", "miRNA", "methylation", "CNV"],
    "ibdmdb": ["MGX", "MGX_func", "MPX", "MBX"],
    "ccle": ["mRNA", "CNV", "proteomics"],
    "tcga_gbm": ["mRNA", "methylation", "CNV"],
}
REGIME_DI_COUPLED, REGIME_DI_ANTI = 0.85, 1.0


def find_shap_npz(outputs_dir: Path, dataset: str, view: str, model: str) -> Optional[Path]:
    base = outputs_dir / "03_supervised"
    model_short = model.replace("_bal", "")
    for pattern in [f"tree_models_{model}/importance/shap__{dataset}__{view}__{model_short}.npz",
                    f"tree_models_*/importance/shap__{dataset}__{view}__*.npz"]:
        matches = list(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_variance_csv(outputs_dir: Path, dataset: str, view: str) -> Optional[Path]:
    for ext in [".csv.gz", ".csv"]:
        p = outputs_dir / "02_unsupervised" / "variance_scores" / f"variance_scores__{dataset}__{view}{ext}"
        if p.exists():
            return p
    return None


def load_shap_importance_per_repeat(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load per-repeat SHAP. Prints keys on failure."""
    z = np.load(path, allow_pickle=False)
    keys = list(z.files)
    
    feat = None
    for k in ["feature_names", "features", "feature_ids"]:
        if k in keys:
            feat = np.array([str(x) for x in z[k]])
            break
    
    for k in ["mean_abs_shap_per_repeat", "shap_abs_mean_per_repeat", "importance_per_repeat"]:
        if k in keys:
            imp = np.asarray(z[k])
            return (imp if imp.ndim == 2 else imp.reshape(1, -1)), feat
    
    if "shap_values" in keys:
        sv = np.asarray(z["shap_values"])
        if sv.ndim == 3:
            return np.mean(np.abs(sv), axis=1), feat
        elif sv.ndim == 2:
            return np.mean(np.abs(sv), axis=0, keepdims=True), feat
    
    if "mean_abs_shap" in keys:
        imp = np.asarray(z["mean_abs_shap"])
        return (imp.reshape(1, -1) if imp.ndim == 1 else imp), feat
    
    raise KeyError(f"No per-repeat SHAP in {path.name}. Keys: {keys}")


def load_variance_scores(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    feat_col = next((c for c in ["feature", "feature_name", "gene"] if c in df.columns), df.columns[0])
    var_col = next((c for c in ["v_marginal", "v_total", "variance", "var"] if c in df.columns), None)
    if not var_col:
        var_col = next((c for c in df.columns if "var" in c.lower()), None)
    if not var_col:
        raise ValueError(f"No variance column in {path.name}")
    return df[var_col].values.astype(np.float64), df[feat_col].astype(str).values


def compute_one_di(v_rank: np.ndarray, p_rank: np.ndarray, k_frac: float) -> Dict[str, Any]:
    n = len(v_rank)
    k_n = max(1, int(n * k_frac))
    top_v, top_p = set(np.where(v_rank <= k_n)[0]), set(np.where(p_rank <= k_n)[0])
    intersection, union = len(top_v & top_p), len(top_v | top_p)
    J = intersection / union if union > 0 else 0.0
    q = k_frac
    J_rand = q / (2 - q)
    J_tilde = (J - J_rand) / (1 - J_rand) if J_rand < 1 else 0.0
    return {"k_pct": int(k_frac * 100), "k_n": k_n, "n_features": n, "J": J,
            "J_rand": J_rand, "delta_J": J - J_rand, "DI": 1 - J_tilde, "overlap": intersection}


def compute_di_across_repeats(var_scores, var_feat, shap_imp, shap_feat) -> Tuple[pd.DataFrame, Dict]:
    var_dict = dict(zip(var_feat, var_scores))
    shap_idx = {f: i for i, f in enumerate(shap_feat)}
    common = [f for f in shap_feat if f in var_dict]
    
    n_var, n_shap, n_common = len(var_feat), len(shap_feat), len(common)
    align = {"n_var": n_var, "n_shap": n_shap, "n_common": n_common, "common_frac": n_common / max(n_shap, 1)}
    
    if n_common < 100:
        raise ValueError(f"Alignment: only {n_common} common features")
    
    var_aligned = np.array([var_dict[f] for f in common])
    shap_aligned = shap_imp[:, [shap_idx[f] for f in common]]
    v_rank = stats.rankdata(-var_aligned, method="average")
    
    results = []
    for r in range(shap_imp.shape[0]):
        p_rank = stats.rankdata(-shap_aligned[r], method="average")
        rho, _ = stats.spearmanr(var_aligned, shap_aligned[r])
        for k in K_FRACTIONS:
            row = compute_one_di(v_rank, p_rank, k)
            row["repeat"], row["spearman_rho"] = r, rho
            results.append(row)
    return pd.DataFrame(results), align


def classify_regime(di: float) -> str:
    return "ANTI_ALIGNED" if di >= REGIME_DI_ANTI else ("COUPLED" if di < REGIME_DI_COUPLED else "MIXED")


def aggregate_di_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile intervals from repeat-CV variability (NOT bootstrap CI)."""
    results = []
    for k_pct in [1, 5, 10, 20]:
        kdf = df[df["k_pct"] == k_pct]
        if len(kdf) == 0:
            continue
        di_vals, n = kdf["DI"].values, len(kdf)
        regimes = [classify_regime(d) for d in di_vals]
        rc = pd.Series(regimes).value_counts()
        results.append({
            "k_pct": k_pct, "n_repeats": n,
            "DI_mean": np.mean(di_vals), "DI_std": np.std(di_vals, ddof=1) if n > 1 else 0,
            "DI_pctl_2.5": np.percentile(di_vals, 2.5) if n > 1 else np.mean(di_vals),
            "DI_pctl_97.5": np.percentile(di_vals, 97.5) if n > 1 else np.mean(di_vals),
            "J_mean": kdf["J"].mean(), "delta_J_mean": kdf["delta_J"].mean(), "rho_mean": kdf["spearman_rho"].mean(),
            "regime_COUPLED_frac": rc.get("COUPLED", 0) / n,
            "regime_MIXED_frac": rc.get("MIXED", 0) / n,
            "regime_ANTI_ALIGNED_frac": rc.get("ANTI_ALIGNED", 0) / n,
            "consensus_regime": rc.idxmax(), "regime_confidence": rc.max() / n,
        })
    return pd.DataFrame(results)


def process_view(outputs_dir: Path, dataset: str, view: str, model: str):
    print(f"  {dataset}/{view}...", end=" ")
    shap_path, var_path = find_shap_npz(outputs_dir, dataset, view, model), find_variance_csv(outputs_dir, dataset, view)
    if not shap_path or not var_path:
        print("WARNING: files not found"); return None, None
    
    try:
        shap_imp, shap_feat = load_shap_importance_per_repeat(shap_path)
        var_scores, var_feat = load_variance_scores(var_path)
        if shap_feat is None:
            print("WARNING: no feature names"); return None, None
        di_per, align = compute_di_across_repeats(var_scores, var_feat, shap_imp, shap_feat)
    except Exception as e:
        print(f"FAILED: {e}"); return None, None
    
    di_per["dataset"], di_per["view"], di_per["model"] = dataset, view, model
    di_sum = aggregate_di_uncertainty(di_per)
    di_sum["dataset"], di_sum["view"], di_sum["model"] = dataset, view, model
    
    k10 = di_sum[di_sum["k_pct"] == 10]
    if len(k10) > 0:
        r = k10.iloc[0]
        print(f"DI={r['DI_mean']:.3f}±{r['DI_std']:.3f} [{r['DI_pctl_2.5']:.2f},{r['DI_pctl_97.5']:.2f}] "
              f"{r['consensus_regime']}({r['regime_confidence']:.0%}) align={align['common_frac']:.0%}")
    return di_per, di_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--view", default=None)
    parser.add_argument("--model", default="xgb_bal")
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / "04_importance" / "uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DI UNCERTAINTY FROM CV REPEATS")
    print("NOTE: Percentile intervals from repeat-CV (not bootstrap CI)")
    print("=" * 70)
    
    tasks = ([(args.dataset, args.view)] if args.dataset and args.view else
             [(args.dataset, v) for v in VIEW_REGISTRY.get(args.dataset, [])] if args.dataset else
             [(d, v) for d, vs in VIEW_REGISTRY.items() for v in vs])
    
    all_sum = []
    for ds, vw in tasks:
        di_per, di_sum = process_view(outputs_dir, ds, vw, args.model)
        if di_per is not None:
            di_per.to_csv(out_dir / f"di_per_repeat__{ds}__{vw}.csv", index=False)
            all_sum.append(di_sum)
    
    if all_sum:
        summary = pd.concat(all_sum, ignore_index=True)
        summary.to_csv(out_dir / "di_summary.csv", index=False)
        print(f"\nSaved: di_summary.csv ({len(summary)} rows)")
        
        # Sanity checks
        print("\nSANITY CHECKS (K=10%):")
        for _, r in summary[summary["k_pct"] == 10].iterrows():
            issues = []
            if r["n_repeats"] < 3: issues.append("n<3")
            if r["DI_pctl_2.5"] == r["DI_pctl_97.5"] and r["n_repeats"] > 1: issues.append("pctl=")
            if 0.2 < r["regime_confidence"] < 0.4: issues.append("random?")
            print(f"  {'OK' if not issues else 'WARNING'} {r['dataset']}/{r['view']}: " +
                  f"conf={r['regime_confidence']:.0%}" + (f" [{','.join(issues)}]" if issues else ""))
    print("\nDONE")


if __name__ == "__main__":
    main()
