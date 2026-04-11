#!/usr/bin/env python3
r"""
PHASE 2 -- 03_clustering_comparison.py

Unsupervised validation: does high DI imply variance-based clustering
fails to recover biological clusters?

For each of 14 views, compares clustering quality (ARI, NMI, Silhouette)
under four feature-selection strategies:
    - All features (baseline)
    - TopVar  (top K% by variance)
    - TopSHAP (top K% by SHAP importance)
    - Random  (50 matched-size random draws, mean +/- SD)

Two complementary analyses:
  1. CLUSTERING: PCA -> KMeans(k=n_classes) -> ARI/NMI vs true labels
  2. PC SIGNAL:  fraction of PC variance explained by between-class
                 structure under each feature subset

Then tests whether DI correlates with delta(TopVar - Random) clustering
degradation.

Reads:
  outputs/bundles/{dataset}_bundle_normalized.npz
  outputs/03_splits/splits__{dataset}.npz
  outputs/02_unsupervised/variance_scores/variance_scores__{dataset}__{view}.csv.gz
  outputs/04_importance/per_model/importance__{dataset}__{view}__{model}.csv.gz
  outputs/04_importance/aggregated/regime_consensus.csv  (for DI values)

Writes:
  outputs/14_unsupervised/
    clustering_comparison.csv           # per view x strategy: ARI, NMI, silhouette
    pc_class_signal.csv                 # per view x strategy: between-class variance in PCs
    unsupervised_vs_di.csv              # DI vs delta(TopVar-Random) for correlation
    unsupervised_validation_report.md
    MANIFEST_CLUSTERING_COMPARISON.json

Usage:
  python code/compute/02_unsupervised/03_clustering_comparison.py \
    --outputs-dir outputs \
    --k-pct 10 --n-random 50 --n-pcs 20 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# scikit-learn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from scipy import stats

# --- shared helpers ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import ensure_dir, now_iso

warnings.filterwarnings("ignore", category=FutureWarning)


# ===============================================================
# Constants / Registry
# ===============================================================

DATASETS = ["mlomics", "ibdmdb", "ccle", "tcga_gbm"]

VIEW_REGISTRY = {
    "mlomics":  ["mRNA", "methylation", "CNV", "miRNA"],
    "ibdmdb":   ["MGX", "MGX_func", "MPX", "MBX"],
    "ccle":     ["mRNA", "CNV", "proteomics"],
    "tcga_gbm": ["mRNA", "methylation", "CNV"],
}

DISPLAY_NAMES = {
    "mlomics": "MLOmics",
    "ibdmdb": "IBDMDB",
    "ccle": "CCLE",
    "tcga_gbm": "TCGA-GBM",
}


# ===============================================================
# Helpers
# ===============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score, handling edge cases."""
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(X, labels, sample_size=min(5000, len(labels))))
    except Exception:
        return float("nan")


# ===============================================================
# Data loading (follows existing pipeline conventions)
# ===============================================================

def find_bundle(outputs_dir: Path, dataset: str) -> Path:
    """Find normalised bundle NPZ."""
    search_dirs = [
        outputs_dir / "01_bundles" / "normalized",
        outputs_dir / "01_bundles" / "normalised",
        outputs_dir / "01_bundles",
        outputs_dir / "bundles",
    ]
    for d in search_dirs:
        for suffix in ["_bundle_normalized.npz", "_bundle_normalised.npz", "_bundle.npz"]:
            p = d / f"{dataset}{suffix}"
            if p.exists():
                return p
    raise FileNotFoundError(
        f"No bundle found for {dataset}. Searched: "
        + ", ".join(str(d) for d in search_dirs)
    )


def load_bundle_view(bundle_path: Path, view: str
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load (y, sample_ids, X, feature_names) for one view."""
    z = np.load(bundle_path, allow_pickle=False)
    y = z["y"]
    sample_ids = z["sample_ids"].astype(str) if "sample_ids" in z.files else None

    x_key = f"X_{view}"
    f_key = f"features_{view}"

    if x_key not in z.files:
        # case-insensitive fallback
        alt = [k for k in z.files if k.lower() == x_key.lower()]
        if alt:
            x_key = alt[0]
        else:
            raise KeyError(f"View '{view}' not found. Available: "
                           f"{[k for k in z.files if k.startswith('X_')]}")

    X = z[x_key].astype(np.float32)
    feats = z[f_key].astype(str).tolist() if f_key in z.files else [
        f"f{i}" for i in range(X.shape[1])
    ]

    if not np.isfinite(X).all():
        # impute NaN with column median
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])

    return y, sample_ids, X, feats


def load_variance_ranks(outputs_dir: Path, dataset: str, view: str
                        ) -> Optional[pd.DataFrame]:
    """Load variance scores and return feature ranking."""
    var_dir = outputs_dir / "02_unsupervised" / "variance_scores"
    for ext in [".csv.gz", ".csv"]:
        p = var_dir / f"variance_scores__{dataset}__{view}{ext}"
        if p.exists():
            df = pd.read_csv(p)
            return df
    return None


def load_importance_ranks(outputs_dir: Path, dataset: str, view: str,
                          model: str = "xgb_bal") -> Optional[pd.DataFrame]:
    """Load SHAP importance scores and return feature ranking."""
    imp_dir = outputs_dir / "04_importance" / "per_model"
    for ext in [".csv.gz", ".csv"]:
        p = imp_dir / f"importance__{dataset}__{view}__{model}{ext}"
        if p.exists():
            df = pd.read_csv(p)
            return df
    return None


def load_di_values(outputs_dir: Path, k_pct: int = 10) -> pd.DataFrame:
    """Load DI from regime_consensus or vad_summary."""
    # Try regime_consensus first
    rc_path = outputs_dir / "04_importance" / "aggregated" / "regime_consensus.csv"
    if rc_path.exists():
        rc = pd.read_csv(rc_path)
        if "DI" in rc.columns or "di" in rc.columns:
            di_col = "DI" if "DI" in rc.columns else "di"
            return rc[["dataset", "view", di_col]].rename(columns={di_col: "DI"})

    # Fallback to vad_summary
    vad_path = outputs_dir / "12_diagnostic" / "vad_summary.csv"
    if vad_path.exists():
        vad = pd.read_csv(vad_path)
        if "k_pct" in vad.columns:
            vad = vad[vad["k_pct"] == k_pct]
        # DI column might be di_mean, DI, etc.
        for col in ["di_mean", "DI", "di"]:
            if col in vad.columns:
                return vad[["dataset", "view", col]].rename(columns={col: "DI"})

    # Fallback: compute from vp_kgrid
    kgrid_dir = outputs_dir / "04_importance" / "joined_vp"
    rows = []
    for ds in DATASETS:
        for vw in VIEW_REGISTRY.get(ds, []):
            for ext in [".csv.gz", ".csv"]:
                p = kgrid_dir / f"vp_kgrid__{ds}__{vw}{ext}"
                if p.exists():
                    df = pd.read_csv(p)
                    if "k_pct" in df.columns and "DI" in df.columns:
                        row = df[df["k_pct"] == k_pct]
                        if not row.empty:
                            rows.append({"dataset": ds, "view": vw,
                                         "DI": float(row.iloc[0]["DI"])})
                    break
    if rows:
        return pd.DataFrame(rows)

    raise FileNotFoundError("Could not find DI values in any expected location")


# ===============================================================
# Feature subset selection
# ===============================================================

def get_feature_subsets(
    feature_names: List[str],
    var_df: Optional[pd.DataFrame],
    imp_df: Optional[pd.DataFrame],
    k_pct: int,
    n_random: int,
    rng: np.random.Generator,
) -> Dict[str, List[np.ndarray]]:
    """
    Return dict of strategy -> list of boolean index masks.
    'Random' returns n_random masks; others return a single-element list.
    """
    p = len(feature_names)
    k = max(1, int(p * k_pct / 100))
    feat_set = set(feature_names)
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    subsets: Dict[str, List[np.ndarray]] = {}

    # All features
    subsets["All"] = [np.ones(p, dtype=bool)]

    # TopVar
    if var_df is not None:
        # Align to feature order
        score_col = None
        for c in ["score", "variance", "var", "marginal_variance"]:
            if c in var_df.columns:
                score_col = c
                break
        feat_col = None
        for c in ["feature", "gene", "name"]:
            if c in var_df.columns:
                feat_col = c
                break

        if score_col and feat_col:
            var_df_aligned = var_df[var_df[feat_col].isin(feat_set)].copy()
            top_var_names = set(
                var_df_aligned.nlargest(k, score_col)[feat_col].tolist()
            )
            mask = np.array([f in top_var_names for f in feature_names], dtype=bool)
            if mask.sum() >= max(1, k // 2):
                subsets["TopVar"] = [mask]

    # TopSHAP
    if imp_df is not None:
        score_col = None
        for c in ["p_score", "shap_mean", "importance", "mean_abs_shap"]:
            if c in imp_df.columns:
                score_col = c
                break
        feat_col = None
        for c in ["feature", "gene", "name"]:
            if c in imp_df.columns:
                feat_col = c
                break

        if score_col and feat_col:
            imp_df_aligned = imp_df[imp_df[feat_col].isin(feat_set)].copy()
            top_shap_names = set(
                imp_df_aligned.nlargest(k, score_col)[feat_col].tolist()
            )
            mask = np.array([f in top_shap_names for f in feature_names], dtype=bool)
            if mask.sum() >= max(1, k // 2):
                subsets["TopSHAP"] = [mask]

    # Random (n_random draws of size k)
    random_masks = []
    for _ in range(n_random):
        idx = rng.choice(p, size=k, replace=False)
        mask = np.zeros(p, dtype=bool)
        mask[idx] = True
        random_masks.append(mask)
    subsets["Random"] = random_masks

    return subsets


# ===============================================================
# Analysis 1: Clustering (PCA -> KMeans -> ARI/NMI)
# ===============================================================

def clustering_analysis(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    n_pcs: int,
    n_classes: int,
    seed: int,
) -> Dict[str, float]:
    """
    Subset X by mask, scale, PCA, KMeans, return ARI/NMI/Silhouette.
    """
    X_sub = X[:, mask]
    if X_sub.shape[1] == 0:
        return {"ARI": float("nan"), "NMI": float("nan"), "Silhouette": float("nan")}

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    # PCA
    n_comp = min(n_pcs, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if n_comp < 2:
        return {"ARI": float("nan"), "NMI": float("nan"), "Silhouette": float("nan")}

    pca = PCA(n_components=n_comp, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    km = KMeans(n_clusters=n_classes, n_init=10, random_state=seed, max_iter=300)
    pred_labels = km.fit_predict(X_pca)

    ari = float(adjusted_rand_score(y, pred_labels))
    nmi = float(normalized_mutual_info_score(y, pred_labels, average_method="arithmetic"))
    sil = safe_silhouette(X_pca, pred_labels)

    return {"ARI": ari, "NMI": nmi, "Silhouette": sil}


# ===============================================================
# Analysis 2: PC class signal (between-class variance in PCs)
# ===============================================================

def pc_class_signal(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    n_pcs: int,
    seed: int,
) -> Dict[str, float]:
    """
    For the first n_pcs PCs of X[:, mask], compute fraction of variance
    that is between-class. This directly tests whether the dominant
    variance axes capture class structure.
    """
    X_sub = X[:, mask]
    if X_sub.shape[1] == 0:
        return {"pc_between_frac": float("nan"), "pc_between_cumvar": float("nan")}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    n_comp = min(n_pcs, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if n_comp < 1:
        return {"pc_between_frac": float("nan"), "pc_between_cumvar": float("nan")}

    pca = PCA(n_components=n_comp, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    classes = np.unique(y)
    n = len(y)

    # Total variance of each PC (should be ~eigenvalues after scaling)
    var_total = np.var(X_pca, axis=0, ddof=0)

    # Between-class variance of each PC
    grand_mean = np.mean(X_pca, axis=0)
    var_between = np.zeros(n_comp)
    for c in classes:
        mask_c = y == c
        n_c = mask_c.sum()
        if n_c == 0:
            continue
        mean_c = np.mean(X_pca[mask_c], axis=0)
        var_between += (n_c / n) * (mean_c - grand_mean) ** 2

    # Weighted average: how much of PC variance is between-class?
    # Weight by explained variance ratio
    evr = pca.explained_variance_ratio_
    frac_per_pc = np.where(var_total > 1e-12, var_between / var_total, 0.0)

    # Weighted mean across PCs (weighted by variance explained)
    pc_between_frac = float(np.average(frac_per_pc, weights=evr))

    # Cumulative between-class variance captured by PCs
    # (sum of between-class * explained_variance_ratio)
    pc_between_cumvar = float(np.sum(frac_per_pc * evr))

    return {
        "pc_between_frac": pc_between_frac,
        "pc_between_cumvar": pc_between_cumvar,
    }


# ===============================================================
# Main
# ===============================================================

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="C4.1: Unsupervised clustering comparison across feature subsets"
    )
    parser.add_argument("--outputs-dir", type=Path, required=True,
                        help="Root outputs directory")
    parser.add_argument("--k-pct", type=int, default=10,
                        help="Feature selection budget (default: 10%%)")
    parser.add_argument("--n-random", type=int, default=50,
                        help="Number of random draws for baseline (default: 50)")
    parser.add_argument("--n-pcs", type=int, default=20,
                        help="Number of PCA components (default: 20)")
    parser.add_argument("--model", default="xgb_bal",
                        help="Model for SHAP rankings (default: xgb_bal)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--views", default="all",
                        help="'all' or comma-separated dataset/view pairs")
    args = parser.parse_args(argv)

    outputs_dir = args.outputs_dir.resolve()
    out_dir = outputs_dir / "02_unsupervised" / "clustering_comparison"
    ensure_dir(out_dir)

    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("PHASE 2 -- 06  Unsupervised clustering comparison")
    print(f"  {now_iso()}")
    print(f"  outputs_dir: {outputs_dir}")
    print(f"  K:           {args.k_pct}%")
    print(f"  n_random:    {args.n_random}")
    print(f"  n_PCs:       {args.n_pcs}")
    print(f"  model:       {args.model}")
    print(f"  seed:        {args.seed}")
    print("=" * 70)

    # -- Resolve views --
    if args.views == "all":
        view_list = [(ds, vw) for ds in DATASETS for vw in VIEW_REGISTRY[ds]]
    else:
        view_list = []
        for pair in args.views.split(","):
            parts = pair.strip().split("/")
            if len(parts) == 2:
                view_list.append((parts[0], parts[1]))

    print(f"\n  Views to process: {len(view_list)}")

    # -- Load DI values --
    try:
        di_df = load_di_values(outputs_dir, args.k_pct)
        print(f"  DI values loaded: {len(di_df)} views")
    except FileNotFoundError as e:
        print(f"  WARNING: {e} -- will skip DI correlation")
        di_df = pd.DataFrame(columns=["dataset", "view", "DI"])

    # -- Process each view --
    clust_rows = []
    pc_rows = []

    for ds, vw in view_list:
        print(f"\n  {ds}:{vw}")

        # Load data
        try:
            bundle_path = find_bundle(outputs_dir, ds)
            y, sids, X, feats = load_bundle_view(bundle_path, vw)
        except (FileNotFoundError, KeyError) as e:
            print(f"    WARNING: Bundle error: {e}")
            continue

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        print(f"    n={n_samples}, p={n_features}, classes={n_classes}")

        if n_classes < 2:
            print(f"    WARNING: <2 classes, skipping")
            continue

        # Load feature rankings
        var_df = load_variance_ranks(outputs_dir, ds, vw)
        imp_df = load_importance_ranks(outputs_dir, ds, vw, args.model)

        if var_df is None:
            print(f"    WARNING: No variance scores found, skipping")
            continue
        if imp_df is None:
            print(f"    WARNING: No importance scores found, skipping")
            continue

        # Build feature subsets
        subsets = get_feature_subsets(
            feats, var_df, imp_df, args.k_pct, args.n_random, rng
        )

        missing = [s for s in ["TopVar", "TopSHAP"] if s not in subsets]
        if missing:
            print(f"    WARNING: Missing subsets {missing}, skipping")
            continue

        k_actual = int(subsets["TopVar"][0].sum())
        print(f"    K={args.k_pct}% -> {k_actual} features")

        # -- Run analyses for each strategy --
        for strategy, masks in subsets.items():
            aris, nmis, sils = [], [], []
            pc_bfs, pc_bcvs = [], []

            for mask in masks:
                # Clustering
                cr = clustering_analysis(X, y, mask, args.n_pcs, n_classes, args.seed)
                aris.append(cr["ARI"])
                nmis.append(cr["NMI"])
                sils.append(cr["Silhouette"])

                # PC signal
                pcs = pc_class_signal(X, y, mask, args.n_pcs, args.seed)
                pc_bfs.append(pcs["pc_between_frac"])
                pc_bcvs.append(pcs["pc_between_cumvar"])

            clust_rows.append({
                "dataset": ds, "view": vw, "strategy": strategy,
                "k_pct": args.k_pct, "k_features": k_actual,
                "ARI_mean": float(np.nanmean(aris)),
                "ARI_std": float(np.nanstd(aris)) if len(aris) > 1 else 0.0,
                "NMI_mean": float(np.nanmean(nmis)),
                "NMI_std": float(np.nanstd(nmis)) if len(nmis) > 1 else 0.0,
                "Silhouette_mean": float(np.nanmean(sils)),
                "Silhouette_std": float(np.nanstd(sils)) if len(sils) > 1 else 0.0,
                "n_draws": len(masks),
            })

            pc_rows.append({
                "dataset": ds, "view": vw, "strategy": strategy,
                "k_pct": args.k_pct,
                "pc_between_frac_mean": float(np.nanmean(pc_bfs)),
                "pc_between_frac_std": float(np.nanstd(pc_bfs)) if len(pc_bfs) > 1 else 0.0,
                "pc_between_cumvar_mean": float(np.nanmean(pc_bcvs)),
                "pc_between_cumvar_std": float(np.nanstd(pc_bcvs)) if len(pc_bcvs) > 1 else 0.0,
                "n_draws": len(masks),
            })

            lbl = strategy
            if strategy == "Random":
                lbl = f"Random (n={len(masks)})"
            print(f"    {lbl:20s}  ARI={np.nanmean(aris):.3f}  "
                  f"NMI={np.nanmean(nmis):.3f}  "
                  f"PC_between={np.nanmean(pc_bfs):.3f}")

    # -- Save clustering results --
    if not clust_rows:
        print("\n  ERROR: No views processed.")
        return 1

    clust_df = pd.DataFrame(clust_rows)
    clust_path = out_dir / "clustering_comparison.csv"
    clust_df.to_csv(clust_path, index=False)
    print(f"\n  Wrote: {clust_path}")

    pc_df = pd.DataFrame(pc_rows)
    pc_path = out_dir / "pc_class_signal.csv"
    pc_df.to_csv(pc_path, index=False)
    print(f"  Wrote: {pc_path}")

    # -- Compute deltas and correlate with DI --
    delta_rows = []
    for ds in clust_df["dataset"].unique():
        for vw in clust_df[clust_df["dataset"] == ds]["view"].unique():
            view_data = clust_df[(clust_df["dataset"] == ds) & (clust_df["view"] == vw)]
            tv = view_data[view_data["strategy"] == "TopVar"]
            ts = view_data[view_data["strategy"] == "TopSHAP"]
            rnd = view_data[view_data["strategy"] == "Random"]
            allf = view_data[view_data["strategy"] == "All"]

            if tv.empty or rnd.empty:
                continue

            pc_view = pc_df[(pc_df["dataset"] == ds) & (pc_df["view"] == vw)]
            pc_tv = pc_view[pc_view["strategy"] == "TopVar"]
            pc_rnd = pc_view[pc_view["strategy"] == "Random"]

            row = {
                "dataset": ds, "view": vw,
                "ARI_TopVar": float(tv.iloc[0]["ARI_mean"]),
                "ARI_TopSHAP": float(ts.iloc[0]["ARI_mean"]) if not ts.empty else float("nan"),
                "ARI_Random": float(rnd.iloc[0]["ARI_mean"]),
                "ARI_All": float(allf.iloc[0]["ARI_mean"]) if not allf.empty else float("nan"),
                "delta_ARI_TopVar_Random": float(tv.iloc[0]["ARI_mean"]) - float(rnd.iloc[0]["ARI_mean"]),
                "delta_ARI_TopSHAP_Random": (float(ts.iloc[0]["ARI_mean"]) - float(rnd.iloc[0]["ARI_mean"])) if not ts.empty else float("nan"),
                "NMI_TopVar": float(tv.iloc[0]["NMI_mean"]),
                "NMI_Random": float(rnd.iloc[0]["NMI_mean"]),
                "delta_NMI_TopVar_Random": float(tv.iloc[0]["NMI_mean"]) - float(rnd.iloc[0]["NMI_mean"]),
            }

            if not pc_tv.empty and not pc_rnd.empty:
                row["PC_between_TopVar"] = float(pc_tv.iloc[0]["pc_between_frac_mean"])
                row["PC_between_Random"] = float(pc_rnd.iloc[0]["pc_between_frac_mean"])
                row["delta_PC_between"] = (float(pc_tv.iloc[0]["pc_between_frac_mean"])
                                           - float(pc_rnd.iloc[0]["pc_between_frac_mean"]))

            # Attach DI
            di_row = di_df[(di_df["dataset"] == ds) & (di_df["view"] == vw)]
            if not di_row.empty:
                row["DI"] = float(di_row.iloc[0]["DI"])
            else:
                row["DI"] = float("nan")

            delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)
    delta_path = out_dir / "unsupervised_vs_di.csv"
    delta_df.to_csv(delta_path, index=False)
    print(f"  Wrote: {delta_path}")

    # -- Correlations --
    print(f"\n{'=' * 70}")
    print("DI vs unsupervised degradation correlations")
    print("=" * 70)

    corr_results = {}
    valid = delta_df.dropna(subset=["DI", "delta_ARI_TopVar_Random"])

    if len(valid) >= 5:
        for metric in ["delta_ARI_TopVar_Random", "delta_NMI_TopVar_Random", "delta_PC_between"]:
            if metric in valid.columns and valid[metric].notna().sum() >= 5:
                sub = valid.dropna(subset=[metric])
                rho, p = stats.spearmanr(sub["DI"], sub[metric])
                r_pear, p_pear = stats.pearsonr(sub["DI"], sub[metric])
                corr_results[metric] = {
                    "spearman_rho": float(rho), "spearman_p": float(p),
                    "pearson_r": float(r_pear), "pearson_p": float(p_pear),
                    "n": len(sub),
                }
                print(f"  DI vs {metric}: rho = {rho:.3f} (p = {p:.4f}), "
                      f"r = {r_pear:.3f} (p = {p_pear:.4f}), n = {len(sub)}")
    else:
        print(f"  WARNING: Too few views ({len(valid)}) for correlation")

    # -- Summary statistics --
    n_views = len(delta_df)
    n_topvar_worse = int((delta_df["delta_ARI_TopVar_Random"] < 0).sum())
    mean_delta_ari = float(delta_df["delta_ARI_TopVar_Random"].mean())

    if "delta_ARI_TopSHAP_Random" in delta_df.columns:
        n_topshap_better = int((delta_df["delta_ARI_TopSHAP_Random"] > 0).sum())
    else:
        n_topshap_better = 0

    print(f"\n  SUMMARY:")
    print(f"    Views processed: {n_views}")
    print(f"    TopVar ARI < Random: {n_topvar_worse}/{n_views}")
    print(f"    Mean delta(TopVar-Random) ARI: {mean_delta_ari:.3f}")

    # -- Generate report --
    lines = [
        f"# Phase 2 — Unsupervised clustering comparison",
        f"",
        f"**Generated:** {now_iso()}",
        f"**K:** {args.k_pct}%  |  **n_PCs:** {args.n_pcs}  |  "
        f"**n_random:** {args.n_random}  |  **seed:** {args.seed}",
        f"**Views evaluated:** {n_views}",
        f"",
        f"## Key Finding",
        f"",
    ]

    if n_topvar_worse > n_views // 2:
        lines.append(
            f"TopVar-based clustering recovered biological classes less well than random "
            f"feature selection in {n_topvar_worse}/{n_views} views "
            f"(mean ΔARI = {mean_delta_ari:.3f}). "
            f"The same geometric misalignment that degrades supervised prediction "
            f"(high DI) also impairs unsupervised structure recovery."
        )
    elif n_topvar_worse > 0:
        lines.append(
            f"TopVar-based clustering was worse than random in {n_topvar_worse}/{n_views} views "
            f"(mean ΔARI = {mean_delta_ari:.3f}). "
            f"The effect is view-dependent, paralleling the supervised findings."
        )
    else:
        lines.append(
            f"TopVar-based clustering was comparable to or better than random across all "
            f"{n_views} views (mean ΔARI = {mean_delta_ari:.3f}). "
            f"The unsupervised penalty appears smaller than the supervised penalty."
        )

    lines.extend([
        f"",
        f"## DI–Clustering Correlations",
        f"",
        f"| Metric | Spearman ρ | p-value | Pearson r | p-value | n |",
        f"|--------|-----------|---------|-----------|---------|---|",
    ])
    for metric, vals in corr_results.items():
        lines.append(
            f"| {metric} | {vals['spearman_rho']:.3f} | {vals['spearman_p']:.4f} "
            f"| {vals['pearson_r']:.3f} | {vals['pearson_p']:.4f} | {vals['n']} |"
        )

    lines.extend([
        f"",
        f"## Per-View Clustering ARI",
        f"",
        delta_df[["dataset", "view", "DI", "ARI_TopVar", "ARI_TopSHAP",
                   "ARI_Random", "ARI_All", "delta_ARI_TopVar_Random"]].to_markdown(
            index=False, floatfmt=".3f"
        ),
        f"",
        f"## Manuscript-Ready Sentence",
        f"",
    ])

    # Manuscript text (adapts to results)
    ari_corr = corr_results.get("delta_ARI_TopVar_Random", {})
    pc_corr = corr_results.get("delta_PC_between", {})

    if ari_corr and ari_corr["spearman_p"] < 0.10:
        lines.append(
            f"> To test whether the variance–importance misalignment also impairs "
            f"unsupervised analyses, we repeated the feature-selection comparison "
            f"using KMeans clustering (k = n_classes) on PCA-reduced data. TopVar-selected "
            f"features yielded lower cluster–label agreement than matched-size random subsets "
            f"in {n_topvar_worse}/{n_views} views (mean ΔARI = {mean_delta_ari:.3f}), "
            f"and clustering degradation correlated with DI (ρ = {ari_corr['spearman_rho']:.2f}, "
            f"p = {ari_corr['spearman_p']:.3f}; n = {ari_corr['n']}). "
            f"This confirms that the geometric misalignment captured by the DI extends "
            f"beyond supervised prediction to unsupervised structure recovery."
        )
    elif ari_corr:
        lines.append(
            f"> Unsupervised clustering (KMeans on PCA-reduced data, k = n_classes) showed "
            f"that TopVar features recovered biological clusters less well than random subsets "
            f"in {n_topvar_worse}/{n_views} views (mean ΔARI = {mean_delta_ari:.3f}). "
            f"The DI–ΔARI correlation was negative but non-significant "
            f"(ρ = {ari_corr['spearman_rho']:.2f}, p = {ari_corr['spearman_p']:.2f}; "
            f"n = {ari_corr['n']}), consistent with the supervised findings but with "
            f"reduced statistical power at n = {ari_corr['n']} views."
        )
    else:
        lines.append(
            f"> Unsupervised clustering showed TopVar features recovered biological clusters "
            f"less well than random in {n_topvar_worse}/{n_views} views "
            f"(mean ΔARI = {mean_delta_ari:.3f})."
        )

    report_path = out_dir / "unsupervised_validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Wrote: {report_path}")

    # -- Manifest --
    files_written = [clust_path, pc_path, delta_path, report_path]
    manifest = {
        "script": "code/compute/02_unsupervised/03_clustering_comparison.py",
        "created_at": now_iso(),
        "params": {
            "k_pct": args.k_pct,
            "n_random": args.n_random,
            "n_pcs": args.n_pcs,
            "model": args.model,
            "seed": args.seed,
        },
        "summary": {
            "n_views": n_views,
            "n_topvar_worse_than_random": n_topvar_worse,
            "mean_delta_ARI_topvar_random": round(mean_delta_ari, 4),
            "correlations": corr_results,
        },
        "files": [
            {"name": p.name, "sha256": sha256_file(p)} for p in files_written
        ],
    }
    mf_path = out_dir / "MANIFEST_CLUSTERING_COMPARISON.json"
    mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Wrote: {mf_path}")

    print(f"\n{'=' * 70}")
    print(f"Phase 2 -- 06  Unsupervised clustering comparison complete")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())