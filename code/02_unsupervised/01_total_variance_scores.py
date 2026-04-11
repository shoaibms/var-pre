#!/usr/bin/env python3
"""
01_total_variance_scores.py (Phase 2 / .npz bundles)

Phase 2 / Unsupervised (Variance–Prediction Paradox)
----------------------------------------------------
Computes per-feature "variance-driving" scores for each dataset × view from the
normalized .npz bundles produced in Phase 1.

Inputs
------
Normalized bundles (Phase 1):
    outputs/bundles/{dataset}_bundle_normalized.npz

Bundle schema (summary):
    - X_{view}: (n_samples, n_features) float32
    - features_{view}: (n_features,) str
    - y: (n_samples,) int32
    - sample_ids: (n_samples,) str
    - info: JSON string (includes normalization.variance_approach)

Variance approach
-----------------
- marginal (default): score_j = Var(X_j)  (ddof=1 when n_samples>=2)
- latent_axis (mlomics): PCA-based contribution:
      score_j = sum_k (loading_{j,k}^2 * explained_variance_k)
  because per-feature variance is collapsed by pre-standardization.

Outputs
-------
For each dataset × view (and derived MGX variants if detected):
    outputs/02_unsupervised/variance_scores/
        - variance_scores__{dataset}__{view}.parquet (or .csv.gz)
        - top_sets__{dataset}__{view}.json
    and an overall:
        - variance_scores_summary.json

Usage
-----
    python code/compute/02_unsupervised/01_total_variance_scores.py --dataset all
    python code/compute/02_unsupervised/01_total_variance_scores.py --dataset mlomics --k-pcs 50
    python code/compute/02_unsupervised/01_total_variance_scores.py --dataset ibdmdb --include-sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: sklearn for PCA latent-axis scoring
try:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# Optional: Spearman diagnostics
try:
    from scipy.stats import spearmanr

    HAS_SPEARMANR = True
except Exception:
    HAS_SPEARMANR = False


# --------------------------------------------------------------------------------------
# Fallback view registry (used only if importing project view_registry.py fails)
# --------------------------------------------------------------------------------------
FALLBACK_VIEW_REGISTRY = {
    "mlomics": {
        "core_views": ["mRNA", "miRNA", "methylation", "CNV"],
        "sensitivity_views": [],
        "variance_approach": "latent_axis",
    },
    "ibdmdb": {
        "core_views": ["MGX", "MGX_func", "MPX", "MBX"],
        "sensitivity_views": ["MGX_CLR"],
        "variance_approach": "marginal",
    },
    "ccle": {
        "core_views": ["mRNA", "CNV", "proteomics"],
        "sensitivity_views": [],
        "variance_approach": "marginal",
    },
    "tcga_gbm": {
        "core_views": ["mRNA", "methylation", "CNV"],
        "sensitivity_views": ["methylation_Mval"],
        "variance_approach": "marginal",
    },
}

# Attempt to import project registry helpers; fallback if unavailable.
def _get_registry_helpers():
    try:
        # Typical repo location per Phase 1 docs:
        # code/compute/01_bundles/view_registry.py
        from view_registry import get_core_views, get_sensitivity_views, get_variance_approach  # type: ignore

        return get_core_views, get_sensitivity_views, get_variance_approach
    except Exception:
        def get_core_views(ds: str) -> List[str]:
            return list(FALLBACK_VIEW_REGISTRY[ds]["core_views"])

        def get_sensitivity_views(ds: str) -> List[str]:
            return list(FALLBACK_VIEW_REGISTRY[ds]["sensitivity_views"])

        def get_variance_approach(ds: str) -> str:
            return str(FALLBACK_VIEW_REGISTRY[ds]["variance_approach"])

        return get_core_views, get_sensitivity_views, get_variance_approach


GET_CORE_VIEWS, GET_SENS_VIEWS, GET_VAR_APPROACH = _get_registry_helpers()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parquet_engine_available() -> bool:
    try:
        import importlib.util

        return (
            importlib.util.find_spec("pyarrow") is not None
            or importlib.util.find_spec("fastparquet") is not None
        )
    except Exception:
        return False


def save_table(df: pd.DataFrame, out_base: Path) -> Tuple[Path, str]:
    """
    Save df to out_base with parquet if available, else csv.gz.
    Returns (path, format_str).
    """
    if _parquet_engine_available():
        out_path = out_base.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        return out_path, "parquet"
    out_path = out_base.with_suffix(".csv.gz")
    df.to_csv(out_path, index=False, compression="gzip")
    return out_path, "csv.gz"


def load_bundle_npz(dataset: str, bundle_dir: Path, normalized: bool = True) -> Dict[str, Any]:
    """
    Load bundle from outputs/bundles/{dataset}_bundle[_normalized].npz (allow_pickle=False).
    """
    suffix = "_normalized" if normalized else ""
    path = bundle_dir / f"{dataset}_bundle{suffix}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")

    npz = np.load(path, allow_pickle=False)

    info = {}
    if "info" in npz.files:
        info = json.loads(str(npz["info"]))
    else:
        warn(f"{dataset}: missing 'info' field in npz; proceeding with empty info")

    X_views: Dict[str, np.ndarray] = {}
    features: Dict[str, np.ndarray] = {}

    for key in npz.files:
        if key.startswith("X_"):
            v = key[2:]
            X_views[v] = npz[key]
        elif key.startswith("features_"):
            v = key[9:]
            features[v] = npz[key]

    return {
        "X_views": X_views,
        "feature_names": features,
        "y": npz["y"] if "y" in npz.files else None,
        "sample_ids": npz["sample_ids"] if "sample_ids" in npz.files else None,
        "info": info,
        "path": str(path),
    }


def _safe_feature_names(feature_arr: Optional[np.ndarray], n_features: int) -> np.ndarray:
    if feature_arr is None or len(feature_arr) != n_features:
        return np.asarray([f"f{j}" for j in range(n_features)], dtype=str)
    # ensure unicode str dtype
    return feature_arr.astype(str, copy=False)


def _cast64(X: np.ndarray) -> np.ndarray:
    # Bundles are float32; cast to float64 to avoid overflow/precision issues
    return np.asarray(X, dtype=np.float64)


def compute_marginal_variance_scores(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, marginal_var). For marginal approach, score == marginal_var.
    """
    X64 = _cast64(X)
    n = X64.shape[0]
    ddof = 1 if n >= 2 else 0
    # Use nanvar defensively
    v = np.nanvar(X64, axis=0, ddof=ddof)
    return v, v


def compute_latent_axis_scores_pca(
    X: np.ndarray, k_pcs: int = 50, seed: int = 1
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    PCA latent-axis contribution score:
        score_j = sum_k (loading_{j,k}^2 * explained_variance_k)

    Returns (scores, marginal_var, meta).
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for latent_axis scoring (PCA) but is not available.")

    X64 = _cast64(X)
    n_samples, n_features = X64.shape
    ddof = 1 if n_samples >= 2 else 0
    marginal_var = np.nanvar(X64, axis=0, ddof=ddof)

    # Mean-impute NaNs only for PCA fitting
    if np.isnan(X64).any():
        imp = SimpleImputer(strategy="mean")
        X_fit = imp.fit_transform(X64)
    else:
        X_fit = X64

    n_components = int(min(k_pcs, n_samples, n_features))
    if n_components < 1:
        raise ValueError(f"Cannot run PCA: n_components={n_components} for shape {X_fit.shape}")

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(X_fit)

    # components_: (n_components, n_features) => transpose to (n_features, n_components)
    load_sq = (pca.components_.T ** 2)
    ev = pca.explained_variance_.reshape(1, -1)
    scores = np.sum(load_sq * ev, axis=1)

    meta = {
        "method": "PCA",
        "n_components": n_components,
        "explained_variance_sum": float(np.sum(pca.explained_variance_)),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return scores, marginal_var, meta


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """
    Returns 1..n ranks (1 = highest), stable for ties by index order.
    """
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _percentile_from_rank(rank: np.ndarray) -> np.ndarray:
    n = len(rank)
    if n <= 1:
        return np.ones(n, dtype=float)
    # 1.0 = top-ranked
    return 1.0 - (rank.astype(float) - 1.0) / (n - 1.0)


def _spearman(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if not HAS_SPEARMANR:
        return {"available": False}
    try:
        rho, p = spearmanr(a, b)
        return {"available": True, "rho": float(rho) if rho == rho else None, "p": float(p) if p == p else None}
    except Exception:
        return {"available": True, "rho": None, "p": None}


def _detect_metaphlan_multilevel(feature_names: np.ndarray) -> bool:
    # Heuristic: MetaPhlAn style taxonomy prefixes
    tokens = ("k__", "p__", "c__", "o__", "f__", "g__", "s__", "t__")
    return any(t in str(f) for f in feature_names[: min(len(feature_names), 500)] for t in tokens)


def _species_mask(feature_names: np.ndarray) -> np.ndarray:
    # species-only: contains 's__' and not 't__'
    return np.array([("s__" in str(f) and "t__" not in str(f)) for f in feature_names], dtype=bool)


@dataclass
class ViewVarianceRecord:
    dataset: str
    view: str
    bundle_path: str
    n_samples: int
    n_features: int
    variance_approach: str
    score_kind: str
    pca_meta: Optional[Dict[str, Any]]
    spearman_score_vs_marginal: Dict[str, Any]
    wrote: str
    wrote_format: str
    top_sets_json: str
    timestamp: str


def compute_and_write_view(
    dataset: str,
    view_out: str,
    X: np.ndarray,
    feature_names: np.ndarray,
    variance_approach: str,
    out_dir: Path,
    bundle_path: str,
    k_pcs: int,
    seed: int,
) -> ViewVarianceRecord:
    n_samples, n_features = X.shape

    pca_meta: Optional[Dict[str, Any]] = None
    if variance_approach == "latent_axis":
        scores, marginal_var, pca_meta = compute_latent_axis_scores_pca(X, k_pcs=k_pcs, seed=seed)
        score_kind = "latent_axis_pca"
        variance_collapsed = True
    else:
        scores, marginal_var = compute_marginal_variance_scores(X)
        score_kind = "marginal_variance"
        variance_collapsed = False

    # ranks / percentiles
    rank = _rank_desc(scores)
    marginal_rank = _rank_desc(marginal_var)
    percentile = _percentile_from_rank(rank)

    df = pd.DataFrame(
        {
            "feature": feature_names.astype(str),
            "score": scores.astype(float),
            "rank": rank.astype(int),
            "percentile": percentile.astype(float),
            "marginal_variance": marginal_var.astype(float),
            "marginal_rank": marginal_rank.astype(int),
            "variance_collapsed": np.full(n_features, bool(variance_collapsed)),
        }
    )

    # Save
    out_base = out_dir / f"variance_scores__{dataset}__{view_out}"
    wrote_path, wrote_fmt = save_table(df, out_base)

    # Top sets JSON
    top_ks = [10, 25, 50, 100, 200, 500, 1000]
    top_ks = [k for k in top_ks if k <= n_features]
    df_sorted = df.sort_values("rank", ascending=True)
    top_sets = {
        "dataset": dataset,
        "view": view_out,
        "variance_approach": variance_approach,
        "score_kind": score_kind,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "top_features": {str(k): df_sorted["feature"].head(k).tolist() for k in top_ks},
    }
    top_json_path = out_dir / f"top_sets__{dataset}__{view_out}.json"
    with open(top_json_path, "w", encoding="utf-8") as f:
        json.dump(top_sets, f, indent=2)

    # Diagnostics
    sp = _spearman(scores, marginal_var)

    return ViewVarianceRecord(
        dataset=dataset,
        view=view_out,
        bundle_path=bundle_path,
        n_samples=int(n_samples),
        n_features=int(n_features),
        variance_approach=str(variance_approach),
        score_kind=score_kind,
        pca_meta=pca_meta,
        spearman_score_vs_marginal=sp,
        wrote=str(wrote_path),
        wrote_format=wrote_fmt,
        top_sets_json=str(top_json_path),
        timestamp=_now_iso(),
    )


# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute total-variance feature scores per dataset × view (Phase 2)")
    p.add_argument(
        "--dataset",
        default="all",
        choices=["all", "mlomics", "ibdmdb", "ccle", "tcga_gbm"],
        help="Which dataset to process",
    )
    p.add_argument("--bundle-dir", default="outputs/bundles", help="Directory containing bundle .npz files")
    p.add_argument("--out-dir", default="outputs/02_unsupervised/variance_scores", help="Output directory")
    p.add_argument(
        "--raw",
        action="store_true",
        help="Load raw (non-normalized) bundles instead of *_normalized.npz",
    )
    p.add_argument(
        "--include-sensitivity",
        action="store_true",
        help="Include sensitivity views (e.g., MGX_CLR, methylation_Mval) in addition to core views",
    )
    p.add_argument(
        "--k-pcs",
        type=int,
        default=50,
        help="Number of PCs for latent_axis PCA scoring (mlomics)",
    )
    p.add_argument("--seed", type=int, default=1, help="Random seed for PCA randomized SVD")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    datasets = ["mlomics", "ibdmdb", "ccle", "tcga_gbm"] if args.dataset == "all" else [args.dataset]
    normalized = not args.raw

    log("=" * 80)
    log("PHASE 2: TOTAL VARIANCE SCORES (.npz)")
    log(f"timestamp: {_now_iso()}")
    log(f"bundle_dir: {bundle_dir}")
    log(f"out_dir:    {out_dir}")
    log(f"datasets:   {datasets}")
    log(f"normalized: {normalized}")
    log(f"include_sensitivity_views: {bool(args.include_sensitivity)}")
    log(f"k_pcs (latent_axis): {args.k_pcs}")
    log("=" * 80)

    records: List[ViewVarianceRecord] = []

    for ds in datasets:
        log("-" * 80)
        log(f"Dataset: {ds}")

        bundle = load_bundle_npz(ds, bundle_dir=bundle_dir, normalized=normalized)
        X_views: Dict[str, np.ndarray] = bundle["X_views"]
        feat_map: Dict[str, np.ndarray] = bundle["feature_names"]
        info: Dict[str, Any] = bundle.get("info", {})
        bundle_path = str(bundle.get("path", ""))

        # variance approach: prefer bundle info if present, else registry
        variance_approach = (
            info.get("normalization", {}).get("variance_approach", None)
            or GET_VAR_APPROACH(ds)
        )

        core = GET_CORE_VIEWS(ds)
        sens = GET_SENS_VIEWS(ds)
        wanted = list(core) + (list(sens) if args.include_sensitivity else [])

        # process only views present in bundle
        present = set(X_views.keys())
        wanted_present = [v for v in wanted if v in present]
        missing = [v for v in wanted if v not in present]
        if missing:
            warn(f"{ds}: requested views missing in bundle: {missing}")
        if not wanted_present:
            warn(f"{ds}: no requested views found; skipping dataset")
            continue

        log(f"variance_approach: {variance_approach}")
        log(f"views_to_process:  {wanted_present}")

        for view_name in wanted_present:
            X = X_views[view_name]
            if X.ndim != 2:
                warn(f"{ds}:{view_name}: expected 2D matrix, got shape={X.shape}; skipping")
                continue

            n_samples, n_features = X.shape
            feats = _safe_feature_names(feat_map.get(view_name, None), n_features)

            # IBDMDB MGX: only split if both multi-level taxonomy is present AND species is a strict subset
            if ds == "ibdmdb" and view_name in {"MGX", "MGX_CLR"}:
                mask = _species_mask(feats)

                has_multilevel = _detect_metaphlan_multilevel(feats) and (0 < int(mask.sum()) < n_features)

                if has_multilevel:
                    # all-level (original view name with _all suffix)
                    rec_all = compute_and_write_view(
                        dataset=ds,
                        view_out=f"{view_name}_all",
                        X=X,
                        feature_names=feats,
                        variance_approach=variance_approach,
                        out_dir=out_dir,
                        bundle_path=bundle_path,
                        k_pcs=args.k_pcs,
                        seed=args.seed,
                    )
                    records.append(rec_all)

                    # species-only
                    X_sp = X[:, mask]
                    feats_sp = feats[mask]
                    rec_sp = compute_and_write_view(
                        dataset=ds,
                        view_out=f"{view_name}_species",
                        X=X_sp,
                        feature_names=feats_sp,
                        variance_approach=variance_approach,
                        out_dir=out_dir,
                        bundle_path=bundle_path,
                        k_pcs=args.k_pcs,
                        seed=args.seed,
                    )
                    records.append(rec_sp)

                    continue  # prevent default write

                # else: fall through to default write with view_out=view_name (MGX or MGX_CLR)

            # default: write as-is
            rec = compute_and_write_view(
                dataset=ds,
                view_out=view_name,
                X=X,
                feature_names=feats,
                variance_approach=variance_approach,
                out_dir=out_dir,
                bundle_path=bundle_path,
                k_pcs=args.k_pcs,
                seed=args.seed,
            )
            records.append(rec)

    # summary
    summary_path = out_dir / "variance_scores_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": _now_iso(),
                "normalized": bool(normalized),
                "records": [asdict(r) for r in records],
            },
            f,
            indent=2,
        )

    log("-" * 80)
    log(f"Wrote summary: {summary_path}")
    log(f"Views processed: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
