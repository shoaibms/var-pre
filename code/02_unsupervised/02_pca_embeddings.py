#!/usr/bin/env python3
"""
02_pca_embeddings.py (PHASE 2 / .npz bundles)

Consumes normalized bundles:
    outputs/bundles/{dataset}_bundle_normalized.npz

Bundle schema:
    X_{view}, features_{view}, y, sample_ids, info (JSON)

Produces per dataset×view PCA embeddings + top loadings:
    outputs/02_unsupervised/pca/
        pca__{dataset}__{view}.npz
        pca_embeddings_summary.json

Notes:
- Uses sklearn PCA (randomized) on dense matrices.
- Mean-imputes NaNs (per feature) for PCA only.
- Stores FULL components only if n_features <= --full-loadings-max-features (AUTO),
  otherwise stores only top-|loading| features per PC.

Run:
    python code/compute/02_unsupervised/02_pca_embeddings.py --dataset all
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

try:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# ----------------------------
# Fallback view registry
# ----------------------------
FALLBACK_VIEW_REGISTRY = {
    "mlomics": {
        "core_views": ["mRNA", "miRNA", "methylation", "CNV"],
        "sensitivity_views": [],
    },
    "ibdmdb": {
        "core_views": ["MGX", "MGX_func", "MPX", "MBX"],
        "sensitivity_views": ["MGX_CLR"],
    },
    "ccle": {
        "core_views": ["mRNA", "CNV", "proteomics"],
        "sensitivity_views": [],
    },
    "tcga_gbm": {
        "core_views": ["mRNA", "methylation", "CNV"],
        "sensitivity_views": ["methylation_Mval"],
    },
}


def _get_registry_helpers():
    try:
        from view_registry import get_core_views, get_sensitivity_views  # type: ignore

        return get_core_views, get_sensitivity_views
    except Exception:

        def get_core_views(ds: str) -> List[str]:
            return list(FALLBACK_VIEW_REGISTRY[ds]["core_views"])

        def get_sensitivity_views(ds: str) -> List[str]:
            return list(FALLBACK_VIEW_REGISTRY[ds]["sensitivity_views"])

        return get_core_views, get_sensitivity_views


GET_CORE_VIEWS, GET_SENS_VIEWS = _get_registry_helpers()


# ----------------------------
# Logging / utils
# ----------------------------
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_feature_names(feature_arr: Optional[np.ndarray], n_features: int) -> np.ndarray:
    if feature_arr is None or len(feature_arr) != n_features:
        return np.asarray([f"f{j}" for j in range(n_features)], dtype=str)
    return feature_arr.astype(str, copy=False)


def _detect_metaphlan_multilevel(feature_names: np.ndarray) -> bool:
    tokens = ("k__", "p__", "c__", "o__", "f__", "g__", "s__", "t__")
    return any(t in str(f) for f in feature_names[: min(len(feature_names), 500)] for t in tokens)


def _species_mask(feature_names: np.ndarray) -> np.ndarray:
    return np.array([("s__" in str(f) and "t__" not in str(f)) for f in feature_names], dtype=bool)


# ----------------------------
# IO: bundle loader
# ----------------------------
def load_bundle_npz(dataset: str, bundle_dir: Path, normalized: bool = True) -> Dict[str, Any]:
    suffix = "_normalized" if normalized else ""
    path = bundle_dir / f"{dataset}_bundle{suffix}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")

    npz = np.load(path, allow_pickle=False)

    info = {}
    if "info" in npz.files:
        info = json.loads(str(npz["info"]))

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


# ----------------------------
# PCA core
# ----------------------------
def fit_pca_dense(
    X: np.ndarray,
    n_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      scores: (n_samples, k)
      components: (k, n_features)
      explained_variance: (k,)
      explained_variance_ratio: (k,)
      singular_values: (k,)
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for PCA embeddings but is not available.")

    X_in = np.asarray(X)

    # Mean-impute NaNs for PCA only
    if np.isnan(X_in).any():
        imp = SimpleImputer(strategy="mean")
        X_fit = imp.fit_transform(X_in)
    else:
        X_fit = X_in

    n_samples, n_features = X_fit.shape
    k = int(min(n_components, n_samples - 1 if n_samples > 1 else 1, n_features))
    if k < 1:
        raise ValueError(f"Invalid PCA: k={k} for shape={X_fit.shape}")

    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(X_fit)

    components = pca.components_
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_

    return (
        scores.astype(np.float32, copy=False),
        components.astype(np.float32, copy=False),
        explained_variance.astype(np.float32, copy=False),
        explained_variance_ratio.astype(np.float32, copy=False),
        singular_values.astype(np.float32, copy=False),
    )


def top_loadings(components: np.ndarray, feature_names: np.ndarray, top_m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each PC, return indices and signed loadings for top |loading|.
    Returns:
      idx: (k, m) int32
      loading: (k, m) float32 (signed)
      names: (k, m) str
    """
    k, p = components.shape
    m = int(min(top_m, p))
    idx = np.zeros((k, m), dtype=np.int32)
    val = np.zeros((k, m), dtype=np.float32)
    names = np.empty((k, m), dtype=object)

    for i in range(k):
        comp = components[i]
        order = np.argsort(np.abs(comp))[::-1][:m]
        idx[i, :] = order.astype(np.int32)
        val[i, :] = comp[order].astype(np.float32)
        names[i, :] = feature_names[order].astype(str)

    return idx, val, names.astype(str)


@dataclass
class PCARecord:
    dataset: str
    view: str
    bundle_path: str
    out_npz: str
    n_samples: int
    n_features: int
    k: int
    stored_full_components: bool
    timestamp: str


def save_pca_npz(
    out_path: Path,
    sample_ids: Optional[np.ndarray],
    scores: np.ndarray,
    explained_variance: np.ndarray,
    explained_variance_ratio: np.ndarray,
    singular_values: np.ndarray,
    top_idx: np.ndarray,
    top_val: np.ndarray,
    top_names: np.ndarray,
    store_full_components: bool,
    components: Optional[np.ndarray],
    feature_names: Optional[np.ndarray],
) -> None:
    payload: Dict[str, Any] = {
        "sample_ids": sample_ids.astype(str) if sample_ids is not None else np.asarray([], dtype=str),
        "scores": scores,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "singular_values": singular_values,
        "top_idx": top_idx,
        "top_val": top_val,
        "top_names": top_names.astype(str),
        "stored_full_components": np.asarray([bool(store_full_components)], dtype=bool),
    }

    if store_full_components and components is not None and feature_names is not None:
        payload["components"] = components
        payload["feature_names"] = feature_names.astype(str)

    np.savez_compressed(out_path, **payload)


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 PCA embeddings from .npz bundles")
    p.add_argument("--dataset", default="all", choices=["all", "mlomics", "ibdmdb", "ccle", "tcga_gbm"])
    p.add_argument("--bundle-dir", default="outputs/bundles")
    p.add_argument("--out-dir", default="outputs/02_unsupervised/pca")
    p.add_argument("--raw", action="store_true", help="Use raw (non-normalized) bundle npz")
    p.add_argument("--include-sensitivity", action="store_true", help="Include sensitivity views in addition to core")
    p.add_argument("--n-components", type=int, default=50)
    p.add_argument("--top-loadings", type=int, default=200)
    p.add_argument("--full-loadings", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--full-loadings-max-features", type=int, default=50000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--ibdmdb-species-split",
        action="store_true",
        help="If MGX contains multilevel taxonomy, also output MGX_all and MGX_species variants",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    datasets = ["mlomics", "ibdmdb", "ccle", "tcga_gbm"] if args.dataset == "all" else [args.dataset]
    normalized = not args.raw

    log("=" * 80)
    log("02_UNSUPERVISED: PCA EMBEDDINGS (PHASE 2 .npz)")
    log("=" * 80)
    log(f"Started: {_now_iso()}")
    log(f"bundle_dir: {bundle_dir}")
    log(f"out_dir:    {out_dir}")
    log(f"datasets:   {datasets}")
    log(f"normalized: {normalized}")
    log(f"include_sensitivity: {bool(args.include_sensitivity)}")
    log(f"n_components: {args.n_components}")
    log(f"top_loadings: {args.top_loadings}")
    log(f"full_loadings: {args.full_loadings} (max_features={args.full_loadings_max_features})")
    log(f"ibdmdb_species_split: {bool(args.ibdmdb_species_split)}")
    log("=" * 80)

    records: List[PCARecord] = []

    for ds in datasets:
        log("-" * 80)
        log(f"Dataset: {ds}")

        bundle = load_bundle_npz(ds, bundle_dir=bundle_dir, normalized=normalized)
        X_views: Dict[str, np.ndarray] = bundle["X_views"]
        feat_map: Dict[str, np.ndarray] = bundle["feature_names"]
        sample_ids = bundle.get("sample_ids", None)
        bundle_path = str(bundle.get("path", ""))

        core = GET_CORE_VIEWS(ds)
        sens = GET_SENS_VIEWS(ds)
        wanted = list(core) + (list(sens) if args.include_sensitivity else [])
        present = set(X_views.keys())
        views = [v for v in wanted if v in present]
        missing = [v for v in wanted if v not in present]
        if missing:
            warn(f"{ds}: requested views missing in bundle: {missing}")
        if not views:
            warn(f"{ds}: no requested views found; skipping")
            continue

        log(f"Views: {views}")

        for view in views:
            X = X_views[view]
            if X.ndim != 2:
                warn(f"{ds}:{view}: expected 2D matrix, got shape={X.shape}; skipping")
                continue

            n_samples, n_features = X.shape
            feature_names = _safe_feature_names(feat_map.get(view, None), n_features)

            # Optional IBDMDB split (only if true multilevel)
            if ds == "ibdmdb" and args.ibdmdb_species_split and view in {"MGX", "MGX_CLR"}:
                mask = _species_mask(feature_names)
                has_multilevel = _detect_metaphlan_multilevel(feature_names) and (0 < int(mask.sum()) < n_features)
                if has_multilevel:
                    # ALL
                    _process_one(
                        ds=ds,
                        view_out=f"{view}_all",
                        X=X,
                        feature_names=feature_names,
                        sample_ids=sample_ids,
                        bundle_path=bundle_path,
                        out_dir=out_dir,
                        args=args,
                        records=records,
                    )
                    # SPECIES
                    _process_one(
                        ds=ds,
                        view_out=f"{view}_species",
                        X=X[:, mask],
                        feature_names=feature_names[mask],
                        sample_ids=sample_ids,
                        bundle_path=bundle_path,
                        out_dir=out_dir,
                        args=args,
                        records=records,
                    )
                    continue  # skip default

            # Default: view as-is
            _process_one(
                ds=ds,
                view_out=view,
                X=X,
                feature_names=feature_names,
                sample_ids=sample_ids,
                bundle_path=bundle_path,
                out_dir=out_dir,
                args=args,
                records=records,
            )

    summary_path = out_dir / "pca_embeddings_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": _now_iso(), "records": [asdict(r) for r in records]}, f, indent=2)

    log("-" * 80)
    log(f"Wrote summary: {summary_path}")
    log(f"Total views processed: {len(records)}")
    return 0


def _process_one(
    ds: str,
    view_out: str,
    X: np.ndarray,
    feature_names: np.ndarray,
    sample_ids: Optional[np.ndarray],
    bundle_path: str,
    out_dir: Path,
    args: argparse.Namespace,
    records: List[PCARecord],
) -> None:
    n_samples, n_features = X.shape

    scores, components, ev, evr, sv = fit_pca_dense(
        X=X,
        n_components=args.n_components,
        seed=args.seed,
    )
    k = int(scores.shape[1])

    top_idx, top_val, top_names = top_loadings(components, feature_names, top_m=args.top_loadings)

    # Decide full loadings storage
    if args.full_loadings == "on":
        store_full = True
    elif args.full_loadings == "off":
        store_full = False
    else:
        store_full = bool(n_features <= int(args.full_loadings_max_features))

    out_path = out_dir / f"pca__{ds}__{view_out}.npz"
    save_pca_npz(
        out_path=out_path,
        sample_ids=sample_ids,
        scores=scores,
        explained_variance=ev,
        explained_variance_ratio=evr,
        singular_values=sv,
        top_idx=top_idx,
        top_val=top_val,
        top_names=top_names,
        store_full_components=store_full,
        components=components if store_full else None,
        feature_names=feature_names if store_full else None,
    )

    records.append(
        PCARecord(
            dataset=ds,
            view=view_out,
            bundle_path=bundle_path,
            out_npz=str(out_path),
            n_samples=int(n_samples),
            n_features=int(n_features),
            k=int(k),
            stored_full_components=bool(store_full),
            timestamp=_now_iso(),
        )
    )

    log(f"  wrote: {out_path}  (n={n_samples}, p={n_features}, k={k}, full_components={store_full})")


if __name__ == "__main__":
    raise SystemExit(main())
