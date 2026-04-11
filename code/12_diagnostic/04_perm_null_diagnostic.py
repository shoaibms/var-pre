#!/usr/bin/env python3
"""
PHASE 12 — Optional A: Diagnostic permutation null (reviewer-bulletproof)

Goal
----
Create an empirical *label-permutation* null distribution for the supervision-light
diagnostics, without training any ML model:

  - VSA(K):  Mann–Whitney AUROC(eta^2; TopVar vs Rest) - 0.5   (null ~ 0)
  - etaES(K): mean(eta^2 in TopVar(K%)) / mean(eta^2 overall)  (null ~ 1)
  - PCLA:    PCA-weighted eta^2 of PCs                         (null low)

We permute labels globally (optionally within groups) and recompute metrics fold-by-fold
on TRAIN indices only (no leakage). Then we aggregate to view-level:
  - observed_mean (across folds)
  - null_mean, null_sd (across permutations of the view-level mean)
  - z-score and empirical p-value

Outputs
-------
<outputs_dir>/<out_dirname>/
  - per_view/permnull__<dataset>__<view>.csv.gz     (fold × perm long)
  - permnull_long.csv.gz                           (all views)
  - permnull_summary.csv                           (view-level obs vs null, z, p, q)
  - PERMNULL_REPORT.md                             (paper-ready summary)

Usage (PowerShell)
------------------
python .\\code\\compute\\12_diagnostic\\04_perm_null_diagnostic.py `
  --outputs-dir "<path-to-outputs>" `
  --out-dirname "12_diagnostic_permnull" `
  --all-views `
  --k 10 `
  --n-perm 100 `
  --max-repeats 3 `
  --pca-components 30 `
  --within-groups `
  --n-jobs 12 `
  --seed 123

Speed note (optional)
---------------------
If your p is extremely large and n_perm=100 is heavy, set:
  --max-features 20000

This approximates VSA/etaES by evaluating eta^2 on:
  [TopVar(K%) features] ∪ [random sample of remaining features]
and is usually sufficient for a null/p-value argument.
Default (0) uses all features.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from tqdm.contrib.concurrent import thread_map
except Exception:  # pragma: no cover
    thread_map = None

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None


# allow imports from code/compute/_shared
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.vad_metrics import eta2_features, eta2_1d  # type: ignore


_EPS = 1e-12


# -----------------------------
# Defaults / views
# -----------------------------
HERO_VIEWS = [
    ("mlomics", "methylation"),
    ("ibdmdb", "MGX"),
    ("ccle", "mRNA"),
]

ALL_VIEWS = [
    ("mlomics", "mRNA"), ("mlomics", "miRNA"), ("mlomics", "methylation"), ("mlomics", "CNV"),
    ("ibdmdb", "MGX"), ("ibdmdb", "MGX_func"), ("ibdmdb", "MPX"), ("ibdmdb", "MBX"),
    ("ccle", "mRNA"), ("ccle", "CNV"), ("ccle", "proteomics"),
    ("tcga_gbm", "mRNA"), ("tcga_gbm", "methylation"), ("tcga_gbm", "CNV"),
]


# -----------------------------
# Dataclasses
# -----------------------------
@dataclass(frozen=True)
class SplitData:
    dataset: str
    y: np.ndarray
    sample_ids: np.ndarray
    groups: np.ndarray
    fold_ids: np.ndarray  # (n_repeats, n_samples)
    info: Dict[str, object]


# -----------------------------
# IO helpers
# -----------------------------
def load_npz_pickle_free(path: Path):
    return np.load(path, allow_pickle=False)


def load_splits_json(path: Path) -> SplitData:
    d = json.loads(path.read_text(encoding="utf-8"))
    info = d.get("info", {})
    dataset = str(info.get("dataset", path.stem.replace("splits__", "")))

    y = np.asarray(d["y"])
    sample_ids = np.asarray(d["sample_ids"]).astype(str)
    groups = np.asarray(d.get("groups", ["NA"] * len(sample_ids))).astype(str)

    fold_ids = np.asarray(d["fold_ids"])
    if fold_ids.ndim == 1:
        fold_ids = fold_ids[None, :]
    fold_ids = fold_ids.astype(np.int16)

    if fold_ids.shape[1] != len(sample_ids):
        raise ValueError(f"fold_ids shape {fold_ids.shape} != n_samples {len(sample_ids)} for {path}")

    return SplitData(dataset=dataset, y=y, sample_ids=sample_ids, groups=groups, fold_ids=fold_ids, info=info)


def resolve_bundle_path(outputs_dir: Path, splits: SplitData) -> Path:
    bp = splits.info.get("bundle_path", None)
    if bp is not None:
        bp = Path(str(bp))
        if not bp.is_absolute():
            cand = (outputs_dir / bp).resolve()
            if cand.exists():
                return cand
        if bp.exists():
            return bp

    ds = splits.dataset
    cands: List[Path] = []
    for root in [outputs_dir / "01_bundles", outputs_dir, outputs_dir / "archive_pre_migration_20260127"]:
        if root.exists():
            cands += list(root.rglob(f"*{ds}*bundle*normalized*.npz"))
    if not cands:
        raise FileNotFoundError(f"Could not resolve bundle_path for dataset={ds}. info.bundle_path missing/invalid.")
    cands = sorted(cands, key=lambda p: (len(str(p)), str(p)))
    return cands[0]


def load_bundle_view(bundle_path: Path, view: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    z = load_npz_pickle_free(bundle_path)
    info = json.loads(str(z["info"])) if "info" in z.files else {}

    y = z["y"] if "y" in z.files else None
    sample_ids = z["sample_ids"].astype(str) if "sample_ids" in z.files else None

    x_key = f"X_{view}"
    if x_key not in z.files:
        alt = [k for k in z.files if k.lower() == x_key.lower()]
        if alt:
            x_key = alt[0]
        else:
            raise KeyError(
                f"Bundle missing view '{view}' (expected '{x_key}'). "
                f"Available X_ keys: {[k for k in z.files if k.startswith('X_')]}"
            )

    X = z[x_key].astype(np.float32)

    f_key1 = f"features_{view}"
    f_key2 = f"feature_names_{view}"
    if f_key1 in z.files:
        feats = z[f_key1].astype(str)
    elif f_key2 in z.files:
        feats = z[f_key2].astype(str)
    else:
        feats = np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)

    if y is None or sample_ids is None:
        raise KeyError(f"Bundle missing required keys 'y' and/or 'sample_ids': {bundle_path}")

    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)

    return np.asarray(y), sample_ids, X, feats, info


def align_X_to_splits_order(X: np.ndarray, bundle_ids: np.ndarray, splits_ids: np.ndarray) -> np.ndarray:
    if np.array_equal(bundle_ids, splits_ids):
        return X
    idx = {sid: i for i, sid in enumerate(bundle_ids)}
    order: List[int] = []
    missing = 0
    for sid in splits_ids:
        if sid not in idx:
            missing += 1
            order.append(-1)
        else:
            order.append(idx[sid])
    if missing > 0:
        raise ValueError(f"Cannot align X to splits: {missing} split sample_ids not found in bundle sample_ids")
    return X[np.asarray(order, dtype=int), :]


# -----------------------------
# Utility
# -----------------------------
def stable_u64(*parts: object) -> int:
    s = "|".join([str(p) for p in parts]).encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def parse_views_arg(s: str) -> List[Tuple[str, str]]:
    out = []
    for item in [x.strip() for x in str(s).split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"Bad --views item '{item}'. Expected dataset:view")
        ds, vw = item.split(":", 1)
        out.append((ds.strip(), vw.strip()))
    return out


def infer_task_type(y: np.ndarray) -> str:
    if np.issubdtype(y.dtype, np.floating):
        u = np.unique(y[~np.isnan(y)])
        if len(u) <= 20 and np.all(np.isclose(u, np.round(u))):
            return "classification"
        return "regression"
    return "classification"


def mean_impute(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32, copy=False)
    out = X.copy()
    m = ~np.isfinite(out)
    if m.any():
        out[m] = np.take(col_means, np.where(m)[1])
    return out


def mann_whitney_effect(x1: np.ndarray, x0: np.ndarray) -> float:
    """Return AUROC(x1 vs x0) - 0.5; robust to missing/empty."""
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    x1 = x1[np.isfinite(x1)]
    x0 = x0[np.isfinite(x0)]
    if x1.size == 0 or x0.size == 0:
        return float("nan")
    if _stats is not None:
        u, _p = _stats.mannwhitneyu(x1, x0, alternative="two-sided")
        auroc = float(u) / float(x1.size * x0.size)
        return float(auroc - 0.5)

    # fallback ranks
    x = np.concatenate([x1, x0])
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    r1 = ranks[: x1.size].sum()
    u = r1 - x1.size * (x1.size + 1) / 2.0
    auroc = float(u) / float(x1.size * x0.size)
    return float(auroc - 0.5)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    if _stats is not None:
        r, _p = _stats.spearmanr(a[m], b[m])
        return float(r) if np.isfinite(r) else float("nan")
    # fallback: pandas
    return float(pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman"))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR q-values."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    m = np.isfinite(p)
    if m.sum() == 0:
        return q
    pv = p[m]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q_raw = ranked * n / (np.arange(1, n + 1))
    # enforce monotone
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_mono = np.clip(q_mono, 0.0, 1.0)
    out = np.empty_like(pv)
    out[order] = q_mono
    q[m] = out
    return q


def permute_labels(y_codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(y_codes.size)
    rng.shuffle(idx)
    return y_codes[idx]


def permute_labels_within_groups(y_codes: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_perm = y_codes.copy()
    groups = np.asarray(groups).astype(str)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if idx.size <= 1:
            continue
        rng.shuffle(idx)  # shuffle indices in-place
        # assign permuted labels within group (shuffle label positions)
        y_perm[np.where(groups == g)[0]] = y_codes[idx]
    return y_perm


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--out-dirname", default="12_diagnostic_permnull")
    ap.add_argument("--views", default="", help='Comma list: "dataset:view,dataset:view"')
    ap.add_argument("--hero-views", action="store_true")
    ap.add_argument("--all-views", action="store_true")
    ap.add_argument("--k", type=int, default=10, help="K-percent for TopVar in VSA/etaES (usually 10)")
    ap.add_argument("--primary-k", type=int, default=None, help="Alias for --k (kept for consistency)")
    ap.add_argument("--n-perm", type=int, default=100, help="Number of label permutations")
    ap.add_argument("--max-repeats", type=int, default=3)
    ap.add_argument("--pca-components", type=int, default=30)
    ap.add_argument("--within-groups", action="store_true", help="Permute labels within splits.groups")
    ap.add_argument("--max-features", type=int, default=0, help="0=all features; else evaluate eta^2 on TopVar(K) + sampled rest (speed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--backend", default="threading", choices=["threading"])
    ap.add_argument("--force", action="store_true", help="Overwrite existing per_view outputs")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


# -----------------------------
# Core: fold computation
# -----------------------------
def compute_pca_cache(Xtr: np.ndarray, n_components: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA on Xtr (mean-imputed). Returns (Z, lambdas).
    """
    if PCA is None:
        return np.empty((Xtr.shape[0], 0), dtype=np.float32), np.empty((0,), dtype=float)

    Ximp = mean_impute(Xtr)
    n, p = Ximp.shape
    m = int(min(max(1, n_components), max(1, n - 1), max(1, p)))
    if m < 2:
        return np.empty((n, 0), dtype=np.float32), np.empty((0,), dtype=float)

    pca = PCA(n_components=m, svd_solver="randomized", random_state=int(random_state))
    Z = pca.fit_transform(Ximp).astype(np.float32, copy=False)
    lambdas = np.asarray(pca.explained_variance_, dtype=float)
    lambdas = np.where(np.isfinite(lambdas), lambdas, 0.0)
    return Z, lambdas


def pcla_from_cache(Z: np.ndarray, lambdas: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    PCLA = sum( w_k * eta^2(PC_k, y) ), w_k = lambda_k / sum(lambda)
    SAS  = spearman(lambda, eta2_pc)
    """
    if Z.size == 0 or lambdas.size < 2:
        return float("nan"), float("nan")
    eta2_pcs = np.array([eta2_1d(Z[:, k], y) for k in range(Z.shape[1])], dtype=float)
    w = lambdas / (float(lambdas.sum()) + _EPS)
    pcla = float(np.sum(w * eta2_pcs))
    sas = float(spearman(lambdas, eta2_pcs))
    return pcla, sas


def select_feature_subset_by_variance(
    Xtr: np.ndarray,
    k_pct: int,
    max_features: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """
    Always return indices ordered as:
      [TopVar(K% of full p)] + [rest...]
    so that the first top_n columns are truly the top-variance block.

    If max_features==0: evaluate ALL features (TopVar + all remaining).
    If max_features>0: evaluate TopVar + a random sample of the remaining.
    """
    p = int(Xtr.shape[1])
    top_n = max(1, int(p * (float(k_pct) / 100.0)))

    # variance ranking (nan-robust)
    v_total = np.nanvar(Xtr, axis=0, ddof=0)
    v = np.where(np.isfinite(v_total), v_total, -np.inf)

    # stable sort (deterministic tie-breaking)
    var_rank = np.argsort(-v, kind="mergesort").astype(int, copy=False)

    top_idx = var_rank[:top_n]
    rest_pool = var_rank[top_n:]

    # evaluate all features
    if max_features <= 0 or max_features >= p:
        subset_idx = np.concatenate([top_idx, rest_pool], axis=0)
        return subset_idx.astype(int, copy=False), top_n

    # evaluate TopVar + sampled rest
    rest_budget = int(max_features) - int(top_n)
    if rest_budget <= 0:
        return top_idx.astype(int, copy=False), top_n

    rest_n = min(rest_budget, rest_pool.size)
    if rest_n <= 0:
        return top_idx.astype(int, copy=False), top_n

    rest_idx = rng.choice(rest_pool, size=rest_n, replace=False).astype(int, copy=False)
    subset_idx = np.concatenate([top_idx, rest_idx], axis=0)
    return subset_idx.astype(int, copy=False), top_n


def compute_fold_metrics(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    k_pct: int,
    subset_idx: np.ndarray,
    top_n_full: int,
    Z: np.ndarray,
    lambdas: np.ndarray,
) -> Dict[str, float]:
    """
    Compute etaES/VSA (feature-level via eta^2) on subset_idx, and PCLA/SAS on PCA cache.
    """
    # subset for feature-level metrics
    Xsub = Xtr[:, subset_idx]
    _v_total, _v_between, eta2 = eta2_features(Xsub, ytr)

    # top block is always first top_n_full entries by construction when max_features>0
    top_n = min(int(top_n_full), int(eta2.size))
    if top_n < 1:
        return {"eta_es": float("nan"), "vsa": float("nan"), "pcla": float("nan"), "sas": float("nan")}

    eta2_top = eta2[:top_n]
    eta2_rest = eta2[top_n:] if eta2.size > top_n else np.array([], dtype=float)

    eta_all = float(np.mean(eta2[np.isfinite(eta2)])) if np.isfinite(eta2).any() else 0.0
    eta_top = float(np.mean(eta2_top[np.isfinite(eta2_top)])) if np.isfinite(eta2_top).any() else float("nan")
    eta_es = float(eta_top / (eta_all + _EPS))

    vsa = mann_whitney_effect(eta2_top, eta2_rest)

    pcla, sas = pcla_from_cache(Z, lambdas, ytr)

    return {"eta_es": float(eta_es), "vsa": float(vsa), "pcla": float(pcla), "sas": float(sas)}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)

    k_pct = int(args.primary_k) if args.primary_k is not None else int(args.k)
    n_perm = int(args.n_perm)

    out_dir = outputs_dir / args.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    per_view_dir = out_dir / "per_view"
    per_view_dir.mkdir(parents=True, exist_ok=True)

    if args.views.strip():
        targets = parse_views_arg(args.views)
    elif args.all_views:
        targets = list(ALL_VIEWS)
    elif args.hero_views:
        targets = list(HERO_VIEWS)
    else:
        targets = list(ALL_VIEWS)

    # group by dataset (splits + permutations per dataset)
    ds_to_views: Dict[str, List[str]] = {}
    for ds, vw in targets:
        ds_to_views.setdefault(ds, []).append(vw)

    rows_all: List[Dict[str, object]] = []
    t0 = time.time()

    for dataset, views in ds_to_views.items():
        splits_path = outputs_dir / "01_bundles" / "splits" / f"splits__{dataset}.json"
        if not splits_path.exists():
            raise FileNotFoundError(f"Missing splits: {splits_path}")

        splits = load_splits_json(splits_path)
        task_type = infer_task_type(splits.y)
        if task_type != "classification":
            print(f"[WARN] dataset={dataset}: task_type={task_type}; permutation null is defined for classification. Skipping.")
            continue

        # label-encode y (stable)
        y = np.asarray(splits.y)
        classes, y_codes = np.unique(y, return_inverse=True)
        y_codes = y_codes.astype(np.int32, copy=False)

        groups = np.asarray(splits.groups).astype(str)

        # build global permutations for this dataset (coherent across folds/views)
        y_perm_list: List[np.ndarray] = []
        for perm_id in range(n_perm):
            rng = np.random.default_rng(int(stable_u64("perm", dataset, args.seed, perm_id)))
            if args.within_groups:
                yp = permute_labels_within_groups(y_codes, groups, rng)
            else:
                yp = permute_labels(y_codes, rng)
            y_perm_list.append(yp.astype(np.int32, copy=False))

        bundle_path = resolve_bundle_path(outputs_dir, splits)

        fold_ids = splits.fold_ids
        n_repeats_total, n_samples = fold_ids.shape
        n_repeats = min(int(args.max_repeats), int(n_repeats_total))

        # fold tasks (repeat × fold)
        tasks: List[Tuple[int, int, np.ndarray]] = []
        for r in range(n_repeats):
            fold_r = fold_ids[r, :]
            for f in np.unique(fold_r):
                train_idx = np.where(fold_r != f)[0]
                tasks.append((int(r), int(f), train_idx))

        for view in views:
            view_safe = view.replace("/", "__")
            out_path = per_view_dir / f"permnull__{dataset}__{view_safe}.csv.gz"
            if out_path.exists() and not args.force:
                print(f"  Skipping {dataset}/{view} (already saved). Use --force to overwrite.")
                df_view = pd.read_csv(out_path, compression="gzip")
                rows_all.extend(df_view.to_dict("records"))
                continue

            # load view matrix
            _y_bundle, bundle_ids, X, _feats, _info_bundle = load_bundle_view(bundle_path, view)
            X = align_X_to_splits_order(X, bundle_ids, splits.sample_ids)

            # per-view fold job
            def _fold_job(t: Tuple[int, int, np.ndarray]) -> List[Dict[str, object]]:
                r, f, train_idx = t
                Xtr = X[train_idx, :]
                y_true = y_codes[train_idx]

                # deterministic RNG for feature subset sampling per fold
                rng_fold = np.random.default_rng(int(stable_u64("feat", dataset, view, args.seed, r, f)))

                subset_idx, top_n_full = select_feature_subset_by_variance(
                    Xtr=Xtr,
                    k_pct=k_pct,
                    max_features=int(args.max_features),
                    rng=rng_fold,
                )

                # PCA cache once per fold (no label randomness in PCA itself)
                Z, lambdas = compute_pca_cache(
                    Xtr,
                    n_components=int(args.pca_components),
                    random_state=int(stable_u64("pca", dataset, view, args.seed, r, f) % (2**31 - 1)),
                )

                # observed
                m_obs = compute_fold_metrics(
                    Xtr=Xtr,
                    ytr=y_true,
                    k_pct=k_pct,
                    subset_idx=subset_idx,
                    top_n_full=top_n_full,
                    Z=Z,
                    lambdas=lambdas,
                )
                out_rows: List[Dict[str, object]] = []
                out_rows.append({
                    "dataset": dataset,
                    "view": view,
                    "k_pct": int(k_pct),
                    "repeat": int(r),
                    "fold": int(f),
                    "perm_id": -1,
                    "is_perm": 0,
                    "n_train": int(train_idx.size),
                    "p_full": int(Xtr.shape[1]),
                    "p_eval": int(subset_idx.size),
                    "top_n_full": int(top_n_full),
                    "eta_es": float(m_obs["eta_es"]),
                    "vsa": float(m_obs["vsa"]),
                    "pcla": float(m_obs["pcla"]),
                    "sas": float(m_obs["sas"]),
                })

                # permutations
                for perm_id, y_perm_full in enumerate(y_perm_list):
                    y_perm = y_perm_full[train_idx]
                    m = compute_fold_metrics(
                        Xtr=Xtr,
                        ytr=y_perm,
                        k_pct=k_pct,
                        subset_idx=subset_idx,
                        top_n_full=top_n_full,
                        Z=Z,
                        lambdas=lambdas,
                    )
                    out_rows.append({
                        "dataset": dataset,
                        "view": view,
                        "k_pct": int(k_pct),
                        "repeat": int(r),
                        "fold": int(f),
                        "perm_id": int(perm_id),
                        "is_perm": 1,
                        "n_train": int(train_idx.size),
                        "p_full": int(Xtr.shape[1]),
                        "p_eval": int(subset_idx.size),
                        "top_n_full": int(top_n_full),
                        "eta_es": float(m["eta_es"]),
                        "vsa": float(m["vsa"]),
                        "pcla": float(m["pcla"]),
                        "sas": float(m["sas"]),
                    })
                return out_rows

            if int(args.n_jobs) <= 1 or thread_map is None:
                pbar = tqdm(tasks, desc=f"permnull {dataset}/{view}", unit="fold")
                view_rows: List[Dict[str, object]] = []
                for t in pbar:
                    view_rows.extend(_fold_job(t))
            else:
                if str(args.backend) != "threading":
                    raise ValueError("Only --backend threading is supported.")
                fold_rows = thread_map(
                    _fold_job,
                    tasks,
                    max_workers=int(args.n_jobs),
                    desc=f"permnull {dataset}/{view}",
                    unit="fold",
                )
                view_rows = []
                for rr in fold_rows:
                    view_rows.extend(rr)

            df_view = pd.DataFrame(view_rows)
            df_view.to_csv(out_path, index=False, compression="gzip")
            rows_all.extend(view_rows)

    # write long
    long_df = pd.DataFrame(rows_all)
    long_path = out_dir / "permnull_long.csv.gz"
    long_df.to_csv(long_path, index=False, compression="gzip")

    # summarise: view-level observed mean vs permutation null of view-level mean
    if long_df.empty:
        print("[WARN] No rows produced.")
        return

    metrics = ["eta_es", "vsa", "pcla", "sas"]
    summaries: List[Dict[str, object]] = []

    for (ds, vw), g in long_df.groupby(["dataset", "view"]):
        g_obs = g[g["perm_id"].astype(int) == -1]
        g_perm = g[g["perm_id"].astype(int) >= 0].copy()

        row: Dict[str, object] = {"dataset": ds, "view": vw, "k_pct": int(g["k_pct"].iloc[0])}
        row["n_folds"] = int(g_obs.shape[0])
        row["n_perm"] = int(g_perm["perm_id"].nunique())

        # observed: mean across folds
        for m in metrics:
            row[f"{m}_obs_mean"] = float(pd.to_numeric(g_obs[m], errors="coerce").mean())

        # null: per-permutation mean across folds, then distribution over permutations
        perm_means = (
            g_perm.groupby("perm_id")[metrics]
            .mean(numeric_only=True)
            .reset_index()
        )

        for m in metrics:
            null_vals = pd.to_numeric(perm_means[m], errors="coerce").astype(float).to_numpy()
            null_vals = null_vals[np.isfinite(null_vals)]
            if null_vals.size == 0:
                row[f"{m}_null_mean"] = np.nan
                row[f"{m}_null_sd"] = np.nan
                row[f"{m}_z"] = np.nan
                row[f"{m}_p"] = np.nan
                continue

            obs = float(row[f"{m}_obs_mean"])
            mu = float(np.mean(null_vals))
            sd = float(np.std(null_vals, ddof=0))

            row[f"{m}_null_mean"] = mu
            row[f"{m}_null_sd"] = sd
            row[f"{m}_z"] = float((obs - mu) / (sd + _EPS))

            # empirical p-values
            if m == "vsa":
                # two-sided around 0 (symmetry argument; robust even if null mean not exactly 0)
                p_emp = (1.0 + float(np.sum(np.abs(null_vals) >= abs(obs)))) / (null_vals.size + 1.0)
            elif m == "eta_es":
                # two-sided around 1
                d_obs = abs(obs - 1.0)
                d_null = np.abs(null_vals - 1.0)
                p_emp = (1.0 + float(np.sum(d_null >= d_obs))) / (null_vals.size + 1.0)
            elif m == "pcla":
                # one-sided: large pcla indicates alignment
                p_emp = (1.0 + float(np.sum(null_vals >= obs))) / (null_vals.size + 1.0)
            else:
                # SAS: two-sided around 0
                p_emp = (1.0 + float(np.sum(np.abs(null_vals) >= abs(obs)))) / (null_vals.size + 1.0)

            row[f"{m}_p"] = float(p_emp)

        summaries.append(row)

    sum_df = pd.DataFrame(summaries).sort_values(["dataset", "view"]).reset_index(drop=True)

    # FDR (BH) per metric across views
    for m in metrics:
        pcol = f"{m}_p"
        if pcol in sum_df.columns:
            sum_df[f"{m}_q"] = bh_fdr(sum_df[pcol].to_numpy(dtype=float))

    sum_path = out_dir / "permnull_summary.csv"
    sum_df.to_csv(sum_path, index=False)

    # report
    rep_path = out_dir / "PERMNULL_REPORT.md"
    lines: List[str] = []
    lines.append("# Phase 12 — Diagnostic permutation null (no-model)\n\n")
    lines.append(f"- K = {int(k_pct)}%\n")
    lines.append(f"- n_perm = {int(n_perm)}\n")
    lines.append(f"- within_groups = {bool(args.within_groups)}\n")
    lines.append(f"- max_features = {int(args.max_features)} (0=all)\n\n")
    lines.append("## View-level significance (empirical permutation p-values; BH-FDR q-values)\n\n")

    show_cols = [
        "dataset", "view",
        "vsa_obs_mean", "vsa_z", "vsa_p", "vsa_q",
        "eta_es_obs_mean", "eta_es_z", "eta_es_p", "eta_es_q",
        "pcla_obs_mean", "pcla_z", "pcla_p", "pcla_q",
    ]
    avail = [c for c in show_cols if c in sum_df.columns]
    table = sum_df[avail].copy()
    # nicer rounding
    for c in table.columns:
        if c.endswith(("_mean", "_z", "_p", "_q")):
            table[c] = pd.to_numeric(table[c], errors="coerce")
    lines.append(table.to_markdown(index=False))
    lines.append("\n\n## Notes\n")
    lines.append("- VSA null is centred near 0; etaES null is centred near 1; PCLA null is low.\n")
    lines.append("- p-values are permutation-empirical with +1 smoothing: (1 + count)/(N + 1).\n")
    lines.append("- No model training is used; computation is fold-restricted to TRAIN indices.\n")

    rep_path.write_text("".join(lines), encoding="utf-8")

    dt = time.time() - t0
    print(f"\n[OK] Phase12 permutation-null complete in {dt/60.0:.1f} min")
    print(f"  wrote: {long_path}")
    print(f"  wrote: {sum_path}")
    print(f"  wrote: {rep_path}")


if __name__ == "__main__":
    main()
