#!/usr/bin/env python
"""
Label permutation negative control test.

Permutes class labels and re-evaluates feature-subset strategies (ALL, VAR-topK,
SHAP-topK, random-topK) to verify that observed performance differences are
driven by true signal rather than overfitting or selection bias. Supports
parallel execution via threading backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None


HERO_VIEWS = [
    ("mlomics", "methylation"),
    ("ibdmdb", "MGX"),
    ("ccle", "mRNA"),
]


# -----------------------------
# dataclasses
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
# misc utils
# -----------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def infer_task_type(y: np.ndarray) -> str:
    # match project logic: floats are regression unless small integer-like set
    if np.issubdtype(y.dtype, np.floating):
        u = np.unique(y[~np.isnan(y)])
        if len(u) <= 20 and np.all(np.isclose(u, np.round(u))):
            return "classification"
        return "regression"
    return "classification"


def k_to_n(n_features: int, k_pct: int) -> int:
    # match DI rounding: floor
    return max(1, int(n_features * (k_pct / 100.0)))


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
        raise ValueError(f"fold_ids shape {fold_ids.shape} does not match n_samples {len(sample_ids)} for {path}")

    return SplitData(dataset=dataset, y=y, sample_ids=sample_ids, groups=groups, fold_ids=fold_ids, info=info)


def resolve_bundle_path(outputs_dir: Path, splits: SplitData) -> Path:
    bp = splits.info.get("bundle_path", None)
    if bp is not None:
        bp = Path(str(bp))
        # allow relative bundle_path in info
        if not bp.is_absolute():
            cand = (outputs_dir / bp).resolve()
            if cand.exists():
                return cand
        if bp.exists():
            return bp

    # fallback search (covers migrations / older manifests)
    ds = splits.dataset
    cands = []
    for root in [outputs_dir / "01_bundles", outputs_dir, outputs_dir / "archive_pre_migration_20260127"]:
        if root.exists():
            cands += list(root.rglob(f"*{ds}*bundle*normalized*.npz"))
    if not cands:
        raise FileNotFoundError(f"Could not resolve bundle_path for dataset={ds}. info.bundle_path missing or invalid.")
    # prefer shortest path
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
                f"Bundle missing view '{view}' (expected key '{x_key}'). "
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
    order = []
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
# ranks: variance + importance
# -----------------------------
def find_variance_scores(outputs_dir: Path, dataset: str, view: str) -> Path:
    vs_dir = outputs_dir / "02_unsupervised" / "variance_scores"
    p = vs_dir / f"variance_scores__{dataset}__{view}.csv.gz"
    if p.exists():
        return p
    # fallback: case-insensitive or different extension
    hits = list(vs_dir.glob(f"variance_scores__{dataset}__{view}*.csv*"))
    if hits:
        return sorted(hits)[0]
    raise FileNotFoundError(f"Missing variance scores for {dataset}/{view} under {vs_dir}")


def load_variance_rank(outputs_dir: Path, dataset: str, view: str) -> List[str]:
    p = find_variance_scores(outputs_dir, dataset, view)
    df = pd.read_csv(p)
    if "feature" in df.columns:
        feat_col = "feature"
    else:
        # try common alternatives
        for c in ["features", "feature_name", "name"]:
            if c in df.columns:
                feat_col = c
                break
        else:
            raise ValueError(f"Cannot find feature column in {p}. cols={list(df.columns)}")

    # pick a numeric score column
    score_col = None
    for c in ["variance", "var_score", "score", "V", "variance_score"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        num_cols = [c for c in df.columns if c != feat_col and np.issubdtype(df[c].dtype, np.number)]
        if not num_cols:
            raise ValueError(f"Cannot find numeric variance column in {p}. cols={list(df.columns)}")
        score_col = num_cols[0]

    df = df[[feat_col, score_col]].copy()
    df[feat_col] = df[feat_col].astype(str)
    df = df.sort_values(score_col, ascending=False)
    return df[feat_col].tolist()


def find_importance_table(outputs_dir: Path, dataset: str, view: str, model: str) -> Path:
    per_model = outputs_dir / "04_importance" / "per_model"
    p = per_model / f"importance__{dataset}__{view}__{model}.csv.gz"
    if p.exists():
        return p
    hits = list(per_model.glob(f"importance__{dataset}__{view}__{model}.csv*"))
    if hits:
        return sorted(hits)[0]
    raise FileNotFoundError(f"Missing importance table for {dataset}/{view}/{model} under {per_model}")


def load_importance_rank(outputs_dir: Path, dataset: str, view: str, model: str) -> List[str]:
    p = find_importance_table(outputs_dir, dataset, view, model)
    df = pd.read_csv(p)
    need = {"feature", "p_rank"}
    if not need.issubset(df.columns):
        raise ValueError(f"Importance table missing required cols {need} in {p}. cols={list(df.columns)}")
    df["feature"] = df["feature"].astype(str)
    df = df.sort_values("p_rank", ascending=True)
    return df["feature"].tolist()


# -----------------------------
# modeling (match ablation choices)
# -----------------------------
def make_model(model: str, task_type: str, seed: int, n_jobs_model: int, n_classes: int):
    if task_type == "classification":
        if model == "rf":
            return (
                RandomForestClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    max_depth=None,
                    class_weight="balanced_subsample",
                    n_jobs=n_jobs_model,
                    random_state=seed,
                ),
                False,
            )
        if model in ("xgb", "xgb_bal"):
            if XGBClassifier is None:
                raise RuntimeError("xgboost is not installed but model=xgb/xgb_bal was requested")
            objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
            eval_metric = "mlogloss" if n_classes > 2 else "logloss"
            xgb_kwargs = dict(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective=objective,
                eval_metric=eval_metric,
                tree_method="hist",
                random_state=seed,
                n_jobs=n_jobs_model,
                verbosity=0,
            )
            if n_classes > 2:
                xgb_kwargs["num_class"] = int(n_classes)
            return (
                XGBClassifier(**xgb_kwargs),
                (model == "xgb_bal"),
            )
        raise ValueError(f"Unknown model={model}")
    else:
        if model == "rf":
            return (RandomForestRegressor(n_estimators=500, max_features="sqrt", n_jobs=n_jobs_model, random_state=seed), False)
        if model in ("xgb", "xgb_bal"):
            if XGBRegressor is None:
                raise RuntimeError("xgboost is not installed but model=xgb/xgb_bal was requested")
            return (
                XGBRegressor(
                    n_estimators=800,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=n_jobs_model,
                    verbosity=0,
                ),
                False,
            )
        raise ValueError(f"Unknown model={model}")


def compute_metric(metric: str, task_type: str, y_true: np.ndarray, y_pred: np.ndarray, proba: Optional[np.ndarray]) -> float:
    if task_type != "classification":
        raise ValueError("This label-permutation test currently supports classification metrics only.")

    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))

    if metric in ("auroc_ovr_macro", "auroc_macro"):
        # AUROC can be undefined if a fold's y_true is missing classes
        if proba is None:
            return 0.5  # conservative "chance" fallback

        y_true = np.asarray(y_true)
        proba = np.asarray(proba)

        # If fold has <2 classes, AUROC is undefined
        n_true = len(np.unique(y_true))
        if n_true < 2:
            return 0.5

        # Binary probability vector case
        if proba.ndim == 1:
            try:
                return float(roc_auc_score(y_true, proba))
            except ValueError:
                return 0.5

        # Multiclass proba matrix case
        n_cols = proba.shape[1]

        # Key guard: fold missing one or more classes → AUROC undefined
        if n_true != n_cols:
            return 0.5

        try:
            if n_true == 2:
                return float(roc_auc_score(y_true, proba[:, 1]))
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except ValueError:
            return 0.5

    raise ValueError(f"Unsupported metric: {metric}")


def train_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    metrics: List[str],
    seed: int,
    n_jobs_model: int = 1,
) -> Dict[str, float]:
    task_type = infer_task_type(y_train)
    if task_type != "classification":
        raise ValueError("Label permutation test implemented for classification tasks only.")

    classes = np.unique(y_train)
    n_classes = int(len(classes))

    est, needs_sw = make_model(model_name, task_type, seed=seed, n_jobs_model=n_jobs_model, n_classes=n_classes)
    fit_kwargs = {}
    if needs_sw:
        fit_kwargs["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_train)

    est.fit(X_train, y_train, **fit_kwargs)

    y_pred = est.predict(X_test)
    proba = est.predict_proba(X_test) if hasattr(est, "predict_proba") else None

    out = {}
    for m in metrics:
        out[m] = compute_metric(m, "classification", y_test, y_pred, proba)
    return out


# -----------------------------
# permutation + evaluation
# -----------------------------
def permute_labels(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator, within_groups: bool) -> np.ndarray:
    y = np.asarray(y)
    if not within_groups:
        return y[rng.permutation(len(y))]

    _, counts = np.unique(groups, return_counts=True)
    if counts.max() <= 1:
        raise ValueError(
            "within_groups=True but all groups are singletons (max_group_size=1). "
            "Permutation would be identity. Disable --within-groups or provide non-singleton groups."
        )

    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) <= 1:
            continue
        y_perm[idx] = y_perm[idx][rng.permutation(len(idx))]
    return y_perm


def parse_views_arg(s: str) -> List[Tuple[str, str]]:
    # format: "dataset:view,dataset:view"
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        ds, vw = item.split(":")
        out.append((ds.strip(), vw.strip()))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label permutation negative control (ALL vs VAR-topK vs SHAP-topK).")
    p.add_argument("--outputs-dir", required=True)
    p.add_argument("--model", default="xgb_bal", choices=["rf", "xgb", "xgb_bal"])
    p.add_argument("--metrics", default="balanced_accuracy,auroc_ovr_macro")
    p.add_argument("--k-pcts", default="10", help="Comma-separated K percentages (default 10 for fast null control)")
    p.add_argument("--hero-views", action="store_true")
    p.add_argument("--views", default="", help='Optional explicit list "dataset:view,dataset:view" (overrides --hero-views)')
    p.add_argument("--n-perm", type=int, default=3, help="Number of permutation seeds (default 3)")
    p.add_argument("--perm-seed0", type=int, default=0, help="Base seed; uses seed0..seed0+n_perm-1")
    p.add_argument("--within-groups", action="store_true", help="Permute labels within groups (if groups are meaningful)")
    p.add_argument("--max-repeats", type=int, default=2, help="Use first R repeats only (default 2 for speed)")
    p.add_argument("--n-random-draws", type=int, default=0, help="Optional random-topK mean (expensive). Default 0.")
    p.add_argument("--out-dirname", default="06_robustness/label_perm", help="Relative path under outputs-dir")
    # Parallelism controls (optional)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of parallel fold-tasks (threading backend only)")
    p.add_argument("--backend", default="threading", choices=["sequential","threading","multiprocessing"],
                   help="Execution backend (multiprocessing falls back to threading on Windows)")
    p.add_argument("--model-threads", type=int, default=1, help="Threads per model fit (estimator n_jobs)")
    return p.parse_args()


def _run_fold_task(
    X: np.ndarray,
    y_perm: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: str,
    view: str,
    model_name: str,
    metrics: List[str],
    perm_seed: int,
    repeat: int,
    fold: int,
    k_pcts: List[int],
    n_features: int,
    var_rank_idx: List[int],
    imp_rank_idx: List[int],
    n_random_draws: int,
    model_threads: int,
) -> Tuple[List[Dict[str, object]], int]:
    """Run one permutation/repeat/fold unit.

    Optimisations:
      - Fit 'all' once per fold (reused across k_pcts).
      - VAR-topK and SHAP-topK are fit per k_pct.
      - Optional random-topK mean fits n_random_draws per k_pct.
    Returns (rows, n_model_fits).
    """
    rows: List[Dict[str, object]] = []

    y_train = y_perm[train_idx]
    y_test = y_perm[test_idx]

    # ---- Fit ALL once ----
    Xtr_all = X[train_idx, :]
    Xte_all = X[test_idx, :]
    seed_all = int(perm_seed + 10_000 + repeat * 100 + int(fold))
    res_all = train_eval(
        Xtr_all, y_train, Xte_all, y_test,
        model_name, metrics,
        seed=seed_all,
        n_jobs_model=int(model_threads),
    )

    for k_pct in k_pcts:
        for m, v in res_all.items():
            rows.append({
                "dataset": dataset, "view": view, "model": model_name,
                "perm_seed": int(perm_seed), "repeat": int(repeat), "fold": int(fold),
                "k_pct": int(k_pct), "k_n": int(n_features), "n_features": int(n_features),
                "strategy": "all", "metric": m, "value": float(v),
            })

    n_fits = 1

    # ---- Subset fits per k ----
    for k_pct in k_pcts:
        k_n = k_to_n(int(n_features), int(k_pct))

        # VAR-topK
        idxs = var_rank_idx[:k_n]
        Xtr = X[train_idx, :][:, idxs]
        Xte = X[test_idx, :][:, idxs]
        seed_var = int(perm_seed + 20_000 + repeat * 100 + int(fold))
        res = train_eval(Xtr, y_train, Xte, y_test, model_name, metrics, seed=seed_var, n_jobs_model=int(model_threads))
        for m, v in res.items():
            rows.append({
                "dataset": dataset, "view": view, "model": model_name,
                "perm_seed": int(perm_seed), "repeat": int(repeat), "fold": int(fold),
                "k_pct": int(k_pct), "k_n": int(k_n), "n_features": int(n_features),
                "strategy": "var_topk", "metric": m, "value": float(v),
            })
        n_fits += 1

        # SHAP-topK
        idxs = imp_rank_idx[:k_n]
        Xtr = X[train_idx, :][:, idxs]
        Xte = X[test_idx, :][:, idxs]
        seed_shap = int(perm_seed + 30_000 + repeat * 100 + int(fold))
        res = train_eval(Xtr, y_train, Xte, y_test, model_name, metrics, seed=seed_shap, n_jobs_model=int(model_threads))
        for m, v in res.items():
            rows.append({
                "dataset": dataset, "view": view, "model": model_name,
                "perm_seed": int(perm_seed), "repeat": int(repeat), "fold": int(fold),
                "k_pct": int(k_pct), "k_n": int(k_n), "n_features": int(n_features),
                "strategy": "shap_topk", "metric": m, "value": float(v),
            })
        n_fits += 1

        # RANDOM-topK mean (optional)
        if int(n_random_draws) > 0:
            vals = {m: [] for m in metrics}
            rng2 = np.random.default_rng(int(perm_seed * 100000 + repeat * 1000 + int(fold) * 10 + int(k_pct)))
            for _ in range(int(n_random_draws)):
                idxs = rng2.choice(int(n_features), size=int(k_n), replace=False)
                Xtr = X[train_idx, :][:, idxs]
                Xte = X[test_idx, :][:, idxs]
                res = train_eval(
                    Xtr, y_train, Xte, y_test,
                    model_name, metrics,
                    seed=int(rng2.integers(1, 2**31 - 1)),
                    n_jobs_model=int(model_threads),
                )
                for m, v in res.items():
                    vals[m].append(float(v))
                n_fits += 1
            for m in metrics:
                rows.append({
                    "dataset": dataset, "view": view, "model": model_name,
                    "perm_seed": int(perm_seed), "repeat": int(repeat), "fold": int(fold),
                    "k_pct": int(k_pct), "k_n": int(k_n), "n_features": int(n_features),
                    "strategy": "random_topk_mean", "metric": m, "value": float(np.mean(vals[m])),
                })

    return rows, int(n_fits)

def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / args.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    per_view_dir = out_dir / "per_view"
    per_view_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    k_pcts = [int(x.strip()) for x in args.k_pcts.split(",") if x.strip()]

    # decide targets
    if args.views.strip():
        targets = parse_views_arg(args.views)
    elif args.hero_views:
        targets = list(HERO_VIEWS)
    else:
        # default: hero views (safe)
        targets = list(HERO_VIEWS)

    # group targets by dataset (splits file is per dataset)
    ds_to_views: Dict[str, List[str]] = {}
    for ds, vw in targets:
        ds_to_views.setdefault(ds, []).append(vw)

    rows = []
    t0 = time.time()

    for dataset, views in ds_to_views.items():
        splits_path = outputs_dir / "01_bundles" / "splits" / f"splits__{dataset}.json"
        if not splits_path.exists():
            raise FileNotFoundError(f"Missing splits: {splits_path}")

        splits = load_splits_json(splits_path)
        bundle_path = resolve_bundle_path(outputs_dir, splits)

        # perm seeds
        perm_seeds = list(range(args.perm_seed0, args.perm_seed0 + args.n_perm))

        for view in views:
            out_path = per_view_dir / f"label_perm__{dataset}__{view.replace('/', '__')}.csv.gz"
            if out_path.exists():
                print(f"  Skipping {dataset}/{view} (already saved)")
                continue

            view_rows = []

            # load bundle view
            _y_bundle, bundle_ids, X, feats, _info_bundle = load_bundle_view(bundle_path, view)
            X = align_X_to_splits_order(X, bundle_ids, splits.sample_ids)

            # precompute feature-index mapping
            feat_to_idx = {str(f): i for i, f in enumerate(feats.tolist())}
            n_features = X.shape[1]

            # ranks for subset selection
            var_rank = load_variance_rank(outputs_dir, dataset, view)
            imp_rank = load_importance_rank(outputs_dir, dataset, view, args.model)

            # convert ranks to index lists (drop features not present in bundle)
            var_rank_idx = [feat_to_idx[f] for f in var_rank if f in feat_to_idx]
            imp_rank_idx = [feat_to_idx[f] for f in imp_rank if f in feat_to_idx]

            fold_ids = splits.fold_ids
            n_repeats_total, _n_samples = fold_ids.shape
            n_repeats = min(args.max_repeats, n_repeats_total)

            unique_folds = np.unique(fold_ids[0, :])
            n_folds = len(unique_folds)

            backend = str(getattr(args, "backend", "threading")).lower().strip()
            if backend == "multiprocessing":
                # Multiprocessing is often fragile on Windows for large numpy workloads; prefer threading.
                backend = "threading"
            use_parallel = (backend == "threading" and int(getattr(args, "n_jobs", 1)) > 1)

            # progress total is in terms of *actual model fits*
            n_strat_per_k = 2 + (int(args.n_random_draws) if int(args.n_random_draws) > 0 else 0)  # var + shap + random draws
            total_fits = len(perm_seeds) * n_repeats * n_folds * (1 + len(k_pcts) * n_strat_per_k)
            pbar = tqdm(total=total_fits, desc=f"label_perm {dataset}/{view} {args.model}", unit="fit")

            max_inflight = max(1, int(getattr(args, "n_jobs", 1)) * 2)

            def _submit(ex, y_perm, train_idx, test_idx, perm_seed, r, f):
                return ex.submit(
                    _run_fold_task,
                    X, y_perm, train_idx, test_idx,
                    dataset, view, args.model, metrics,
                    int(perm_seed), int(r), int(f),
                    k_pcts, int(n_features),
                    var_rank_idx, imp_rank_idx,
                    int(args.n_random_draws),
                    int(args.model_threads),
                )

            if use_parallel:
                with ThreadPoolExecutor(max_workers=int(args.n_jobs)) as ex:
                    futures = set()

                    for perm_seed in perm_seeds:
                        rng = np.random.default_rng(int(perm_seed))
                        y_perm = permute_labels(splits.y, splits.groups, rng, within_groups=args.within_groups)

                        for r in range(n_repeats):
                            fold_r = fold_ids[r, :]
                            for f in np.unique(fold_r):
                                test_idx = np.where(fold_r == f)[0]
                                train_idx = np.where(fold_r != f)[0]

                                if len(futures) >= max_inflight:
                                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                                    for fut in done:
                                        out_rows, n_fits = fut.result()
                                        view_rows.extend(out_rows)
                                        pbar.update(n_fits)

                                futures.add(_submit(ex, y_perm, train_idx, test_idx, perm_seed, r, int(f)))

                    for fut in futures:
                        out_rows, n_fits = fut.result()
                        view_rows.extend(out_rows)
                        pbar.update(n_fits)

            else:
                for perm_seed in perm_seeds:
                    rng = np.random.default_rng(int(perm_seed))
                    y_perm = permute_labels(splits.y, splits.groups, rng, within_groups=args.within_groups)

                    for r in range(n_repeats):
                        fold_r = fold_ids[r, :]
                        for f in np.unique(fold_r):
                            test_idx = np.where(fold_r == f)[0]
                            train_idx = np.where(fold_r != f)[0]
                            out_rows, n_fits = _run_fold_task(
                                X, y_perm, train_idx, test_idx,
                                dataset, view, args.model, metrics,
                                int(perm_seed), int(r), int(f),
                                k_pcts, int(n_features),
                                var_rank_idx, imp_rank_idx,
                                int(args.n_random_draws),
                                int(args.model_threads),
                            )
                            view_rows.extend(out_rows)
                            pbar.update(n_fits)

            # Save regardless of backend
            pbar.close()
            pd.DataFrame(view_rows).to_csv(out_path, index=False, compression="gzip")
            rows.extend(view_rows)

    long_df = pd.DataFrame(rows)
    long_path = out_dir / "label_perm_long.csv.gz"
    long_df.to_csv(long_path, index=False, compression="gzip")

    # Summary: per dataset/view/model/metric/k_pct across folds+repeats per perm_seed, then aggregated
    # 1) within perm_seed
    g1 = (long_df
          .groupby(["dataset", "view", "model", "perm_seed", "metric", "k_pct", "strategy"], as_index=False)["value"]
          .mean())

    # 2) compute deltas inside each perm_seed
    pivot = g1.pivot_table(index=["dataset", "view", "model", "perm_seed", "metric", "k_pct"],
                           columns="strategy", values="value", aggfunc="first").reset_index()
    if "shap_topk" in pivot.columns and "var_topk" in pivot.columns:
        pivot["delta_shap_var"] = pivot["shap_topk"] - pivot["var_topk"]
    else:
        pivot["delta_shap_var"] = np.nan

    # 3) aggregate across perm seeds
    def q(x, p): return float(np.quantile(x, p)) if len(x) else float("nan")

    summ = []
    for (ds, vw, model, metric, k_pct), sub in pivot.groupby(["dataset", "view", "model", "metric", "k_pct"]):
        vals = sub["delta_shap_var"].dropna().to_numpy(dtype=float)
        summ.append({
            "dataset": ds, "view": vw, "model": model, "metric": metric, "k_pct": int(k_pct),
            "n_perm": int(sub["perm_seed"].nunique()),
            "delta_shap_var_mean": float(np.mean(vals)) if vals.size else float("nan"),
            "delta_shap_var_q05": q(vals, 0.05),
            "delta_shap_var_q95": q(vals, 0.95),
        })
    summ_df = pd.DataFrame(summ).sort_values(["dataset", "view", "metric", "k_pct"])
    summ_path = out_dir / "label_perm_summary.csv"
    summ_df.to_csv(summ_path, index=False)

    manifest = {
        "script": "03_label_permutation_test.py",
        "created_utc": _now_utc(),
        "args": vars(args),
        "outputs": {
            "long": str(long_path),
            "summary": str(summ_path),
        },
        "hashes": {
            long_path.name: sha256_file(long_path),
            summ_path.name: sha256_file(summ_path),
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    (out_dir / "MANIFEST__label_perm.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
