#!/usr/bin/env python3
"""
PHASE 3 — 03_train_tree_models.py

Train tree-based classifiers per dataset × view using deterministic CV splits
(from 03_supervised/01_define_tasks_and_splits.py) and compute prediction-driving
feature importance P_j as cross-validated mean absolute SHAP value.

Core guarantees
- Strict leakage control: SHAP computed on held-out folds only
- Reproducible: seeds fixed; bundle + splits sha256 recorded
- Reviewer-proof I/O: no pickles; allow_pickle=False everywhere
- Memory-lean: per-task loads a single view matrix from disk

Outputs (per dataset × view × model)
- outputs/models/tree_models/preds__{dataset}__{view}__{model}.npz
    y, sample_ids, fold_ids_used, class_order,
    oof_pred (n_repeats,n_samples),
    oof_proba (n_repeats,n_samples,n_classes)
- outputs/metrics/tree_models/metrics__{dataset}__{view}__{model}.json
    pooled OOF metrics per repeat + summary stats + provenance
- outputs/importance/tree_models/shap__{dataset}__{view}__{model}.npz
    feature_names, mean_abs_shap_per_repeat (n_repeats,n_features),
    mean_abs_shap (n_features), shap_meta (json str)
- outputs/importance/tree_models/prediction_importance__{dataset}__{view}__{model}.csv.gz
    feature, importance, rank, percentile

Manifest
- outputs/metrics/tree_models/tree_models_manifest.json

Notes
- For CCLE, splits may contain label collapsing to enable stratified CV. We train
  on splits.y and verify bundle alignment using splits.y_raw when present.
- For TCGA_GBM (N=47), interpret results as sensitivity only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

import shap
import xgboost as xgb


# -----------------------------
# View registry (must match bundles + splits)
# -----------------------------
VIEW_REGISTRY = {
    "mlomics": {
        "core_views": ["mRNA", "miRNA", "methylation", "CNV"],
        "sensitivity_views": [],
        "analysis_role": "primary",
    },
    "ibdmdb": {
        "core_views": ["MGX", "MGX_func", "MPX", "MBX"],
        "sensitivity_views": ["MGX_CLR"],
        "analysis_role": "primary",
    },
    "ccle": {
        "core_views": ["mRNA", "CNV", "proteomics"],
        "sensitivity_views": [],
        "analysis_role": "primary",
    },
    "tcga_gbm": {
        "core_views": ["mRNA", "methylation", "CNV"],
        "sensitivity_views": ["methylation_Mval"],
        "analysis_role": "sensitivity",
    },
}

TCGA_GBM_NOTE = "N=47; AUROC/importance high-variance — treat as sensitivity analysis"


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_npz_pickle_free(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(str(path), allow_pickle=False)


def unique_sorted_int(y: np.ndarray) -> np.ndarray:
    return np.array(sorted(np.unique(y).tolist()), dtype=np.int32)


def ci95(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(values, 2.5))
    hi = float(np.percentile(values, 97.5))
    return (lo, hi)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_order: np.ndarray | None = None,
) -> Dict[str, float]:
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None and class_order is not None:
        try:
            metrics["macro_auroc"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=class_order,
                )
            )
        except Exception:
            metrics["macro_auroc"] = float("nan")
    return metrics


def summarize_metrics(per_repeat: List[Dict[str, float]]) -> Dict[str, object]:
    if not per_repeat:
        return {}
    out: Dict[str, object] = {}
    keys = list(per_repeat[0].keys())
    for k in keys:
        vals = [float(d[k]) for d in per_repeat]
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        lo, hi = ci95(vals)
        out[k] = {"mean": mu, "std": sd, "ci95": [lo, hi], "per_repeat": vals}
    return out


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


@dataclass
class SplitData:
    dataset: str
    y: np.ndarray
    y_raw: Optional[np.ndarray]
    sample_ids: np.ndarray
    groups: np.ndarray
    fold_ids: np.ndarray  # (n_repeats, n_samples)
    info: Dict[str, object]


def load_splits(splits_path: Path) -> SplitData:
    z = load_npz_pickle_free(splits_path)
    info = json.loads(str(z["info"]))
    y = z["y"].astype(np.int32)
    y_raw_arr = z["y_raw"].astype(np.int32)
    y_raw = y_raw_arr if y_raw_arr.size > 0 else None
    sample_ids = z["sample_ids"].astype(str)
    groups = z["groups"].astype(str)
    fold_ids = z["fold_ids"].astype(np.int16)
    dataset = str(info.get("dataset", splits_path.stem.replace("splits__", "")))
    return SplitData(
        dataset=dataset,
        y=y,
        y_raw=y_raw,
        sample_ids=sample_ids,
        groups=groups,
        fold_ids=fold_ids,
        info=info,
    )


def load_bundle_view(bundle_path: Path, view: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    z = load_npz_pickle_free(bundle_path)
    y = z["y"].astype(np.int32)
    sample_ids = z["sample_ids"].astype(str)
    info = json.loads(str(z["info"]))
    x_key = f"X_{view}"
    f_key = f"features_{view}"
    if x_key not in z.files:
        raise KeyError(f"Bundle missing view '{view}' (key '{x_key}'). Available: {[k for k in z.files if k.startswith('X_')]}")
    X = z[x_key].astype(np.float32)
    feats = z[f_key].astype(str) if f_key in z.files else np.array([], dtype=str)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)
    return y, sample_ids, X, feats, info


def assert_alignment(bundle_y: np.ndarray, bundle_ids: np.ndarray, split_y_for_alignment: np.ndarray, split_ids: np.ndarray) -> None:
    if bundle_y.shape != split_y_for_alignment.shape or bundle_ids.shape != split_ids.shape:
        raise ValueError("Bundle and split shapes mismatch.")
    if not np.array_equal(bundle_ids, split_ids):
        raise ValueError("sample_ids in bundle and splits do not match (order mismatch).")
    return


# -----------------------------
# Model constructors
# -----------------------------
def make_xgb_classifier(
    seed: int,
    n_classes: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
) -> xgb.XGBClassifier:
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss" if n_classes > 2 else "logloss"
    return xgb.XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        min_child_weight=1.0,
        gamma=0.0,
        objective=objective,
        eval_metric=eval_metric,
        tree_method="hist",
        n_jobs=1,
        random_state=int(seed),
        verbosity=0,
    )


def make_rf_classifier(
    seed: int,
    n_estimators: int,
    max_features: str,
    min_samples_leaf: int,
    max_depth: Optional[int],
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_features=max_features,
        min_samples_leaf=int(min_samples_leaf),
        max_depth=None if max_depth is None else int(max_depth),
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=int(seed),
    )


# -----------------------------
# SHAP helpers
# -----------------------------
def _safe_shap_explainer(model, X_background: Optional[np.ndarray]):
    try:
        if X_background is not None and X_background.size > 0:
            return shap.TreeExplainer(
                model,
                data=X_background,
                feature_perturbation="tree_path_dependent",
            )
        return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    except Exception as e:
        msg = str(e)

        # Fix for RF leaf-coverage failure under tree_path_dependent
        if (
            "does not cover all the leaves" in msg
            and X_background is not None
            and X_background.size > 0
        ):
            return shap.TreeExplainer(
                model,
                data=X_background,
                feature_perturbation="interventional",
            )

        # Last resort fallbacks
        if X_background is not None and X_background.size > 0:
            return shap.Explainer(model, X_background)
        return shap.Explainer(model)


def _mean_abs_shap_per_feature(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        per_class = []
        for sv in shap_values:
            sv = np.asarray(sv)
            per_class.append(np.mean(np.abs(sv), axis=0))
        return np.mean(np.stack(per_class, axis=0), axis=0)

    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        return np.mean(np.abs(sv), axis=(0, 2))
    if sv.ndim == 2:
        return np.mean(np.abs(sv), axis=0)
    raise ValueError(f"Unexpected SHAP value shape: {sv.shape}")


def _sample_rows(rng: np.random.Generator, X: np.ndarray, max_rows: Optional[int]) -> np.ndarray:
    if max_rows is None:
        return X
    max_rows = int(max_rows)
    if X.shape[0] <= max_rows:
        return X
    idx = rng.choice(X.shape[0], size=max_rows, replace=False)
    return X[idx]


def xgb_mean_abs_shap_pred_contribs(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """
    Returns mean(|SHAP|) per feature using native XGBoost pred_contribs (TreeSHAP).
    Works for binary and multiclass.
    """
    booster = model.get_booster()
    dm = xgb.DMatrix(X)
    contribs = booster.predict(dm, pred_contribs=True, approx_contribs=False)
    arr = np.asarray(contribs)

    # arr shapes:
    #  - binary: (n_samples, n_features+1)  last column = bias
    #  - multiclass: (n_samples, n_classes, n_features+1) last = bias
    if arr.ndim == 2:
        vals = arr[:, :-1]
        return np.mean(np.abs(vals), axis=0).astype(np.float32)
    if arr.ndim == 3:
        vals = arr[:, :, :-1]
        return np.mean(np.abs(vals), axis=(0, 1)).astype(np.float32)

    raise ValueError(f"Unexpected pred_contribs shape: {arr.shape}")


def train_oof_and_shap(
    model_name: str,
    model_ctor,
    X: np.ndarray,
    y: np.ndarray,
    class_order: np.ndarray,
    fold_ids_used: np.ndarray,
    seed: int,
    n_classes: int,
    compute_shap: bool,
    shap_background: int,
    shap_max_test: Optional[int],
    shap_check_additivity: bool,
    xgb_balance: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]], Optional[np.ndarray], Dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    n_repeats, n_samples = fold_ids_used.shape
    n_features = X.shape[1]

    oof_pred = np.empty((n_repeats, n_samples), dtype=np.int32)
    oof_proba = np.empty((n_repeats, n_samples, n_classes), dtype=np.float32)
    per_repeat: List[Dict[str, float]] = []

    shap_per_repeat = np.empty((n_repeats, n_features), dtype=np.float32) if compute_shap else None
    shap_sample_counts: List[int] = []
    warn_counts = {"warnings": 0}

    for r in tqdm(range(n_repeats), desc=f"{model_name} repeats", unit="rep"):
        y_pred_r = np.full(n_samples, -1, dtype=np.int32)
        y_proba_r = np.full((n_samples, n_classes), np.nan, dtype=np.float32)

        if compute_shap:
            shap_sum = np.zeros(n_features, dtype=np.float64)
            shap_n = 0

        n_folds = int(fold_ids_used[r].max()) + 1
        for k in tqdm(range(n_folds), desc=f"repeat {r+1}/{n_repeats} folds", unit="fold", leave=False):
            test_idx = np.where(fold_ids_used[r] == k)[0]
            train_idx = np.where(fold_ids_used[r] != k)[0]
            if test_idx.size == 0 or train_idx.size == 0:
                raise RuntimeError(f"Empty fold encountered: repeat={r}, fold={k}")

            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_te = X[test_idx]

            model = model_ctor(seed=seed + 1009 * r + 17 * k, n_classes=n_classes)

            fit_kwargs = {}
            if model_name == "xgb" and xgb_balance:
                fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_tr).astype(np.float32)

            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                with threadpool_limits(limits=1):
                    model.fit(X_tr, y_tr, **fit_kwargs)
                if wlist:
                    warn_counts["warnings"] += len(wlist)

            with threadpool_limits(limits=1):
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_te)
                    proba = np.asarray(proba, dtype=np.float32)
                    if proba.ndim == 1:
                        proba = np.vstack([1.0 - proba, proba]).T.astype(np.float32)
                else:
                    pred = model.predict(X_te)
                    proba = np.zeros((X_te.shape[0], n_classes), dtype=np.float32)
                    proba[np.arange(X_te.shape[0]), pred.astype(int)] = 1.0

            if proba.shape[1] != n_classes:
                raise RuntimeError(f"{model_name}: probability shape mismatch {proba.shape} vs n_classes={n_classes}")

            pred = np.argmax(proba, axis=1).astype(np.int32)
            y_pred_r[test_idx] = pred
            y_proba_r[test_idx] = proba

            if compute_shap:
                X_bg = _sample_rows(rng, X_tr, max_rows=int(shap_background) if shap_background is not None else None)
                X_te_shap = _sample_rows(rng, X_te, max_rows=shap_max_test)

                if model_name == "xgb":
                    fold_mean_abs = xgb_mean_abs_shap_pred_contribs(model, X_te_shap).astype(np.float64)
                else:
                    explainer = _safe_shap_explainer(model, X_bg)

                    with threadpool_limits(limits=1):
                        try:
                            if hasattr(explainer, "shap_values"):
                                sv = explainer.shap_values(X_te_shap, check_additivity=shap_check_additivity)
                            else:
                                sv = explainer(X_te_shap).values
                        except Exception as e:
                            if "does not cover all the leaves" in str(e):
                                # Retry with interventional perturbation
                                explainer = shap.TreeExplainer(
                                    model,
                                    data=X_bg,
                                    feature_perturbation="interventional",
                                )
                                sv = explainer.shap_values(X_te_shap, check_additivity=False)
                            else:
                                raise

                    fold_mean_abs = _mean_abs_shap_per_feature(sv).astype(np.float64)
                shap_sum += fold_mean_abs * float(X_te_shap.shape[0])
                shap_n += int(X_te_shap.shape[0])

        if np.any(y_pred_r < 0) or not np.isfinite(y_proba_r).all():
            raise RuntimeError(f"{model_name}: OOF predictions incomplete for repeat={r}")

        oof_pred[r] = y_pred_r
        oof_proba[r] = y_proba_r
        per_repeat.append(compute_metrics(y, y_pred_r, y_proba=y_proba_r, class_order=class_order))

        if compute_shap:
            if shap_n <= 0:
                raise RuntimeError(f"{model_name}: SHAP accumulation failed (shap_n=0) for repeat={r}")
            shap_per_repeat[r] = (shap_sum / float(shap_n)).astype(np.float32)
            shap_sample_counts.append(int(shap_n))

    meta = {
        "model_name": model_name,
        "warnings_count": warn_counts["warnings"],
        "shap_sample_counts_per_repeat": shap_sample_counts,
        "xgb_balance": bool(xgb_balance),
    }
    return oof_pred, oof_proba, fold_ids_used, per_repeat, shap_per_repeat, meta


def process_dataset_view_task(
    ds: str,
    view: str,
    repo_root: Path,
    bundle_dir: Path,
    splits_dir: Path,
    out_models_dir: Path,
    out_metrics_dir: Path,
    out_importance_dir: Path,
    models_to_run: List[str],
    seed: int,
    max_repeats: Optional[int],
    compute_shap: bool,
    shap_background: int,
    shap_max_test: Optional[int],
    shap_check_additivity: bool,
    xgb_params: Dict[str, object],
    rf_params: Dict[str, object],
    xgb_balance: bool,
) -> Tuple[str, str, Dict[str, object]]:
    bundle_path = bundle_dir / f"{ds}_bundle_normalized.npz"
    splits_path = splits_dir / f"splits__{ds}.npz"

    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle: {bundle_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits: {splits_path}")

    bundle_y, bundle_ids, X, feat_names, _bundle_info = load_bundle_view(bundle_path, view)
    splits = load_splits(splits_path)

    y_for_alignment = splits.y_raw if splits.y_raw is not None else splits.y
    assert_alignment(bundle_y, bundle_ids, y_for_alignment, splits.sample_ids)

    y_train = splits.y.astype(np.int32)
    class_order = unique_sorted_int(y_train)
    n_classes = int(len(class_order))

    fold_ids = splits.fold_ids
    n_repeats_total = fold_ids.shape[0]
    n_repeats_used = n_repeats_total if max_repeats is None else int(min(max_repeats, n_repeats_total))
    fold_ids_used = fold_ids[:n_repeats_used]

    if X.shape[0] != y_train.shape[0]:
        raise ValueError(f"{ds}::{view}: n_samples mismatch (X={X.shape[0]} vs y={y_train.shape[0]})")
    if not np.isfinite(X).all():
        raise ValueError(f"{ds}::{view}: non-finite values found (NaN/inf). Fix bundle normalization first.")

    view_entry: Dict[str, object] = {
        "analysis_role": VIEW_REGISTRY[ds]["analysis_role"],
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "splits_path": str(splits_path),
        "splits_sha256": sha256_file(splits_path),
        "n_samples": int(y_train.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "n_splits": int(splits.info.get("n_splits", int(fold_ids_used.max()) + 1)),
        "n_repeats_used": int(n_repeats_used),
        "label_meta": splits.info.get("label_meta", {}),
        "group_meta": splits.info.get("group_meta", {}),
        "models": {},
    }
    if ds == "tcga_gbm":
        view_entry["tcga_gbm_note"] = TCGA_GBM_NOTE

    def _ctor_xgb(seed: int, n_classes: int):
        return make_xgb_classifier(
            seed=seed,
            n_classes=n_classes,
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            subsample=xgb_params["subsample"],
            colsample_bytree=xgb_params["colsample_bytree"],
            reg_lambda=xgb_params["reg_lambda"],
        )

    def _ctor_rf(seed: int, n_classes: int):
        return make_rf_classifier(
            seed=seed,
            n_estimators=rf_params["n_estimators"],
            max_features=rf_params["max_features"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            max_depth=rf_params["max_depth"],
        )

    model_map = {
        "xgb": ("xgb", _ctor_xgb),
        "rf": ("rf", _ctor_rf),
    }

    for mkey in models_to_run:
        if mkey not in model_map:
            raise ValueError(f"Unknown model '{mkey}'. Supported: {', '.join(sorted(model_map))}")

        short_name, ctor = model_map[mkey]
        metrics_json_path = out_metrics_dir / f"metrics__{ds}__{view}__{short_name}.json"
        if metrics_json_path.exists():
            print(f"Skipping (exists): {ds} :: {view} :: {short_name}")
            continue
        t0 = time.time()

        oof_pred, oof_proba, fold_ids_echo, per_repeat_metrics, shap_per_repeat, meta = train_oof_and_shap(
            model_name=short_name,
            model_ctor=ctor,
            X=X,
            y=y_train,
            class_order=class_order,
            fold_ids_used=fold_ids_used,
            seed=int(seed),
            n_classes=n_classes,
            compute_shap=compute_shap,
            shap_background=int(shap_background),
            shap_max_test=None if shap_max_test is None else int(shap_max_test),
            shap_check_additivity=bool(shap_check_additivity),
            xgb_balance=bool(xgb_balance),
        )
        elapsed = float(time.time() - t0)

        preds_npz_path = out_models_dir / f"preds__{ds}__{view}__{short_name}.npz"
        np.savez_compressed(
            preds_npz_path,
            y=y_train.astype(np.int32),
            sample_ids=splits.sample_ids.astype(str),
            fold_ids=fold_ids_echo.astype(np.int16),
            class_order=class_order.astype(np.int32),
            oof_pred=oof_pred.astype(np.int32),
            oof_proba=oof_proba.astype(np.float32),
        )

        shap_npz_path = None
        importance_csv_path = None
        shap_summary = None

        if compute_shap and shap_per_repeat is not None:
            mean_abs_shap = np.mean(shap_per_repeat, axis=0).astype(np.float32)

            shap_meta = {
                "model": short_name,
                "shap_background": int(shap_background),
                "shap_max_test": None if shap_max_test is None else int(shap_max_test),
                "shap_check_additivity": bool(shap_check_additivity),
                "timestamp": now_iso(),
                "training_seconds_total": elapsed,
                "meta": meta,
            }
            shap_npz_path = out_importance_dir / f"shap__{ds}__{view}__{short_name}.npz"
            np.savez_compressed(
                shap_npz_path,
                feature_names=feat_names.astype(str),
                mean_abs_shap_per_repeat=shap_per_repeat.astype(np.float32),
                mean_abs_shap=mean_abs_shap.astype(np.float32),
                shap_meta=json.dumps(shap_meta),
            )

            df = pd.DataFrame(
                {
                    "feature": feat_names.astype(str),
                    "importance": mean_abs_shap.astype(np.float64),
                }
            )
            df["rank"] = df["importance"].rank(ascending=False, method="min").astype(int)
            df = df.sort_values("rank")
            df["percentile"] = 100.0 * (1.0 - (df["rank"] - 1) / max(1, (len(df) - 1)))
            importance_csv_path = out_importance_dir / f"prediction_importance__{ds}__{view}__{short_name}.csv.gz"
            df.to_csv(importance_csv_path, index=False, compression="gzip")

            shap_summary = {
                "shap_npz": str(shap_npz_path),
                "importance_csv": str(importance_csv_path),
                "top10_features": df.head(10)["feature"].tolist(),
            }

        metrics_json_path = out_metrics_dir / f"metrics__{ds}__{view}__{short_name}.json"
        metrics_payload: Dict[str, object] = {
            "dataset": ds,
            "view": view,
            "model": short_name,
            "analysis_role": VIEW_REGISTRY[ds]["analysis_role"],
            "n_samples": int(y_train.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(n_classes),
            "cv_config": {
                "n_splits": int(splits.info.get("n_splits", int(fold_ids_used.max()) + 1)),
                "n_repeats": int(n_repeats_used),
                "group_aware": bool(splits.info.get("group_meta", {}).get("group_mode", "") == "ok"),
            },
            "bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
            "splits": {
                "path": str(splits_path),
                "sha256": sha256_file(splits_path),
                "label_meta": splits.info.get("label_meta", {}),
                "group_meta": splits.info.get("group_meta", {}),
            },
            "performance": {
                "per_repeat": per_repeat_metrics,
                "summary": summarize_metrics(per_repeat_metrics),
            },
            "shap": shap_summary,
            "timing": {"seconds_total": elapsed},
            "timestamp": now_iso(),
        }
        if ds == "tcga_gbm":
            metrics_payload["tcga_gbm_note"] = TCGA_GBM_NOTE

        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        view_entry["models"][short_name] = {
            "preds_npz": str(preds_npz_path),
            "metrics_json": str(metrics_json_path),
            "shap_npz": None if shap_npz_path is None else str(shap_npz_path),
            "importance_csv": None if importance_csv_path is None else str(importance_csv_path),
            "seconds_total": elapsed,
        }

    return ds, view, view_entry


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Progress bar for joblib. Updates when each batch of jobs completes."""
    old_cb = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(old_cb):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tree models and compute held-out SHAP importance using saved CV splits.")
    parser.add_argument("--dataset", type=str, default="all", help=f"Dataset name or 'all'. Options: {', '.join(sorted(VIEW_REGISTRY))}")
    parser.add_argument("--views", type=str, default="core", help="Which views to train: core | all | sensitivity")
    parser.add_argument("--only-view", type=str, default=None, help="If provided, only process this specific view (filters after --views)")
    parser.add_argument("--models", type=str, default="xgb", help="Comma-separated list: xgb,rf (default: xgb)")
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root")
    parser.add_argument("--bundle-dir", type=str, default="outputs/bundles", help="Directory containing *_bundle_normalized.npz")
    parser.add_argument("--splits-dir", type=str, default="outputs/splits", help="Directory containing splits__{dataset}.npz")
    parser.add_argument("--out-models-dir", type=str, default=None, help="Output directory for OOF predictions (default: outputs/03_supervised/tree_models_xgb_{bal|unbal}/models)")
    parser.add_argument("--out-metrics-dir", type=str, default=None, help="Output directory for metrics JSONs + manifest (default: outputs/03_supervised/tree_models_xgb_{bal|unbal}/metrics)")
    parser.add_argument("--out-importance-dir", type=str, default=None, help="Output directory for SHAP + importance CSVs (default: outputs/03_supervised/tree_models_xgb_{bal|unbal}/importance)")

    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--max-repeats", type=int, default=None, help="Optionally cap number of repeats used (fast debug)")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs across dataset×view tasks (default: min(cpu, 24, #tasks))")
    parser.add_argument("--sequential", action="store_true", help="Disable parallel execution (debug / low-RAM mode)")

    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP (train + OOF metrics only)")
    parser.add_argument("--shap-background", type=int, default=200, help="Max background rows from train fold for SHAP explainer (default: 200)")
    parser.add_argument("--shap-max-test", type=int, default=None, help="Optionally cap held-out rows used for SHAP per fold (speed)")
    parser.add_argument("--shap-check-additivity", action="store_true", help="Enable SHAP additivity check (slower; can error for multiclass)")

    parser.add_argument("--xgb-n-estimators", type=int, default=400)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument(
        "--xgb-balance",
        action="store_true",
        help="Apply balanced sample weights for XGBoost (like sklearn class_weight='balanced'), per fold."
    )

    parser.add_argument("--rf-n-estimators", type=int, default=600)
    parser.add_argument("--rf-max-features", type=str, default="sqrt", help="sqrt | log2 | float fraction")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-max-depth", type=int, default=None)

    args = parser.parse_args()

    # Ensure tqdm is enabled even in classic Windows terminals
    os.environ["TQDM_DISABLE"] = "0"

    # Set default output directories based on --xgb-balance flag
    if args.out_models_dir is None:
        suffix = "xgb_bal" if args.xgb_balance else "xgb_unbal"
        args.out_models_dir = f"outputs/03_supervised/tree_models_{suffix}/models"
    if args.out_metrics_dir is None:
        suffix = "xgb_bal" if args.xgb_balance else "xgb_unbal"
        args.out_metrics_dir = f"outputs/03_supervised/tree_models_{suffix}/metrics"
    if args.out_importance_dir is None:
        suffix = "xgb_bal" if args.xgb_balance else "xgb_unbal"
        args.out_importance_dir = f"outputs/03_supervised/tree_models_{suffix}/importance"

    repo_root = Path(args.repo_root).resolve()
    bundle_dir = (repo_root / args.bundle_dir).resolve() if not Path(args.bundle_dir).is_absolute() else Path(args.bundle_dir)
    splits_dir = (repo_root / args.splits_dir).resolve() if not Path(args.splits_dir).is_absolute() else Path(args.splits_dir)
    out_models_dir = (repo_root / args.out_models_dir).resolve() if not Path(args.out_models_dir).is_absolute() else Path(args.out_models_dir)
    out_metrics_dir = (repo_root / args.out_metrics_dir).resolve() if not Path(args.out_metrics_dir).is_absolute() else Path(args.out_metrics_dir)
    out_importance_dir = (repo_root / args.out_importance_dir).resolve() if not Path(args.out_importance_dir).is_absolute() else Path(args.out_importance_dir)

    ensure_dir(out_models_dir)
    ensure_dir(out_metrics_dir)
    ensure_dir(out_importance_dir)

    datasets = sorted(VIEW_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    compute_shap = not bool(args.no_shap)

    xgb_params = {
        "n_estimators": int(args.xgb_n_estimators),
        "max_depth": int(args.xgb_max_depth),
        "learning_rate": float(args.xgb_learning_rate),
        "subsample": float(args.xgb_subsample),
        "colsample_bytree": float(args.xgb_colsample_bytree),
        "reg_lambda": float(args.xgb_reg_lambda),
    }
    rf_max_depth = None if args.rf_max_depth is None else int(args.rf_max_depth)
    rf_params = {
        "n_estimators": int(args.rf_n_estimators),
        "max_features": args.rf_max_features,
        "min_samples_leaf": int(args.rf_min_samples_leaf),
        "max_depth": rf_max_depth,
    }

    print("=" * 80)
    print("PHASE 3 — TRAIN TREE MODELS + SHAP")
    print("=" * 80)
    print(f"Repo root:       {repo_root}")
    print(f"Bundle dir:      {bundle_dir}")
    print(f"Splits dir:      {splits_dir}")
    print(f"Out models:      {out_models_dir}")
    print(f"Out metrics:     {out_metrics_dir}")
    print(f"Out importance:  {out_importance_dir}")
    print(f"Views mode:      {args.views}")
    print(f"Models:          {models_to_run}")
    print(f"Compute SHAP:    {compute_shap}")
    if compute_shap:
        print(f"  shap_background={args.shap_background}  shap_max_test={args.shap_max_test}  additivity_check={bool(args.shap_check_additivity)}")
    print()

    tasks: List[Tuple[str, str]] = []
    for ds in datasets:
        bundle_path = bundle_dir / f"{ds}_bundle_normalized.npz"
        splits_path = splits_dir / f"splits__{ds}.npz"
        if not bundle_path.exists():
            print("-" * 80)
            print(f"Dataset: {ds}")
            print(f"  Missing bundle: {bundle_path}")
            continue
        if not splits_path.exists():
            print("-" * 80)
            print(f"Dataset: {ds}")
            print(f"  Missing splits: {splits_path}")
            continue

        ds_views = resolve_views(ds, args.views)
        for view in ds_views:
            if args.only_view is not None and view != args.only_view:
                continue
            tasks.append((ds, view))

    if len(tasks) == 0:
        print("No dataset×view tasks to run. Exiting.")
        return

    cpu = os.cpu_count() or 1
    default_jobs = min(cpu, 24, len(tasks))
    n_jobs = default_jobs if args.n_jobs is None else int(max(1, min(args.n_jobs, cpu, len(tasks))))

    print("-" * 80)
    print(f"Tasks: {len(tasks)} dataset×view jobs")
    print(f"Parallel: {'OFF' if args.sequential else 'ON'}  (n_jobs={1 if args.sequential else n_jobs})")
    print("Note: models are single-threaded (n_jobs=1) inside each worker to avoid oversubscription.")
    print("-" * 80)

    manifest: Dict[str, object] = {
        "created_at": now_iso(),
        "repo_root": str(repo_root),
        "script": "03_supervised/03_train_tree_models.py",
        "views_mode": args.views,
        "models": models_to_run,
        "compute_shap": compute_shap,
        "xgb_params": xgb_params,
        "rf_params": rf_params,
        "datasets": {},
    }

    results: List[Tuple[str, str, Dict[str, object]]] = []
    errors: List[str] = []

    if args.sequential or n_jobs == 1:
        for ds, view in tasks:
            try:
                print(f"Running: {ds} :: {view}")
                res = process_dataset_view_task(
                    ds=ds,
                    view=view,
                    repo_root=repo_root,
                    bundle_dir=bundle_dir,
                    splits_dir=splits_dir,
                    out_models_dir=out_models_dir,
                    out_metrics_dir=out_metrics_dir,
                    out_importance_dir=out_importance_dir,
                    models_to_run=models_to_run,
                    seed=int(args.seed),
                    max_repeats=args.max_repeats,
                    compute_shap=compute_shap,
                    shap_background=int(args.shap_background),
                    shap_max_test=args.shap_max_test,
                    shap_check_additivity=bool(args.shap_check_additivity),
                    xgb_params=xgb_params,
                    rf_params=rf_params,
                    xgb_balance=bool(args.xgb_balance),
                )
                results.append(res)
            except Exception as e:
                errors.append(f"{ds}::{view}: {e}")
    else:
        def _run(ds: str, view: str):
            return process_dataset_view_task(
                ds=ds,
                view=view,
                repo_root=repo_root,
                bundle_dir=bundle_dir,
                splits_dir=splits_dir,
                out_models_dir=out_models_dir,
                out_metrics_dir=out_metrics_dir,
                out_importance_dir=out_importance_dir,
                models_to_run=models_to_run,
                seed=int(args.seed),
                max_repeats=args.max_repeats,
                compute_shap=compute_shap,
                shap_background=int(args.shap_background),
                shap_max_test=args.shap_max_test,
                shap_check_additivity=bool(args.shap_check_additivity),
                xgb_params=xgb_params,
                rf_params=rf_params,
                xgb_balance=bool(args.xgb_balance),
            )

        with tqdm_joblib(tqdm(total=len(tasks), desc="tree jobs (dataset×view)")):
            out = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                delayed(_run)(ds, view) for (ds, view) in tasks
            )
        results.extend(out)

    for ds, view, view_entry in results:
        if ds not in manifest["datasets"]:
            manifest["datasets"][ds] = {"views": {}}
        manifest["datasets"][ds]["views"][view] = view_entry

    if errors:
        print("-" * 80)
        print("Some tasks failed:")
        for msg in errors:
            print(f"  FAILED: {msg}")
        print("Re-run with --sequential (and optionally --dataset <name>) to debug.")

    manifest_path = out_metrics_dir / "tree_models_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print(f"Manifest written: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
