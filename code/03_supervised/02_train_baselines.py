#!/usr/bin/env python3
"""
PHASE 3 — 02_train_baselines.py

Train simple, reviewer-proof baseline classifiers per dataset × view using
the deterministic CV splits produced by 03_supervised/01_define_tasks_and_splits.py.

Design goals
- No pickles (all outputs are .npz + .json)
- Strictly out-of-fold (OOF) evaluation; metrics computed on pooled OOF per repeat
- Deterministic given the saved split artefacts
- Lightweight baselines intended for sanity checks and performance context

Outputs
- outputs/models/baselines/preds__{dataset}__{view}.npz
    Contains y, sample_ids, fold_ids, class_order, and per-model OOF predictions per repeat
- outputs/metrics/baselines/metrics__{dataset}__{view}.json
    Contains per-repeat pooled OOF metrics + summary stats (mean/std/95% CI)

Author: variance-prediction paradox pipeline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

# -----------------------------
# View registry (must match bundle reality)
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

TCGA_GBM_NOTE = "N=47; AUROC high-variance — treat as sensitivity analysis"


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
    # allow_pickle MUST remain False for reviewer-proof reproducibility
    return np.load(str(path), allow_pickle=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_sorted_int(y: np.ndarray) -> np.ndarray:
    return np.array(sorted(np.unique(y).tolist()), dtype=np.int32)


def ci95(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(values, 2.5))
    hi = float(np.percentile(values, 97.5))
    return (lo, hi)


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


def load_bundle(bundle_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    z = load_npz_pickle_free(bundle_path)
    y = z["y"].astype(np.int32)
    sample_ids = z["sample_ids"].astype(str)
    info = json.loads(str(z["info"]))

    X_views: Dict[str, np.ndarray] = {}
    feature_names: Dict[str, np.ndarray] = {}
    for key in z.files:
        if key.startswith("X_"):
            view = key[2:]
            X_views[view] = z[key].astype(np.float32)
        elif key.startswith("features_"):
            view = key[len("features_") :]
            feature_names[view] = z[key].astype(str)
    return y, sample_ids, X_views, feature_names, info


def assert_alignment(bundle_y: np.ndarray, bundle_ids: np.ndarray, split_y: np.ndarray, split_ids: np.ndarray) -> None:
    if bundle_y.shape != split_y.shape or bundle_ids.shape != split_ids.shape:
        raise ValueError("Bundle and split shapes mismatch.")
    if not np.array_equal(bundle_ids, split_ids):
        # If this ever happens, we'd need to reindex consistently.
        raise ValueError("sample_ids in bundle and splits do not match (order mismatch).")
    if not np.array_equal(bundle_y, split_y):
        # CCLE label collapsing means split_y may differ from bundle_y.
        # That is OK only if y_raw matches bundle_y.
        pass


def make_baseline_models(seed: int, scale_linear: bool) -> Dict[str, object]:
    models: Dict[str, object] = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=seed),
    }

    scaler = StandardScaler(with_mean=True, with_std=True) if scale_linear else "passthrough"

    # Note: class_weight='balanced' improves robustness under imbalance (esp. CCLE)
    models["logistic_l2"] = Pipeline(
        steps=[
            ("scaler", scaler),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="saga",              # robust in high-d; supports multinomial
                multi_class="multinomial",
                class_weight="balanced",
                max_iter=2000,
                tol=1e-3,
                random_state=seed,
                n_jobs=-1,
            )),
        ]
    )

    models["ridge"] = Pipeline(
        steps=[
            ("scaler", scaler),
            ("clf", RidgeClassifier(alpha=1.0)),
        ]
    )

    models["linear_svm"] = Pipeline(
        steps=[
            ("scaler", scaler),
            ("clf", LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            )),
        ]
    )
    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train_oof_for_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    seed: int,
    model_name: str,
) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, object]]:
    """
    Returns:
      oof_pred: (n_repeats, n_samples) int32
      per_repeat_metrics: list length n_repeats
      meta: includes convergence warnings count etc.
    """
    n_repeats, n_samples = fold_ids.shape
    oof_pred = np.empty((n_repeats, n_samples), dtype=np.int32)
    per_repeat: List[Dict[str, float]] = []
    warn_counts = {"convergence_warnings": 0, "other_warnings": 0}

    # Ensure deterministic numpy RNG is not implicitly used by sklearn models (we set random_state in models where relevant)
    for r in range(n_repeats):
        y_pred_r = np.full(n_samples, -1, dtype=np.int32)
        for k in range(int(fold_ids[r].max()) + 1):
            test_idx = np.where(fold_ids[r] == k)[0]
            train_idx = np.where(fold_ids[r] != k)[0]
            if test_idx.size == 0 or train_idx.size == 0:
                raise RuntimeError(f"Empty fold encountered: repeat={r}, fold={k}")

            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_te = X[test_idx]

            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)

                # Warning accounting (lightweight provenance)
                for w in wlist:
                    msg = str(w.message).lower()
                    if "converg" in msg:
                        warn_counts["convergence_warnings"] += 1
                    else:
                        warn_counts["other_warnings"] += 1

            y_pred_r[test_idx] = pred.astype(np.int32)

        if np.any(y_pred_r < 0):
            raise RuntimeError(f"OOF prediction incomplete for model={model_name}, repeat={r}")

        oof_pred[r] = y_pred_r
        per_repeat.append(compute_metrics(y, y_pred_r))

    meta = {"warning_counts": warn_counts}
    return oof_pred, per_repeat, meta


def summarize_metrics(per_repeat: List[Dict[str, float]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if len(per_repeat) == 0:
        return out

    keys = list(per_repeat[0].keys())
    for m in keys:
        vals = [float(d[m]) for d in per_repeat]
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        lo, hi = ci95(vals)
        out[m] = {
            "mean": mu,
            "std": sd,
            "ci95": [lo, hi],
            "per_repeat": vals,
        }
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models using saved CV splits (OOF pooled per repeat).")
    parser.add_argument("--dataset", type=str, default="all", help=f"Dataset name or 'all'. Options: {', '.join(sorted(VIEW_REGISTRY))}")
    parser.add_argument("--views", type=str, default="core", help="Which views to train: core | all | sensitivity")
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root")
    parser.add_argument("--bundle-dir", type=str, default="outputs/bundles", help="Directory containing *_bundle_normalized.npz")
    parser.add_argument("--splits-dir", type=str, default="outputs/splits", help="Directory containing splits__{dataset}.npz")
    parser.add_argument("--out-models-dir", type=str, default="outputs/models/baselines", help="Output directory for baseline predictions")
    parser.add_argument("--out-metrics-dir", type=str, default="outputs/metrics/baselines", help="Output directory for baseline metrics JSONs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic baselines")
    parser.add_argument("--max-repeats", type=int, default=None, help="Optionally cap number of repeats used (for fast debug)")
    parser.add_argument("--no-scale", action="store_true", help="Disable StandardScaler for linear baselines")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    bundle_dir = (repo_root / args.bundle_dir).resolve() if not Path(args.bundle_dir).is_absolute() else Path(args.bundle_dir)
    splits_dir = (repo_root / args.splits_dir).resolve() if not Path(args.splits_dir).is_absolute() else Path(args.splits_dir)
    out_models_dir = (repo_root / args.out_models_dir).resolve() if not Path(args.out_models_dir).is_absolute() else Path(args.out_models_dir)
    out_metrics_dir = (repo_root / args.out_metrics_dir).resolve() if not Path(args.out_metrics_dir).is_absolute() else Path(args.out_metrics_dir)

    ensure_dir(out_models_dir)
    ensure_dir(out_metrics_dir)

    datasets = sorted(VIEW_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    scale_linear = not args.no_scale

    print("=" * 80)
    print("PHASE 3 — TRAIN BASELINES")
    print("=" * 80)
    print(f"Repo root:     {repo_root}")
    print(f"Bundle dir:    {bundle_dir}")
    print(f"Splits dir:    {splits_dir}")
    print(f"Out models:    {out_models_dir}")
    print(f"Out metrics:   {out_metrics_dir}")
    print(f"Views mode:    {args.views}")
    print(f"Scale linear:  {scale_linear}")
    print()

    manifest: Dict[str, object] = {
        "created_at": now_iso(),
        "repo_root": str(repo_root),
        "datasets": {},
        "script": "03_supervised/02_train_baselines.py",
    }

    for ds in datasets:
        print("-" * 80)
        print(f"Dataset: {ds}")

        bundle_path = bundle_dir / f"{ds}_bundle_normalized.npz"
        splits_path = splits_dir / f"splits__{ds}.npz"

        if not bundle_path.exists():
            print(f"  Missing bundle: {bundle_path}")
            continue
        if not splits_path.exists():
            print(f"  Missing splits: {splits_path}")
            continue

        # Load inputs
        bundle_y, bundle_ids, X_views, feat_names, bundle_info = load_bundle(bundle_path)
        splits = load_splits(splits_path)

        # Alignment checks
        try:
            assert_alignment(bundle_y, bundle_ids, splits.y if splits.y_raw is None else splits.y_raw, splits.sample_ids)
        except Exception as e:
            print(f"  FAILED alignment: {e}")
            continue

        # Training labels for baselines must match split y (which may be transformed for CCLE)
        y_train = splits.y.astype(np.int32)
        class_order = unique_sorted_int(y_train)

        fold_ids = splits.fold_ids
        n_repeats_total = fold_ids.shape[0]
        n_repeats_used = n_repeats_total if args.max_repeats is None else int(min(args.max_repeats, n_repeats_total))
        fold_ids_used = fold_ids[:n_repeats_used]

        ds_views = resolve_views(ds, args.views)
        missing_views = [v for v in ds_views if f"X_{v}" not in [f"X_{k}" for k in X_views.keys()] and v not in X_views]
        # (X_views keys are already view names)
        missing_views = [v for v in ds_views if v not in X_views]
        if missing_views:
            print(f"  WARNING: missing views in bundle (skipping): {missing_views}")

        models = make_baseline_models(seed=int(args.seed), scale_linear=scale_linear)

        ds_entry: Dict[str, object] = {
            "bundle_path": str(bundle_path),
            "bundle_sha256": sha256_file(bundle_path),
            "splits_path": str(splits_path),
            "splits_sha256": sha256_file(splits_path),
            "n_samples": int(y_train.shape[0]),
            "n_classes": int(len(class_order)),
            "n_repeats_used": int(n_repeats_used),
            "n_splits": int(splits.info.get("n_splits", int(fold_ids_used.max()) + 1)),
            "group_meta": splits.info.get("group_meta", {}),
            "label_meta": splits.info.get("label_meta", {}),
            "views": {},
        }
        if ds == "tcga_gbm":
            ds_entry["tcga_gbm_note"] = TCGA_GBM_NOTE

        for view in ds_views:
            if view not in X_views:
                continue

            X = X_views[view]
            if X.shape[0] != y_train.shape[0]:
                print(f"  FAILED view {view}: n_samples mismatch (X={X.shape[0]} vs y={y_train.shape[0]})")
                continue
            if not np.isfinite(X).all():
                print(f"  FAILED view {view}: non-finite values found (NaN/inf). Fix bundle normalization first.")
                continue

            print(f"  View: {view}  X={tuple(X.shape)}  features={X.shape[1]}")

            # Train each model OOF
            preds_npz_path = out_models_dir / f"preds__{ds}__{view}.npz"
            metrics_json_path = out_metrics_dir / f"metrics__{ds}__{view}.json"

            preds_payload: Dict[str, object] = {
                "y": y_train.astype(np.int32),
                "sample_ids": splits.sample_ids.astype(str),
                "fold_ids": fold_ids_used.astype(np.int16),
                "class_order": class_order.astype(np.int32),
            }

            metrics_payload: Dict[str, object] = {
                "dataset": ds,
                "view": view,
                "analysis_role": VIEW_REGISTRY[ds]["analysis_role"],
                "n_samples": int(y_train.shape[0]),
                "n_features": int(X.shape[1]),
                "n_classes": int(len(class_order)),
                "cv_config": {
                    "n_splits": int(ds_entry["n_splits"]),
                    "n_repeats": int(n_repeats_used),
                    "group_aware": bool(splits.info.get("group_meta", {}).get("group_mode", "") == "ok"),
                },
                "bundle": {
                    "path": str(bundle_path),
                    "sha256": ds_entry["bundle_sha256"],
                },
                "splits": {
                    "path": str(splits_path),
                    "sha256": ds_entry["splits_sha256"],
                    "label_meta": ds_entry.get("label_meta", {}),
                    "group_meta": ds_entry.get("group_meta", {}),
                },
                "models": {},
                "timestamp": now_iso(),
            }
            if ds == "tcga_gbm":
                metrics_payload["tcga_gbm_note"] = TCGA_GBM_NOTE

            for model_name, model in models.items():
                # Fresh clone per model to avoid cross-fit state issues
                # (Pipelines are safe to reuse if refit each time, but cloning avoids edge cases)
                from sklearn.base import clone
                m = clone(model)

                oof_pred, per_repeat, meta = train_oof_for_model(
                    m, X, y_train, fold_ids_used, seed=int(args.seed), model_name=model_name
                )

                preds_payload[f"pred__{model_name}"] = oof_pred.astype(np.int32)
                metrics_payload["models"][model_name] = {
                    "summary": summarize_metrics(per_repeat),
                    "meta": meta,
                }

            # Save outputs (pickle-free)
            np.savez_compressed(preds_npz_path, **preds_payload)
            with open(metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2)

            ds_entry["views"][view] = {
                "preds_npz": str(preds_npz_path),
                "metrics_json": str(metrics_json_path),
                "n_features": int(X.shape[1]),
            }

            print(f"    saved preds:   {preds_npz_path}")
            print(f"    saved metrics: {metrics_json_path}")

        manifest["datasets"][ds] = ds_entry

    manifest_path = out_metrics_dir / "baselines_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print(f"Manifest written: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
