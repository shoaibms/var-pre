#!/usr/bin/env python
"""
Feature subset ablation study.

Compares classification performance across feature-selection strategies
(ALL features, variance-topK, SHAP-topK, random-topK) using repeated
stratified cross-validation. Supports per-view and global scheduling
with optional parallelism via joblib.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import sklearn
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb  # noqa
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    xgb = None
    XGBClassifier = None
    XGBRegressor = None


# -----------------------------
# Defaults / hero views
# -----------------------------
HERO_VIEWS = [
    ("mlomics", "methylation"),
    ("ibdmdb", "MGX"),
    ("ccle", "mRNA"),
]

DATASET_VIEWS_REGISTRY: Dict[str, Dict[str, List[str]]] = {
    "mlomics": {"core": ["mRNA", "miRNA", "methylation", "CNV"], "sensitivity": []},
    "ibdmdb": {"core": ["MGX", "MGX_func", "MPX", "MBX"], "sensitivity": ["MGX_CLR"]},
    "ccle": {"core": ["mRNA", "CNV", "proteomics"], "sensitivity": []},
    "tcga_gbm": {"core": ["mRNA", "methylation", "CNV"], "sensitivity": []},
}


# -----------------------------
# Dataclasses
# -----------------------------
@dataclass(frozen=True)
class SplitData:
    dataset: str
    y: np.ndarray
    y_raw: Optional[np.ndarray]
    sample_ids: np.ndarray
    groups: np.ndarray
    fold_ids: np.ndarray  # (n_repeats, n_samples)
    info: Dict[str, object]


# -----------------------------
# Utilities
# -----------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        df.to_csv(f, index=False)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def append_runlog(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")


def _prefer_paths(paths: List[Path]) -> List[Path]:
    def score(p: Path) -> Tuple[int, int, int]:
        s = str(p).lower()
        is_archive = int(("_archive" in s) or ("archive_pre_migration" in s) or ("\\archive" in s) or ("/archive" in s))
        is_01_bundles = int(("01_bundles" in s))
        depth = len(p.parts)
        return (is_archive, -is_01_bundles, depth)

    return sorted(paths, key=score)


def stable_int_seed(*parts: object) -> int:
    s = "|".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def jaccard_idx(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    den = len(sa | sb)
    return 0.0 if den == 0 else len(sa & sb) / den


def balanced_accuracy_no_warn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Same balanced accuracy as sklearn, but avoids the warning by
    restricting labels to classes present in y_true.
    """
    labels = np.unique(y_true)
    C = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C) / C.sum(axis=1)
    per_class = per_class[~np.isnan(per_class)]
    return float(np.mean(per_class)) if per_class.size else float("nan")


# -----------------------------
# IO + discovery
# -----------------------------
def load_npz_pickle_free(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=False)


def find_splits_npz(outputs_dir: Path, dataset: str) -> Path:
    cands = list(outputs_dir.glob(f"**/splits__{dataset}.npz"))
    if not cands:
        cands = list(outputs_dir.glob(f"**/splits/splits__{dataset}.npz"))
    if not cands:
        raise FileNotFoundError(f"Could not find splits__{dataset}.npz under {outputs_dir}")
    return _prefer_paths(cands)[0]


def find_bundle_npz(outputs_dir: Path, dataset: str) -> Path:
    pats = [
        f"**/{dataset}*bundle*normalized*.npz",
        f"**/{dataset}*normalized*bundle*.npz",
        f"**/{dataset}*bundle*.npz",
    ]
    cands: List[Path] = []
    for pat in pats:
        cands.extend(outputs_dir.glob(pat))
    if not cands:
        raise FileNotFoundError(f"Could not find bundle npz for dataset={dataset} under {outputs_dir}")
    cands = _prefer_paths(cands)
    cands_norm = [p for p in cands if "normalized" in p.name.lower()]
    return (cands_norm[0] if cands_norm else cands[0])


def load_splits(splits_path: Path) -> SplitData:
    z = load_npz_pickle_free(splits_path)
    info = json.loads(str(z["info"])) if "info" in z.files else {}
    y = z["y"]
    y_raw = z["y_raw"] if "y_raw" in z.files else None
    sample_ids = z["sample_ids"].astype(str)
    groups = z["groups"].astype(str) if "groups" in z.files else np.array(["NA"] * len(sample_ids), dtype=str)
    fold_ids = z["fold_ids"].astype(np.int16)
    dataset_name = str(info.get("dataset", splits_path.stem.replace("splits__", "")))
    return SplitData(dataset=dataset_name, y=y, y_raw=y_raw, sample_ids=sample_ids, groups=groups, fold_ids=fold_ids, info=info)


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
            raise KeyError(f"Bundle missing view '{view}' (expected key '{x_key}'). Available X_ keys: {[k for k in z.files if k.startswith('X_')]}")

    X = z[x_key].astype(np.float32)
    f_key1 = f"features_{view}"
    f_key2 = f"feature_names_{view}"
    if f_key1 in z.files:
        feats = z[f_key1].astype(str)
    elif f_key2 in z.files:
        feats = z[f_key2].astype(str)
    else:
        feats = np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)

    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)

    if y is None or sample_ids is None:
        raise KeyError(f"Bundle missing required keys 'y' and/or 'sample_ids': {bundle_path}")

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


def discover_views(outputs_dir: Path, dataset: str) -> List[str]:
    vs_dir = outputs_dir / "02_unsupervised" / "variance_scores"
    views = set()
    if vs_dir.exists():
        for p in vs_dir.glob(f"variance_scores__{dataset}__*.csv.gz"):
            m = re.match(rf"variance_scores__{re.escape(dataset)}__(.+)\.csv\.gz$", p.name)
            if m:
                views.add(m.group(1))
    if views:
        return sorted(views)
    reg = DATASET_VIEWS_REGISTRY.get(dataset, {})
    return sorted(list(dict.fromkeys(reg.get("core", []) + reg.get("sensitivity", []))))


def resolve_views(dataset: str, mode: str, discovered: List[str]) -> List[str]:
    reg = DATASET_VIEWS_REGISTRY.get(dataset, {})
    core = reg.get("core", [])
    sens = reg.get("sensitivity", [])
    if mode == "all":
        return discovered
    if mode == "core":
        return core if core else discovered
    if mode == "sensitivity":
        return sens if sens else []
    raise ValueError(f"Unknown views mode: {mode}")


def find_variance_scores(outputs_dir: Path, dataset: str, view: str) -> Path:
    p = outputs_dir / "02_unsupervised" / "variance_scores" / f"variance_scores__{dataset}__{view}.csv.gz"
    if p.exists():
        return p
    cands = list(outputs_dir.glob(f"**/variance_scores__{dataset}__{view}.csv.gz"))
    if not cands:
        raise FileNotFoundError(f"Missing variance_scores__{dataset}__{view}.csv.gz under {outputs_dir}")
    return _prefer_paths(cands)[0]


def find_shap_importance(outputs_dir: Path, dataset: str, view: str, model: str) -> Path:
    p = outputs_dir / "04_importance" / "per_model" / f"importance__{dataset}__{view}__{model}.csv.gz"
    if p.exists():
        return p
    cands = list(outputs_dir.glob(f"**/prediction_importance__{dataset}__{view}__*.csv.gz"))
    if not cands:
        raise FileNotFoundError(
            f"Could not find SHAP importance for {dataset}/{view}/{model}. Expected {p} or prediction_importance__... under {outputs_dir}."
        )
    def score(path: Path) -> Tuple[int, int]:
        s = str(path).lower()
        want = model.lower()
        hit = int(want in s)
        depth = len(path.parts)
        return (-hit, depth)
    cands = sorted(cands, key=score)
    return cands[0]


def infer_task_type(y: np.ndarray) -> str:
    if np.issubdtype(y.dtype, np.floating):
        u = np.unique(y[~np.isnan(y)])
        if len(u) <= 20 and np.all(np.isclose(u, np.round(u))):
            return "classification"
        return "regression"
    return "classification"


# -----------------------------
# Regime lookup (authoritative: di_summary)
# -----------------------------
def _find_di_summary(outputs_dir: Path) -> Optional[Path]:
    cands = [
        outputs_dir / "04_importance" / "uncertainty" / "di_summary.csv",
        outputs_dir / "04_importance" / "uncertainty" / "di_summary.csv.gz",
    ]
    for p in cands:
        if p.exists():
            return p
    found = list(outputs_dir.glob("**/di_summary.csv")) + list(outputs_dir.glob("**/di_summary.csv.gz"))
    if not found:
        return None
    return _prefer_paths(found)[0]


def get_regime_info(outputs_dir: Path, dataset: str, view: str, model: str, k_pct: int = 10) -> dict:
    """
    Attach regime (+ DI summary if available) for a (dataset, view, k_pct).

    Priority:
      1) di_summary exact match on (dataset, view, model, k_pct) if model column exists
      2) di_summary fallback match on (dataset, view, k_pct) ignoring model (prefer xgb_bal)
      3) vp_summary json (try both legacy + joined_vp locations)
      4) joined_vp CSV mode(regime)
      5) UNKNOWN
    """
    dataset = str(dataset).strip()
    view = str(view).strip()
    model = str(model).strip()
    k_pct = int(k_pct)

    def _empty(source: str) -> dict:
        return {
            "regime": "UNKNOWN",
            "regime_confidence": np.nan,
            "regime_k_pct": int(k_pct),
            "DI_mean": np.nan,
            "DI_pctl_2.5": np.nan,
            "DI_pctl_97.5": np.nan,
            "regime_source": source,
        }

    def _warn_once(msg: str) -> None:
        # avoids spam in parallel runs; prints each distinct message once per process
        warned = getattr(get_regime_info, "_warned_msgs", set())
        if msg not in warned:
            print(msg)
            warned.add(msg)
            setattr(get_regime_info, "_warned_msgs", warned)

    def _pack_from_row(r: dict, source: str) -> dict:
        out = _empty(source)

        # regime label: prefer consensus (view-level)
        reg = r.get("consensus_regime", None)
        if reg is None:
            reg = r.get("regime", None)
            if isinstance(reg, dict):
                reg = reg.get("consensus", None) or reg.get(model, None)
        if reg is None:
            reg = "UNKNOWN"
        out["regime"] = str(reg)

        # confidence (if present)
        if "regime_confidence" in r:
            out["regime_confidence"] = float(r.get("regime_confidence"))
        elif "confidence" in r:
            out["regime_confidence"] = float(r.get("confidence"))

        # DI stats (tolerate underscore variants)
        out["DI_mean"] = float(r.get("DI_mean", np.nan))

        lo = r.get("DI_pctl_2.5", r.get("DI_pctl_2_5", np.nan))
        hi = r.get("DI_pctl_97.5", r.get("DI_pctl_97_5", np.nan))
        out["DI_pctl_2.5"] = float(lo) if lo is not None else np.nan
        out["DI_pctl_97.5"] = float(hi) if hi is not None else np.nan

        return out

    # --- 1/2) authoritative: di_summary ---
    di_path = _find_di_summary(outputs_dir)
    if di_path is not None:
        try:
            di = pd.read_csv(di_path)

            # Normalise columns
            di = di.rename(columns={c: str(c).strip() for c in di.columns})
            if "k_pct" not in di.columns and "K_pct" in di.columns:
                di = di.rename(columns={"K_pct": "k_pct"})

            if {"dataset", "view", "k_pct"}.issubset(set(di.columns)):
                ds = di["dataset"].astype(str).str.strip()
                vw = di["view"].astype(str).str.strip()
                kk = pd.to_numeric(di["k_pct"], errors="coerce")

                base = (ds == dataset) & (vw == view) & (kk == k_pct)

                # 1) exact-model match (only if model column exists)
                if "model" in di.columns:
                    mm = di["model"].astype(str).str.strip()
                    sub = di[base & (mm == model)]
                    if len(sub) > 0:
                        r = sub.iloc[0].to_dict()
                        used = str(r.get("model", "")).strip()
                        return _pack_from_row(r, f"di_summary:{di_path.name}:exact_model:{used}")

                    # 2) fallback any model (prefer xgb_bal)
                    sub_any = di[base]
                    if len(sub_any) > 0:
                        prefer = ["xgb_bal", "xgb", "rf"]
                        chosen = sub_any
                        for pm in prefer:
                            hit = sub_any[sub_any["model"].astype(str).str.strip() == pm]
                            if len(hit) > 0:
                                chosen = hit
                                break
                        r = chosen.iloc[0].to_dict()
                        used = str(r.get("model", "")).strip()
                        return _pack_from_row(r, f"di_summary:{di_path.name}:fallback_any_model:{used}")

                else:
                    # model-independent di_summary
                    sub = di[base]
                    if len(sub) > 0:
                        r = sub.iloc[0].to_dict()
                        return _pack_from_row(r, f"di_summary:{di_path.name}:no_model_col")

        except Exception as e:
            _warn_once(f"[WARN] get_regime_info: failed reading/parsing di_summary {di_path}: {e}")

    # --- 3) vp_summary fallback (try both locations) ---
    vp_candidates = [
        outputs_dir / "04_importance" / f"vp_summary__{dataset}__{view}.json",
        outputs_dir / "04_importance" / "joined_vp" / f"vp_summary__{dataset}__{view}.json",
    ]
    for vp_summary in vp_candidates:
        if vp_summary.exists():
            try:
                js = json.loads(vp_summary.read_text())
                out = _empty(f"{vp_summary.name}")

                if isinstance(js, dict):
                    # attempt to fill DI fields if present
                    if "regime_confidence" in js:
                        out["regime_confidence"] = float(js.get("regime_confidence", np.nan))
                    for k in ["DI_mean", "DI_pctl_2.5", "DI_pctl_97.5", "DI_pctl_2_5", "DI_pctl_97_5"]:
                        if k in js:
                            # map underscore keys into dotted keys
                            if k == "DI_pctl_2_5":
                                out["DI_pctl_2.5"] = float(js.get(k, np.nan))
                            elif k == "DI_pctl_97_5":
                                out["DI_pctl_97.5"] = float(js.get(k, np.nan))
                            elif k in out:
                                out[k] = float(js.get(k, np.nan))

                    # regime extraction
                    if "regime" in js:
                        r = js["regime"]
                        if isinstance(r, dict):
                            if r.get("consensus"):
                                out["regime"] = str(r["consensus"])
                                out["regime_source"] = f"{vp_summary.name}:consensus"
                                return out
                            if r.get(model):
                                out["regime"] = str(r[model])
                                out["regime_source"] = f"{vp_summary.name}:model"
                                return out
                        if isinstance(r, str):
                            out["regime"] = r
                            out["regime_source"] = f"{vp_summary.name}:str"
                            return out

                    if js.get("consensus_regime"):
                        out["regime"] = str(js["consensus_regime"])
                        out["regime_source"] = f"{vp_summary.name}:consensus_regime"
                        return out

            except Exception:
                pass

    # --- 4) joined_vp CSV mode(regime) ---
    joined = outputs_dir / "04_importance" / "joined_vp" / f"vp_joined__{dataset}__{view}.csv.gz"
    if joined.exists():
        try:
            df = pd.read_csv(joined)
            if "regime" in df.columns:
                out = _empty("joined_vp:mode")
                out["regime"] = str(df["regime"].dropna().astype(str).mode().iloc[0])
                return out
        except Exception:
            pass

    # --- 5) nothing found ---
    return _empty("none")


# -----------------------------
# Modeling
# -----------------------------
def make_model(model: str, task_type: str, seed: int, n_jobs_model: int, n_classes: int) -> Tuple[object, bool]:
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
            return (
                XGBClassifier(
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
                    **({"num_class": n_classes} if n_classes > 2 else {}),
                ),
                (model == "xgb_bal"),
            )
        raise ValueError(f"Unknown model={model}")
    else:
        if model == "rf":
            return (
                RandomForestRegressor(n_estimators=500, max_features="sqrt", n_jobs=n_jobs_model, random_state=seed),
                False,
            )
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
    if task_type == "classification":
        if metric == "balanced_accuracy":
            return balanced_accuracy_no_warn(y_true, y_pred)
        if metric in ("auroc_ovr_macro", "auroc_macro"):
            if proba is None:
                return float("nan")
            classes_in_y = np.unique(y_true)
            n_classes_proba = proba.shape[1]
            if len(classes_in_y) != n_classes_proba:
                return float("nan")
            if len(classes_in_y) == 2:
                return float(roc_auc_score(y_true, proba[:, 1]))
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        raise ValueError(f"Unsupported classification metric: {metric}")

    raise ValueError(f"Regression metrics not enabled in this script (got metric={metric})")


def train_eval_many(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    task_type: str,
    metrics: List[str],
    seed: int,
    n_classes: int,
) -> Dict[str, float]:
    est, needs_sw = make_model(model=model_name, task_type=task_type, seed=seed, n_jobs_model=1, n_classes=n_classes)

    fit_kwargs = {}
    if needs_sw and task_type == "classification":
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        fit_kwargs["sample_weight"] = sw

    est.fit(X_train, y_train, **fit_kwargs)

    y_pred = est.predict(X_test)
    proba = est.predict_proba(X_test) if (task_type == "classification" and hasattr(est, "predict_proba")) else None

    out = {}
    for m in metrics:
        out[m] = compute_metric(m, task_type, y_test, y_pred, proba)
    return out


# -----------------------------
# Ranking helpers
# -----------------------------
def topk_feature_indices_from_rank_table(
    rank_df: pd.DataFrame,
    feature_col: str,
    rank_col: Optional[str],
    score_col: Optional[str],
    bundle_features: np.ndarray,
    n_select: int,
) -> Tuple[np.ndarray, List[str]]:
    feats_in_bundle = bundle_features.astype(str)
    feat_to_idx: Dict[str, int] = {f: i for i, f in enumerate(feats_in_bundle)}

    df = rank_df.copy()
    if rank_col and rank_col in df.columns:
        df = df.sort_values(rank_col, ascending=True)
    elif score_col and score_col in df.columns:
        df = df.sort_values(score_col, ascending=False)
    else:
        raise ValueError(f"Rank table missing both rank_col={rank_col} and score_col={score_col}")

    selected_idxs: List[int] = []
    selected_names: List[str] = []
    for f in df[feature_col].astype(str).tolist():
        if f in feat_to_idx:
            selected_idxs.append(feat_to_idx[f])
            selected_names.append(f)
        if len(selected_idxs) >= n_select:
            break

    if len(selected_idxs) < n_select:
        raise ValueError(f"Only matched {len(selected_idxs)}/{n_select} features between rank table and bundle features.")

    return np.asarray(selected_idxs, dtype=int), selected_names


# -----------------------------
# Repeat-level stats
# -----------------------------
def repeat_level_stats(vals_by_repeat: np.ndarray) -> dict:
    vals = np.asarray(vals_by_repeat, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    lo, hi = (mean, mean) if n == 1 else np.percentile(vals, [2.5, 97.5]).tolist()
    return {"mean": mean, "std": std, "ci95": [float(lo), float(hi)], "n": n}


# -----------------------------
# Core ablation per view
# -----------------------------
def ablation_for_view(
    outputs_dir: Path,
    dataset: str,
    view: str,
    model: str,
    k_pcts: List[int],
    n_random_draws: int,
    metrics: List[str],
    max_repeats: Optional[int],
    seed: int,
    regime_k_pct: int,
    save_feature_selections: bool,
    fold_n_jobs: int = 1,
    backend: str = "threading",
) -> Tuple[pd.DataFrame, dict]:
    t_start = time.time()

    bundle_path = find_bundle_npz(outputs_dir, dataset)
    splits_path = find_splits_npz(outputs_dir, dataset)
    var_path = find_variance_scores(outputs_dir, dataset, view)
    shap_path = find_shap_importance(outputs_dir, dataset, view, model)

    splits = load_splits(splits_path)
    _, bundle_ids, X, feats, _ = load_bundle_view(bundle_path, view)
    X = align_X_to_splits_order(X, bundle_ids, splits.sample_ids)

    y = np.asarray(splits.y)
    task_type = infer_task_type(y)
    if task_type != "classification":
        raise ValueError(f"This script currently supports classification only. Got task_type={task_type} for {dataset}/{view}")

    classes = np.unique(y)
    n_classes = int(len(classes))

    fold_ids = splits.fold_ids
    if max_repeats is not None:
        fold_ids = fold_ids[: int(max_repeats), :]

    n_repeats = int(fold_ids.shape[0])
    n_features_total = int(X.shape[1])

    var_df = pd.read_csv(var_path)
    shap_df = pd.read_csv(shap_path)

    v_feature_col = "feature" if "feature" in var_df.columns else var_df.columns[0]
    v_rank_col = "v_rank" if "v_rank" in var_df.columns else ("rank" if "rank" in var_df.columns else None)
    v_score_col = "v_score" if "v_score" in var_df.columns else ("score" if "score" in var_df.columns else None)

    p_feature_col = "feature" if "feature" in shap_df.columns else shap_df.columns[0]
    p_rank_col = "p_rank" if "p_rank" in shap_df.columns else ("rank" if "rank" in shap_df.columns else None)
    p_score_col = "p_score" if "p_score" in shap_df.columns else ("importance" if "importance" in shap_df.columns else None)

    var_idx_byK: Dict[int, np.ndarray] = {}
    shap_idx_byK: Dict[int, np.ndarray] = {}
    n_at_K: Dict[int, int] = {}
    selected_hashes: Dict[int, Dict[str, str]] = {}
    jacc_byK: Dict[int, float] = {}

    feature_selections_rows: List[dict] = []

    for K in k_pcts:
        n_select = max(1, int(round((K / 100.0) * n_features_total)))
        n_at_K[K] = n_select

        v_idx, v_names = topk_feature_indices_from_rank_table(var_df, v_feature_col, v_rank_col, v_score_col, feats, n_select)
        p_idx, p_names = topk_feature_indices_from_rank_table(shap_df, p_feature_col, p_rank_col, p_score_col, feats, n_select)

        var_idx_byK[K] = v_idx
        shap_idx_byK[K] = p_idx
        jacc_byK[K] = float(jaccard_idx(v_idx, p_idx))

        selected_hashes[K] = {
            "var_topk_sha1": hashlib.sha1((",".join(v_names)).encode("utf-8")).hexdigest(),
            "shap_topk_sha1": hashlib.sha1((",".join(p_names)).encode("utf-8")).hexdigest(),
        }

        if save_feature_selections:
            s_var = set(v_idx.tolist())
            s_shp = set(p_idx.tolist())
            overlap = sorted(s_var & s_shp)
            shap_only = sorted(s_shp - s_var)
            var_only = sorted(s_var - s_shp)

            def add_rows(label: str, idxs: List[int]):
                for i in idxs:
                    feature_selections_rows.append(
                        {"dataset": dataset, "view": view, "model": model, "K_pct": int(K), "set": label, "feature": str(feats[i])}
                    )

            add_rows("var_topk", v_idx.tolist())
            add_rows("shap_topk", p_idx.tolist())
            add_rows("overlap", overlap)
            add_rows("shap_only", shap_only)
            add_rows("var_only", var_only)

    rows: List[dict] = []

    # Build repeat×fold tasks
    fold_tasks = []
    for r in range(n_repeats):
        fold_row = fold_ids[r, :]
        for fold in np.unique(fold_row).astype(int).tolist():
            train_idx = np.where(fold_row != fold)[0]
            test_idx = np.where(fold_row == fold)[0]
            fold_tasks.append((r, int(fold), train_idx, test_idx))

    def _fold_worker(r: int, fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
        X_train_full = X[train_idx, :]
        y_train = y[train_idx]
        X_test_full = X[test_idx, :]
        y_test = y[test_idx]

        # Diagnostic: fold missing classes?
        n_cls_test = int(len(np.unique(y_test)))
        missing_classes = int(n_cls_test < n_classes)

        local_rows: List[dict] = []

        # ALL (train once per fold)
        seed_all = stable_int_seed(seed, dataset, view, model, r, fold, "ALL")
        perf_all = train_eval_many(X_train_full, y_train, X_test_full, y_test, model, task_type, metrics, seed_all, n_classes)

        for K in k_pcts:
            n_select = n_at_K[K]

            # ALL row for this K (same perf_all)
            for m, val in perf_all.items():
                local_rows.append(dict(
                    repeat=r, fold=int(fold), K_pct=int(K),
                    strategy="all", n_features=int(n_features_total),
                    metric=m, value=float(val)
                ))

            # VAR-topK
            v_idx = var_idx_byK[K]
            seed_var = stable_int_seed(seed, dataset, view, model, r, fold, K, "VAR")
            perf_var = train_eval_many(
                X_train_full[:, v_idx], y_train,
                X_test_full[:, v_idx], y_test,
                model, task_type, metrics, seed_var, n_classes
            )
            for m, val in perf_var.items():
                local_rows.append(dict(
                    repeat=r, fold=int(fold), K_pct=int(K),
                    strategy="var_topk", n_features=int(n_select),
                    metric=m, value=float(val)
                ))

            # SHAP-topK
            p_idx = shap_idx_byK[K]
            seed_shp = stable_int_seed(seed, dataset, view, model, r, fold, K, "SHAP")
            perf_shp = train_eval_many(
                X_train_full[:, p_idx], y_train,
                X_test_full[:, p_idx], y_test,
                model, task_type, metrics, seed_shp, n_classes
            )
            for m, val in perf_shp.items():
                local_rows.append(dict(
                    repeat=r, fold=int(fold), K_pct=int(K),
                    strategy="shap_topk", n_features=int(n_select),
                    metric=m, value=float(val)
                ))

            # RANDOM draws (mean across draws)
            rand_vals: Dict[str, List[float]] = {m: [] for m in metrics}
            for d in range(n_random_draws):
                rng = np.random.default_rng(stable_int_seed(seed, dataset, view, model, r, fold, K, "RAND", d))
                rand_idx = rng.choice(n_features_total, size=n_select, replace=False)
                seed_rfit = stable_int_seed(seed, dataset, view, model, r, fold, K, "RANDFIT", d)
                perf_r = train_eval_many(
                    X_train_full[:, rand_idx], y_train,
                    X_test_full[:, rand_idx], y_test,
                    model, task_type, metrics, seed_rfit, n_classes
                )
                for m, val in perf_r.items():
                    rand_vals[m].append(float(val))

            for m in metrics:
                local_rows.append(dict(
                    repeat=r, fold=int(fold), K_pct=int(K),
                    strategy="random_mean", n_features=int(n_select),
                    metric=m, value=float(np.mean(rand_vals[m]))
                ))

        return local_rows, missing_classes

    # Dispatch fold tasks
    missing_count = 0
    if int(fold_n_jobs) > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=int(fold_n_jobs), backend=str(backend))(
            delayed(_fold_worker)(r, fold, tr, te) for (r, fold, tr, te) in fold_tasks
        )
    else:
        outs = [_fold_worker(r, fold, tr, te) for (r, fold, tr, te) in fold_tasks]

    for local_rows, miss in outs:
        rows.extend(local_rows)
        missing_count += int(miss)

    df = pd.DataFrame(rows)

    # repeat-means tidy output (plot-ready)
    repeat_means = df.groupby(["repeat", "K_pct", "strategy", "metric"])["value"].mean().reset_index()

    # Summary: per metric -> per K -> stats (repeat-level)
    results_by_metric: Dict[str, Dict[str, dict]] = {}

    for m in metrics:
        results_by_metric[m] = {}
        for K in k_pcts:
            sub = df[(df["K_pct"] == K) & (df["metric"] == m)].copy()
            rep = sub.groupby(["repeat", "strategy"])["value"].mean().unstack("strategy")

            out = {}
            # strategy stats across repeats
            def add_stat(strategy: str, key: str):
                if strategy in rep.columns:
                    out[key] = repeat_level_stats(rep[strategy].to_numpy())
                else:
                    out[key] = {"mean": np.nan, "std": np.nan, "ci95": [np.nan, np.nan], "n": int(rep.shape[0])}

            add_stat("all", "perf_all")
            add_stat("var_topk", "perf_var")
            add_stat("shap_topk", "perf_shap")
            add_stat("random_mean", "perf_random")

            # paired deltas per repeat
            if {"shap_topk", "var_topk"}.issubset(rep.columns):
                out["delta_shap_minus_var"] = repeat_level_stats((rep["shap_topk"] - rep["var_topk"]).to_numpy())
            if {"var_topk", "random_mean"}.issubset(rep.columns):
                out["delta_var_minus_random"] = repeat_level_stats((rep["var_topk"] - rep["random_mean"]).to_numpy())

            out["n_features_at_K"] = int(n_at_K[K])
            out["n_features_total"] = int(n_features_total)
            out["jaccard_var_shap"] = float(jacc_byK[K])
            out["selection_hashes"] = selected_hashes[K]

            results_by_metric[m][str(K)] = out

    reg_info = get_regime_info(outputs_dir, dataset, view, model, k_pct=regime_k_pct)

    runtime_sec = float(time.time() - t_start)

    unique_folds = np.unique(fold_ids[0]).astype(int).tolist()
    summary = {
        "dataset": dataset,
        "view": view,
        "model": model,
        "task_type": task_type,
        "metrics": metrics,
        "n_repeats": int(n_repeats),
        "n_folds": int(len(unique_folds)),
        "n_random_draws": int(n_random_draws),
        "fold_diagnostics": {
            "n_repeat_fold_tasks": int(len(fold_tasks)),
            "n_tasks_missing_classes_in_test": int(missing_count),
        },
        "regime_info": reg_info,
        "runtime_seconds": runtime_sec,
        "inputs": {
            "bundle_path": str(bundle_path),
            "splits_path": str(splits_path),
            "variance_scores_path": str(var_path),
            "importance_path": str(shap_path),
            "bundle_sha256": sha256_file(bundle_path),
            "splits_sha256": sha256_file(splits_path),
            "variance_scores_sha256": sha256_file(var_path),
            "importance_sha256": sha256_file(shap_path),
        },
        "provenance": {
            "script": Path(__file__).name,
            "timestamp": _now(),
            "command": " ".join(sys.argv),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": (xgb.__version__ if xgb is not None else "N/A"),
        },
        "results_by_metric": results_by_metric,
    }

    return df, repeat_means, summary, feature_selections_rows


# -----------------------------
# Master summary + interpretation
# -----------------------------
def build_master_summary(per_view_summaries: List[dict]) -> pd.DataFrame:
    rows = []
    for s in per_view_summaries:
        dataset = s["dataset"]
        view = s["view"]
        model = s["model"]
        metrics = s.get("metrics", [s.get("metric", "balanced_accuracy")])
        reg = s.get("regime_info", {})
        regime = reg.get("regime", "UNKNOWN")
        regime_conf = reg.get("regime_confidence", np.nan)
        DI_mean = reg.get("DI_mean", np.nan)
        DI_lo = reg.get("DI_pctl_2.5", np.nan)
        DI_hi = reg.get("DI_pctl_97.5", np.nan)
        regime_k = reg.get("regime_k_pct", np.nan)
        regime_source = reg.get("regime_source", "none")
        runtime_sec = s.get("runtime_seconds", np.nan)
        n_repeats = s.get("n_repeats", np.nan)
        n_folds = s.get("n_folds", np.nan)

        rbm = s["results_by_metric"]
        for m in metrics:
            for K_str, block in rbm[m].items():
                K = int(K_str)
                def unpack(key: str):
                    b = block.get(key, {})
                    return b.get("mean", np.nan), (b.get("ci95", [np.nan, np.nan])[0]), (b.get("ci95", [np.nan, np.nan])[1])

                pa, pa_lo, pa_hi = unpack("perf_all")
                pv, pv_lo, pv_hi = unpack("perf_var")
                ps, ps_lo, ps_hi = unpack("perf_shap")
                pr, pr_lo, pr_hi = unpack("perf_random")
                dsv, dsv_lo, dsv_hi = unpack("delta_shap_minus_var")
                dvr, dvr_lo, dvr_hi = unpack("delta_var_minus_random")

                rows.append(dict(
                    dataset=dataset, view=view, model=model,
                    metric=m, K_pct=K,
                    regime=regime, regime_confidence=regime_conf,
                    DI_mean=DI_mean, DI_pctl_2_5=DI_lo, DI_pctl_97_5=DI_hi,
                    regime_k_pct=regime_k, regime_source=regime_source,
                    perf_all_mean=pa, perf_all_ci_lo=pa_lo, perf_all_ci_hi=pa_hi,
                    perf_var_mean=pv, perf_var_ci_lo=pv_lo, perf_var_ci_hi=pv_hi,
                    perf_shap_mean=ps, perf_shap_ci_lo=ps_lo, perf_shap_ci_hi=ps_hi,
                    perf_random_mean=pr, perf_random_ci_lo=pr_lo, perf_random_ci_hi=pr_hi,
                    delta_shap_var_mean=dsv, delta_shap_var_ci_lo=dsv_lo, delta_shap_var_ci_hi=dsv_hi,
                    delta_var_random_mean=dvr, delta_var_random_ci_lo=dvr_lo, delta_var_random_ci_hi=dvr_hi,
                    jaccard_var_shap=block.get("jaccard_var_shap", np.nan),
                    n_features_total=block.get("n_features_total", np.nan),
                    n_features_at_K=block.get("n_features_at_K", np.nan),
                    n_repeats=n_repeats, n_folds=n_folds,
                    runtime_seconds=runtime_sec,
                ))
    return pd.DataFrame(rows)


def interpret_master(master: pd.DataFrame) -> dict:
    if master.empty:
        return {"hypothesis_supported": {"anti_aligned_harmed": False, "coupled_acceptable": False, "n_rows": 0, "n_views": 0}}

    # Prefer balanced_accuracy if present
    metric = "balanced_accuracy" if "balanced_accuracy" in set(master["metric"].astype(str)) else str(master["metric"].iloc[0])
    sub = master[master["metric"].astype(str) == metric].copy()

    anti = sub[sub["regime"].astype(str) == "ANTI_ALIGNED"]
    coupled = sub[sub["regime"].astype(str) == "COUPLED"]

    anti_harmed = False
    coupled_ok = False

    if len(anti) > 0:
        # harm: SHAP-VAR CI entirely >0 AND VAR-RANDOM CI includes <=0 (var not better than random)
        anti_harmed = bool(np.any((anti["delta_shap_var_ci_lo"] > 0) & (anti["delta_var_random_ci_hi"] <= 0)))

    if len(coupled) > 0:
        # coupled acceptable: SHAP-VAR CI overlaps 0 AND VAR-RANDOM CI entirely >0
        coupled_ok = bool(np.any((coupled["delta_shap_var_ci_lo"] <= 0) & (coupled["delta_shap_var_ci_hi"] >= 0) & (coupled["delta_var_random_ci_lo"] > 0)))

    # key number: max delta_shap_var_mean
    if np.isfinite(sub["delta_shap_var_mean"]).any():
        max_row = sub.loc[sub["delta_shap_var_mean"].astype(float).idxmax()].to_dict()
        max_delta = {
            "value": float(max_row["delta_shap_var_mean"]),
            "dataset": str(max_row["dataset"]),
            "view": str(max_row["view"]),
            "K": int(max_row["K_pct"]),
            "model": str(max_row["model"]),
            "regime": str(max_row["regime"]),
            "metric": metric,
        }
    else:
        max_delta = None

    return {
        "hypothesis_supported": {
            "anti_aligned_harmed": anti_harmed,
            "coupled_acceptable": coupled_ok,
            "metric_used": metric,
            "n_rows": int(len(master)),
            "n_views": int(master[["dataset", "view"]].drop_duplicates().shape[0]),
        },
        "key_numbers_for_abstract": {"max_delta_shap_var": max_delta},
    }


@contextmanager
def tqdm_joblib(tqdm_object):
    """Progress bar for joblib Parallel. Updates as batches complete."""
    import joblib

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


# -----------------------------
# Entrypoint
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7 ablation: feature subset performance comparison.")
    p.add_argument("--outputs-dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="all")
    p.add_argument("--views", type=str, default="core", choices=["core", "all", "sensitivity"])
    p.add_argument("--model", type=str, default="xgb_bal", choices=["xgb_bal", "xgb", "rf"])
    p.add_argument("--k-pcts", type=str, default="1,5,10,20")
    p.add_argument("--n-random-draws", type=int, default=20)
    # Backward compatible: --metric single OR comma list
    p.add_argument("--metric", type=str, default="balanced_accuracy",
                   help="Single metric or comma list. Supported: balanced_accuracy, auroc_ovr_macro")
    p.add_argument("--max-repeats", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--fold-n-jobs", type=int, default=1,
                   help="Parallelism inside a view across repeat×fold tasks. If >1, view-level parallel is disabled.")
    p.add_argument("--backend", type=str, default="threading", choices=["threading", "loky"],
                   help="joblib backend for fold-level parallelism (threading recommended for XGB with n_jobs_model=1).")
    p.add_argument("--scheduler", type=str, default="per_view", choices=["per_view", "global"],
                   help="per_view: current behaviour. global: flatten repeat×fold tasks across all views for better CPU utilisation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hero-views", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--regime-k-pct", type=int, default=10, help="K% used to fetch regime label from di_summary")
    p.add_argument("--save-feature-selections", action="store_true",
                   help="Write feature selections CSV for Phase 8 (can be large).")
    p.add_argument("--ablation-dirname", default="07_ablation",
                   help="Subfolder name under outputs-dir for this ablation run (prevents overwrite).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.scheduler == "global" and args.backend != "threading":
        print("Forcing backend=threading for global scheduler (Windows + shared arrays).")
        args.backend = "threading"
    outputs_dir = Path(args.outputs_dir)
    out_root = outputs_dir / args.ablation_dirname
    per_view_dir = out_root / "per_view"
    runlog_path = out_root / f"RUNLOG__{args.ablation_dirname}.txt"
    per_view_dir.mkdir(parents=True, exist_ok=True)

    k_pcts = [int(x.strip()) for x in args.k_pcts.split(",") if x.strip()]
    if not k_pcts or any((k <= 0 or k >= 100) for k in k_pcts):
        raise ValueError(f"Invalid k-pcts: {k_pcts}")

    metrics = [m.strip() for m in args.metric.split(",") if m.strip()]
    for m in metrics:
        if m not in ("balanced_accuracy", "auroc_ovr_macro", "auroc_macro"):
            raise ValueError(f"Unsupported metric: {m}")

    # tasks
    if args.hero_views:
        tasks = [(d, v) for (d, v) in HERO_VIEWS]
    else:
        if args.dataset != "all":
            datasets = [args.dataset]
        else:
            datasets = set()
            vs_dir = outputs_dir / "02_unsupervised" / "variance_scores"
            if vs_dir.exists():
                for pth in vs_dir.glob("variance_scores__*__*.csv.gz"):
                    m = re.match(r"variance_scores__(.+)__(.+)\.csv\.gz$", pth.name)
                    if m:
                        datasets.add(m.group(1))
            datasets = sorted(datasets) if datasets else sorted(DATASET_VIEWS_REGISTRY.keys())
        tasks = []
        for ds in datasets:
            discovered = discover_views(outputs_dir, ds)
            views = resolve_views(ds, args.views, discovered)
            for vw in views:
                tasks.append((ds, vw))

    def _validate_one(ds: str, vw: str) -> Tuple[str, str, Optional[str]]:
        try:
            bundle = find_bundle_npz(outputs_dir, ds)
            _ = find_splits_npz(outputs_dir, ds)
            _ = find_variance_scores(outputs_dir, ds, vw)
            _ = find_shap_importance(outputs_dir, ds, vw, args.model)
            z = load_npz_pickle_free(bundle)
            if f"X_{vw}" not in z.files:
                return ds, vw, "bundle_missing_view"
            return ds, vw, None
        except Exception as e:
            return ds, vw, f"error: {e}"

    if args.dry_run:
        msgs = [f"[{_now()}] DRY RUN: {len(tasks)} tasks"]
        for ds, vw in tasks:
            _, _, err = _validate_one(ds, vw)
            msgs.append(f"  - {ds}/{vw}: {'OK' if err is None else err}")
        append_runlog(runlog_path, msgs)
        print("\n".join(msgs))
        print("Dry-run complete.")
        return

    # Avoid nested parallelism: if we parallelise folds, run views sequentially.
    if args.scheduler == "per_view" and int(args.fold_n_jobs) > 1 and int(args.n_jobs) > 1:
        append_runlog(runlog_path, [f"[{_now()}] NOTE: fold-n-jobs={args.fold_n_jobs} -> forcing n_jobs(view)=1 to avoid nested parallelism"])
        args.n_jobs = 1

    # Parallel (optional)
    try:
        from joblib import Parallel, delayed
    except Exception:
        Parallel = None
        delayed = None

    per_view_summaries: List[dict] = []

    def _write_outputs(ds: str, vw: str, df: pd.DataFrame, repeat_means: pd.DataFrame, summary: dict, feature_sel_rows: List[dict]) -> None:
        out_csv = per_view_dir / f"ablation__{ds}__{vw}__{args.model}.csv.gz"
        out_rep = per_view_dir / f"ablation_repeat_means__{ds}__{vw}__{args.model}.csv.gz"
        out_json = per_view_dir / f"ablation_summary__{ds}__{vw}__{args.model}.json"

        write_csv_gz(df, out_csv)
        write_csv_gz(repeat_means, out_rep)
        write_json(summary, out_json)

        if args.save_feature_selections:
            fs = pd.DataFrame(feature_sel_rows)
            out_fs = per_view_dir / f"feature_selections__{ds}__{vw}__{args.model}.csv.gz"
            write_csv_gz(fs, out_fs)

    def _finalize_view_from_rows(cache: dict, rows: List[dict], missing_count: int, runtime_sec: float) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        df = pd.DataFrame(rows)

        # repeat-means tidy output (plot-ready)
        repeat_means = df.groupby(["repeat", "K_pct", "strategy", "metric"])["value"].mean().reset_index()

        # Summary: per metric -> per K -> stats (repeat-level)
        results_by_metric: Dict[str, Dict[str, dict]] = {}

        metrics = cache["metrics"]
        k_pcts = cache["k_pcts"]
        n_at_K = cache["n_at_K"]
        n_features_total = cache["n_features_total"]
        jacc_byK = cache["jacc_byK"]
        selected_hashes = cache["selected_hashes"]

        for m in metrics:
            results_by_metric[m] = {}
            for K in k_pcts:
                sub = df[(df["K_pct"] == K) & (df["metric"] == m)].copy()
                rep = sub.groupby(["repeat", "strategy"])["value"].mean().unstack("strategy")

                out = {}
                # strategy stats across repeats
                def add_stat(strategy: str, key: str):
                    if strategy in rep.columns:
                        out[key] = repeat_level_stats(rep[strategy].to_numpy())
                    else:
                        out[key] = {"mean": np.nan, "std": np.nan, "ci95": [np.nan, np.nan], "n": int(rep.shape[0])}

                add_stat("all", "perf_all")
                add_stat("var_topk", "perf_var")
                add_stat("shap_topk", "perf_shap")
                add_stat("random_mean", "perf_random")

                # paired deltas per repeat
                if {"shap_topk", "var_topk"}.issubset(rep.columns):
                    out["delta_shap_minus_var"] = repeat_level_stats((rep["shap_topk"] - rep["var_topk"]).to_numpy())
                if {"var_topk", "random_mean"}.issubset(rep.columns):
                    out["delta_var_minus_random"] = repeat_level_stats((rep["var_topk"] - rep["random_mean"]).to_numpy())

                out["n_features_at_K"] = int(n_at_K[K])
                out["n_features_total"] = int(n_features_total)
                out["jaccard_var_shap"] = float(jacc_byK[K])
                out["selection_hashes"] = selected_hashes[K]

                results_by_metric[m][str(K)] = out

        summary = {
            "dataset": cache["dataset"],
            "view": cache["view"],
            "model": cache["model"],
            "task_type": cache["task_type"],
            "metrics": metrics,
            "n_repeats": int(cache["n_repeats"]),
            "n_folds": int(len(cache["unique_folds"])),
            "n_random_draws": int(cache["n_random_draws"]),
            "fold_diagnostics": {
                "n_repeat_fold_tasks": int(len(cache["fold_tasks"])),
                "n_tasks_missing_classes_in_test": int(missing_count),
            },
            "regime_info": cache["reg_info"],
            "runtime_seconds": runtime_sec,
            "inputs": {
                "bundle_path": str(cache["bundle_path"]),
                "splits_path": str(cache["splits_path"]),
                "variance_scores_path": str(cache["var_path"]),
                "importance_path": str(cache["shap_path"]),
                "bundle_sha256": sha256_file(cache["bundle_path"]),
                "splits_sha256": sha256_file(cache["splits_path"]),
                "variance_scores_sha256": sha256_file(cache["var_path"]),
                "importance_sha256": sha256_file(cache["shap_path"]),
            },
            "provenance": {
                "script": Path(__file__).name,
                "timestamp": _now(),
                "command": " ".join(sys.argv),
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "sklearn_version": sklearn.__version__,
                "xgboost_version": (xgb.__version__ if xgb is not None else "N/A"),
            },
            "results_by_metric": results_by_metric,
        }

        return df, repeat_means, summary

    def _run_one(ds: str, vw: str) -> Optional[dict]:
        try:
            df, repeat_means, summary, feature_sel_rows = ablation_for_view(
                outputs_dir=outputs_dir,
                dataset=ds,
                view=vw,
                model=args.model,
                k_pcts=k_pcts,
                n_random_draws=int(args.n_random_draws),
                metrics=metrics,
                max_repeats=args.max_repeats,
                seed=int(args.seed),
                regime_k_pct=int(args.regime_k_pct),
                save_feature_selections=bool(args.save_feature_selections),
                fold_n_jobs=int(args.fold_n_jobs),
                backend=str(args.backend),
            )
            _write_outputs(ds, vw, df, repeat_means, summary, feature_sel_rows)

            return summary
        except Exception as e:
            append_runlog(runlog_path, [f"[{_now()}] ERROR {ds}/{vw}/{args.model}: {e}"])
            return None

    append_runlog(runlog_path, [f"[{_now()}] START ablation: tasks={len(tasks)} model={args.model} metrics={metrics} k={k_pcts} draws={args.n_random_draws}"])

    if args.scheduler == "global":
        view_caches: Dict[str, dict] = {}
        rows_by_key: Dict[str, List[dict]] = {}
        missing_by_key: Dict[str, int] = {}
        start_by_key: Dict[str, float] = {}
        end_by_key: Dict[str, float] = {}
        global_tasks = []

        for ds, vw in tasks:
            key = f"{ds}|{vw}"
            try:
                t_start = time.time()
                bundle_path = find_bundle_npz(outputs_dir, ds)
                splits_path = find_splits_npz(outputs_dir, ds)
                var_path = find_variance_scores(outputs_dir, ds, vw)
                shap_path = find_shap_importance(outputs_dir, ds, vw, args.model)

                splits = load_splits(splits_path)
                _, bundle_ids, X, feats, _ = load_bundle_view(bundle_path, vw)
                X = align_X_to_splits_order(X, bundle_ids, splits.sample_ids)

                y = np.asarray(splits.y)
                task_type = infer_task_type(y)
                if task_type != "classification":
                    raise ValueError(f"This script currently supports classification only. Got task_type={task_type} for {ds}/{vw}")

                classes = np.unique(y)
                n_classes = int(len(classes))

                fold_ids = splits.fold_ids
                if args.max_repeats is not None:
                    fold_ids = fold_ids[: int(args.max_repeats), :]

                n_repeats = int(fold_ids.shape[0])
                n_features_total = int(X.shape[1])

                var_df = pd.read_csv(var_path)
                shap_df = pd.read_csv(shap_path)

                v_feature_col = "feature" if "feature" in var_df.columns else var_df.columns[0]
                v_rank_col = "v_rank" if "v_rank" in var_df.columns else ("rank" if "rank" in var_df.columns else None)
                v_score_col = "v_score" if "v_score" in var_df.columns else ("score" if "score" in var_df.columns else None)

                p_feature_col = "feature" if "feature" in shap_df.columns else shap_df.columns[0]
                p_rank_col = "p_rank" if "p_rank" in shap_df.columns else ("rank" if "rank" in shap_df.columns else None)
                p_score_col = "p_score" if "p_score" in shap_df.columns else ("importance" if "importance" in shap_df.columns else None)

                var_idx_byK: Dict[int, np.ndarray] = {}
                shap_idx_byK: Dict[int, np.ndarray] = {}
                n_at_K: Dict[int, int] = {}
                selected_hashes: Dict[int, Dict[str, str]] = {}
                jacc_byK: Dict[int, float] = {}

                feature_selections_rows: List[dict] = []

                for K in k_pcts:
                    n_select = max(1, int(round((K / 100.0) * n_features_total)))
                    n_at_K[K] = n_select

                    v_idx, v_names = topk_feature_indices_from_rank_table(var_df, v_feature_col, v_rank_col, v_score_col, feats, n_select)
                    p_idx, p_names = topk_feature_indices_from_rank_table(shap_df, p_feature_col, p_rank_col, p_score_col, feats, n_select)

                    var_idx_byK[K] = v_idx
                    shap_idx_byK[K] = p_idx
                    jacc_byK[K] = float(jaccard_idx(v_idx, p_idx))

                    selected_hashes[K] = {
                        "var_topk_sha1": hashlib.sha1((",".join(v_names)).encode("utf-8")).hexdigest(),
                        "shap_topk_sha1": hashlib.sha1((",".join(p_names)).encode("utf-8")).hexdigest(),
                    }

                    if args.save_feature_selections:
                        s_var = set(v_idx.tolist())
                        s_shp = set(p_idx.tolist())
                        overlap = sorted(s_var & s_shp)
                        shap_only = sorted(s_shp - s_var)
                        var_only = sorted(s_var - s_shp)

                        def add_rows(label: str, idxs: List[int]):
                            for i in idxs:
                                feature_selections_rows.append(
                                    {"dataset": ds, "view": vw, "model": args.model, "K_pct": int(K), "set": label, "feature": str(feats[i])}
                                )

                        add_rows("var_topk", v_idx.tolist())
                        add_rows("shap_topk", p_idx.tolist())
                        add_rows("overlap", overlap)
                        add_rows("shap_only", shap_only)
                        add_rows("var_only", var_only)

                unique_folds = np.unique(fold_ids[0]).astype(int).tolist()
                fold_tasks = []
                for r in range(n_repeats):
                    fold_row = fold_ids[r, :]
                    for fold in np.unique(fold_row).astype(int).tolist():
                        train_idx = np.where(fold_row != fold)[0]
                        test_idx = np.where(fold_row == fold)[0]
                        fold_tasks.append((r, int(fold), train_idx, test_idx))

                reg_info = get_regime_info(outputs_dir, ds, vw, args.model, k_pct=int(args.regime_k_pct))

                view_caches[key] = {
                    "dataset": ds,
                    "view": vw,
                    "model": args.model,
                    "metrics": metrics,
                    "k_pcts": k_pcts,
                    "n_random_draws": int(args.n_random_draws),
                    "n_repeats": n_repeats,
                    "n_features_total": n_features_total,
                    "n_at_K": n_at_K,
                    "jacc_byK": jacc_byK,
                    "selected_hashes": selected_hashes,
                    "feature_selections_rows": feature_selections_rows,
                    "bundle_path": bundle_path,
                    "splits_path": splits_path,
                    "var_path": var_path,
                    "shap_path": shap_path,
                    "reg_info": reg_info,
                    "task_type": task_type,
                    "X": X,
                    "y": y,
                    "var_idx_byK": var_idx_byK,
                    "shap_idx_byK": shap_idx_byK,
                    "fold_tasks": fold_tasks,
                    "unique_folds": unique_folds,
                    "n_classes": n_classes,
                    "seed": int(args.seed),
                    "t_start": t_start,
                }
                rows_by_key[key] = []
                missing_by_key[key] = 0
                start_by_key[key] = float("inf")
                end_by_key[key] = 0.0

                for r, fold, tr, te in fold_tasks:
                    global_tasks.append((key, r, fold, tr, te))
            except Exception as e:
                append_runlog(runlog_path, [f"[{_now()}] ERROR {ds}/{vw}/{args.model}: {e}"])

        def _fold_worker_global(key: str, r: int, fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
            t0 = time.time()
            cache = view_caches[key]
            X = cache["X"]
            y = cache["y"]
            n_classes = cache["n_classes"]
            var_idx_byK = cache["var_idx_byK"]
            shap_idx_byK = cache["shap_idx_byK"]
            n_at_K = cache["n_at_K"]
            k_pcts = cache["k_pcts"]
            metrics = cache["metrics"]
            dataset = cache["dataset"]
            view = cache["view"]
            model = cache["model"]
            seed = cache["seed"]
            n_features_total = cache["n_features_total"]
            task_type = cache["task_type"]

            X_train_full = X[train_idx, :]
            y_train = y[train_idx]
            X_test_full = X[test_idx, :]
            y_test = y[test_idx]

            # Diagnostic: fold missing classes?
            n_cls_test = int(len(np.unique(y_test)))
            missing_classes = int(n_cls_test < n_classes)

            local_rows: List[dict] = []

            # ALL (train once per fold)
            seed_all = stable_int_seed(seed, dataset, view, model, r, fold, "ALL")
            perf_all = train_eval_many(X_train_full, y_train, X_test_full, y_test, model, task_type, metrics, seed_all, n_classes)

            for K in k_pcts:
                n_select = n_at_K[K]

                # ALL row for this K (same perf_all)
                for m, val in perf_all.items():
                    local_rows.append(dict(
                        repeat=r, fold=int(fold), K_pct=int(K),
                        strategy="all", n_features=int(n_features_total),
                        metric=m, value=float(val)
                    ))

                # VAR-topK
                v_idx = var_idx_byK[K]
                seed_var = stable_int_seed(seed, dataset, view, model, r, fold, K, "VAR")
                perf_var = train_eval_many(
                    X_train_full[:, v_idx], y_train,
                    X_test_full[:, v_idx], y_test,
                    model, task_type, metrics, seed_var, n_classes
                )
                for m, val in perf_var.items():
                    local_rows.append(dict(
                        repeat=r, fold=int(fold), K_pct=int(K),
                        strategy="var_topk", n_features=int(n_select),
                        metric=m, value=float(val)
                    ))

                # SHAP-topK
                p_idx = shap_idx_byK[K]
                seed_shp = stable_int_seed(seed, dataset, view, model, r, fold, K, "SHAP")
                perf_shp = train_eval_many(
                    X_train_full[:, p_idx], y_train,
                    X_test_full[:, p_idx], y_test,
                    model, task_type, metrics, seed_shp, n_classes
                )
                for m, val in perf_shp.items():
                    local_rows.append(dict(
                        repeat=r, fold=int(fold), K_pct=int(K),
                        strategy="shap_topk", n_features=int(n_select),
                        metric=m, value=float(val)
                    ))

                # RANDOM draws (mean across draws)
                rand_vals: Dict[str, List[float]] = {m: [] for m in metrics}
                for d in range(int(args.n_random_draws)):
                    rng = np.random.default_rng(stable_int_seed(seed, dataset, view, model, r, fold, K, "RAND", d))
                    rand_idx = rng.choice(n_features_total, size=n_select, replace=False)
                    seed_rfit = stable_int_seed(seed, dataset, view, model, r, fold, K, "RANDFIT", d)
                    perf_r = train_eval_many(
                        X_train_full[:, rand_idx], y_train,
                        X_test_full[:, rand_idx], y_test,
                        model, task_type, metrics, seed_rfit, n_classes
                    )
                    for m, val in perf_r.items():
                        rand_vals[m].append(float(val))

                for m in metrics:
                    local_rows.append(dict(
                        repeat=r, fold=int(fold), K_pct=int(K),
                        strategy="random_mean", n_features=int(n_select),
                        metric=m, value=float(np.mean(rand_vals[m]))
                    ))

            t1 = time.time()
            return key, local_rows, missing_classes, t0, t1

        if args.n_jobs > 1 and Parallel is not None:
            with tqdm_joblib(tqdm(total=len(global_tasks), desc="07_ablation global tasks")):
                outs = Parallel(n_jobs=int(args.n_jobs), backend=str(args.backend))(
                    delayed(_fold_worker_global)(key, r, fold, tr, te) for (key, r, fold, tr, te) in global_tasks
                )
        else:
            outs = []
            for key, r, fold, tr, te in tqdm(global_tasks, desc="07_ablation global tasks"):
                outs.append(_fold_worker_global(key, r, fold, tr, te))

        for key, local_rows, miss, t0, t1 in outs:
            rows_by_key[key].extend(local_rows)
            missing_by_key[key] += int(miss)
            start_by_key[key] = min(start_by_key[key], t0)
            end_by_key[key] = max(end_by_key[key], t1)

        for key, rows in rows_by_key.items():
            cache = view_caches[key]
            runtime_sec = float(end_by_key[key] - start_by_key[key]) if np.isfinite(start_by_key[key]) else float("nan")
            df, repeat_means, summary = _finalize_view_from_rows(cache, rows, missing_by_key[key], runtime_sec)
            _write_outputs(cache["dataset"], cache["view"], df, repeat_means, summary, cache["feature_selections_rows"])
            per_view_summaries.append(summary)
    else:
        if args.n_jobs > 1 and Parallel is not None:
            with tqdm_joblib(tqdm(total=len(tasks), desc="07_ablation tasks")):
                outs = Parallel(n_jobs=int(args.n_jobs))(
                    delayed(_run_one)(ds, vw) for ds, vw in tasks
                )
        else:
            outs = []
            for ds, vw in tqdm(tasks, desc="07_ablation tasks"):
                outs.append(_run_one(ds, vw))

        for s in outs:
            if s is not None:
                per_view_summaries.append(s)

    master = build_master_summary(per_view_summaries)
    master_path = out_root / "ablation_master_summary.csv"
    master.to_csv(master_path, index=False)

    interp = interpret_master(master)
    interp_path = out_root / "ablation_interpretation.json"
    write_json(interp, interp_path)

    append_runlog(runlog_path, [f"[{_now()}] DONE ablation: completed_views={len(per_view_summaries)}/{len(tasks)}",
                               f"[{_now()}] WROTE master: {master_path}",
                               f"[{_now()}] WROTE interpretation: {interp_path}"])

    print(f"Done. Completed {len(per_view_summaries)}/{len(tasks)} views.")
    print(f"Master: {master_path}")
    print(f"Interpretation: {interp_path}")


if __name__ == "__main__":
    main()