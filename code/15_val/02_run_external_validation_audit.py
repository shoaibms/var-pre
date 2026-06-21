#!/usr/bin/env python3
r"""
02_run_external_validation_audit.py

Manuscript-standard external-validation audit runner.

Runs the manuscript-standard audit workflow for existing variance-filtering validation bundles.

This script does NOT download or parse new datasets. It scans existing bundle files:
    data/raw/val/<dataset>/*__bundle.npz

It then runs the manuscript-aligned validation stack:
  1) vardiag fold-wise TRAIN-only diagnostics (eta_ES, VSA, PCLA, F_DI)
  2) TopVar-vs-matched-Random ablation with balanced XGBoost primary
  3) Optional coherent tree-ensemble sensitivity panel

Outputs are written to a timestamped folder by default so previous results are not overwritten:
    outputs/15_val/_manuscript_audit/<run_tag>/

PowerShell examples
-------------------
cd <project-root>
.\venv\Scripts\activate

# inventory only
python code\compute\15_val\02_run_external_validation_audit.py --inventory

# canonical VAD across all existing bundles
python code\compute\15_val\02_run_external_validation_audit.py --vad --run-tag vardiag_all_v1

# primary manuscript-style XGBoost ablation only for VAD-promising bundles
python code\compute\15_val\02_run_external_validation_audit.py --ablate --model xgb --only-promising --run-tag vardiag_all_v1

# Optional tree-panel sensitivity for the same promising bundles
python code\compute\15_val\02_run_external_validation_audit.py --ablate --model all_tree_panel --only-promising --run-tag vardiag_all_v1

# final integrated table
python code\compute\15_val\02_run_external_validation_audit.py --final --run-tag vardiag_all_v1
"""
from __future__ import annotations

import argparse
import math
import os
import time

# Avoid repeated joblib/loky physical-core warnings on some Windows/pyenv setups.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


TREE_PANEL_MODELS = ["xgb", "hgb", "gbc", "rf", "extratrees"]
BOOSTING_MODELS = {"xgb", "hgb", "gbc"}
BAGGING_MODELS = {"rf", "extratrees"}


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def find_project_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start] + list(start.parents):
        if (p / "data" / "raw" / "val").exists() or (p / "outputs").exists() or (p / "code").exists():
            return p
    return start


def raw_validation_dir(project_root: Path) -> Path:
    return project_root / "data" / "raw" / "val"


def audit_dir(project_root: Path, run_tag: str) -> Path:
    return project_root / "outputs" / "15_val" / "_manuscript_audit" / run_tag


def safe_load_npz(path: Path):
    try:
        return np.load(path, allow_pickle=False)
    except ValueError as e:
        if "Object arrays cannot be loaded" in str(e):
            log(f"legacy object array bundle detected; loading with allow_pickle=True: {path.name}")
            return np.load(path, allow_pickle=True)
        raise


def bundle_metadata(bundle: Path) -> Dict[str, object]:
    try:
        z = safe_load_npz(bundle)
        X_shape = z["X"].shape if "X" in z.files else (np.nan, np.nan)
        y = z["y"].astype(int) if "y" in z.files else np.array([])
        dataset = str(z["source_accession"][0]) if "source_accession" in z.files else bundle.parent.name
        label = str(z["label_column"][0]) if "label_column" in z.files else bundle.stem
        matrix_file = str(z["matrix_file"][0]) if "matrix_file" in z.files else ""
        class_names = z["class_names"].astype(str) if "class_names" in z.files else np.array([str(c) for c in np.unique(y)])
        classes, counts = np.unique(y, return_counts=True) if y.size else (np.array([]), np.array([]))
        class_counts = "; ".join([
            f"{class_names[int(c)] if int(c) < len(class_names) else c}:{int(n)}"
            for c, n in zip(classes, counts)
        ])
        min_class = int(counts.min()) if counts.size else 0
        status = "ok" if "X" in z.files and "y" in z.files else "missing_X_or_y"
        return {
            "bundle_path": str(bundle),
            "dataset": dataset,
            "bundle": bundle.name,
            "label_column": label,
            "matrix_file": matrix_file,
            "n_samples": int(X_shape[0]) if np.isfinite(X_shape[0]) else np.nan,
            "n_features_raw": int(X_shape[1]) if np.isfinite(X_shape[1]) else np.nan,
            "n_classes": int(len(classes)),
            "min_class_n": min_class,
            "class_counts": class_counts,
            "status": status,
        }
    except Exception as e:
        return {"bundle_path": str(bundle), "dataset": bundle.parent.name, "bundle": bundle.name, "status": "load_failed", "error": str(e)}


def discover_bundles(project_root: Path, targets: Optional[Sequence[str]] = None, labels: Optional[Sequence[str]] = None) -> List[Path]:
    raw = raw_validation_dir(project_root)
    target_set = {t.strip() for t in targets if t.strip()} if targets else None
    label_set = {l.strip() for l in labels if l.strip()} if labels else None
    bundles = sorted(raw.glob("*/*__bundle.npz"))
    out = []
    for b in bundles:
        if target_set is not None and b.parent.name not in target_set:
            # also allow source_accession inside bundle to match target
            try:
                ds = bundle_metadata(b).get("dataset", b.parent.name)
                if str(ds) not in target_set:
                    continue
            except Exception:
                continue
        if label_set is not None:
            try:
                lab = str(bundle_metadata(b).get("label_column", b.stem))
                if lab not in label_set:
                    continue
            except Exception:
                continue
        out.append(b)
    return out


def write_inventory(project_root: Path, out_dir: Path, targets: Optional[Sequence[str]], labels: Optional[Sequence[str]]) -> pd.DataFrame:
    bundles = discover_bundles(project_root, targets=targets, labels=labels)
    rows = [bundle_metadata(b) for b in bundles]
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bundle_inventory.csv"
    df.to_csv(path, index=False)
    log(f"Wrote {path} ({len(df)} bundles)")
    if not df.empty:
        print(df[[c for c in ["dataset", "bundle", "label_column", "n_samples", "n_features_raw", "n_classes", "min_class_n", "status", "class_counts"] if c in df.columns]].to_string(index=False))
    return df


# ------------------------------
# Canonical vardiag VAD
# ------------------------------

def vad_for_bundle(bundle: Path, k_values: Sequence[float], folds: int, repeats: int, pca_components: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from vardiag import eta2_features, eta_enrichment, vsa_mannwhitney, alpha_prime, pca_alignment, f_di, classify_zone
    except Exception as e:
        raise RuntimeError("vardiag is required. Activate venv or install: pip install vardiag") from e

    z = safe_load_npz(bundle)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(int)
    dataset = str(z["source_accession"][0]) if "source_accession" in z.files else bundle.parent.name
    label = str(z["label_column"][0]) if "label_column" in z.files else bundle.stem
    matrix_file = str(z["matrix_file"][0]) if "matrix_file" in z.files else ""
    class_names = z["class_names"].astype(str) if "class_names" in z.files else np.array([str(c) for c in np.unique(y)])

    var_global = np.nanvar(X, axis=0)
    keep = np.isfinite(var_global) & (var_global > 0)
    X = X[:, keep]
    p = int(X.shape[1])
    if p < 2:
        raise RuntimeError(f"No usable features after zero-variance filtering: p={p}")

    classes, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if counts.size else 0
    if min_class < int(folds):
        raise RuntimeError(f"min_class_n={min_class} < folds={folds}; skip for manuscript-standard {folds}-fold VAD")

    splitter = RepeatedStratifiedKFold(n_splits=int(folds), n_repeats=int(repeats), random_state=int(seed))
    raw_rows: List[Dict[str, object]] = []
    for split_i, (tr, _te) in enumerate(splitter.split(X, y)):
        repeat_i = int(split_i // int(folds))
        fold_i = int(split_i % int(folds))
        Xtr = X[tr, :]
        ytr = y[tr]
        v_total, _v_between, eta2 = eta2_features(Xtr, ytr)
        a_prime = alpha_prime(v_total, eta2)
        try:
            pa = pca_alignment(Xtr, ytr, n_components=int(pca_components), random_state=int(seed) + 1000 * repeat_i + fold_i)
            sas = float(pa.get("sas", np.nan))
            pcla = float(pa.get("pcla", np.nan))
        except Exception:
            sas = np.nan
            pcla = np.nan
        for k in k_values:
            k_int = int(round(float(k)))
            eta_es, eta_topv, eta_all = eta_enrichment(eta2, v_total, k_pct=k_int)
            vsa = vsa_mannwhitney(eta2, v_total, k_pct=k_int)
            fdi = f_di(eta2, v_total, k_pct=k_int)
            n_sel = max(1, int(math.ceil(p * float(k) / 100.0)))
            raw_rows.append({
                "dataset": dataset,
                "bundle": bundle.name,
                "bundle_path": str(bundle),
                "matrix_file": matrix_file,
                "label_column": label,
                "repeat": repeat_i,
                "fold": fold_i,
                "k_pct": float(k),
                "n_train": int(len(tr)),
                "n_samples": int(X.shape[0]),
                "n_features": p,
                "n_classes": int(len(classes)),
                "min_class_n": min_class,
                "class_counts": "; ".join([f"{class_names[int(c)] if int(c) < len(class_names) else c}:{int(n)}" for c, n in zip(classes, counts)]),
                "n_selected": n_sel,
                "eta_ES": float(eta_es),
                "eta_topv": float(eta_topv),
                "eta_all": float(eta_all),
                "VSA": float(vsa),
                "alpha_prime": float(a_prime),
                "SAS": float(sas),
                "PCLA": float(pcla),
                "F_DI": float(fdi),
                "vad_source": "vardiag_foldwise_train_only",
            })
    raw = pd.DataFrame(raw_rows)
    summary_rows: List[Dict[str, object]] = []
    for k, g in raw.groupby("k_pct", dropna=False):
        row: Dict[str, object] = {
            "dataset": dataset,
            "bundle": bundle.name,
            "bundle_path": str(bundle),
            "matrix_file": matrix_file,
            "label_column": label,
            "k_pct": float(k),
            "n_samples": int(X.shape[0]),
            "n_features": p,
            "n_classes": int(len(classes)),
            "min_class_n": min_class,
            "class_counts": "; ".join([f"{class_names[int(c)] if int(c) < len(class_names) else c}:{int(n)}" for c, n in zip(classes, counts)]),
            "n_selected": max(1, int(math.ceil(p * float(k) / 100.0))),
            "vad_folds": int(folds),
            "vad_repeats": int(repeats),
            "vad_source": "vardiag_foldwise_train_only",
        }
        for col in ["eta_ES", "eta_topv", "eta_all", "VSA", "alpha_prime", "SAS", "PCLA", "F_DI"]:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[col] = float(vals.mean()) if vals.size else np.nan
            row[f"{col}_lo"] = float(np.quantile(vals, 0.025)) if vals.size else np.nan
            row[f"{col}_hi"] = float(np.quantile(vals, 0.975)) if vals.size else np.nan
        try:
            zone_cons = classify_zone(
                eta_es=row["eta_ES"], vsa=row["VSA"],
                eta_es_lo=row.get("eta_ES_lo", np.nan), eta_es_hi=row.get("eta_ES_hi", np.nan),
                vsa_lo=row.get("VSA_lo", np.nan), vsa_hi=row.get("VSA_hi", np.nan), margin=0.05)
        except Exception:
            zone_cons = "BORDERLINE_KEEP"
        if np.isfinite(row["eta_ES"]) and np.isfinite(row["VSA"]):
            if row["eta_ES"] < 1.0 and row["VSA"] < 0:
                diag = "VARIANCE_SIGNAL_DECOUPLED"
                zone_nom = "RED_RISK"
            elif row["eta_ES"] > 1.0 and row["VSA"] > 0:
                diag = "VARIANCE_SIGNAL_ALIGNED"
                zone_nom = "GREEN_SAFE"
            else:
                diag = "INCONCLUSIVE"
                zone_nom = "YELLOW_INCONCLUSIVE"
        else:
            diag = "UNKNOWN"
            zone_nom = "UNKNOWN"
        row["diagnostic_state"] = diag
        row["zone_nominal"] = zone_nom
        row["zone_conservative"] = str(zone_cons)
        row["promising_for_ablation"] = bool(diag in {"VARIANCE_SIGNAL_DECOUPLED", "INCONCLUSIVE"} and str(zone_cons) != "STRONG_GREEN_EXCLUDE")
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "label_column", "k_pct"]).reset_index(drop=True)
    return summary, raw


def run_vad(project_root: Path, out_dir: Path, bundles: List[Path], k_values: Sequence[float], folds: int, repeats: int, pca_components: int, seed: int) -> pd.DataFrame:
    frames = []
    raw_frames = []
    failures = []
    for b in bundles:
        log(f"VAD/vardiag: {b}")
        try:
            s, r = vad_for_bundle(b, k_values, folds, repeats, pca_components, seed)
            frames.append(s)
            raw_frames.append(r)
        except Exception as e:
            log(f"  FAILED VAD {b.name}: {e}")
            meta = bundle_metadata(b)
            meta.update({"error": str(e)})
            failures.append(meta)
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    ranked = rank_vad(summary)
    summary.to_csv(out_dir / "vad_multi_k_summary.csv", index=False)
    raw.to_csv(out_dir / "vad_cv_raw_scores.csv", index=False)
    ranked.to_csv(out_dir / "vad_ranked_promising.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "vad_failures.csv", index=False)
    log(f"Wrote VAD outputs to {out_dir}")
    if not ranked.empty:
        cols = ["dataset", "label_column", "k_pct", "eta_ES", "VSA", "PCLA", "diagnostic_state", "zone_conservative", "class_counts"]
        print("\nTop diagnostic-risk rows:")
        with pd.option_context("display.max_colwidth", 80, "display.width", 220):
            print(ranked[cols].head(40).to_string(index=False))
    return summary


def rank_vad(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["risk_score"] = 0.0
    m = d["eta_ES"] < 1.0
    d.loc[m, "risk_score"] += (1.0 - d.loc[m, "eta_ES"]).clip(0, 2) * 3
    m = d["VSA"] < 0.0
    d.loc[m, "risk_score"] += (-d.loc[m, "VSA"]).clip(0, 0.5) * 6
    d.loc[d["diagnostic_state"].eq("VARIANCE_SIGNAL_DECOUPLED"), "risk_score"] += 5
    d.loc[d["zone_conservative"].astype(str).eq("STRONG_GREEN_EXCLUDE"), "risk_score"] -= 3
    return d.sort_values(["risk_score", "dataset", "label_column", "k_pct"], ascending=[False, True, True, True]).reset_index(drop=True)


# ------------------------------
# Ablation
# ------------------------------

def make_model(model: str, n_classes: int, seed: int, estimator_n_jobs: int):
    """Return a tree-based classifier.

    Model panel is deliberately restricted to tree ensembles so that sensitivity
    tests ask a coherent question: is variance-filter harm specific to boosted
    trees, or shared across related non-parametric tree learners?
    """
    model = model.lower()
    if model == "xgb":
        try:
            from xgboost import XGBClassifier
            params = dict(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_lambda=1.0,
                objective="binary:logistic" if n_classes == 2 else "multi:softprob",
                eval_metric="logloss" if n_classes == 2 else "mlogloss",
                tree_method="hist",
                random_state=int(seed),
                n_jobs=max(1, int(estimator_n_jobs)),
            )
            if n_classes > 2:
                params["num_class"] = int(n_classes)
            return XGBClassifier(**params), "xgb"
        except Exception as e:
            raise RuntimeError(f"XGBoost requested but unavailable/failed: {e}") from e

    if model == "hgb":
        # sklearn histogram gradient boosting: the closest built-in analogue to
        # XGBoost/LightGBM-style boosted trees without requiring a new package.
        return HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=5,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=int(seed),
        ), "hgb"

    if model == "gbc":
        # Classic gradient boosting. Slower and less modern than XGB/HGB, but a
        # useful boosting-family sanity check on small n omics matrices.
        return GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            random_state=int(seed),
        ), "gbc"

    if model == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=max(1, int(estimator_n_jobs)),
        ), "rf"

    if model == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=max(1, int(estimator_n_jobs)),
        ), "extratrees"

    raise ValueError(f"Unknown model={model}")


def load_for_ablation(bundle_path: str):
    z = safe_load_npz(Path(bundle_path))
    X = z["X"].astype(np.float32)
    y = z["y"].astype(int)
    dataset = str(z["source_accession"][0]) if "source_accession" in z.files else Path(bundle_path).parent.name
    label = str(z["label_column"][0]) if "label_column" in z.files else Path(bundle_path).stem
    var = np.nanvar(X.astype(np.float64), axis=0, ddof=1)
    keep = np.isfinite(var) & (var > 0)
    X = X[:, keep]
    var = var[keep]
    return X, y, var, dataset, label


def eval_subset_worker(args: Tuple[str, float, str, int, int, int, int, str, int]) -> Dict[str, object]:
    bundle_path, k_pct, subset_type, draw_idx, seed, folds, repeats, model, estimator_n_jobs = args
    X, y, var, dataset, label = load_for_ablation(bundle_path)
    classes, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if counts.size else 0
    if min_class < int(folds):
        raise RuntimeError(f"min_class_n={min_class} < folds={folds}; skip manuscript-standard ablation")
    p = int(X.shape[1])
    if p < 2:
        raise RuntimeError(f"No usable features: p={p}")
    n_sel = max(1, int(math.ceil(p * float(k_pct) / 100.0)))
    draw_seed = int(draw_idx) if int(draw_idx) >= 0 else 0
    rng_seed = int(seed) + 10000 * draw_seed + int(round(float(k_pct) * 100))
    if subset_type == "topvar":
        cols = np.argsort(var)[-n_sel:]
    elif subset_type == "random":
        rng = np.random.default_rng(rng_seed)
        cols = rng.choice(p, size=n_sel, replace=False)
    else:
        raise ValueError(f"Unknown subset_type={subset_type}")
    Xs = X[:, cols]
    n_classes = int(len(classes))
    model_seed = int(seed) + draw_seed + int(round(float(k_pct) * 1000))
    clf, model_used = make_model(model, n_classes=n_classes, seed=model_seed, estimator_n_jobs=estimator_n_jobs)
    splitter = RepeatedStratifiedKFold(n_splits=int(folds), n_repeats=int(repeats), random_state=int(seed))
    scores = []

    # Balance classes exactly once. XGB/HGB/GBC need explicit sample weights;
    # RF/ExtraTrees already receive class_weight in make_model, so passing
    # sample_weight as well would double-weight minority classes.
    use_fit_sample_weight = model_used in {"xgb", "hgb", "gbc"}

    for tr, te in splitter.split(Xs, y):
        if use_fit_sample_weight:
            try:
                sw = compute_sample_weight(class_weight="balanced", y=y[tr])
                clf.fit(Xs[tr], y[tr], sample_weight=sw)
            except TypeError:
                clf.fit(Xs[tr], y[tr])
        else:
            clf.fit(Xs[tr], y[tr])
        pred = clf.predict(Xs[te])
        scores.append(balanced_accuracy_score(y[te], pred))
    return {
        "dataset": dataset,
        "bundle": Path(bundle_path).name,
        "bundle_path": bundle_path,
        "label_column": label,
        "k_pct": float(k_pct),
        "subset_type": subset_type,
        "draw_idx": int(draw_idx),
        "n_samples": int(X.shape[0]),
        "n_features": p,
        "n_classes": n_classes,
        "min_class_n": min_class,
        "n_selected": int(n_sel),
        "folds": int(folds),
        "repeats": int(repeats),
        "model": model_used,
        "score_bal_acc": float(np.mean(scores)) if scores else np.nan,
    }


def select_promising_from_vad(out_dir: Path, bundles: List[Path], max_bundles: int) -> List[Path]:
    vad_path = out_dir / "vad_ranked_promising.csv"
    if not vad_path.exists():
        raise RuntimeError(f"Missing {vad_path}; run --vad first or omit --only-promising")
    vad = pd.read_csv(vad_path)
    if vad.empty:
        return []
    # Bundle-level: any non-strong-green diagnostic-risk row sends the bundle to ablation.
    vad = vad.loc[vad["promising_for_ablation"].astype(bool)].copy() if "promising_for_ablation" in vad.columns else vad.copy()
    if "risk_score" in vad.columns:
        vad = vad.sort_values("risk_score", ascending=False)
    bundle_map = {b.name: b for b in bundles}
    selected = []
    seen = set()
    for _, row in vad.iterrows():
        bname = str(row["bundle"])
        if bname in seen:
            continue
        seen.add(bname)
        b = bundle_map.get(bname)
        if b is not None and b.exists():
            selected.append(b)
        if len(selected) >= int(max_bundles):
            break
    return selected


def summarise_ablation(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "bundle", "label_column", "k_pct", "n_samples", "n_features", "n_classes", "min_class_n", "n_selected", "folds", "repeats", "model"]
    rows = []
    for keys, sub in raw.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        top = sub.loc[sub["subset_type"].eq("topvar"), "score_bal_acc"].astype(float)
        rnd = sub.loc[sub["subset_type"].eq("random"), "score_bal_acc"].astype(float)
        if top.empty or rnd.empty:
            continue
        top_score = float(top.iloc[0])
        rand_mean = float(rnd.mean())
        rand_sd = float(rnd.std(ddof=1)) if len(rnd) > 1 else 0.0
        delta_pp = 100.0 * (top_score - rand_mean)
        if delta_pp <= -5:
            verdict = "CONFIRMED_HARM"
        elif delta_pp < -1:
            verdict = "MILD_HARM"
        elif abs(delta_pp) <= 1:
            verdict = "TIE"
        else:
            verdict = "TOPVAR_HELP"
        rec.update({
            "topvar_bal_acc": top_score,
            "random_bal_acc_mean": rand_mean,
            "random_bal_acc_sd": rand_sd,
            "random_draws": int(len(rnd)),
            "delta_pp": float(delta_pp),
            "ablation_verdict": verdict,
        })
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["dataset", "label_column", "k_pct"]).reset_index(drop=True) if rows else pd.DataFrame()


def run_ablation(out_dir: Path, bundles: List[Path], k_values: Sequence[float], random_draws: int, folds: int, repeats: int, model: str, n_jobs: int, estimator_n_jobs: int, seed: int) -> pd.DataFrame:
    jobs = []
    failures = []
    # Pre-screen impossible bundles for cleaner logs.
    runnable = []
    for b in bundles:
        meta = bundle_metadata(b)
        if int(meta.get("min_class_n", 0) or 0) < int(folds):
            meta.update({"error": f"min_class_n < folds ({meta.get('min_class_n')} < {folds})"})
            failures.append(meta)
        elif meta.get("status") == "ok":
            runnable.append(b)
        else:
            meta.update({"error": meta.get("error", "bundle not ok")})
            failures.append(meta)
    for b in runnable:
        for k in k_values:
            jobs.append((str(b), float(k), "topvar", -1, int(seed), int(folds), int(repeats), str(model), int(estimator_n_jobs)))
            for draw in range(int(random_draws)):
                jobs.append((str(b), float(k), "random", int(draw), int(seed), int(folds), int(repeats), str(model), int(estimator_n_jobs)))
    log(f"Ablation model={model}: selected {len(runnable)} bundles; {len(jobs)} subset evaluations; n_jobs={n_jobs}")
    rows = []
    if jobs:
        if int(n_jobs) <= 1:
            for i, job in enumerate(jobs, 1):
                log(f"  job {i}/{len(jobs)} {Path(job[0]).name} {job[2]} K={job[1]:g} draw={job[3]}")
                try:
                    rows.append(eval_subset_worker(job))
                except Exception as e:
                    failures.append({"bundle_path": job[0], "bundle": Path(job[0]).name, "error": str(e)})
        else:
            with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
                futs = {ex.submit(eval_subset_worker, job): job for job in jobs}
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    job = futs[fut]
                    try:
                        rows.append(fut.result())
                    except Exception as e:
                        failures.append({"bundle_path": job[0], "bundle": Path(job[0]).name, "error": str(e)})
                        log(f"  FAILED {Path(job[0]).name} {job[2]} K={job[1]} draw={job[3]}: {e}")
                    if done % max(1, min(20, len(jobs)//10 or 1)) == 0 or done == len(jobs):
                        log(f"  completed {done}/{len(jobs)} jobs")
    raw_new = pd.DataFrame(rows)
    summary_new = summarise_ablation(raw_new)

    # Preserve earlier broader model runs. A focused sanity run should not erase
    # previously computed rows for the same model but different bundles/labels/K.
    # If the same row is recomputed, the newest result replaces the old one.
    raw_path = out_dir / f"ablation_raw_scores_{model}.csv"
    summary_path = out_dir / f"ablation_summary_{model}.csv"
    fail_path = out_dir / f"ablation_failures_{model}.csv"

    if raw_path.exists():
        try:
            raw_old = pd.read_csv(raw_path)
        except Exception:
            raw_old = pd.DataFrame()
    else:
        raw_old = pd.DataFrame()
    if not raw_old.empty and not raw_new.empty:
        raw = pd.concat([raw_old, raw_new], ignore_index=True)
        raw_keys = [c for c in ["dataset", "bundle", "label_column", "k_pct", "subset_type", "draw_idx", "folds", "repeats", "model"] if c in raw.columns]
        raw = raw.drop_duplicates(subset=raw_keys, keep="last") if raw_keys else raw
    elif not raw_new.empty:
        raw = raw_new
    else:
        raw = raw_old

    if summary_path.exists():
        try:
            summary_old = pd.read_csv(summary_path)
        except Exception:
            summary_old = pd.DataFrame()
    else:
        summary_old = pd.DataFrame()
    if not summary_old.empty and not summary_new.empty:
        summary = pd.concat([summary_old, summary_new], ignore_index=True)
        summary_keys = [c for c in ["dataset", "bundle", "label_column", "k_pct", "folds", "repeats", "model"] if c in summary.columns]
        summary = summary.drop_duplicates(subset=summary_keys, keep="last") if summary_keys else summary
    elif not summary_new.empty:
        summary = summary_new
    else:
        summary = summary_old

    failures_df = pd.DataFrame(failures)
    if fail_path.exists() and not failures_df.empty:
        try:
            failures_df = pd.concat([pd.read_csv(fail_path), failures_df], ignore_index=True)
        except Exception:
            pass

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    failures_df.to_csv(fail_path, index=False)
    if raw_new.empty:
        log(f"No new successful ablation rows for model={model}; preserved existing outputs if present")
    log(f"Wrote ablation outputs for model={model} to {out_dir}")
    if not summary_new.empty:
        print(f"\nAblation summary ({model}; newly computed rows):")
        with pd.option_context("display.max_colwidth", 80, "display.width", 240):
            print(summary_new.sort_values(["delta_pp", "dataset", "label_column", "k_pct"]).to_string(index=False))
    return summary


def _panel_call(group: pd.DataFrame) -> str:
    """Interpret tree-panel evidence at one dataset/label/K row.

    This is deliberately conservative. It does not call anything model-agnostic.
    It distinguishes primary XGBoost harm, boosting-family harm, and broader
    tree-panel harm.
    """
    deltas = {}
    verdicts = {}
    for c in group.index:
        pass
    row = group.iloc[0]
    for m in TREE_PANEL_MODELS:
        d = row.get(f"{m}_delta_pp", np.nan)
        v = row.get(f"{m}_verdict", np.nan)
        if pd.notna(d):
            deltas[m] = float(d)
        if pd.notna(v):
            verdicts[m] = str(v)
    if not deltas:
        diag = str(row.get("diagnostic_state", "UNKNOWN"))
        if diag == "VARIANCE_SIGNAL_DECOUPLED":
            return "VARIANCE_SIGNAL_DECOUPLED_ONLY"
        if diag == "VARIANCE_SIGNAL_ALIGNED":
            return "VARIANCE_SIGNAL_ALIGNED"
        return "INCONCLUSIVE"

    harm = {m for m, d in deltas.items() if d <= -5}
    mild_or_harm = {m for m, d in deltas.items() if d < -1}
    help_ = {m for m, d in deltas.items() if d > 1}
    boost_done = [m for m in BOOSTING_MODELS if m in deltas]
    bag_done = [m for m in BAGGING_MODELS if m in deltas]
    boost_harm = [m for m in boost_done if deltas[m] <= -5]
    bag_harm = [m for m in bag_done if deltas[m] <= -5]

    if "xgb" in harm and len(harm) >= 3:
        return "TREE_PANEL_CONFIRMED_HARM"
    if "xgb" in harm and boost_done and len(boost_harm) >= max(2, int(math.ceil(len(boost_done) / 2))):
        return "BOOSTING_FAMILY_CONFIRMED_HARM"
    if "xgb" in harm:
        return "XGB_CONFIRMED_HARM_MODEL_DEPENDENT"
    if len(harm) >= 2:
        return "NON_XGB_TREE_HARM_CHECK_REQUIRED"
    if len(help_) >= max(2, int(math.ceil(len(deltas) / 2))):
        return "TOPVAR_HELP_TREE_PANEL"
    diag = str(row.get("diagnostic_state", "UNKNOWN"))
    if diag == "VARIANCE_SIGNAL_DECOUPLED":
        return "VARIANCE_SIGNAL_DECOUPLED_ONLY"
    if mild_or_harm:
        return "MILD_OR_MODEL_SPECIFIC_HARM"
    if diag == "VARIANCE_SIGNAL_ALIGNED":
        return "VARIANCE_SIGNAL_ALIGNED"
    return "INCONCLUSIVE"


def final_calls(out_dir: Path) -> pd.DataFrame:
    vad_path = out_dir / "vad_multi_k_summary.csv"
    if not vad_path.exists():
        raise RuntimeError("Run --vad first")
    vad = pd.read_csv(vad_path)
    if vad.empty:
        return pd.DataFrame()
    base_cols = ["dataset", "bundle", "label_column", "k_pct"]
    merged = vad.copy()
    detected_models = []
    for p in sorted(out_dir.glob("ablation_summary_*.csv")):
        model = p.stem.replace("ablation_summary_", "")
        a = pd.read_csv(p)
        if a.empty:
            continue
        detected_models.append(model)
        keep = base_cols + ["topvar_bal_acc", "random_bal_acc_mean", "random_bal_acc_sd", "random_draws", "delta_pp", "ablation_verdict"]
        a = a[[c for c in keep if c in a.columns]].copy()
        a = a.rename(columns={
            "topvar_bal_acc": f"{model}_topvar_bal_acc",
            "random_bal_acc_mean": f"{model}_random_bal_acc_mean",
            "random_bal_acc_sd": f"{model}_random_bal_acc_sd",
            "random_draws": f"{model}_random_draws",
            "delta_pp": f"{model}_delta_pp",
            "ablation_verdict": f"{model}_verdict",
        })
        merged = merged.merge(a, on=base_cols, how="left")

    # Per-row integrated call.
    calls = []
    harm_counts = []
    tested_counts = []
    boosting_harm_counts = []
    bagging_harm_counts = []
    for _, r in merged.iterrows():
        row_df = pd.DataFrame([r])
        calls.append(_panel_call(row_df))
        deltas = {m: r.get(f"{m}_delta_pp", np.nan) for m in TREE_PANEL_MODELS}
        tested = [m for m, d in deltas.items() if pd.notna(d)]
        harms = [m for m in tested if float(deltas[m]) <= -5]
        harm_counts.append(len(harms))
        tested_counts.append(len(tested))
        boosting_harm_counts.append(len([m for m in harms if m in BOOSTING_MODELS]))
        bagging_harm_counts.append(len([m for m in harms if m in BAGGING_MODELS]))
    merged["final_call"] = calls
    merged["tree_models_tested_n"] = tested_counts
    merged["tree_models_harm_n"] = harm_counts
    merged["boosting_models_harm_n"] = boosting_harm_counts
    merged["bagging_models_harm_n"] = bagging_harm_counts

    path = out_dir / "final_integrated_calls.csv"
    merged.to_csv(path, index=False)

    # Dataset/label-level summary across K.
    label_rows = []
    for keys, g in merged.groupby(["dataset", "label_column"], dropna=False):
        rec = {"dataset": keys[0], "label_column": keys[1]}
        rec["n_k"] = int(g["k_pct"].nunique())
        rec["diagnostic_decoupled_k"] = int((g["diagnostic_state"].astype(str) == "VARIANCE_SIGNAL_DECOUPLED").sum())
        for m in detected_models:
            col = f"{m}_delta_pp"
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                rec[f"{m}_tested_k"] = int(vals.notna().sum())
                rec[f"{m}_harm_k"] = int((vals <= -5).sum())
                rec[f"{m}_mean_delta_pp"] = float(vals.mean()) if vals.notna().any() else np.nan
                rec[f"{m}_min_delta_pp"] = float(vals.min()) if vals.notna().any() else np.nan
        # Conservative label-level call.
        if (g["final_call"].astype(str) == "TREE_PANEL_CONFIRMED_HARM").any():
            rec["label_level_call"] = "TREE_PANEL_CONFIRMED_HARM"
        elif (g["final_call"].astype(str) == "BOOSTING_FAMILY_CONFIRMED_HARM").any():
            rec["label_level_call"] = "BOOSTING_FAMILY_CONFIRMED_HARM"
        elif (g["final_call"].astype(str) == "XGB_CONFIRMED_HARM_MODEL_DEPENDENT").any():
            rec["label_level_call"] = "XGB_CONFIRMED_HARM_MODEL_DEPENDENT"
        elif (g["final_call"].astype(str).str.contains("TOPVAR_HELP", na=False)).any():
            rec["label_level_call"] = "TOPVAR_HELP_OR_MIXED"
        elif (g["diagnostic_state"].astype(str) == "VARIANCE_SIGNAL_DECOUPLED").any():
            rec["label_level_call"] = "VARIANCE_SIGNAL_DECOUPLED_ONLY"
        elif (g["diagnostic_state"].astype(str) == "VARIANCE_SIGNAL_ALIGNED").all():
            rec["label_level_call"] = "VARIANCE_SIGNAL_ALIGNED"
        else:
            rec["label_level_call"] = "INCONCLUSIVE"
        label_rows.append(rec)
    label_summary = pd.DataFrame(label_rows)
    label_summary.to_csv(out_dir / "tree_panel_label_summary.csv", index=False)

    log(f"Wrote {path}")
    log(f"Detected ablation models in final table: {detected_models if detected_models else 'none'}")
    cols = [c for c in ["dataset", "label_column", "k_pct", "eta_ES", "VSA", "PCLA", "diagnostic_state", "xgb_delta_pp", "hgb_delta_pp", "gbc_delta_pp", "rf_delta_pp", "extratrees_delta_pp", "final_call", "class_counts"] if c in merged.columns]
    if cols:
        print("\nFinal integrated calls:")
        with pd.option_context("display.max_colwidth", 90, "display.width", 280):
            print(merged[cols].sort_values(["final_call", "dataset", "label_column", "k_pct"]).to_string(index=False))
    if not label_summary.empty:
        print("\nTree-panel label summary:")
        with pd.option_context("display.max_colwidth", 90, "display.width", 260):
            print(label_summary.sort_values(["label_level_call", "dataset", "label_column"]).to_string(index=False))
    return merged


def parse_csv_arg(s: str) -> Optional[List[str]]:
    vals = [x.strip() for x in str(s or "").split(",") if x.strip()]
    return vals or None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Manuscript-standard audit runner for existing validation bundles")
    ap.add_argument("--run-tag", default="", help="Output subfolder under outputs/15_val/_manuscript_audit. Default: timestamped")
    ap.add_argument("--targets", default="", help="Optional comma-separated dataset folder/source_accession filters, e.g. GSE148725,PXD007160")
    ap.add_argument("--labels", default="", help="Optional comma-separated exact label_column filters")
    ap.add_argument("--inventory", action="store_true", help="Write bundle inventory only")
    ap.add_argument("--vad", action="store_true", help="Run vardiag fold-wise TRAIN-only diagnostics")
    ap.add_argument("--ablate", action="store_true", help="Run TopVar-vs-Random ablation")
    ap.add_argument("--final", action="store_true", help="Merge VAD and ablation summaries into final_integrated_calls.csv")
    ap.add_argument("--only-promising", action="store_true", help="For ablation, use VAD-promising bundles from this run-tag")
    ap.add_argument("--max-bundles", type=int, default=9999, help="Maximum bundles to run for ablation/inventory after filtering")
    ap.add_argument("--k-values", default="1,5,10,20", help="Comma-separated K percentages")
    ap.add_argument("--vad-folds", type=int, default=5)
    ap.add_argument("--vad-repeats", type=int, default=5)
    ap.add_argument("--vad-pca-components", type=int, default=30)
    ap.add_argument("--full-folds", type=int, default=5)
    ap.add_argument("--full-repeats", type=int, default=5)
    ap.add_argument("--full-random-draws", type=int, default=10, help="Manuscript-equivalent default is 10; use 30/50 for precision sensitivity")
    ap.add_argument("--model", default="xgb", choices=["xgb", "hgb", "gbc", "rf", "extratrees", "all_tree_panel"], help="Single model or all_tree_panel")
    ap.add_argument("--models", default="", help="Optional comma-separated models for --ablate, e.g. xgb,hgb,gbc,rf,extratrees")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--estimator-n-jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_root = find_project_root()
    tag = args.run_tag.strip() or now_tag()
    out_dir = audit_dir(project_root, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = parse_csv_arg(args.targets)
    labels = parse_csv_arg(args.labels)
    k_values = [float(x.strip()) for x in args.k_values.split(",") if x.strip()]
    log(f"Project root: {project_root}")
    log(f"Audit output: {out_dir}")
    log(f"Targets filter: {targets if targets else 'ALL existing bundles'}")
    log(f"Labels filter: {labels if labels else 'ALL'}")
    bundles = discover_bundles(project_root, targets=targets, labels=labels)
    if int(args.max_bundles) > 0:
        bundles = bundles[: int(args.max_bundles)]
    if args.inventory or not any([args.vad, args.ablate, args.final]):
        write_inventory(project_root, out_dir, targets=targets, labels=labels)
    if args.vad:
        write_inventory(project_root, out_dir, targets=targets, labels=labels)
        run_vad(project_root, out_dir, bundles, k_values, folds=args.vad_folds, repeats=args.vad_repeats, pca_components=args.vad_pca_components, seed=args.seed)
    if args.ablate:
        if args.only_promising:
            all_bundles = discover_bundles(project_root, targets=targets, labels=labels)
            bundles = select_promising_from_vad(out_dir, all_bundles, max_bundles=args.max_bundles)
        if args.models.strip():
            model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        elif args.model == "all_tree_panel":
            model_list = TREE_PANEL_MODELS
        else:
            model_list = [args.model]
        bad = [m for m in model_list if m not in TREE_PANEL_MODELS]
        if bad:
            raise ValueError(f"Unknown model(s): {bad}; valid: {TREE_PANEL_MODELS}")
        for m in model_list:
            run_ablation(out_dir, bundles, k_values, random_draws=args.full_random_draws, folds=args.full_folds, repeats=args.full_repeats, model=m, n_jobs=args.n_jobs, estimator_n_jobs=args.estimator_n_jobs, seed=args.seed)
    if args.final:
        final_calls(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

