#!/usr/bin/env python3
"""
PHASE 12 — Pre-flight Diagnostic (VAD)

Compute supervision-light diagnostics to predict whether variance-filtering will
likely be SAFE vs HARMFUL for a given (dataset, view), without training models.

Diagnostics (per fold on TRAIN split only):
  - etaES(K): signal enrichment of TopVar(K%) features (eta^2 enrichment)
  - VSA(K): Mann–Whitney effect between eta^2 distributions (TopVar vs Rest)
  - alpha': Spearman(Var_total, eta^2) (K-free, monotone association)
  - SAS/PCLA: PCA alignment diagnostics (multivariate)

Outputs under: <outputs_dir>/<out_dirname>/
  - per_view/vad__<dataset>__<view>.csv.gz  (fold-level)
  - vad_long.csv.gz                         (all views)
  - vad_summary.csv                         (aggregated per dataset/view/k_pct)

Usage (PowerShell):
  python .\\code\\compute\12_diagnostic\01_compute_vad.py `
    --outputs-dir "<path-to-outputs>" `
    --out-dirname "12_diagnostic" `
    --views "mlomics:methylation,ibdmdb:MGX,ccle:mRNA" `
    --k-pcts "1,5,10,20" `
    --max-repeats 3 `
    --pca-components 30
"""

from __future__ import annotations

import argparse
import gzip
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

# allow imports from code/compute/_shared
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.vad_metrics import (
    eta2_features, eta_enrichment, vsa_mannwhitney, alpha_prime, pca_alignment,
    f_di, classify_zone,
)

_EPS = 1e-12


# -----------------------------
# Defaults / hero views
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
# IO helpers (minimal, consistent with existing scripts)
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
# CLI helpers
# -----------------------------
def parse_views_arg(s: str) -> List[Tuple[str, str]]:
    """
    Parse:
      "mlomics:methylation,ccle:mRNA"
    """
    out = []
    for item in [x.strip() for x in str(s).split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"Bad --views item '{item}'. Expected dataset:view")
        ds, vw = item.split(":", 1)
        out.append((ds.strip(), vw.strip()))
    return out


def infer_task_type(y: np.ndarray) -> str:
    # project logic: floats are regression unless small integer-like set
    if np.issubdtype(y.dtype, np.floating):
        u = np.unique(y[~np.isnan(y)])
        if len(u) <= 20 and np.all(np.isclose(u, np.round(u))):
            return "classification"
        return "regression"
    return "classification"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--out-dirname", default="12_diagnostic")
    ap.add_argument("--views", default="", help='Comma list: "dataset:view,dataset:view"')
    ap.add_argument("--hero-views", action="store_true", help="Use HERO_VIEWS (3 key views)")
    ap.add_argument("--all-views", action="store_true", help="Use ALL 14 views (full validation)")
    ap.add_argument("--k-pcts", default="1,5,10,20")
    ap.add_argument("--max-repeats", type=int, default=3)
    ap.add_argument("--pca-components", type=int, default=30)
    ap.add_argument("--write-feature-tables", action="store_true",
                   help="Write per-feature eta^2 tables (can be large)")
    ap.add_argument("--skip-pca", action="store_true")
    ap.add_argument(
        "--primary-k",
        type=int,
        default=10,
        help="Only assign predicted_zone for this K-percent; other K get blank zone.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel workers (threading only; avoids copying large X)")
    ap.add_argument("--backend", default="threading", choices=["threading"], help="Parallel backend (only threading is supported)")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def _summarise(long_df: pd.DataFrame, primary_k: int = 10) -> pd.DataFrame:
    """
    Aggregate fold-level rows into view-level summary with simple uncertainty
    across folds/repeats (percentiles).
    """
    if long_df.empty:
        return long_df

    agg = []
    group_cols = ["dataset", "view", "k_pct"]

    for (ds, vw, k), g in long_df.groupby(group_cols):
        row = {
            "dataset": ds,
            "view": vw,
            "k_pct": int(k),
            "n_rows": int(len(g)),
        }

        for col in ["eta_es", "eta_topv", "eta_all", "vsa", "alpha_prime", "sas", "pcla", "f_di"]:
            x = pd.to_numeric(g[col], errors="coerce").astype(float).to_numpy()
            x = x[np.isfinite(x)]
            if x.size == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_pctl_2.5"] = np.nan
                row[f"{col}_pctl_97.5"] = np.nan
            else:
                row[f"{col}_mean"] = float(np.mean(x))
                row[f"{col}_pctl_2.5"] = float(np.quantile(x, 0.025))
                row[f"{col}_pctl_97.5"] = float(np.quantile(x, 0.975))

        # basic meta
        for col in ["n_train", "p", "n_classes"]:
            x = pd.to_numeric(g[col], errors="coerce").astype(float).to_numpy()
            x = x[np.isfinite(x)]
            row[f"{col}_mean"] = float(np.mean(x)) if x.size else np.nan

        # ── Zones only at primary K ──
        k_pct = row.get("k_pct", np.nan)
        if int(k_pct) != int(primary_k):
            row["predicted_zone"] = ""
        else:
            row["predicted_zone"] = classify_zone(
                eta_es=row["eta_es_mean"],
                vsa=row["vsa_mean"],
                eta_es_lo=row["eta_es_pctl_2.5"],
                eta_es_hi=row["eta_es_pctl_97.5"],
                vsa_lo=row["vsa_pctl_2.5"],
                vsa_hi=row["vsa_pctl_97.5"],
                margin=0.05,
            )

        agg.append(row)

    return pd.DataFrame(agg).sort_values(["dataset", "view", "k_pct"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / args.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    per_view_dir = out_dir / "per_view"
    per_view_dir.mkdir(parents=True, exist_ok=True)

    k_pcts = [int(x.strip()) for x in str(args.k_pcts).split(",") if x.strip()]
    primary_k = int(args.primary_k)

    if args.views.strip():
        targets = parse_views_arg(args.views)
    elif args.all_views:
        targets = list(ALL_VIEWS)
    elif args.hero_views:
        targets = list(HERO_VIEWS)
    else:
        targets = list(ALL_VIEWS)   # default: run everything for full validation

    # group targets by dataset (splits per dataset)
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
            print(f"[WARN] dataset={dataset}: task_type={task_type}; VAD (eta^2) is defined for classification. Skipping.")
            continue

        bundle_path = resolve_bundle_path(outputs_dir, splits)

        for view in views:
            view_safe = view.replace("/", "__")
            out_path = per_view_dir / f"vad__{dataset}__{view_safe}.csv.gz"
            if out_path.exists():
                print(f"  Skipping {dataset}/{view} (already saved)")
                df_view = pd.read_csv(out_path, compression="gzip")
                rows_all.extend(df_view.to_dict("records"))
                continue

            # load bundle view
            _y_bundle, bundle_ids, X, feats, _info_bundle = load_bundle_view(bundle_path, view)
            X = align_X_to_splits_order(X, bundle_ids, splits.sample_ids)

            fold_ids = splits.fold_ids
            n_repeats_total, _n_samples = fold_ids.shape
            n_repeats = min(int(args.max_repeats), int(n_repeats_total))

            unique_folds = np.unique(fold_ids[0, :])
            n_folds = int(len(unique_folds))

            view_rows: List[Dict[str, object]] = []

            def _fold_job(r: int, f: int, train_idx: np.ndarray) -> List[Dict[str, object]]:
                Xtr = X[train_idx, :]
                ytr = splits.y[train_idx]

                n_train = int(len(train_idx))
                p = int(Xtr.shape[1])
                n_classes = int(np.unique(ytr).size)

                # core univariate primitives
                v_total, v_between, eta2 = eta2_features(Xtr, ytr)

                # K-free + PCA metrics (once per fold)
                a_prime = alpha_prime(v_total, eta2)

                if args.skip_pca:
                    sas = float("nan")
                    pcla = float("nan")
                else:
                    pa = pca_alignment(
                        Xtr, ytr,
                        n_components=int(args.pca_components),
                        random_state=int(args.seed) + 1000 * int(r) + int(f),
                    )
                    sas = float(pa.get("sas", np.nan))
                    pcla = float(pa.get("pcla", np.nan))

                # primary-k VSA once per fold
                vsa = vsa_mannwhitney(eta2, v_total, k_pct=primary_k)

                # F-DI (supervision-free DI analog) once per fold at primary K
                fdi = f_di(eta2, v_total, k_pct=primary_k)

                rows: List[Dict[str, object]] = []
                for k in k_pcts:
                    eta_es, eta_topv, eta_all = eta_enrichment(eta2, v_total, k_pct=int(k))
                    rows.append({
                        "dataset": dataset,
                        "view": view,
                        "repeat": int(r),
                        "fold": int(f),
                        "k_pct": int(k),
                        "primary_k": int(primary_k),
                        "n_train": n_train,
                        "p": p,
                        "n_classes": n_classes,
                        "eta_es": float(eta_es),
                        "eta_topv": float(eta_topv),
                        "eta_all": float(eta_all),
                        "vsa": float(vsa),
                        "alpha_prime": float(a_prime),
                        "sas": float(sas),
                        "pcla": float(pcla),
                        "f_di": float(fdi),
                    })
                return rows

            # Build fold tasks (repeat × fold), expand to k_pcts inside job
            tasks: List[Tuple[int, int, np.ndarray]] = []
            for r in range(n_repeats):
                fold_r = fold_ids[r, :]
                for f in np.unique(fold_r):
                    train_idx = np.where(fold_r != f)[0]
                    tasks.append((int(r), int(f), train_idx))

            if int(args.n_jobs) <= 1 or thread_map is None:
                # Sequential with fine-grained progress (repeat×fold×K)
                pbar_total = len(tasks) * len(k_pcts)
                pbar = tqdm(total=pbar_total, desc=f"VAD {dataset}/{view}", unit="kfold")
                for (r, f, train_idx) in tasks:
                    rows = _fold_job(r, f, train_idx)
                    view_rows.extend(rows)
                    pbar.update(len(k_pcts))
                pbar.close()
            else:
                # Parallel (threading): avoids copying large X to worker processes
                if str(args.backend) != "threading":
                    raise ValueError("Only --backend threading is supported (to avoid duplicating large view matrices).")

                fold_rows = thread_map(
                    lambda t: _fold_job(*t),
                    tasks,
                    max_workers=int(args.n_jobs),
                    desc=f"VAD {dataset}/{view}",
                    unit="fold",
                )
                for rows in fold_rows:
                    view_rows.extend(rows)


            df_view = pd.DataFrame(view_rows)
            df_view.to_csv(out_path, index=False, compression="gzip")
            if args.write_feature_tables:
                # ── Per-feature η² output (FULL DATA; for figures/supplement only) ──
                # Note: this uses all samples (including test folds). Do not use this
                # table for cross-validated validation claims.
                v_total_full, v_between_full, eta2_full = eta2_features(X, splits.y)
                feat_df = pd.DataFrame({
                    "dataset": dataset,
                    "view": view,
                    "feature": feats,
                    "var_total": v_total_full.astype(np.float64),
                    "var_between": v_between_full.astype(np.float64),
                    "var_within": (v_total_full - v_between_full).astype(np.float64),
                    "eta_sq": eta2_full.astype(np.float64),
                })
                feat_df["var_rank"] = feat_df["var_total"].rank(ascending=False, method="first").astype(int)
                feat_df["eta_rank"] = feat_df["eta_sq"].rank(ascending=False, method="first").astype(int)
                feat_path = per_view_dir / f"feature_eta_sq__{dataset}__{view_safe}.csv.gz"
                feat_df.to_csv(feat_path, index=False, compression="gzip")

            rows_all.extend(view_rows)

    long_df = pd.DataFrame(rows_all)
    long_path = out_dir / "vad_long.csv.gz"
    long_df.to_csv(long_path, index=False, compression="gzip")

    sum_df = _summarise(long_df, primary_k=args.primary_k)
    sum_path = out_dir / "vad_summary.csv"
    sum_df.to_csv(sum_path, index=False)

    # quick console summary for primary K=10 and k_pct=10
    if not sum_df.empty:
        p10 = sum_df[sum_df["k_pct"].astype(int) == 10].copy()
        if not p10.empty:
            cols = ["dataset", "view", "eta_es_mean", "alpha_prime_mean", "sas_mean", "pcla_mean", "f_di_mean", "predicted_zone"]
            avail = [c for c in cols if c in p10.columns]
            print("\n=== VAD summary (k=10%) ===")
            print(p10[avail].to_string(index=False))

    dt = time.time() - t0
    print(f"\n[OK] Phase12 VAD complete in {dt/60.0:.1f} min")
    print(f"  wrote: {long_path}")
    print(f"  wrote: {sum_path}")


if __name__ == "__main__":
    main()
