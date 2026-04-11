#!/usr/bin/env python3
"""
Generate per-feature eta^2 tables for SAF integration.

This script produces the feature_eta_sq__<dataset>__<view>.csv.gz files
that Phase 12 would write with --write-feature-tables, but without
re-running the full VAD computation (which skips already-saved views).

Outputs:
  outputs/12_diagnostic/per_view/feature_eta_sq__<dataset>__<view>.csv.gz

Usage:
  python 14_supplementary_analyses/01_generate_feature_eta_tables.py --outputs-dir outputs
  python 14_supplementary_analyses/01_generate_feature_eta_tables.py --outputs-dir outputs --hero-only
  python 14_supplementary_analyses/01_generate_feature_eta_tables.py --outputs-dir outputs --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Import eta2_features from shared module ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.vad_metrics import eta2_features


# ── Views ──
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


# ── Data loading (mirrors Phase 12 exactly) ──
@dataclass(frozen=True)
class SplitData:
    dataset: str
    y: np.ndarray
    sample_ids: np.ndarray
    groups: np.ndarray
    fold_ids: np.ndarray
    info: Dict[str, object]


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
    return SplitData(dataset=dataset, y=y, sample_ids=sample_ids,
                     groups=groups, fold_ids=fold_ids, info=info)


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
    cands = []
    for root in [outputs_dir / "01_bundles", outputs_dir,
                 outputs_dir / "archive_pre_migration_20260127"]:
        if root.exists():
            cands += list(root.rglob(f"*{ds}*bundle*normalized*.npz"))
    if not cands:
        raise FileNotFoundError(f"Cannot resolve bundle for dataset={ds}")
    return sorted(cands, key=lambda p: (len(str(p)), str(p)))[0]


def load_bundle_view(bundle_path: Path, view: str):
    z = np.load(bundle_path, allow_pickle=False)
    info = json.loads(str(z["info"])) if "info" in z.files else {}
    y = z["y"] if "y" in z.files else None
    sample_ids = z["sample_ids"].astype(str) if "sample_ids" in z.files else None

    x_key = f"X_{view}"
    if x_key not in z.files:
        alt = [k for k in z.files if k.lower() == x_key.lower()]
        if alt:
            x_key = alt[0]
        else:
            raise KeyError(f"Bundle missing view '{view}'. "
                           f"Available: {[k for k in z.files if k.startswith('X_')]}")
    X = z[x_key].astype(np.float32)

    for fk in [f"features_{view}", f"feature_names_{view}"]:
        if fk in z.files:
            feats = z[fk].astype(str)
            break
    else:
        feats = np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)

    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)

    return y, sample_ids, X, feats, info


def align_X_to_splits_order(X, bundle_ids, splits_ids):
    if np.array_equal(bundle_ids, splits_ids):
        return X
    idx = {sid: i for i, sid in enumerate(bundle_ids)}
    order = [idx[sid] for sid in splits_ids]
    return X[np.asarray(order, dtype=int), :]


# ── Main ──
def main():
    ap = argparse.ArgumentParser(description="Generate per-feature eta^2 tables for SAF")
    ap.add_argument("--outputs-dir", type=str, required=True)
    ap.add_argument("--hero-only", action="store_true",
                    help="Generate for 3 hero views only (default: all 14)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing feature tables")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    per_view_dir = outputs_dir / "12_diagnostic" / "per_view"
    per_view_dir.mkdir(parents=True, exist_ok=True)

    targets = HERO_VIEWS if args.hero_only else ALL_VIEWS

    # Group by dataset
    ds_to_views: Dict[str, List[str]] = {}
    for ds, vw in targets:
        ds_to_views.setdefault(ds, []).append(vw)

    t0 = time.time()
    n_written = 0
    n_skipped = 0

    for dataset, views in ds_to_views.items():
        splits_path = outputs_dir / "01_bundles" / "splits" / f"splits__{dataset}.json"
        if not splits_path.exists():
            print(f"[ERROR] Missing splits: {splits_path}")
            continue

        splits = load_splits_json(splits_path)
        bundle_path = resolve_bundle_path(outputs_dir, splits)

        for view in views:
            view_safe = view.replace("/", "__")
            feat_path = per_view_dir / f"feature_eta_sq__{dataset}__{view_safe}.csv.gz"

            if feat_path.exists() and not args.force:
                print(f"  [SKIP] {dataset}/{view} — already exists. Use --force to overwrite.")
                n_skipped += 1
                continue

            print(f"  Computing {dataset}/{view} ... ", end="", flush=True)
            tv = time.time()

            _y_bundle, bundle_ids, X, feats, _info = load_bundle_view(bundle_path, view)
            X = align_X_to_splits_order(X, bundle_ids, splits.sample_ids)

            v_total, v_between, eta2 = eta2_features(X, splits.y)

            feat_df = pd.DataFrame({
                "dataset": dataset,
                "view": view,
                "feature": feats,
                "var_total": v_total.astype(np.float64),
                "var_between": v_between.astype(np.float64),
                "var_within": (v_total - v_between).astype(np.float64),
                "eta_sq": eta2.astype(np.float64),
            })
            feat_df["var_rank"] = feat_df["var_total"].rank(ascending=False, method="first").astype(int)
            feat_df["eta_rank"] = feat_df["eta_sq"].rank(ascending=False, method="first").astype(int)

            feat_df.to_csv(feat_path, index=False, compression="gzip")
            dt = time.time() - tv
            print(f"{len(feat_df)} features, {dt:.1f}s")
            n_written += 1

    total = time.time() - t0
    print(f"\n[DONE] {n_written} written, {n_skipped} skipped, {total:.1f}s total")
    print(f"  Output dir: {per_view_dir}")


if __name__ == "__main__":
    main()
