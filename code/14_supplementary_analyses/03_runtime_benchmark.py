#!/usr/bin/env python3
"""
Runtime benchmark: VAD-only vs supervised ablation.

Pulls ablation runtime from existing summary JSONs, times a live
VAD + SAF ranking computation, and outputs a comparison table.

Usage:
  python 14_supplementary_analyses/04_runtime_benchmark.py --outputs-dir outputs

Outputs:
  outputs/14_supplementary_analyses/runtime_benchmark/runtime_benchmark.csv
  outputs/14_supplementary_analyses/runtime_benchmark/runtime_benchmark_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.vad_metrics import eta2_features, eta_enrichment, vsa_mannwhitney, alpha_prime


# ── Views ──
HERO_VIEWS = [
    ("mlomics", "methylation"),
    ("ibdmdb", "MGX"),
    ("ccle", "mRNA"),
]


# ── Data loaders (minimal, reused from generate_feature_eta_tables) ──
def load_splits_json(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    info = d.get("info", {})
    dataset = str(info.get("dataset", path.stem.replace("splits__", "")))
    y = np.asarray(d["y"])
    sample_ids = np.asarray(d["sample_ids"]).astype(str)
    return dataset, y, sample_ids, info


def resolve_bundle(outputs_dir: Path, info: dict, dataset: str) -> Path:
    bp = info.get("bundle_path", None)
    if bp is not None:
        bp = Path(str(bp))
        if not bp.is_absolute():
            cand = (outputs_dir / bp).resolve()
            if cand.exists():
                return cand
        if bp.exists():
            return bp
    cands = list((outputs_dir / "01_bundles").rglob(f"*{dataset}*bundle*normalized*.npz"))
    if not cands:
        raise FileNotFoundError(f"Cannot resolve bundle for {dataset}")
    return sorted(cands, key=lambda p: (len(str(p)), str(p)))[0]


def load_bundle_view(bundle_path: Path, view: str):
    z = np.load(bundle_path, allow_pickle=False)
    x_key = f"X_{view}"
    if x_key not in z.files:
        alt = [k for k in z.files if k.lower() == x_key.lower()]
        x_key = alt[0] if alt else x_key
    X = z[x_key].astype(np.float32)
    sample_ids = z["sample_ids"].astype(str)
    for fk in [f"features_{view}", f"feature_names_{view}"]:
        if fk in z.files:
            return sample_ids, X, z[fk].astype(str)
    return sample_ids, X, np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)


def align_X(X, bundle_ids, splits_ids):
    if np.array_equal(bundle_ids, splits_ids):
        return X
    idx = {sid: i for i, sid in enumerate(bundle_ids)}
    return X[np.asarray([idx[sid] for sid in splits_ids], dtype=int), :]


def get_ablation_runtime(outputs_dir: Path, dataset: str, view: str,
                         model: str = "xgb_bal") -> Optional[float]:
    """Pull runtime_seconds from existing ablation summary JSON."""
    p = outputs_dir / "07_ablation" / "per_view" / f"ablation_summary__{dataset}__{view}__{model}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("runtime_seconds", None)


def main():
    ap = argparse.ArgumentParser(description="Runtime benchmark: VAD vs supervised ablation")
    ap.add_argument("--outputs-dir", type=str, required=True)
    ap.add_argument("--k-pct", type=int, default=10)
    ap.add_argument("--n-timing-repeats", type=int, default=3,
                    help="Repeat VAD timing this many times and take median")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / "14_supplementary_analyses" / "runtime_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    print("=" * 60)
    print("Runtime Benchmark: VAD + SAF vs Supervised Ablation")
    print("=" * 60)

    for dataset, view in HERO_VIEWS:
        print(f"\n── {dataset} / {view} ──")

        # 1. Pull ablation runtime from existing output
        abl_rt = get_ablation_runtime(outputs_dir, dataset, view)
        if abl_rt is not None:
            print(f"  Ablation runtime (from JSON): {abl_rt:.1f}s")
        else:
            print(f"  Ablation runtime: not available")

        # 2. Time live VAD + SAF computation
        splits_path = outputs_dir / "01_bundles" / "splits" / f"splits__{dataset}.json"
        if not splits_path.exists():
            print(f"  [SKIP] splits not found")
            continue

        ds, y, splits_ids, info = load_splits_json(splits_path)
        bundle_path = resolve_bundle(outputs_dir, info, dataset)
        bundle_ids, X, feats = load_bundle_view(bundle_path, view)
        X = align_X(X, bundle_ids, splits_ids)
        n_samples, n_features = X.shape

        # Time VAD diagnostics (eta2 + enrichment + VSA + alpha')
        vad_times = []
        for _ in range(args.n_timing_repeats):
            t0 = time.perf_counter()
            v_total, v_between, eta2 = eta2_features(X, y)
            _ = eta_enrichment(eta2, v_total, k_pct=args.k_pct)
            _ = vsa_mannwhitney(eta2, v_total, k_pct=args.k_pct)
            _ = alpha_prime(v_total, eta2)
            t1 = time.perf_counter()
            vad_times.append(t1 - t0)

        vad_median = float(np.median(vad_times))

        # Time SAF ranking (sort var_between descending + select top-k)
        saf_times = []
        n_select = max(1, int(round((args.k_pct / 100.0) * n_features)))
        for _ in range(args.n_timing_repeats):
            t0 = time.perf_counter()
            order = np.argsort(-v_between)
            _ = order[:n_select]
            t1 = time.perf_counter()
            saf_times.append(t1 - t0)

        saf_median = float(np.median(saf_times))
        vad_plus_saf = vad_median + saf_median

        print(f"  VAD compute:  {vad_median:.4f}s (median of {args.n_timing_repeats})")
        print(f"  SAF ranking:  {saf_median:.6f}s (median of {args.n_timing_repeats})")
        print(f"  VAD + SAF:    {vad_plus_saf:.4f}s total")
        if abl_rt and abl_rt > 0:
            speedup = abl_rt / vad_plus_saf
            print(f"  Speedup:      {speedup:.0f}x vs supervised ablation")

        rows.append({
            "dataset": dataset,
            "view": view,
            "n_samples": n_samples,
            "n_features": n_features,
            "k_pct": args.k_pct,
            "vad_seconds": round(vad_median, 4),
            "saf_ranking_seconds": round(saf_median, 6),
            "vad_plus_saf_seconds": round(vad_plus_saf, 4),
            "ablation_seconds": round(abl_rt, 1) if abl_rt else None,
            "speedup_factor": round(abl_rt / vad_plus_saf, 0) if abl_rt and vad_plus_saf > 0 else None,
            "n_timing_repeats": args.n_timing_repeats,
        })

    # Write outputs
    df = pd.DataFrame(rows)
    csv_path = out_dir / "runtime_benchmark.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "description": "Wall-clock comparison: VAD+SAF (model-free) vs supervised feature-selection ablation",
        "k_pct": args.k_pct,
        "n_hero_views": len(rows),
        "vad_plus_saf_max_seconds": round(float(df["vad_plus_saf_seconds"].max()), 4) if not df.empty else None,
        "ablation_total_seconds": round(float(df["ablation_seconds"].sum()), 1) if not df.empty and df["ablation_seconds"].notna().all() else None,
        "manuscript_line": None,
    }
    if not df.empty and df["ablation_seconds"].notna().all():
        vad_max = df["vad_plus_saf_seconds"].max()
        abl_total = df["ablation_seconds"].sum()
        summary["manuscript_line"] = (
            f"VAD + SAF completes in <{vad_max:.1f} s per view "
            f"(vs {abl_total/60:.0f} min for supervised ablation across {len(rows)} hero views)"
        )

    json_path = out_dir / "runtime_benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    if summary.get("manuscript_line"):
        print(f"\nManuscript-ready line:")
        print(f"  {summary['manuscript_line']}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
