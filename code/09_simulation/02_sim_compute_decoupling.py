#!/usr/bin/env python3
"""PHASE 9 — 02_sim_compute_decoupling.py: Compute DI on synthetic data."""

from __future__ import annotations
import argparse, json, sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import ensure_dir, now_iso
from _shared.decoupling_metrics import compute_overlap_curve, rank_features_desc, di_auc

def compute_fisher(X, y):
    classes = np.unique(y)
    class_means = np.array([X[y == c].mean(axis=0) for c in classes])
    class_counts = np.array([np.sum(y == c) for c in classes])
    between = np.average((class_means - X.mean(axis=0))**2, axis=0, weights=class_counts)
    within = np.zeros(X.shape[1])
    for c, cnt in zip(classes, class_counts):
        if cnt > 1: within += cnt * np.var(X[y == c], axis=0, ddof=1)
    within /= X.shape[0]
    return between / np.maximum(within, 1e-10)

def classify_regime(
    di_10: float,
    dj_10: float,
    di_last: float,
    dj_last: float,
    eps_dj_coup: float = 0.10,
    eps_dj_anti: float = 0.10,
    anti_di_last: float = 1.11,
) -> str:
    """
    - COUPLED: strong positive ΔJ at 10% and DI well below 1
    - ANTI_ALIGNED: negative ΔJ at the largest K and DI_last clearly > 1
    - otherwise: DECOUPLED
    """
    # Coupled: strong above-random overlap
    if (dj_10 >= eps_dj_coup) and (di_10 <= 0.85):
        return "COUPLED"

    # Anti-aligned: below-random overlap is most reliable at largest K%
    if (dj_last <= -eps_dj_anti) and (di_last >= anti_di_last):
        return "ANTI_ALIGNED"

    return "DECOUPLED"

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--k-pcts", type=str, default="1,5,10,20")
    args = parser.parse_args(argv)
    
    k_pcts = [float(k) for k in args.k_pcts.split(",")]
    
    synthetic_dir = args.outputs_dir / "09_simulation" / "synthetic_data"
    output_dir = args.outputs_dir / "09_simulation" / "decoupling_results"
    ensure_dir(output_dir)
    
    npz_files = sorted(synthetic_dir.glob("synthetic__*.npz"))
    print(f"[{now_iso()}] Computing DI on {len(npz_files)} datasets...")
    
    summaries: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    for npz_path in npz_files:
        with np.load(npz_path) as data:
            X = data["X"]
            y = data["y"]
            scenario = data["scenario_name"].item() if hasattr(data["scenario_name"], "item") else str(data["scenario_name"])
            seed = int(data["seed"].item() if hasattr(data["seed"], "item") else data["seed"])
        variance = np.var(X, axis=0, ddof=1)
        importance = compute_fisher(X, y)
        features = [f"f{i}" for i in range(X.shape[1])]
        ranked_v = rank_features_desc(dict(zip(features, variance)))
        ranked_p = rank_features_desc(dict(zip(features, importance)))
        curve = compute_overlap_curve(ranked_v, ranked_p, k_pcts)
        
        for r in curve:
            curve_rows.append({
                "scenario": scenario,
                "seed": seed,
                "k_pct": float(r.k_pct),
                "k_features": int(r.k),
                "J": float(r.J),
                "delta_J": float(r.dJ),
                "DI": float(r.DI),
            })
        
        # r10 and r_last
        r10 = next((r for r in curve if abs(float(r.k_pct) - 10.0) < 1e-9), None)
        r_last = max(curve, key=lambda r: float(r.k_pct)) if curve else None

        di_10 = float(r10.DI) if r10 else float("nan")
        dj_10 = float(r10.dJ) if r10 else float("nan")

        di_last = float(r_last.DI) if r_last else float("nan")
        dj_last = float(r_last.dJ) if r_last else float("nan")

        auc = di_auc(curve) if curve else float("nan")
        regime = classify_regime(di_10, dj_10, di_last, dj_last)
        
        summaries.append({
            "scenario": scenario,
            "seed": seed,
            "DI_10pct": float(di_10),
            "DI_AUC": float(auc),
            "regime": regime,
        })
        print(
            f"  {npz_path.name}: "
            f"DI@10%={di_10:.3f}  DI@{float(r_last.k_pct):g}%={di_last:.3f}  "
            f"dJ@10%={dj_10:.3f}  dJ@{float(r_last.k_pct):g}%={dj_last:.3f}  "
            f"AUC={auc:.3f} → {regime}"
        )
    
    pd.DataFrame(summaries).to_csv(output_dir / "sim_di_summary.csv", index=False)
    params = {
        "k_pcts": k_pcts,
        "classification": {
            "coupled": {"uses": "dJ@10% and DI@10%", "dJ_10_min": 0.10, "DI_10_max": 0.85},
            "anti_aligned": {"uses": "dJ@maxK and DI@maxK", "dJ_last_max": -0.10, "DI_last_min": 1.11},
            "otherwise": "DECOUPLED",
        },
    }
    with open(output_dir / "classification_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(output_dir / "sim_di_curves.csv.gz", index=False, compression="gzip")
    print(f"[{now_iso()}] Done.")

if __name__ == "__main__":
    main()
