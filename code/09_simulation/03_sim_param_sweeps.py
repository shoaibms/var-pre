#!/usr/bin/env python3
"""PHASE 9 — 03_sim_param_sweeps.py: Parameter sweeps for phase diagram."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import ensure_dir, now_iso, classify_regime
from _shared.decoupling_metrics import compute_overlap_curve, rank_features_desc

def sweep_point(n_samples, n_features, between_var, within_var, k_pct, n_repeats=3):
    dis = []
    for seed in range(n_repeats):
        rng = np.random.default_rng(seed)
        # Balanced labels, without dropping samples when n_samples is odd
        n0 = n_samples // 2
        n1 = n_samples - n0
        y = np.array([0] * n0 + [1] * n1, dtype=int)
        rng.shuffle(y)
        means = rng.normal(0, np.sqrt(between_var), (2, n_features))
        means -= means.mean(0)
        X = np.array([
            means[y[i]] + rng.normal(0, np.sqrt(within_var), n_features)
            for i in range(n_samples)
        ])
        var = np.var(X, axis=0, ddof=1)
        imp = np.var(np.array([X[y == 0].mean(0), X[y == 1].mean(0)]), axis=0)
        features = [f"f{i}" for i in range(n_features)]
        curve = compute_overlap_curve(
            rank_features_desc(dict(zip(features, var))),
            rank_features_desc(dict(zip(features, imp))),
            [k_pct],
        )
        if curve:
            dis.append(curve[0].DI)

    if dis:
        return float(np.mean(dis)), float(np.std(dis))
    return 1.0, 0.0

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    
    output_dir = args.outputs_dir / "09_simulation" / "param_sweeps"
    ensure_dir(output_dir)
    
    wb_ratios = np.logspace(-1, 2, 10 if args.quick else 20)
    results = []
    print(f"[{now_iso()}] Running 1D sweep...")
    for wb in wb_ratios:
        di_m, di_s = sweep_point(200, 300 if args.quick else 500, 1.0, wb, 10.0, 2 if args.quick else 3)
        results.append({"wb_ratio": wb, "DI_mean": di_m, "DI_std": di_s, "regime": classify_regime(di_m)})
        print(f"  wb={wb:.2f}: DI={di_m:.3f}")
    
    pd.DataFrame(results).to_csv(output_dir / "param_sweep_1d.csv", index=False)
    print(f"[{now_iso()}] Done.")

if __name__ == "__main__":
    main()
