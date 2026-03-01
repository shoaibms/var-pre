#!/usr/bin/env python3
"""
PHASE 8 — 04_exemplar_panels_data.py

Precompute exemplar feature statistics for manuscript figures.

Quadrant classification:
    Q1: High-V, High-P (ideal)
    Q2: High-V, Low-P (noise)
    Q3: Low-V, Low-P (stable non-predictive)
    Q4: Low-V, High-P (HIDDEN BIOMARKERS)

Outputs:
    outputs/08_biology/exemplar_panels/
        exemplar_features__{dataset}__{view}.csv
        quadrant_counts__{dataset}__{view}.json
        Q4_hidden_biomarkers.csv

Usage:
    python 04_exemplar_panels_data.py --outputs-dir outputs --top-n 20

Author: variance-prediction paradox pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import (
    ensure_dir, now_iso,
    discover_vp_files, discover_hero_views, load_vp_joined, pick_col,
    pick_importance_col, pick_variance_col,
    FEATURE_COL_CANDIDATES,
    classify_regime,
)
from _shared.decoupling_metrics import compute_overlap_curve, rank_features_desc


# =============================================================================
# QUADRANT CLASSIFICATION
# =============================================================================

def classify_quadrants(
    df: pd.DataFrame,
    v_col: str,
    p_col: str,
    v_threshold_pctl: float = 50.0,
    p_threshold_pctl: float = 50.0,
) -> pd.DataFrame:
    """Classify features into four quadrants."""
    df = df.copy()
    df["v_pctl"] = df[v_col].rank(pct=True) * 100
    df["p_pctl"] = df[p_col].rank(pct=True) * 100
    
    high_v = df["v_pctl"] > v_threshold_pctl
    high_p = df["p_pctl"] > p_threshold_pctl
    
    conditions = [high_v & high_p, high_v & ~high_p, ~high_v & ~high_p, ~high_v & high_p]
    choices = ["Q1", "Q2", "Q3", "Q4"]
    df["quadrant"] = np.select(conditions, choices, default="unknown")
    
    return df


def extract_exemplars(df: pd.DataFrame, v_col: str, p_col: str, top_n: int = 10) -> pd.DataFrame:
    """Extract top exemplar features for each quadrant."""
    exemplars = []
    
    for quadrant in ["Q1", "Q2", "Q3", "Q4"]:
        q_df = df[df["quadrant"] == quadrant].copy()
        if len(q_df) == 0:
            continue
        
        if quadrant in ["Q1", "Q4"]:
            q_df = q_df.nlargest(top_n, p_col)
        elif quadrant == "Q2":
            q_df = q_df.nlargest(top_n, v_col)
        else:
            q_df = q_df.nsmallest(top_n, v_col)
        
        q_df["exemplar_rank"] = range(1, len(q_df) + 1)
        exemplars.append(q_df)
    
    return pd.concat(exemplars, ignore_index=True) if exemplars else pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Extract exemplar features")
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--views", type=str, default="all")
    parser.add_argument("--model", type=str, default="xgb_bal")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--v-threshold", type=float, default=50.0)
    parser.add_argument("--p-threshold", type=float, default=50.0)
    
    args = parser.parse_args(argv)
    
    output_dir = args.outputs_dir / "08_biology" / "exemplar_panels"
    ensure_dir(output_dir)
    
    # Discover files
    if args.views.lower() == "hero":
        records = discover_hero_views(args.outputs_dir, model=args.model)
    elif args.views.lower() == "all":
        records = discover_vp_files(args.outputs_dir, model=args.model)
    else:
        pairs = [v.strip().split("/") for v in args.views.split(",")]
        all_records = discover_vp_files(args.outputs_dir, model=args.model)
        records = [r for r in all_records if (r.dataset, r.view) in [(p[0], p[1]) for p in pairs]]
    
    if not records:
        print("No VP files found.")
        return
    
    print(f"[{now_iso()}] Extracting exemplar features for {len(records)} views...")
    
    all_exemplars: List[pd.DataFrame] = []
    all_stats: List[Dict[str, Any]] = []
    
    for record in records:
        print(f"  Processing {record.short_key}...")
        try:
            df = load_vp_joined(record)
            
            feature_col = pick_col(df, FEATURE_COL_CANDIDATES, required=True)
            p_col = pick_importance_col(df, model=args.model)
            v_col = pick_variance_col(df)
            
            # Classify quadrants
            df = classify_quadrants(df, v_col, p_col, args.v_threshold, args.p_threshold)
            
            # Get stats
            counts = df["quadrant"].value_counts().to_dict()
            n = len(df)
            
            stats = {
                "dataset": record.dataset, "view": record.view,
                "n_features": n,
                "Q4_count": counts.get("Q4", 0),
                "Q4_pct": 100 * counts.get("Q4", 0) / n if n > 0 else 0,
            }
            
            # Compute DI for context
            # (avoid iterrows: too slow for large feature tables)
            feats = df[feature_col].astype(str).to_numpy()
            var_dict = dict(zip(feats, df[v_col].astype(float).to_numpy()))
            imp_dict = dict(zip(feats, df[p_col].astype(float).to_numpy()))
            curve = compute_overlap_curve(rank_features_desc(var_dict), rank_features_desc(imp_dict), [10.0])
            if curve:
                stats["DI_10pct"] = curve[0].DI
                stats["regime"] = classify_regime(curve[0].DI)
            
            all_stats.append(stats)
            
            # Extract exemplars
            exemplars = extract_exemplars(df, v_col, p_col, args.top_n)
            if len(exemplars) > 0:
                exemplars["dataset"] = record.dataset
                exemplars["view"] = record.view
                exemplars = exemplars.rename(columns={
                    feature_col: "feature", v_col: "variance_score", p_col: "importance_score",
                })
                exemplars.to_csv(
                    output_dir / f"exemplar_features__{record.dataset}__{record.view}.csv",
                    index=False
                )
                all_exemplars.append(exemplars)
            
            # Save stats
            with open(output_dir / f"quadrant_counts__{record.dataset}__{record.view}.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            print(f"    Q4: {stats['Q4_count']} features ({stats['Q4_pct']:.1f}%)")
            
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Combine Q4 exemplars
    if all_exemplars:
        combined = pd.concat(all_exemplars, ignore_index=True)
        combined.to_csv(output_dir / "all_exemplars_combined.csv", index=False)
        
        q4_only = combined[combined["quadrant"] == "Q4"]
        q4_only.to_csv(output_dir / "Q4_hidden_biomarkers.csv", index=False)
    
    # Save summary
    with open(output_dir / "exemplar_summary.json", "w") as f:
        json.dump({"timestamp": now_iso(), "stats": all_stats}, f, indent=2)
    
    print(f"\n[{now_iso()}] Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
