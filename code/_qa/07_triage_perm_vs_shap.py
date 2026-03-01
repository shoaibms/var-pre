#!/usr/bin/env python3
"""
07_triage_perm_vs_shap.py

Quick triage: Check baseline AUROC and top-K Jaccard overlap
between SHAP and Permutation importance.

Usage:
    python 07_triage_perm_vs_shap.py --outputs-dir outputs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    perm_dir = outputs_dir / "04_importance" / "per_model"  # Fixed: was "permutation"
    shap_dir = outputs_dir / "04_importance" / "joined_vp"
    
    hero_views = [
        ("mlomics", "methylation"),
        ("ibdmdb", "MPX"),
        ("mlomics", "CNV"),
    ]
    
    print("=" * 60)
    print("PERMUTATION vs SHAP TRIAGE")
    print("=" * 60)
    
    for ds, vw in hero_views:
        print(f"\n{ds}/{vw}:")
        print("-" * 40)
        
        # 1. Check baseline AUROC
        json_files = list(perm_dir.glob(f"perm_importance__{ds}__{vw}__*.json"))
        if json_files:
            with open(json_files[0]) as f:
                meta = json.load(f)
            auroc = meta.get("baseline_score_mean", meta.get("baseline_auroc", "?"))
            print(f"  Baseline AUROC: {auroc}")
        else:
            print("  Baseline AUROC: (no JSON found)")
        
        # 2. Load SHAP rankings
        shap_file = shap_dir / f"vp_joined__{ds}__{vw}.csv.gz"
        if not shap_file.exists():
            print("  SHAP file not found")
            continue
        
        shap_df = pd.read_csv(shap_file)
        shap_rank = shap_df.set_index("feature")["p_xgb_bal_rank"].to_dict()
        print(f"  SHAP features: {len(shap_rank)}")
        
        # 3. Load Perm rankings
        perm_files = list(perm_dir.glob(f"importance__{ds}__{vw}__*perm*.csv*"))
        if not perm_files:
            print("  Permutation file not found")
            continue
        
        perm_df = pd.read_csv(perm_files[0])
        print(f"  Perm file: {perm_files[0].name}")
        
        # Find importance column and compute rank
        imp_cols = [c for c in perm_df.columns if c in ["p_score", "importance", "perm_importance", "delta"]]
        if "p_rank" in perm_df.columns:
            perm_df["perm_rank"] = perm_df["p_rank"]
        elif imp_cols:
            perm_df["perm_rank"] = perm_df[imp_cols[0]].rank(ascending=False)
        else:
            print(f"  No importance column found. Cols: {perm_df.columns.tolist()}")
            continue
        
        feat_col = "feature" if "feature" in perm_df.columns else perm_df.columns[0]
        perm_rank = perm_df.set_index(feat_col)["perm_rank"].to_dict()
        
        # 4. Top-K Jaccard
        common = set(shap_rank.keys()) & set(perm_rank.keys())
        print(f"  Common features: {len(common)}")
        
        print(f"\n  {'K%':<6} {'Jaccard':<10} {'Overlap':<15}")
        print(f"  {'-'*35}")
        
        for k_pct in [1, 5, 10, 20]:
            k_n = max(1, int(len(common) * k_pct / 100))
            
            top_shap = set(f for f in common if shap_rank.get(f, 1e9) <= k_n)
            top_perm = set(f for f in common if perm_rank.get(f, 1e9) <= k_n)
            
            union = top_shap | top_perm
            jaccard = len(top_shap & top_perm) / len(union) if union else 0
            overlap = len(top_shap & top_perm)
            
            print(f"  {k_pct:<6} {jaccard:<10.3f} {overlap}/{k_n}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("  - Jaccard > 0.3 at top-10% = reasonable agreement")
    print("  - Low global ρ but decent top-K = feature redundancy")
    print("  - AUROC < 0.65 = permutation unreliable")
    print("=" * 60)


if __name__ == "__main__":
    main()