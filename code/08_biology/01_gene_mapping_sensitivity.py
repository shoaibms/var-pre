#!/usr/bin/env python3
"""
PHASE 8 — 01_gene_mapping_sensitivity.py

Map non-gene features to genes for cross-modality pathway comparison.

Feature type handling:
    - Gene expression (mRNA, proteomics): Direct pass-through
    - Methylation CpGs: Extract from probe names if embedded
    - CNV: Gene-level CNV (direct)
    - Taxonomy/metabolites: Not mappable to human genes (excluded)

Outputs:
    outputs/08_biology/gene_mapping/
        gene_mapping__{dataset}__{view}.csv.gz      # feature → gene mapping
        mapping_stats__{dataset}__{view}.json       # mapping statistics
        gene_lists__{dataset}__{view}.json          # TopV/TopP/Q4 gene lists

Usage:
    python 01_gene_mapping_sensitivity.py --outputs-dir outputs
    python 01_gene_mapping_sensitivity.py --outputs-dir outputs --views hero

Author: variance-prediction paradox pipeline
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Add parent to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import (
    ensure_dir, now_iso,
    discover_vp_files, discover_hero_views, load_vp_joined, pick_col,
    VARIANCE_COL_CANDIDATES, IMPORTANCE_COL_CANDIDATES, FEATURE_COL_CANDIDATES,
    get_feature_type_info, is_gene_mappable, VPRecord,
)
from _shared.decoupling_metrics import rank_features_desc


# =============================================================================
# GENE MAPPING
# =============================================================================

def extract_gene_from_feature_name(feature: str, view: str) -> Optional[str]:
    """Attempt to extract gene symbol from feature name."""
    feature = str(feature).strip()
    
    # Handle "GENE (EntrezID)" format (e.g., "KRT19 (3880)")
    m = re.match(r'^([A-Z][A-Z0-9]{1,14})\s*\(\d+\)$', feature)
    if m:
        return m.group(1)
    
    # Pattern 1: Direct gene symbol (uppercase letters + numbers, 2-15 chars)
    if re.match(r'^[A-Z][A-Z0-9]{1,14}$', feature):
        return feature
    
    # Pattern 2: CpG probe with embedded gene: cg12345_GENE
    m = re.match(r'^cg\d+[_|]([A-Z][A-Z0-9]{1,14})', feature, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Pattern 3: Gene symbol followed by probe ID: BRCA1_cg12345
    m = re.match(r'^([A-Z][A-Z0-9]{1,14})[_|]cg\d+', feature, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Pattern 4: Protein with _HUMAN suffix
    m = re.match(r'^([A-Z][A-Z0-9]{1,14})_HUMAN', feature, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # For views where features ARE gene names
    type_info = get_feature_type_info(view)
    if type_info.get("mapping_method") == "direct":
        gene = re.sub(r'[_\-]\d+$', '', feature)
        if re.match(r'^[A-Z][A-Z0-9]{1,14}$', gene):
            return gene
    
    return None


def map_features_to_genes(
    df: pd.DataFrame,
    view: str,
    feature_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Map features to genes for a given view."""
    type_info = get_feature_type_info(view)
    mappable = type_info.get("mappable_to_gene", False)
    method = type_info.get("mapping_method", None)
    
    results = []
    genes_found: Set[str] = set()
    
    for feature in df[feature_col]:
        feature = str(feature)
        
        if not mappable:
            results.append({
                "feature": feature, "gene": None, 
                "mapped": False, "mapping_method": "not_mappable",
            })
        else:
            gene = extract_gene_from_feature_name(feature, view)
            if gene:
                genes_found.add(gene)
                results.append({
                    "feature": feature, "gene": gene,
                    "mapped": True, "mapping_method": method or "extracted",
                })
            else:
                results.append({
                    "feature": feature, "gene": None,
                    "mapped": False, "mapping_method": "failed",
                })
    
    mapping_df = pd.DataFrame(results)
    n_mapped = len(mapping_df[mapping_df["mapped"]])
    
    stats = {
        "view": view,
        "feature_type": type_info.get("type", "unknown"),
        "mappable_to_gene": mappable,
        "mapping_method": method,
        "n_features": len(df),
        "n_mapped": n_mapped,
        "n_unmapped": len(df) - n_mapped,
        "n_unique_genes": len(genes_found),
        "mapping_rate": n_mapped / max(len(df), 1),
    }
    
    return mapping_df, stats


def extract_gene_lists(
    vp_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    k_pcts: List[float] = [5, 10, 20],
) -> Dict[str, Any]:
    """Extract TopV, TopP, and Q4 gene lists at various K thresholds."""
    feature_col = pick_col(vp_df, FEATURE_COL_CANDIDATES, required=True)
    v_col = pick_col(vp_df, VARIANCE_COL_CANDIDATES, required=True)
    p_col = pick_col(vp_df, IMPORTANCE_COL_CANDIDATES, required=True)
    
    merged = vp_df.merge(mapping_df, left_on=feature_col, right_on="feature", how="left")
    merged = merged[merged["mapped"] == True].copy()
    
    if len(merged) == 0:
        return {"warning": "No genes mapped", "topV": {}, "topP": {}, "Q4": {}}
    
    # Aggregate by gene (max score)
    var_unique = merged.groupby("gene")[v_col].max().to_dict()
    imp_unique = merged.groupby("gene")[p_col].max().to_dict()
    
    ranked_by_var = rank_features_desc(var_unique)
    ranked_by_imp = rank_features_desc(imp_unique)
    n_genes = len(ranked_by_var)
    
    gene_lists: Dict[str, Any] = {"topV": {}, "topP": {}, "Q4": {}, "n_genes_mapped": n_genes}
    
    for k_pct in k_pcts:
        k = max(1, int(n_genes * k_pct / 100))
        k_key = f"K{int(k_pct)}pct"
        
        topV = set(ranked_by_var[:k])
        topP = set(ranked_by_imp[:k])
        lowV = set(ranked_by_var[-k:])
        Q4 = lowV & topP
        
        gene_lists["topV"][k_key] = list(topV)
        gene_lists["topP"][k_key] = list(topP)
        gene_lists["Q4"][k_key] = list(Q4)
        gene_lists[f"overlap_K{int(k_pct)}pct"] = {
            "topV_size": len(topV), "topP_size": len(topP),
            "Q4_size": len(Q4), "topV_topP_overlap": len(topV & topP),
        }
    
    return gene_lists


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Map features to genes")
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--views", type=str, default="all")
    parser.add_argument("--model", type=str, default="xgb_bal")
    parser.add_argument("--k-pcts", type=str, default="5,10,20")
    
    args = parser.parse_args(argv)
    k_pcts = [float(k) for k in args.k_pcts.split(",")]
    
    output_dir = args.outputs_dir / "08_biology" / "gene_mapping"
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
    
    print(f"[{now_iso()}] Gene mapping for {len(records)} views...")
    
    for record in records:
        print(f"  Processing {record.short_key}...")
        try:
            df = load_vp_joined(record)
            feature_col = pick_col(df, FEATURE_COL_CANDIDATES, required=True)
            
            mapping_df, stats = map_features_to_genes(df, record.view, feature_col)
            
            # Save mapping
            mapping_df.to_csv(
                output_dir / f"gene_mapping__{record.dataset}__{record.view}.csv.gz",
                index=False, compression="gzip"
            )
            
            # Save stats
            with open(output_dir / f"mapping_stats__{record.dataset}__{record.view}.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            # Extract gene lists
            gene_lists = extract_gene_lists(df, mapping_df, k_pcts)
            gene_lists["dataset"] = record.dataset
            gene_lists["view"] = record.view
            
            with open(output_dir / f"gene_lists__{record.dataset}__{record.view}.json", "w") as f:
                json.dump(gene_lists, f, indent=2)
            
            print(f"    Mapped: {stats['n_mapped']}/{stats['n_features']} ({stats['mapping_rate']:.1%})")
            
        except Exception as e:
            print(f"    ERROR: {e}")
    
    print(f"\n[{now_iso()}] Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
