#!/usr/bin/env python3
"""
PHASE 8 — 02_pathway_enrichment.py

Run pathway enrichment on TopV vs TopP gene lists.

Methods:
    1. g:Profiler (gprofiler-official) - primary, well-maintained
    2. Local hypergeometric test (fallback)

Outputs:
    outputs/08_biology/pathway_enrichment/
        enrichment__{dataset}__{view}__{gene_set}.csv.gz
        enrichment_summary__{dataset}__{view}.json

Usage:
    python 02_pathway_enrichment.py --outputs-dir outputs --method gprofiler

Author: variance-prediction paradox pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import (
    ensure_dir, now_iso,
    discover_vp_files, discover_hero_views, VPRecord,
)
from _shared.decoupling_metrics import bh_fdr, jaccard


# =============================================================================
# G:PROFILER ENRICHMENT
# =============================================================================

def run_gprofiler_enrichment(
    gene_list: List[str],
    organism: str = "hsapiens",
    significance_threshold: float = 0.05,
) -> pd.DataFrame:
    """Run enrichment using g:Profiler API."""
    try:
        from gprofiler import GProfiler
    except ImportError:
        raise ImportError("gprofiler-official not installed. Run: pip install gprofiler-official")
    
    if not gene_list:
        return pd.DataFrame()
    
    gp = GProfiler(return_dataframe=True)
    sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]
    
    try:
        results = gp.profile(
            organism=organism,
            query=gene_list,
            sources=sources,
            significance_threshold_method="fdr",
            user_threshold=significance_threshold,
        )
        
        if results is None or len(results) == 0:
            return pd.DataFrame()
        
        results = results.rename(columns={
            "native": "pathway_id", "name": "pathway_name",
            "source": "source", "p_value": "pval",
            "term_size": "set_size", "query_size": "query_size",
            "intersection_size": "overlap",
        })
        
        if "fdr" not in results.columns:
            results["fdr"] = bh_fdr(results["pval"].values)
        
        return results
        
    except Exception as e:
        print(f"    g:Profiler error: {e}")
        return pd.DataFrame()


# =============================================================================
# ENRICHMENT WORKFLOW
# =============================================================================

def run_enrichment_for_view(
    record: VPRecord,
    gene_lists_dir: Path,
    output_dir: Path,
    fdr_threshold: float,
    k_pct: int,
    organism: str,
) -> Dict[str, Any]:
    """Run enrichment for a single view's gene lists."""
    gene_lists_path = gene_lists_dir / f"gene_lists__{record.dataset}__{record.view}.json"
    if not gene_lists_path.exists():
        return {"error": f"Gene lists not found: {gene_lists_path}"}
    
    with open(gene_lists_path) as f:
        gene_data = json.load(f)
    
    k_key = f"K{k_pct}pct"
    
    results_summary = {
        "dataset": record.dataset, "view": record.view,
        "k_pct": k_pct, "fdr_threshold": fdr_threshold,
    }
    
    gene_sets_to_test = {
        "topV": gene_data.get("topV", {}).get(k_key, []),
        "topP": gene_data.get("topP", {}).get(k_key, []),
        "Q4": gene_data.get("Q4", {}).get(k_key, []),
    }
    
    all_enrichments: Dict[str, pd.DataFrame] = {}
    
    for set_name, genes in gene_sets_to_test.items():
        if not genes:
            print(f"    {set_name}: No genes")
            continue
        
        print(f"    {set_name}: {len(genes)} genes")
        
        df = run_gprofiler_enrichment(genes, organism=organism, significance_threshold=fdr_threshold)
        
        if len(df) > 0:
            df = df[df["fdr"] <= fdr_threshold].copy()
            df["gene_set"] = set_name
            
            out_path = output_dir / f"enrichment__{record.dataset}__{record.view}__{set_name}.csv.gz"
            df.to_csv(out_path, index=False, compression="gzip")
            
            all_enrichments[set_name] = df
            results_summary[f"{set_name}_n_pathways"] = len(df)
            print(f"      {len(df)} significant pathways")
        else:
            results_summary[f"{set_name}_n_pathways"] = 0
            print(f"      No significant pathways")
    
    # Compute overlap
    if "topV" in all_enrichments and "topP" in all_enrichments:
        pathways_V = set(all_enrichments["topV"]["pathway_id"])
        pathways_P = set(all_enrichments["topP"]["pathway_id"])
        genes_V = set(gene_sets_to_test["topV"])
        genes_P = set(gene_sets_to_test["topP"])
        
        results_summary["pathway_jaccard"] = jaccard(pathways_V, pathways_P)
        results_summary["gene_jaccard"] = jaccard(genes_V, genes_P)
        
        if results_summary["gene_jaccard"] > 0:
            results_summary["convergence_ratio"] = results_summary["pathway_jaccard"] / results_summary["gene_jaccard"]
    
    # Save summary
    with open(output_dir / f"enrichment_summary__{record.dataset}__{record.view}.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    return results_summary


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run pathway enrichment")
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--views", type=str, default="all")
    parser.add_argument("--model", type=str, default="xgb_bal")
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--k-pct", type=int, default=10)
    parser.add_argument("--organism", type=str, default="hsapiens")
    
    args = parser.parse_args(argv)
    
    # Check dependencies
    try:
        from gprofiler import GProfiler
    except ImportError:
        print("ERROR: gprofiler-official not installed.")
        sys.exit(1)
    
    gene_lists_dir = args.outputs_dir / "08_biology" / "gene_mapping"
    output_dir = args.outputs_dir / "08_biology" / "pathway_enrichment"
    ensure_dir(output_dir)
    
    if not gene_lists_dir.exists():
        print(f"ERROR: Gene lists directory not found: {gene_lists_dir}")
        sys.exit(1)
    
    # Discover files
    if args.views.lower() == "hero":
        records = discover_hero_views(args.outputs_dir, model=args.model)
    elif args.views.lower() == "all":
        records = discover_vp_files(args.outputs_dir, model=args.model)
    else:
        pairs = [v.strip().split("/") for v in args.views.split(",")]
        all_records = discover_vp_files(args.outputs_dir, model=args.model)
        records = [r for r in all_records if (r.dataset, r.view) in [(p[0], p[1]) for p in pairs]]
    
    records = [r for r in records if (gene_lists_dir / f"gene_lists__{r.dataset}__{r.view}.json").exists()]
    
    if not records:
        print("No views with gene lists found.")
        return
    
    print(f"[{now_iso()}] Pathway enrichment for {len(records)} views...")
    
    for record in records:
        print(f"\n  Processing {record.short_key}...")
        try:
            run_enrichment_for_view(
                record=record,
                gene_lists_dir=gene_lists_dir,
                output_dir=output_dir,
                fdr_threshold=args.fdr,
                k_pct=args.k_pct,
                organism=args.organism,
            )
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"    ERROR: {e}")
    
    print(f"\n[{now_iso()}] Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
