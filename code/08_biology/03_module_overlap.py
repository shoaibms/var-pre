#!/usr/bin/env python3
"""
PHASE 8 — 03_module_overlap.py

Analyze gene vs pathway overlap to demonstrate:
gene-level divergence with partial pathway convergence.

Outputs:
    outputs/08_biology/module_overlap/
        overlap_analysis__{dataset}__{view}.json
        convergence_summary.csv

Usage:
    python 03_module_overlap.py --outputs-dir outputs

Author: variance-prediction paradox pipeline
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import (
    ensure_dir, now_iso,
    discover_vp_files, discover_hero_views, VPRecord,
)


def safe_jaccard(a, b) -> float:
    a = set(a) if a is not None else set()
    b = set(b) if b is not None else set()
    if len(a) == 0 and len(b) == 0:
        return float("nan")
    return len(a & b) / max(1, len(a | b))


# =============================================================================
# PATHWAY CATEGORIZATION
# =============================================================================

PATHWAY_CATEGORIES = {
    "cell_cycle": ["cell cycle", "mitotic", "mitosis", "checkpoint", "DNA replication"],
    "metabolism": ["metabolic", "glycolysis", "TCA cycle", "fatty acid", "lipid"],
    "signaling": ["signaling", "receptor", "kinase", "MAPK", "PI3K", "Wnt"],
    "immune": ["immune", "inflammatory", "cytokine", "T cell", "B cell"],
    "apoptosis": ["apoptosis", "cell death", "caspase", "BCL2"],
    "transcription": ["transcription", "RNA polymerase", "chromatin", "histone"],
}


def categorize_pathway(pathway_name: str) -> str:
    name_lower = pathway_name.lower()
    for category, keywords in PATHWAY_CATEGORIES.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                return category
    return "other"


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

def analyze_view_overlap(
    record: VPRecord,
    gene_lists_dir: Path,
    enrichment_dir: Path,
    k_pct: int,
    top_n_pathways: int,
) -> Dict[str, Any]:
    """Analyze gene and pathway overlap for a single view."""
    result = {"dataset": record.dataset, "view": record.view, "k_pct": k_pct}
    
    # Load gene lists
    gene_lists_path = gene_lists_dir / f"gene_lists__{record.dataset}__{record.view}.json"
    if not gene_lists_path.exists():
        result["error"] = "Gene lists not found"
        return result
    
    with open(gene_lists_path) as f:
        gene_data = json.load(f)
    
    k_key = f"K{k_pct}pct"
    genes_V = set(gene_data.get("topV", {}).get(k_key, []))
    genes_P = set(gene_data.get("topP", {}).get(k_key, []))
    
    # Gene overlap metrics
    result["gene_metrics"] = {
        "topV_size": len(genes_V),
        "topP_size": len(genes_P),
        "intersection": len(genes_V & genes_P),
        "gene_jaccard": safe_jaccard(genes_V, genes_P),
    }
    
    # Load enrichment results
    pathways_V: Set[str] = set()
    pathways_P: Set[str] = set()
    
    enrichment_files = [
        enrichment_dir / f"enrichment__{record.dataset}__{record.view}__topV.csv.gz",
        enrichment_dir / f"enrichment__{record.dataset}__{record.view}__topP.csv.gz",
    ]
    enrichment_ok = all(path.exists() for path in enrichment_files)

    for set_name, pathways_set in [("topV", pathways_V), ("topP", pathways_P)]:
        enrich_path = enrichment_dir / f"enrichment__{record.dataset}__{record.view}__{set_name}.csv.gz"
        if enrich_path.exists():
            df = pd.read_csv(enrich_path).nsmallest(top_n_pathways, "fdr")
            pathways_set.update(df["pathway_id"].astype(str))

    if enrichment_ok:
        result["pathway_metrics"] = {
            "topV_pathways": len(pathways_V),
            "topP_pathways": len(pathways_P),
            "intersection": len(pathways_V & pathways_P),
            "pathway_jaccard": safe_jaccard(pathways_V, pathways_P),
        }
    else:
        result["pathway_metrics"] = {
            "topV_pathways": float("nan"),
            "topP_pathways": float("nan"),
            "intersection": float("nan"),
            "pathway_jaccard": float("nan"),
        }
    
    # Convergence ratio
    gj = result["gene_metrics"]["gene_jaccard"]
    pj = result["pathway_metrics"]["pathway_jaccard"]
    if (not math.isnan(gj)) and (not math.isnan(pj)) and gj > 0:
        result["convergence_ratio"] = pj / gj
    else:
        result["convergence_ratio"] = float("nan")
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze gene vs pathway overlap")
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--views", type=str, default="all")
    parser.add_argument("--model", type=str, default="xgb_bal")
    parser.add_argument("--k-pct", type=int, default=10)
    parser.add_argument("--top-n-pathways", type=int, default=50)
    
    args = parser.parse_args(argv)
    
    gene_lists_dir = args.outputs_dir / "08_biology" / "gene_mapping"
    enrichment_dir = args.outputs_dir / "08_biology" / "pathway_enrichment"
    output_dir = args.outputs_dir / "08_biology" / "module_overlap"
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
    
    records = [r for r in records if (gene_lists_dir / f"gene_lists__{r.dataset}__{r.view}.json").exists()]
    
    if not records:
        print("No views with gene lists found.")
        return
    
    print(f"[{now_iso()}] Module overlap analysis for {len(records)} views...")
    
    convergence_rows: List[Dict[str, Any]] = []
    
    for record in records:
        print(f"  Processing {record.short_key}...")
        try:
            result = analyze_view_overlap(
                record=record,
                gene_lists_dir=gene_lists_dir,
                enrichment_dir=enrichment_dir,
                k_pct=args.k_pct,
                top_n_pathways=args.top_n_pathways,
            )
            
            # Save individual result
            with open(output_dir / f"overlap_analysis__{record.dataset}__{record.view}.json", "w") as f:
                json.dump(result, f, indent=2)
            
            if "error" not in result:
                convergence_rows.append({
                    "dataset": record.dataset,
                    "view": record.view,
                    "gene_jaccard": result["gene_metrics"]["gene_jaccard"],
                    "pathway_jaccard": result["pathway_metrics"]["pathway_jaccard"],
                    "convergence_ratio": result.get("convergence_ratio"),
                })
                
                convergence = result.get("convergence_ratio")
                pathway_J = result["pathway_metrics"]["pathway_jaccard"]
                gene_J = result["gene_metrics"]["gene_jaccard"]

                conv_str = "NA" if (
                    convergence is None or (isinstance(convergence, float) and np.isnan(convergence))
                ) else f"{convergence:.2f}x"
                path_str = "NA" if np.isnan(pathway_J) else f"{pathway_J:.3f}"
                gene_str = "NA" if np.isnan(gene_J) else f"{gene_J:.3f}"

                print(f"    Gene J: {gene_str}, Pathway J: {path_str}, Convergence: {conv_str}")
                
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Save convergence summary
    if convergence_rows:
        pd.DataFrame(convergence_rows).to_csv(output_dir / "convergence_summary.csv", index=False)
    
    print(f"\n[{now_iso()}] Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
