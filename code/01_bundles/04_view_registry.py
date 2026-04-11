#!/usr/bin/env python3
"""
view_registry.py
================

Central registry of view metadata for variance-prediction paradox analysis.

This module provides:
- Core vs sensitivity view classification
- Modality descriptions for paper text
- Analysis role (primary vs sensitivity dataset)
- View count expectations for integrity checks

Usage:
    from view_registry import VIEW_REGISTRY, get_core_views, get_analysis_datasets

Key Decisions Documented:
- IBDMDB MGX_func is functional metagenomics (HUMAnN on DNA reads), NOT metatranscriptomics
- TCGA-GBM (N=47) is treated as sensitivity dataset due to small sample size
- Sensitivity views (MGX_CLR, methylation_Mval) are for robustness checks only
"""

from typing import Dict, List, Optional


VIEW_REGISTRY: Dict[str, Dict] = {
    'mlomics': {
        'name': 'MLOmics BRCA',
        'domain': 'Human Cancer (Breast)',
        'core_views': ['mRNA', 'miRNA', 'methylation', 'CNV'],
        'sensitivity_views': [],
        'n_core_views': 4,
        'analysis_role': 'primary',
        'variance_approach': 'latent_axis',  # Pre-standardized data
        'view_descriptions': {
            'mRNA': 'Gene expression (mRNA-seq)',
            'miRNA': 'MicroRNA expression',
            'methylation': 'DNA methylation (450K array)',
            'CNV': 'Copy number variation',
        },
        'paper_description': '4-view multi-omics breast cancer dataset',
    },
    
    'ibdmdb': {
        'name': 'IBDMDB/HMP2',
        'domain': 'Gut Microbiome',
        'core_views': ['MGX', 'MGX_func', 'MPX', 'MBX'],
        'sensitivity_views': ['MGX_CLR'],
        'n_core_views': 4,
        'analysis_role': 'primary',
        'variance_approach': 'marginal',
        'view_descriptions': {
            'MGX': 'Taxonomic profiles (MetaPhlAn, species-level)',
            'MGX_func': 'Functional profiles (HUMAnN gene families)',  # NOTE: MGX-derived, not MTX!
            'MPX': 'Proteomics (enzyme activities)',
            'MBX': 'Metabolomics (untargeted)',
            'MGX_CLR': 'Taxonomic profiles (CLR-transformed) [sensitivity]',
        },
        'view_modality_map': {
            # Maps view names to true modality for paper accuracy
            'MGX': 'metagenomics',
            'MGX_func': 'metagenomics',  # Same sequencing data, different analysis
            'MPX': 'proteomics',
            'MBX': 'metabolomics',
        },
        'paper_description': '4-view gut microbiome dataset (3 modalities + functional metagenomics)',
        'notes': [
            'MGX_func is derived from metagenomic (DNA) reads via HUMAnN, NOT metatranscriptomics',
            'MGX and MGX_func share the same sequencing data but represent different analytical views',
        ],
    },
    
    'ccle': {
        'name': 'CCLE/DepMap',
        'domain': 'Cell Lines',
        'core_views': ['mRNA', 'CNV', 'proteomics'],
        'sensitivity_views': [],
        'n_core_views': 3,
        'analysis_role': 'primary',
        'variance_approach': 'marginal',
        'view_descriptions': {
            'mRNA': 'Gene expression (RNA-seq TPM)',
            'CNV': 'Copy number (gene-level)',
            'proteomics': 'Protein abundance (mass spec)',
        },
        'paper_description': '3-view cancer cell line dataset',
        'notes': [
            'Proteomics has ~36% features dropped due to >30% missingness',
        ],
    },
    
    'tcga_gbm': {
        'name': 'TCGA-GBM',
        'domain': 'Human Cancer (Brain)',
        'core_views': ['mRNA', 'methylation', 'CNV'],
        'sensitivity_views': ['methylation_Mval'],
        'n_core_views': 3,
        'analysis_role': 'sensitivity',  # N=47 too small for stable prediction claims
        'variance_approach': 'marginal',
        'view_descriptions': {
            'mRNA': 'Gene expression (RNA-seq)',
            'methylation': 'DNA methylation (450K array, beta values)',
            'CNV': 'Copy number (GISTIC)',
            'methylation_Mval': 'DNA methylation (M-values) [sensitivity]',
        },
        'paper_description': '3-view glioblastoma dataset (sensitivity analysis)',
        'notes': [
            'N=47 common samples - use for robustness checks, not primary claims',
            'Excluded from predictive performance comparisons due to sample size',
        ],
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_core_views(dataset: str) -> List[str]:
    """Get list of core (non-sensitivity) views for a dataset."""
    if dataset not in VIEW_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset}")
    return VIEW_REGISTRY[dataset]['core_views']


def get_sensitivity_views(dataset: str) -> List[str]:
    """Get list of sensitivity views for a dataset."""
    if dataset not in VIEW_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset}")
    return VIEW_REGISTRY[dataset]['sensitivity_views']


def get_all_views(dataset: str) -> List[str]:
    """Get all views (core + sensitivity) for a dataset."""
    return get_core_views(dataset) + get_sensitivity_views(dataset)


def get_analysis_datasets(role: str = 'primary') -> List[str]:
    """Get datasets by analysis role ('primary' or 'sensitivity')."""
    return [ds for ds, info in VIEW_REGISTRY.items() if info['analysis_role'] == role]


def get_variance_approach(dataset: str) -> str:
    """Get variance approach for a dataset ('marginal' or 'latent_axis')."""
    if dataset not in VIEW_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset}")
    return VIEW_REGISTRY[dataset]['variance_approach']


def is_core_view(dataset: str, view: str) -> bool:
    """Check if a view is a core view (not sensitivity)."""
    return view in get_core_views(dataset)


def get_paper_description(dataset: str) -> str:
    """Get paper-ready description for a dataset."""
    if dataset not in VIEW_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset}")
    return VIEW_REGISTRY[dataset]['paper_description']


# =============================================================================
# Summary for Documentation
# =============================================================================

def print_registry_summary():
    """Print a summary of all datasets and views."""
    print("=" * 70)
    print("VIEW REGISTRY SUMMARY")
    print("=" * 70)
    
    for ds_name, info in VIEW_REGISTRY.items():
        role_marker = "[P]" if info['analysis_role'] == 'primary' else "[S]"
        print(f"\n{role_marker} {info['name']} ({ds_name})")
        print(f"   Domain: {info['domain']}")
        print(f"   Role: {info['analysis_role']}")
        print(f"   Variance: {info['variance_approach']}")
        print(f"   Core views ({info['n_core_views']}): {', '.join(info['core_views'])}")
        if info['sensitivity_views']:
            print(f"   Sensitivity views: {', '.join(info['sensitivity_views'])}")
        print(f"   Paper: \"{info['paper_description']}\"")
    
    print("\n" + "=" * 70)
    print("PRIMARY DATASETS:", ", ".join(get_analysis_datasets('primary')))
    print("SENSITIVITY DATASETS:", ", ".join(get_analysis_datasets('sensitivity')))
    print("=" * 70)


if __name__ == '__main__':
    print_registry_summary()
