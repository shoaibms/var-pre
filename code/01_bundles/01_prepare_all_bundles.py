#!/usr/bin/env python3
"""
Bundle Preparation Script v2 - Based on inspect_data_files_v2.py findings
=========================================================================

Key findings from inspection:
- MLOmics: features × samples, TCGA IDs with dots, labels are positional
- IBDMDB: features × samples, CSM IDs with suffixes, BIOM works
- CCLE: samples × features (opposite!), ACH IDs in index
- TCGA-GBM: features × samples, drop miRNA (only 5 samples)
- NCI-60: needs xlrd for .xls files
- Arabidopsis: correct filename is E-MTAB-7978-tpms.tsv

Bundle Schema (saved as .npz):
- X_{view_name}: np.ndarray - samples × features matrix
- y: np.ndarray - classification labels  
- sample_ids: np.ndarray - sample identifiers
- feature_names_{view_name}: np.ndarray - feature names per view
- info: JSON string - dataset metadata

Usage:
    python 01_prepare_all_bundles_v2.py --data-dir data/raw --output-dir outputs/bundles
    python 01_prepare_all_bundles_v2.py --data-dir data/raw --output-dir outputs/bundles --datasets mlomics,ccle
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# =============================================================================
# Deterministic, variance-neutral feature capping (for manuscript integrity)
# =============================================================================

def _drop_all_nan_and_zerovar_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are all-NaN or have zero variance (robustly)."""
    # Drop all-NaN rows
    df = df.loc[~df.isna().all(axis=1)]
    # Drop zero-variance rows (ignore NaNs)
    try:
        v = df.var(axis=1, skipna=True)
    except TypeError:
        v = df.apply(lambda x: np.nanvar(pd.to_numeric(x, errors="coerce").values), axis=1)
    df = df.loc[v > 0]
    return df


def _cap_rows_neutral(df: pd.DataFrame, max_rows: int, seed: int, label: str) -> pd.DataFrame:
    """
    Variance-neutral cap: after minimal QC, randomly subsample rows with a fixed seed.
    IMPORTANT: Do NOT use variance-based ranking here (preserves manuscript logic).
    """
    if df.shape[0] <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy()
    chosen = rng.choice(idx, size=max_rows, replace=False)
    out = df.loc[chosen].copy()
    # Keep stable ordering for reproducibility/log-diff friendliness
    out = out.sort_index()
    print(f"    {label}: capped to {max_rows:,} features (neutral random cap; seed={seed})")
    return out


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_tcga_id(s: str) -> str:
    """Normalize TCGA ID: replace dots with hyphens, truncate to 15 chars."""
    s = str(s).replace('.', '-')
    if s.startswith('TCGA') and len(s) >= 15:
        return s[:15]
    return s


def extract_hmp2_sample_id(col_name: str) -> str:
    """Extract base HMP2 sample ID by removing suffixes."""
    suffixes = [
        '_Abundance-RPKs', '_P_profile', '_P_pathabundance_cpm',
        '_profile', '_Abundance', '_TPM', '_FPKM'
    ]
    result = str(col_name)
    for suffix in suffixes:
        if result.endswith(suffix):
            result = result[:-len(suffix)]
            break
    return result


def load_tsv_or_csv(filepath: Path) -> pd.DataFrame:
    """Load TSV or CSV with auto-detection and fallback for large files."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        sep = '\t' if '\t' in first_line else ','
    return _safe_read_csv(filepath, sep=sep, index_col=0)


def _safe_read_csv(filepath: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with automatic fallback to Python engine for large/problematic files."""
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception:
        print(f"    (using Python parser for {filepath.name})")
        return pd.read_csv(filepath, engine='python', **kwargs)


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_mlomics(data_dir: Path) -> Dict[str, Any]:
    """
    Load MLOmics BRCA dataset.
    
    Findings from inspector:
    - All views: features × samples (671 columns = samples)
    - Sample IDs use dots: TCGA.3C.AAAU.01
    - Labels file: just values, no sample IDs (positional)
    - Clinical matrix: samples × features (TCGA IDs with hyphens in index)
    """
    src = data_dir / 'mlomics'
    print("  Loading MLOmics BRCA...")
    
    view_files = {
        'mRNA': 'BRCA_mRNA_aligned.csv',
        'miRNA': 'BRCA_miRNA_aligned.csv', 
        'methylation': 'BRCA_Methy_aligned.csv',
        'CNV': 'BRCA_CNV_aligned.csv',
    }
    
    views = {}
    for view_name, filename in view_files.items():
        filepath = src / filename
        if not filepath.exists():
            print(f"    [WARN] {view_name}: not found")
            continue

        df = _safe_read_csv(filepath, index_col=0)
        # Data is features × samples, normalize sample IDs (columns)
        df.columns = [normalize_tcga_id(c) for c in df.columns]
        views[view_name] = df
        print(f"    {view_name}: {df.shape[0]:,} features × {df.shape[1]} samples")
    
    if not views:
        raise ValueError("No views loaded")
    
    # Get common samples across all views
    sample_sets = [set(df.columns) for df in views.values()]
    common_samples = sorted(list(set.intersection(*sample_sets)))
    print(f"  Common samples: {len(common_samples)}")
    
    # Load labels (positional - matches sample order in views)
    label_file = src / 'BRCA_label_num.csv'
    y = None
    if label_file.exists():
        # Labels file has weird format - just a column of values
        df_labels = pd.read_csv(label_file)
        if 'Label' in df_labels.columns:
            all_labels = df_labels['Label'].values
        else:
            all_labels = df_labels.iloc[:, 0].values
        
        # Labels are in same order as original sample columns
        # Get original sample order from first view
        first_view_df = list(views.values())[0]
        original_sample_order = list(first_view_df.columns)
        
        # Create label mapping
        if len(all_labels) == len(original_sample_order):
            label_map = dict(zip(original_sample_order, all_labels))
            y = np.array([label_map[s] for s in common_samples])
            print(f"  Labels: {len(y)} values, classes: {np.unique(y)}")
        else:
            print(f"  [WARN] Label count mismatch: {len(all_labels)} vs {len(original_sample_order)}")
    
    if y is None:
        y = np.zeros(len(common_samples), dtype=int)
    
    # Build output arrays (samples × features)
    X_views = {}
    feature_names = {}
    for view_name, df in views.items():
        aligned = df[common_samples]  # Select common samples
        X_views[view_name] = aligned.values.T  # Transpose to samples × features
        feature_names[view_name] = aligned.index.values
    
    # ASSERT alignment invariant
    for vname, X in X_views.items():
        assert X.shape[0] == len(common_samples), f"{vname}: {X.shape[0]} != {len(common_samples)}"
    assert len(y) == len(common_samples), f"y: {len(y)} != {len(common_samples)}"
    
    return {
        'X_views': X_views,
        'y': y,
        'sample_ids': np.array(common_samples),
        'feature_names': feature_names,
        'info': {
            'name': 'MLOmics BRCA',
            'domain': 'Human Cancer (Breast)',
            'n_samples': len(common_samples),
            'n_views': len(X_views),
            'views': {k: X_views[k].shape[1] for k in X_views},
            'task': 'PAM50 subtype classification',
            'n_classes': len(np.unique(y)),
        }
    }


def load_ibdmdb(data_dir: Path) -> Dict[str, Any]:
    """
    Load IBDMDB/HMP2 dataset - 4 modalities: MGX, MGX_func, MPX, MBX
    
    Note on naming:
    - MGX = taxonomic profiles (species-level abundances)
    - MGX_func = gene families from HUMAnN (functional profiling, same DNA source)
    - MPX = proteomics
    - MBX = metabolomics
    
    genefamilies.tsv is MGX-derived (DNA), NOT metatranscriptomics (RNA).
    
    Methodological note (manuscript-critical):
    We avoid variance-based feature preselection because the paper tests
    "variance-driving vs prediction-driving". Instead, we apply only:
      - semantic filters (e.g., species-only; remove strain rows),
      - minimal QC (drop all-NaN / zero-variance),
      - deterministic, variance-neutral caps (fixed-seed random subsample)
    """
    src = data_dir / 'ibdmdb'
    print("  Loading IBDMDB/HMP2 (4 modalities)...")
    
    # Step 1: Load all views RAW (no feature filtering yet)
    views_raw = {}
    
    # MGX: Taxonomic profiles (metagenomics)
    tax_file = src / 'taxonomic_profiles_3.tsv'
    if not tax_file.exists():
        tax_file = src / 'taxonomic_profiles.tsv'
    if tax_file.exists():
        df = load_tsv_or_csv(tax_file)
        df.columns = [extract_hmp2_sample_id(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        # Filter to species level only (this is semantic, not variance-based)
        species_mask = df.index.str.contains('s__') & ~df.index.str.contains('t__')
        views_raw['MGX'] = df[species_mask].copy()
        print(f"    MGX (taxa): {views_raw['MGX'].shape[0]:,} species × {views_raw['MGX'].shape[1]} samples")
    
    # MGX_func: Gene families from metagenomic HUMAnN (functional profiling)
    func_file = src / 'genefamilies.tsv'
    if func_file.exists():
        df = load_tsv_or_csv(func_file)
        df.columns = [extract_hmp2_sample_id(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        # Remove strain-specific (semantic filter, not variance-based)
        gene_level_mask = ~df.index.str.contains('\\|')
        views_raw['MGX_func'] = df[gene_level_mask].copy()
        print(f"    MGX_func (genes): {views_raw['MGX_func'].shape[0]:,} gene families × {views_raw['MGX_func'].shape[1]} samples (raw)")
    
    # MPX: Proteomics
    prot_file = src / 'HMP2_proteomics_ecs.tsv'
    if prot_file.exists():
        df = load_tsv_or_csv(prot_file)
        df.columns = [extract_hmp2_sample_id(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[df.index != 'UNGROUPED']
        views_raw['MPX'] = df
        print(f"    MPX (proteomics): {views_raw['MPX'].shape[0]:,} ECs × {views_raw['MPX'].shape[1]} samples")
    
    # MBX: Metabolomics (BIOM format) - load raw
    biom_file = src / 'HMP2_metabolomics_w_metadata.biom'
    if biom_file.exists():
        try:
            import biom
            table = biom.load_table(str(biom_file))
            data = table.to_dataframe()
            data.columns = [extract_hmp2_sample_id(str(c)) for c in data.columns]
            data = data.loc[:, ~data.columns.duplicated()]
            # Convert sparse to dense
            if hasattr(data, 'sparse'):
                data = data.sparse.to_dense()
            views_raw['MBX'] = data
            print(f"    MBX (metabolomics): {views_raw['MBX'].shape[0]:,} metabolites × {views_raw['MBX'].shape[1]} samples (raw)")
        except ImportError:
            print("    [WARN] MBX: biom package not installed, skipping")
        except Exception as e:
            print(f"    [WARN] MBX: {e}")
    
    if not views_raw:
        raise ValueError("No views loaded")
    
    # Step 2: Find common samples across ALL views
    print("  Sample ID formats:")
    for vname, df in views_raw.items():
        print(f"    {vname}: {list(df.columns[:3])} ... ({df.shape[1]} total)")
    
    sample_sets = [set(df.columns) for df in views_raw.values()]
    common_samples = sorted(list(set.intersection(*sample_sets)))
    print(f"  Common samples (all {len(views_raw)} views): {len(common_samples)}")
    
    # If overlap is too small, find best subset of views
    MIN_SAMPLES = 50
    if len(common_samples) < MIN_SAMPLES and len(views_raw) > 2:
        print(f"  [WARN] Overlap < {MIN_SAMPLES}, finding best view combination...")
        from itertools import combinations
        view_names = list(views_raw.keys())
        best_combo = None
        best_count = 0
        
        for r in range(len(view_names), 1, -1):
            for combo in combinations(view_names, r):
                combo_samples = set.intersection(*[set(views_raw[v].columns) for v in combo])
                if len(combo_samples) > best_count:
                    best_count = len(combo_samples)
                    best_combo = combo
                    if best_count >= MIN_SAMPLES:
                        break
            if best_count >= MIN_SAMPLES:
                break
        
        if best_combo and best_count >= MIN_SAMPLES // 2:
            print(f"  Using views {best_combo}: {best_count} samples")
            views_raw = {k: views_raw[k] for k in best_combo}
            common_samples = sorted(list(set.intersection(*[set(v.columns) for v in views_raw.values()])))
    
    # Step 3: Load metadata and filter to labeled samples
    meta_file = src / 'hmp2_metadata.csv'
    y = None
    label_encoding = {'nonIBD': 0, 'UC': 1, 'CD': 2}
    diag_map = {}
    
    ID_COLUMNS = ['External ID', 'ExternalID', 'external_id', 'External_ID', 'sample_id', 'SampleID']
    DIAG_COLUMNS = ['diagnosis', 'Diagnosis', 'disease', 'Disease', 'dx']
    
    id_col, diag_col = None, None
    if meta_file.exists():
        meta_df = pd.read_csv(meta_file)
        id_col = next((c for c in ID_COLUMNS if c in meta_df.columns), None)
        diag_col = next((c for c in DIAG_COLUMNS if c in meta_df.columns), None)
        
        print(f"  Metadata: id_col='{id_col}', diag_col='{diag_col}'")
        
        if diag_col and id_col:
            meta_df[id_col] = meta_df[id_col].astype(str)
            diag_map = dict(zip(meta_df[id_col], meta_df[diag_col]))
            
            # Filter to samples with valid labels
            valid = [(s, label_encoding[diag_map[s]]) for s in common_samples 
                     if s in diag_map and diag_map[s] in label_encoding]
            if valid:
                common_samples = [v[0] for v in valid]
                y = np.array([v[1] for v in valid])
                print(f"  Labels matched: {len(y)}, distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    if y is None:
        y = np.zeros(len(common_samples), dtype=int)
        print("  [WARN] No labels matched, using zeros")
    
    # Step 4: Apply variance-NEUTRAL feature capping on COMMON SAMPLES ONLY
    # NOTE: We avoid variance-based selection to preserve manuscript logic
    print(f"  Applying feature QC on {len(common_samples)} common samples...")
    views = {}
    
    for vname, df_raw in views_raw.items():
        # Subset to common samples FIRST
        df = df_raw[common_samples].copy()
        
        # Minimal QC (variance-neutral)
        df = _drop_all_nan_and_zerovar_rows(df)
        
        # Variance-neutral caps for very large views (deterministic)
        # NOTE: This is a compute cap, not a selection criterion.
        if vname == 'MGX_func':
            df = _cap_rows_neutral(df, max_rows=10000, seed=1, label="MGX_func")
        elif vname == 'MBX':
            df = _cap_rows_neutral(df, max_rows=20000, seed=1, label="MBX")
        else:
            print(f"    {vname}: {df.shape[0]:,} features (no cap needed)")
        
        views[vname] = df
    
    # Step 5: Build final output arrays
    X_views = {}
    feature_names = {}
    for view_name, df in views.items():
        X_views[view_name] = df.values.T  # Transpose to samples × features
        feature_names[view_name] = df.index.values
    
    # ASSERT alignment invariant
    for vname, X in X_views.items():
        assert X.shape[0] == len(common_samples), f"{vname}: {X.shape[0]} != {len(common_samples)}"
    assert len(y) == len(common_samples), f"y: {len(y)} != {len(common_samples)}"
    
    print(f"  [OK] Alignment verified: {len(common_samples)} samples x {len(X_views)} views")
    
    return {
        'X_views': X_views,
        'y': y,
        'sample_ids': np.array(common_samples),
        'feature_names': feature_names,
        'info': {
            'name': 'IBDMDB/HMP2',
            'domain': 'Gut Microbiome',
            'n_samples': len(common_samples),
            'n_views': len(X_views),
            'views': {k: X_views[k].shape[1] for k in X_views},
            'task': 'IBD diagnosis classification',
            'n_classes': len(np.unique(y)),
        }
    }


def load_ccle(data_dir: Path) -> Dict[str, Any]:
    """
    Load CCLE/DepMap dataset.
    
    CRITICAL: CCLE is samples × features (opposite of others!)
    
    Findings from inspector:
    - OmicsExpressionProteinCodingGenesTPMLogp1.csv: samples × features, ACH IDs in index
    - OmicsCNGene.csv: samples × features, ACH IDs in index
    - Model.csv: samples × features, ACH IDs in index (metadata)
    - protein_quant: different format (cell line names), skip for now
    """
    src = data_dir / 'ccle'
    print("  Loading CCLE/DepMap...")
    
    views = {}
    
    # Expression (samples × features - ACH IDs in index, genes in columns)
    expr_file = src / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv'
    if expr_file.exists():
        df = _safe_read_csv(expr_file, index_col=0)
        # Already samples × features!
        views['mRNA'] = df
        print(f"    mRNA: {df.shape[0]} samples × {df.shape[1]:,} genes")
    
    # Copy number (samples × features)
    cn_file = src / 'OmicsCNGene.csv'
    if cn_file.exists():
        df = _safe_read_csv(cn_file, index_col=0)
        views['CNV'] = df
        print(f"    CNV: {df.shape[0]} samples × {df.shape[1]:,} genes")
    
    # Proteomics - needs cell line name to ACH ID mapping
    # Format: CELLLINE_TISSUE_TenPxNN (e.g., MDAMB468_BREAST_TenPx01)
    prot_file = src / 'protein_quant_current_normalized.csv'
    model_file = src / 'Model.csv'
    if prot_file.exists() and model_file.exists():
        # Load model for name mapping
        model_df = _safe_read_csv(model_file, index_col=0)
        
        # Create mapping: StrippedCellLineName -> ACH ID
        name_to_ach = {}
        if 'StrippedCellLineName' in model_df.columns:
            for ach_id, row in model_df.iterrows():
                name = str(row['StrippedCellLineName']).upper()
                name_to_ach[name] = ach_id
        
        # Load proteomics
        prot_df = _safe_read_csv(prot_file)
        
        # Find data columns: CELLLINE_TISSUE_TenPxNN format
        data_cols = [c for c in prot_df.columns if '_TenPx' in c and '_Peptides' not in c]
        
        if len(data_cols) > 50:
            # Set gene symbol as index
            if 'Gene_Symbol' in prot_df.columns:
                prot_df = prot_df.set_index('Gene_Symbol')
            
            # Extract cell line names and map to ACH IDs
            # Column format: MDAMB468_BREAST_TenPx01 -> MDAMB468
            col_mapping = {}
            for col in data_cols:
                parts = col.split('_')
                if len(parts) >= 3:
                    cell_line = parts[0].upper()  # MDAMB468
                    if cell_line in name_to_ach:
                        ach_id = name_to_ach[cell_line]
                        # Use first occurrence of each cell line (ignore TenPx batch)
                        if ach_id not in col_mapping.values():
                            col_mapping[col] = ach_id
            
            if len(col_mapping) > 100:
                # Select and rename columns
                prot_data = prot_df[list(col_mapping.keys())].copy()
                prot_data = prot_data.rename(columns=col_mapping)
                
                # Transpose to samples × features (ACH IDs become rows)
                views['proteomics'] = prot_data.T
                print(f"    proteomics: {len(col_mapping)} samples × {prot_data.shape[0]:,} proteins")
            else:
                print(f"    [WARN] proteomics: only {len(col_mapping)} cell lines mapped")
        else:
            print(f"    [WARN] proteomics: unexpected format ({len(data_cols)} data columns)")
    
    if not views:
        raise ValueError("No views loaded")
    
    # Get common samples (ACH IDs in index)
    sample_sets = [set(df.index) for df in views.values()]
    common_samples = sorted(list(set.intersection(*sample_sets)))
    print(f"  Common samples: {len(common_samples)}")
    
    # Load metadata for tissue labels
    model_file = src / 'Model.csv'
    y = None
    tissue_labels = None
    
    if model_file.exists():
        model_df = _safe_read_csv(model_file, index_col=0)
        
        # Find tissue column
        tissue_col = None
        for col in ['OncotreeLineage', 'OncotreePrimaryDisease', 'tissue']:
            if col in model_df.columns:
                tissue_col = col
                break
        
        if tissue_col:
            # Filter to common samples
            model_df = model_df.loc[model_df.index.isin(common_samples)]
            tissues = model_df.loc[common_samples, tissue_col].values
            
            # Encode tissues
            unique_tissues = sorted(set(t for t in tissues if pd.notna(t)))
            tissue_map = {t: i for i, t in enumerate(unique_tissues)}
            
            y = np.array([tissue_map.get(t, -1) for t in tissues])
            
            # Remove samples with unknown tissue
            valid_mask = y >= 0
            if not all(valid_mask):
                common_samples = [s for s, v in zip(common_samples, valid_mask) if v]
                y = y[valid_mask]
            
            tissue_labels = unique_tissues
            print(f"  Tissue types: {len(unique_tissues)} unique")
    
    if y is None:
        y = np.zeros(len(common_samples), dtype=int)
        tissue_labels = ['unknown']
    
    # Build output arrays (already samples × features, just align)
    X_views = {}
    feature_names = {}
    for view_name, df in views.items():
        aligned = df.loc[common_samples]
        X_views[view_name] = aligned.values  # Already samples × features
        feature_names[view_name] = aligned.columns.values
    
    # ASSERT alignment invariant
    for vname, X in X_views.items():
        assert X.shape[0] == len(common_samples), f"{vname}: {X.shape[0]} != {len(common_samples)}"
    assert len(y) == len(common_samples), f"y: {len(y)} != {len(common_samples)}"
    
    return {
        'X_views': X_views,
        'y': y,
        'sample_ids': np.array(common_samples),
        'feature_names': feature_names,
        'info': {
            'name': 'CCLE/DepMap',
            'domain': 'Cell Lines',
            'n_samples': len(common_samples),
            'n_views': len(X_views),
            'views': {k: X_views[k].shape[1] for k in X_views},
            'task': 'Tissue type classification',
            'n_classes': len(tissue_labels),
            'class_labels': tissue_labels[:20] if len(tissue_labels) > 20 else tissue_labels,
        }
    }


def load_tcga_gbm(data_dir: Path) -> Dict[str, Any]:
    """
    Load TCGA-GBM dataset.
    
    Findings from inspector:
    - HiSeqV2: 172 samples, features × samples
    - HumanMethylation450: 155 samples, features × samples  
    - Gistic2_CopyNumber: failed to parse (needs investigation)
    - miRNA: only 5 samples - SKIP
    - All use TCGA-XX-XXXX-XX format
    """
    src = data_dir / 'tcga_gbm'
    print("  Loading TCGA-GBM...")
    
    views = {}
    
    # mRNA expression
    mrna_file = src / 'HiSeqV2'
    if mrna_file.exists():
        df = load_tsv_or_csv(mrna_file)
        df.columns = [normalize_tcga_id(c) for c in df.columns]
        views['mRNA'] = df
        print(f"    mRNA: {df.shape[0]:,} genes × {df.shape[1]} samples")
    
    # Methylation
    methy_file = src / 'HumanMethylation450'
    if methy_file.exists():
        df = load_tsv_or_csv(methy_file)
        df.columns = [normalize_tcga_id(c) for c in df.columns]
        
        # Manuscript-critical: avoid variance-based probe preselection.
        # Apply minimal QC, then a neutral deterministic cap if needed.
        df = _drop_all_nan_and_zerovar_rows(df)
        if df.shape[0] > 20000:
            df = _cap_rows_neutral(df, max_rows=20000, seed=1, label="methylation")
            print(f"    methylation: {df.shape[0]:,} probes × {df.shape[1]} samples (neutral cap)")
        else:
            print(f"    methylation: {df.shape[0]:,} probes × {df.shape[1]} samples")
        views['methylation'] = df
    
    # Copy number - try to load with different parsing
    cnv_file = src / 'Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes'
    if cnv_file.exists():
        try:
            # Try tab-separated
            df = _safe_read_csv(cnv_file, sep='\t', index_col=0)
            df.columns = [normalize_tcga_id(c) for c in df.columns]
            views['CNV'] = df
            print(f"    CNV: {df.shape[0]:,} genes × {df.shape[1]} samples")
        except Exception as e:
            print(f"    CNV: parse error - {e}")
    
    # Skip miRNA (only 5 samples per inspector)
    print("    miRNA: SKIPPED (only 5 samples)")
    
    if not views:
        raise ValueError("No views loaded")
    
    # Get common samples
    sample_sets = [set(df.columns) for df in views.values()]
    common_samples = sorted(list(set.intersection(*sample_sets)))
    print(f"  Common samples (all views): {len(common_samples)}")
    
    # For variance-prediction analysis, prefer MORE VIEWS over more samples
    # Only drop a view if it reduces samples below minimum threshold
    MIN_SAMPLES = 30
    
    if len(common_samples) < MIN_SAMPLES and len(views) > 2:
        # Find which view is limiting sample count
        print("  Checking per-view sample counts...")
        for vname, df in views.items():
            print(f"    {vname}: {df.shape[1]} samples")
        
        # Try removing the view with fewest samples
        view_sample_counts = {k: len(v.columns) for k, v in views.items()}
        smallest_view = min(view_sample_counts, key=view_sample_counts.get)
        
        views_without_smallest = {k: v for k, v in views.items() if k != smallest_view}
        sample_sets_reduced = [set(df.columns) for df in views_without_smallest.values()]
        common_reduced = sorted(list(set.intersection(*sample_sets_reduced)))
        
        if len(common_reduced) >= MIN_SAMPLES and len(common_samples) < MIN_SAMPLES:
            print(f"  Dropping {smallest_view}: {len(common_samples)} → {len(common_reduced)} samples")
            views = views_without_smallest
            common_samples = common_reduced
        else:
            # Keep all views even with fewer samples - more views is better for the paper
            print(f"  Keeping all {len(views)} views with {len(common_samples)} samples")
    
    # Load clinical metadata
    clinical_file = src / 'GBM_clinicalMatrix'
    y = None
    
    if clinical_file.exists():
        clinical_df = load_tsv_or_csv(clinical_file)
        clinical_df.index = [normalize_tcga_id(s) for s in clinical_df.index]
        
        # Look for subtype column
        subtype_col = None
        for col in clinical_df.columns:
            if 'subtype' in col.lower():
                subtype_col = col
                break
        
        if subtype_col:
            matched = [s for s in common_samples if s in clinical_df.index]
            subtypes = clinical_df.loc[matched, subtype_col].values
            
            # Encode
            valid_subtypes = [s for s in subtypes if pd.notna(s)]
            if valid_subtypes:
                unique = sorted(set(valid_subtypes))
                subtype_map = {s: i for i, s in enumerate(unique)}
                
                y = np.array([subtype_map.get(s, -1) for s in subtypes])
                valid_mask = y >= 0
                
                if sum(valid_mask) > 10:  # Enough valid samples
                    common_samples = [s for s, v in zip(matched, valid_mask) if v]
                    y = y[valid_mask]
                    print(f"  Subtypes: {unique}")
    
    if y is None:
        # Use tumor/normal based on barcode (digit 14: 0=tumor, 1=normal)
        y = np.array([0 if s[13] == '0' else 1 for s in common_samples])
        print(f"  Using tumor/normal labels: {np.sum(y==0)} tumor, {np.sum(y==1)} normal")
    
    # Build output arrays
    X_views = {}
    feature_names = {}
    for view_name, df in views.items():
        aligned = df[common_samples]
        X_views[view_name] = aligned.values.T  # Transpose to samples × features
        feature_names[view_name] = aligned.index.values
    
    # ASSERT alignment invariant
    for vname, X in X_views.items():
        assert X.shape[0] == len(common_samples), f"{vname}: {X.shape[0]} != {len(common_samples)}"
    assert len(y) == len(common_samples), f"y: {len(y)} != {len(common_samples)}"
    
    return {
        'X_views': X_views,
        'y': y,
        'sample_ids': np.array(common_samples),
        'feature_names': feature_names,
        'info': {
            'name': 'TCGA-GBM',
            'domain': 'Human Cancer (Brain)',
            'n_samples': len(common_samples),
            'n_views': len(X_views),
            'views': {k: X_views[k].shape[1] for k in X_views},
            'task': 'GBM classification',
            'n_classes': len(np.unique(y)),
        }
    }


def load_nci60(data_dir: Path) -> Dict[str, Any]:
    """
    Load NCI-60 dataset.
    
    Findings from inspector:
    - Files are .xls format (old Excel), need xlrd package
    - Data files in output/ subdirectory
    """
    src = data_dir / 'nci60' / 'output'
    print("  Loading NCI-60...")
    
    if not src.exists():
        src = data_dir / 'nci60'
    
    try:
        import xlrd
    except ImportError:
        print("  [FAIL] xlrd package required for .xls files")
        print("     Run: pip install xlrd")
        return None
    
    views = {}
    
    # Try to load each Excel file
    excel_files = {
        'mRNA': 'RNA__RNA_seq_composite_expression.xls',
        'miRNA': 'RNA__microRNA_OSU_V3_chip_log2.xls',
        'proteomics': 'Protein__SWATH_(Mass_spectrometry)_Protein.xls',
    }
    
    for view_name, filename in excel_files.items():
        filepath = src / filename
        if not filepath.exists():
            print(f"    [WARN] {view_name}: not found")
            continue

        try:
            df = pd.read_excel(filepath, index_col=0, engine='xlrd')
            views[view_name] = df
            print(f"    {view_name}: {df.shape}")
        except Exception as e:
            print(f"    [WARN] {view_name}: {e}")

    if not views:
        print("  [FAIL] No views loaded")
        return None
    
    # NCI-60 has 60 cell lines
    # Determine orientation and find common samples
    # ... (would need to inspect actual file structure)
    
    print("  [WARN] NCI-60 loader needs refinement based on actual file structure")
    return None


def load_arabidopsis(data_dir: Path) -> Dict[str, Any]:
    """
    Load Arabidopsis dataset.
    
    Findings from inspector:
    - Correct filename: E-MTAB-7978-tpms.tsv (not query-results)
    - Excel files need investigation
    """
    src = data_dir / 'arabidopsis'
    print("  Loading Arabidopsis...")
    
    views = {}
    
    # RNA expression - correct filename
    rna_file = src / 'E-MTAB-7978-tpms.tsv'
    if rna_file.exists():
        df = load_tsv_or_csv(rna_file)
        views['RNA'] = df
        print(f"    RNA: {df.shape}")
    else:
        print(f"    [WARN] RNA file not found")

    if not views:
        print("  [FAIL] No views loaded")
        return None

    # Would need more investigation for sample alignment
    print("  [WARN] Arabidopsis loader needs refinement")
    return None


# =============================================================================
# Main
# =============================================================================

LOADERS = {
    'mlomics': load_mlomics,
    'ibdmdb': load_ibdmdb,
    'ccle': load_ccle,
    'tcga_gbm': load_tcga_gbm,
    'nci60': load_nci60,
    'arabidopsis': load_arabidopsis,
}


def create_maxsamples_variant(data_dir: Path, ds_name: str, original_bundle: Dict) -> Optional[Dict]:
    """
    Create a maxsamples variant by dropping the view that limits sample count.
    
    This gives a clean manuscript narrative:
    "Adding modalities shrinks the intersection cohort; 
     variance-prediction relationships remain consistent."
    """
    # Dataset-specific knowledge of which view limits samples
    LIMITING_VIEWS = {
        'ibdmdb': 'MGX_func',  # MGX_func has 735 samples vs MGX 1616
        'ccle': 'proteomics', # proteomics has 375 vs mRNA 1517
        'tcga_gbm': 'methylation',  # methylation has 155 vs mRNA 172
        'mlomics': None,      # All views have same samples
    }
    
    limiting_view = LIMITING_VIEWS.get(ds_name)
    if limiting_view is None or limiting_view not in original_bundle['X_views']:
        return None
    
    # For a proper maxsamples variant, we need to reload the data
    # without the limiting view to get the actual larger intersection
    print(f"\n  Creating maxsamples variant (dropping {limiting_view})...")
    
    # This requires dataset-specific logic, so we'll handle the common cases
    if ds_name == 'ibdmdb':
        # Reload IBDMDB with only MGX, MPX, MBX (drop MGX_func)
        return _load_ibdmdb_maxsamples(data_dir)
    elif ds_name == 'ccle':
        # Reload CCLE with only mRNA, CNV (drop proteomics)
        return _load_ccle_maxsamples(data_dir)
    elif ds_name == 'tcga_gbm':
        # Reload TCGA with only mRNA, CNV (drop methylation)
        return _load_tcga_maxsamples(data_dir)
    
    return None


def _load_ibdmdb_maxsamples(data_dir: Path) -> Dict[str, Any]:
    """Load IBDMDB with 3 views (drop MGX_func) for more samples."""
    src = data_dir / 'ibdmdb'
    
    views = {}
    
    # MGX
    tax_file = src / 'taxonomic_profiles_3.tsv'
    if not tax_file.exists():
        tax_file = src / 'taxonomic_profiles.tsv'
    if tax_file.exists():
        df = load_tsv_or_csv(tax_file)
        df.columns = [extract_hmp2_sample_id(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        species_mask = df.index.str.contains('s__') & ~df.index.str.contains('t__')
        views['MGX'] = df[species_mask].copy()
    
    # MPX
    prot_file = src / 'HMP2_proteomics_ecs.tsv'
    if prot_file.exists():
        df = load_tsv_or_csv(prot_file)
        df.columns = [extract_hmp2_sample_id(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[df.index != 'UNGROUPED']
        views['MPX'] = df
    
    # MBX
    biom_file = src / 'HMP2_metabolomics_w_metadata.biom'
    if biom_file.exists():
        try:
            import biom
            table = biom.load_table(str(biom_file))
            data = table.to_dataframe()
            data.columns = [extract_hmp2_sample_id(str(c)) for c in data.columns]
            data = data.loc[:, ~data.columns.duplicated()]
            # Convert sparse to dense if needed
            if hasattr(data, 'sparse'):
                data = data.sparse.to_dense()
            # Variance-neutral cap (manuscript-critical)
            data = _drop_all_nan_and_zerovar_rows(data)
            if data.shape[0] > 20000:
                data = _cap_rows_neutral(data, max_rows=20000, seed=1, label="MBX")
            views['MBX'] = data
        except:
            pass
    
    # Common samples
    sample_sets = [set(df.columns) for df in views.values()]
    common_samples = sorted(list(set.intersection(*sample_sets)))
    
    # Load labels
    meta_file = src / 'hmp2_metadata.csv'
    y = np.zeros(len(common_samples), dtype=int)
    if meta_file.exists():
        meta_df = pd.read_csv(meta_file)
        id_col = next((c for c in ['External ID', 'ExternalID'] if c in meta_df.columns), None)
        diag_col = next((c for c in ['diagnosis', 'Diagnosis'] if c in meta_df.columns), None)
        if id_col and diag_col:
            meta_df[id_col] = meta_df[id_col].astype(str)
            diag_map = dict(zip(meta_df[id_col], meta_df[diag_col]))
            label_encoding = {'nonIBD': 0, 'UC': 1, 'CD': 2}
            valid = [(s, label_encoding[diag_map[s]]) for s in common_samples 
                     if s in diag_map and diag_map[s] in label_encoding]
            if valid:
                common_samples = [v[0] for v in valid]
                y = np.array([v[1] for v in valid])
    
    # Build arrays
    X_views = {}
    feature_names = {}
    for vname, df in views.items():
        aligned = df[common_samples]
        X_views[vname] = aligned.values.T
        feature_names[vname] = aligned.index.values
    
    print(f"    maxsamples: {len(common_samples)} samples × {len(X_views)} views")
    
    return {
        'X_views': X_views, 'y': y, 'sample_ids': np.array(common_samples),
        'feature_names': feature_names,
        'info': {
            'name': 'IBDMDB/HMP2 (maxsamples)', 'domain': 'Gut Microbiome',
            'n_samples': len(common_samples), 'n_views': len(X_views),
            'views': {k: X_views[k].shape[1] for k in X_views},
            'task': 'IBD diagnosis', 'n_classes': len(np.unique(y)),
            'variant': 'maxsamples', 'dropped_view': 'MGX_func',
        }
    }


def _load_ccle_maxsamples(data_dir: Path) -> Dict[str, Any]:
    """Load CCLE with 2 views (drop proteomics) for more samples."""
    src = data_dir / 'ccle'
    
    views = {}
    
    # mRNA
    expr_file = src / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv'
    if expr_file.exists():
        views['mRNA'] = _safe_read_csv(expr_file, index_col=0)
    
    # CNV
    cn_file = src / 'OmicsCNGene.csv'
    if cn_file.exists():
        views['CNV'] = _safe_read_csv(cn_file, index_col=0)
    
    # Common samples
    sample_sets = [set(df.index) for df in views.values()]
    common_samples = sorted(list(set.intersection(*sample_sets)))
    
    # Labels
    model_file = src / 'Model.csv'
    y = np.zeros(len(common_samples), dtype=int)
    if model_file.exists():
        model_df = _safe_read_csv(model_file, index_col=0)
        if 'OncotreeLineage' in model_df.columns:
            tissues = model_df.loc[common_samples, 'OncotreeLineage'].values
            unique = sorted(set(t for t in tissues if pd.notna(t)))
            tissue_map = {t: i for i, t in enumerate(unique)}
            y = np.array([tissue_map.get(t, 0) for t in tissues])
    
    # Build arrays
    X_views = {}
    feature_names = {}
    for vname, df in views.items():
        aligned = df.loc[common_samples]
        X_views[vname] = aligned.values
        feature_names[vname] = aligned.columns.values
    
    print(f"    maxsamples: {len(common_samples)} samples × {len(X_views)} views")
    
    return {
        'X_views': X_views, 'y': y, 'sample_ids': np.array(common_samples),
        'feature_names': feature_names,
        'info': {
            'name': 'CCLE/DepMap (maxsamples)', 'domain': 'Cell Lines',
            'n_samples': len(common_samples), 'n_views': len(X_views),
            'views': {k: X_views[k].shape[1] for k in X_views},
            'task': 'Tissue type', 'n_classes': len(np.unique(y)),
            'variant': 'maxsamples', 'dropped_view': 'proteomics',
        }
    }


def _load_tcga_maxsamples(data_dir: Path) -> Dict[str, Any]:
    """Load TCGA-GBM with 2 views (drop methylation) for more samples."""
    src = data_dir / 'tcga_gbm'
    
    views = {}
    
    # mRNA
    mrna_file = src / 'HiSeqV2'
    if mrna_file.exists():
        df = load_tsv_or_csv(mrna_file)
        df.columns = [normalize_tcga_id(c) for c in df.columns]
        views['mRNA'] = df
    
    # CNV
    cnv_file = src / 'Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes'
    if cnv_file.exists():
        try:
            df = _safe_read_csv(cnv_file, sep='\t', index_col=0)
            df.columns = [normalize_tcga_id(c) for c in df.columns]
            views['CNV'] = df
        except:
            pass
    
    # Common samples
    sample_sets = [set(df.columns) for df in views.values()]
    common_samples = sorted(list(set.intersection(*sample_sets)))
    
    # Labels (tumor/normal)
    y = np.array([0 if len(s) >= 14 and s[13] == '0' else 1 for s in common_samples])
    
    # Build arrays
    X_views = {}
    feature_names = {}
    for vname, df in views.items():
        aligned = df[common_samples]
        X_views[vname] = aligned.values.T
        feature_names[vname] = aligned.index.values
    
    print(f"    maxsamples: {len(common_samples)} samples × {len(X_views)} views")
    
    return {
        'X_views': X_views, 'y': y, 'sample_ids': np.array(common_samples),
        'feature_names': feature_names,
        'info': {
            'name': 'TCGA-GBM (maxsamples)', 'domain': 'Human Cancer',
            'n_samples': len(common_samples), 'n_views': len(X_views),
            'views': {k: X_views[k].shape[1] for k in X_views},
            'task': 'GBM classification', 'n_classes': len(np.unique(y)),
            'variant': 'maxsamples', 'dropped_view': 'methylation',
        }
    }


def _as_str_array(x) -> np.ndarray:
    """Force stable unicode ndarray (prevents object-array pickling in npz)."""
    if x is None:
        return np.array([], dtype=str)
    if isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S"):
        return x.astype(str, copy=False)
    # Handles list, pandas Index, etc.
    return np.asarray(list(x), dtype=str)


def _as_numeric_1d(y) -> np.ndarray:
    """Stable numeric 1D array for labels/targets."""
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)
    if np.issubdtype(y.dtype, np.integer):
        return y.astype(np.int32, copy=False)
    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float32, copy=False)
    # Fallback: try coercion
    try:
        yy = y.astype(np.float32)
        return yy
    except Exception:
        return np.asarray([str(v) for v in y], dtype=str)


def _as_float32_matrix(X) -> np.ndarray:
    """Stable numeric matrix (float32) for views."""
    if hasattr(X, "to_numpy"):  # pandas
        X = X.to_numpy()
    X = np.asarray(X)
    if X.dtype == np.float32:
        return X
    # Convert ints/bools to float32; keep NaNs
    return X.astype(np.float32, copy=False)


def save_bundle(bundle: Dict[str, Any], output_path: Path) -> None:
    """Save bundle as .npz with NO pickled object arrays (cross-version safe)."""
    info_json = json.dumps(bundle.get("info", {}), ensure_ascii=False)

    save_dict = {
        "y": _as_numeric_1d(bundle["y"]),
        "sample_ids": _as_str_array(bundle["sample_ids"]),
        "info": np.asarray(info_json, dtype=str),  # 0-d unicode array
    }

    for view_name, X in bundle["X_views"].items():
        save_dict[f"X_{view_name}"] = _as_float32_matrix(X)
        save_dict[f"features_{view_name}"] = _as_str_array(bundle["feature_names"][view_name])

    np.savez_compressed(output_path, **save_dict)
    
    # Also save info as JSON for easy inspection
    info_path = output_path.with_suffix('.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(bundle['info'], f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-view bundles (v2)")
    parser.add_argument('--data-dir', type=str, default='data/raw')
    parser.add_argument('--output-dir', type=str, default='outputs/bundles')
    parser.add_argument('--datasets', type=str, default='mlomics,ibdmdb,ccle,tcga_gbm',
                       help='Comma-separated list of datasets')
    parser.add_argument('--create-variants', action='store_true',
                       help='Create both maxviews and maxsamples bundle variants')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    print("=" * 70)
    print("BUNDLE PREPARATION v2")
    print("=" * 70)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Datasets: {datasets}")
    if args.create_variants:
        print("Creating bundle variants: maxviews + maxsamples")
    
    results = []
    
    for ds_name in datasets:
        if ds_name not in LOADERS:
            print(f"\n[WARN] Unknown dataset: {ds_name}")
            continue

        print(f"\n{'='*60}")
        print(f"{ds_name.upper()}")
        print('='*60)
        
        try:
            bundle = LOADERS[ds_name](data_dir)
            
            if bundle is None:
                results.append({'name': ds_name, 'status': 'skipped'})
                continue
            
            # Save bundle
            output_path = output_dir / f'{ds_name}_bundle.npz'
            save_bundle(bundle, output_path)
            
            info = bundle['info']
            print(f"\n  [OK] Saved: {output_path.name}")
            print(f"     {info['n_samples']} samples × {info['n_views']} views")
            print(f"     Views: {info['views']}")
            
            results.append({
                'name': ds_name,
                'status': 'success',
                **info
            })
            
            # Create maxsamples variant if requested
            if args.create_variants:
                try:
                    variant = create_maxsamples_variant(data_dir, ds_name, bundle)
                    if variant:
                        variant_path = output_dir / f'{ds_name}_maxsamples_bundle.npz'
                        save_bundle(variant, variant_path)
                        print(f"  [OK] Saved variant: {variant_path.name}")
                        results.append({
                            'name': f'{ds_name}_maxsamples',
                            'status': 'success',
                            **variant['info']
                        })
                except Exception as e:
                    print(f"  [WARN] Variant error: {e}")
            
        except Exception as e:
            print(f"\n  [FAIL] Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'name': ds_name, 'status': 'error', 'error': str(e)})
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for r in results:
        status_icon = {"success": "[OK]", "skipped": "[WARN]", "error": "[FAIL]"}.get(r['status'], "?")
        print(f"  {status_icon} {r['name']}: {r['status']}")
        if r['status'] == 'success':
            print(f"      {r['n_samples']} samples, views: {r['views']}")
    
    # Save manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'results': results,
    }
    
    manifest_path = output_dir / 'bundle_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()