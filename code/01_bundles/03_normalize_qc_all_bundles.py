#!/usr/bin/env python3
"""
03_normalize_qc_all_bundles.py (v2)
====================================

Domain-standard normalization for variance-prediction paradox analysis.

Key Principles (from normalization_note.md):
1. NO global per-feature scaling (z-score) - this collapses marginal variance
2. Domain-standard transforms to remove technical scale effects
3. For pre-standardized data (MLOmics), use latent-axis variance contribution
4. Same feature space X* for both variance and prediction analysis
5. Drop zero-variance features after imputation (nuisance for PCA/models)

Missingness Policy:
1. Drop features with >30% missing
2. Median-impute remaining NaNs (or zero for compositional data)
3. Drop zero-variance features post-imputation

Dataset-Specific Transforms:
- MLOmics: None (pre-standardized) - flag for latent-axis approach
- IBDMDB MGX: Keep relative abundance + CLR sensitivity
- IBDMDB MGX_func/MPX/MBX: log1p if not already transformed
- CCLE: Already normalized (TPM+log1p, normalized proteomics)
- TCGA-GBM mRNA: log1p if needed
- TCGA-GBM methylation: Keep beta + optional M-value sensitivity

KEY METHODOLOGICAL SENTENCE (for paper Methods):
"We applied domain-standard transforms and missing-data handling but intentionally
avoided global per-feature scaling (z-scoring/robust scaling) to preserve heterogeneity
in marginal variance, which is required to define variance-driving features; for
pre-standardised aligned matrices, variance-driving features were instead defined
by latent-axis (PCA) variance contributions."

Output: outputs/bundles/{dataset}_bundle_normalized.npz

Usage:
    python 03_normalize_qc_all_bundles.py
    python 03_normalize_qc_all_bundles.py --dataset mlomics
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# =============================================================================
# Missingness Handling
# =============================================================================

def handle_missingness(
    X: np.ndarray,
    feature_names: np.ndarray,
    view_name: str,
    max_missing_frac: float = 0.30,
    impute_method: str = 'median',
    drop_zero_variance: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Handle missing values in a view matrix.
    
    Policy:
    1. Drop features with >max_missing_frac missing
    2. Impute remaining NaNs with feature-wise median (or 0)
    3. Drop zero-variance features (post-imputation)
    
    Returns:
        X_clean: cleaned matrix
        feature_names_clean: filtered feature names
        report: missingness handling report
    """
    n_samples, n_features = X.shape
    
    report = {
        'view': view_name,
        'original_features': n_features,
        'max_missing_threshold': max_missing_frac,
        'impute_method': impute_method,
    }
    
    # Step 1: Calculate per-feature missingness
    missing_per_feature = np.sum(np.isnan(X), axis=0) / n_samples
    
    # Drop features above threshold
    keep_mask = missing_per_feature <= max_missing_frac
    n_dropped_missing = np.sum(~keep_mask)
    
    X_filtered = X[:, keep_mask]
    features_filtered = feature_names[keep_mask] if len(feature_names) == n_features else feature_names
    
    report['features_dropped_missing'] = int(n_dropped_missing)
    report['features_dropped_missing_pct'] = float(n_dropped_missing / n_features * 100) if n_features > 0 else 0
    
    # Step 2: Impute remaining NaNs
    remaining_nans = np.sum(np.isnan(X_filtered))
    
    if remaining_nans > 0:
        if impute_method == 'median':
            # Feature-wise median imputation
            feature_medians = np.nanmedian(X_filtered, axis=0)
            nan_indices = np.where(np.isnan(X_filtered))
            X_filtered[nan_indices] = feature_medians[nan_indices[1]]
        elif impute_method == 'zero':
            X_filtered = np.nan_to_num(X_filtered, nan=0.0)
        elif impute_method == 'mean':
            feature_means = np.nanmean(X_filtered, axis=0)
            nan_indices = np.where(np.isnan(X_filtered))
            X_filtered[nan_indices] = feature_means[nan_indices[1]]
        
        report['nans_imputed'] = int(remaining_nans)
        report['nans_imputed_pct'] = float(remaining_nans / X_filtered.size * 100) if X_filtered.size > 0 else 0
    else:
        report['nans_imputed'] = 0
        report['nans_imputed_pct'] = 0.0
    
    # Step 3: Drop zero-variance features (post-imputation)
    n_dropped_zerovar = 0
    if drop_zero_variance and X_filtered.shape[1] > 0:
        feature_vars = np.var(X_filtered, axis=0)
        nonzero_var_mask = feature_vars > 0
        n_dropped_zerovar = np.sum(~nonzero_var_mask)
        
        if n_dropped_zerovar > 0:
            X_filtered = X_filtered[:, nonzero_var_mask]
            features_filtered = features_filtered[nonzero_var_mask] if len(features_filtered) == len(nonzero_var_mask) else features_filtered
    
    report['features_dropped_zerovar'] = int(n_dropped_zerovar)
    report['features_retained'] = int(X_filtered.shape[1])
    report['total_features_dropped'] = int(n_dropped_missing + n_dropped_zerovar)
    report['total_features_dropped_pct'] = float((n_dropped_missing + n_dropped_zerovar) / n_features * 100) if n_features > 0 else 0
    
    # Verify no NaNs remain
    final_nans = np.sum(np.isnan(X_filtered))
    report['final_nans'] = int(final_nans)
    
    return X_filtered, features_filtered, report


# =============================================================================
# Safe Save Function (no pickle)
# =============================================================================

def _as_str_array(x) -> np.ndarray:
    """Force stable unicode ndarray (prevents object-array pickling in npz)."""
    if x is None:
        return np.array([], dtype=str)
    if isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S"):
        return x.astype(str, copy=False)
    return np.asarray(list(x), dtype=str)


def _as_float32_matrix(X) -> np.ndarray:
    """Stable numeric matrix (float32) for views."""
    X = np.asarray(X)
    if X.dtype == np.float32:
        return X
    return X.astype(np.float32, copy=False)


def _as_numeric_1d(y) -> np.ndarray:
    """Stable numeric 1D array for labels/targets."""
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)
    if np.issubdtype(y.dtype, np.integer):
        return y.astype(np.int32, copy=False)
    return y.astype(np.float32, copy=False)


def save_normalized_bundle(bundle: Dict, output_path: Path) -> None:
    """Save normalized bundle with no pickled object arrays."""
    save_dict = {}
    
    for key, val in bundle.items():
        if key.startswith('X_'):
            save_dict[key] = _as_float32_matrix(val)
        elif key.startswith('features_'):
            save_dict[key] = _as_str_array(val)
        elif key == 'sample_ids':
            save_dict[key] = _as_str_array(val)
        elif key == 'y':
            save_dict[key] = _as_numeric_1d(val)
        elif key == 'info':
            # Info is already a JSON string
            save_dict[key] = np.asarray(str(val), dtype=str)
        else:
            save_dict[key] = val
    
    np.savez_compressed(output_path, **save_dict)


# =============================================================================
# Normalization Utilities
# =============================================================================

def check_scale(X: np.ndarray, name: str = "") -> Dict[str, Any]:
    """
    Check the scale/distribution of a data matrix.
    
    Returns dict with:
    - is_standardized: True if variance ~1 across features
    - is_log_transformed: True if looks log-scale (no extreme values, reasonable range)
    - is_compositional: True if rows sum to ~1 or ~100
    - stats: basic statistics
    """
    # Cast to float64 to avoid overflow in variance calculations
    X_clean = np.nan_to_num(X, nan=0.0).astype(np.float64)
    
    # Feature variances (computed in float64 to prevent overflow)
    feat_vars = np.var(X_clean, axis=0)
    mean_var = np.mean(feat_vars)
    std_var = np.std(feat_vars)
    
    # Check if pre-standardized (variance ~1 for most features)
    is_standardized = (0.8 < mean_var < 1.2) and (std_var < 0.5)
    
    # Check if log-transformed (values typically -10 to 20 range, no extreme outliers)
    val_range = np.max(X_clean) - np.min(X_clean)
    max_abs = np.max(np.abs(X_clean))
    is_log_transformed = (max_abs < 50) and (val_range < 100)
    
    # Check if compositional (rows sum to ~1 or ~100)
    row_sums = np.sum(X_clean, axis=1)
    mean_row_sum = np.mean(row_sums)
    is_compositional = (0.9 < mean_row_sum < 1.1) or (90 < mean_row_sum < 110)
    
    return {
        'is_standardized': bool(is_standardized),
        'is_log_transformed': bool(is_log_transformed),
        'is_compositional': bool(is_compositional),
        'stats': {
            'mean_feature_var': float(mean_var),
            'std_feature_var': float(std_var),
            'value_range': float(val_range),
            'max_abs_value': float(max_abs),
            'mean_row_sum': float(mean_row_sum),
            'min': float(np.min(X_clean)),
            'max': float(np.max(X_clean)),
            'mean': float(np.mean(X_clean)),
        }
    }


def log1p_transform(X: np.ndarray) -> np.ndarray:
    """Apply log1p transform, handling negative values."""
    X_pos = np.maximum(X, 0)  # Ensure non-negative
    return np.log1p(X_pos)


def clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """
    Centered log-ratio transform for compositional data.
    
    CLR(x_j) = log(x_j + eps) - mean(log(x + eps))
    """
    X_pos = X + pseudocount
    log_X = np.log(X_pos)
    geometric_mean = np.mean(log_X, axis=1, keepdims=True)
    return log_X - geometric_mean


def beta_to_mvalue(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convert methylation beta values to M-values.
    
    M = log2(beta / (1 - beta))
    """
    # Clip to avoid log(0)
    beta = np.clip(X, eps, 1 - eps)
    return np.log2(beta / (1 - beta))


# =============================================================================
# Dataset-Specific Normalization
# =============================================================================

def normalize_mlomics(bundle: Dict) -> Tuple[Dict, Dict[str, Any]]:
    """
    MLOmics BRCA normalization.
    
    Key finding: Data is pre-standardized (variance ~1 per feature).
    Action: 
    - Handle missingness (drop >30%, impute median)
    - No additional scaling (already z-scored)
    - Flag for latent-axis variance approach
    """
    norm_info = {
        'dataset': 'mlomics',
        'transforms_applied': {},
        'variance_approach': 'marginal',  # May change to 'latent_axis'
        'missingness_reports': {},
        'notes': [],
    }
    
    X_views = {}
    feature_names_out = {}
    
    for view_name in ['mRNA', 'miRNA', 'methylation', 'CNV']:
        key = f'X_{view_name}'
        feat_key = f'features_{view_name}'
        if key not in bundle:
            continue
        
        X = bundle[key]
        features = bundle[feat_key] if feat_key in bundle else np.arange(X.shape[1]).astype(str)
        
        # Handle missingness
        X_clean, features_clean, miss_report = handle_missingness(
            X, features, view_name, max_missing_frac=0.30, impute_method='median'
        )
        norm_info['missingness_reports'][view_name] = miss_report
        
        if miss_report['features_dropped_missing'] > 0:
            norm_info['notes'].append(
                f"{view_name}: Dropped {miss_report['features_dropped_missing']} features ({miss_report['features_dropped_missing_pct']:.1f}%) with >30% missing"
            )
        if miss_report.get('features_dropped_zerovar', 0) > 0:
            norm_info['notes'].append(
                f"{view_name}: Dropped {miss_report['features_dropped_zerovar']} zero-variance features"
            )
        if miss_report['nans_imputed'] > 0:
            norm_info['notes'].append(
                f"{view_name}: Imputed {miss_report['nans_imputed']} NaNs with feature median"
            )
        
        # Check scale (is it pre-standardized?)
        scale_check = check_scale(X_clean, view_name)
        
        if scale_check['is_standardized']:
            norm_info['notes'].append(
                f"{view_name}: Pre-standardized (var={scale_check['stats']['mean_feature_var']:.3f}). "
                f"Use latent-axis variance contribution."
            )
            norm_info['variance_approach'] = 'latent_axis'
            norm_info['transforms_applied'][view_name] = 'none (pre-standardized)'
        else:
            norm_info['transforms_applied'][view_name] = 'none'
        
        X_views[view_name] = X_clean
        feature_names_out[view_name] = features_clean
        norm_info[f'{view_name}_scale'] = scale_check
    
    # Build output bundle
    out_bundle = {
        'y': bundle['y'],
        'sample_ids': bundle['sample_ids'],
    }
    
    for vname, X in X_views.items():
        out_bundle[f'X_{vname}'] = X
        out_bundle[f'features_{vname}'] = feature_names_out[vname]
    
    # Update info
    info = json.loads(str(bundle['info']))
    info['normalization'] = norm_info
    info['n_views'] = len(X_views)
    info['views'] = {k: X_views[k].shape[1] for k in X_views}
    out_bundle['info'] = json.dumps(info)
    
    return out_bundle, norm_info


def normalize_ibdmdb(bundle: Dict) -> Tuple[Dict, Dict[str, Any]]:
    """
    IBDMDB normalization.
    
    - MGX: Keep relative abundance + add CLR as sensitivity
    - MGX_func: log1p if not already log-transformed (gene families from HUMAnN)
    - MPX: Check scale, log1p if needed
    - MBX: log1p (metabolomics intensities)
    
    All views: Handle missingness first (drop >30%, impute median)
    """
    norm_info = {
        'dataset': 'ibdmdb',
        'transforms_applied': {},
        'variance_approach': 'marginal',
        'missingness_reports': {},
        'notes': [],
    }
    
    X_views = {}
    feature_names_out = {}
    
    # MGX - relative abundance (compositional)
    if 'X_MGX' in bundle:
        X = bundle['X_MGX']
        features = bundle.get('features_MGX', np.arange(X.shape[1]).astype(str))
        
        # Handle missingness
        X_clean, features_clean, miss_report = handle_missingness(
            X, features, 'MGX', max_missing_frac=0.30, impute_method='zero'  # Zero for compositional
        )
        norm_info['missingness_reports']['MGX'] = miss_report
        
        scale_check = check_scale(X_clean, 'MGX')
        
        # Keep original as primary
        X_views['MGX'] = X_clean
        feature_names_out['MGX'] = features_clean
        norm_info['transforms_applied']['MGX'] = 'none (relative abundance)'
        norm_info['MGX_scale'] = scale_check
        
        # Add CLR as sensitivity view
        X_clr = clr_transform(X_clean)
        X_views['MGX_CLR'] = X_clr
        feature_names_out['MGX_CLR'] = features_clean
        norm_info['transforms_applied']['MGX_CLR'] = 'CLR (sensitivity)'
        
        norm_info['notes'].append(
            "MGX: Added CLR sensitivity view for compositional data robustness."
        )
    
    # MGX_func - gene families (HUMAnN functional from metagenomic reads)
    if 'X_MGX_func' in bundle:
        X = bundle['X_MGX_func']
        features = bundle.get('features_MGX_func', np.arange(X.shape[1]).astype(str))
        
        X_clean, features_clean, miss_report = handle_missingness(
            X, features, 'MGX_func', max_missing_frac=0.30, impute_method='zero'
        )
        norm_info['missingness_reports']['MGX_func'] = miss_report
        
        scale_check = check_scale(X_clean, 'MGX_func')
        
        if not scale_check['is_log_transformed'] and scale_check['stats']['max_abs_value'] > 100:
            X_clean = log1p_transform(X_clean)
            norm_info['transforms_applied']['MGX_func'] = 'log1p'
            norm_info['notes'].append("MGX_func: Applied log1p transform.")
        else:
            norm_info['transforms_applied']['MGX_func'] = 'none'
        
        X_views['MGX_func'] = X_clean
        feature_names_out['MGX_func'] = features_clean
        norm_info['MGX_func_scale'] = scale_check
    
    # MPX - proteomics
    if 'X_MPX' in bundle:
        X = bundle['X_MPX']
        features = bundle.get('features_MPX', np.arange(X.shape[1]).astype(str))
        
        X_clean, features_clean, miss_report = handle_missingness(
            X, features, 'MPX', max_missing_frac=0.30, impute_method='median'
        )
        norm_info['missingness_reports']['MPX'] = miss_report
        
        scale_check = check_scale(X_clean, 'MPX')
        
        if not scale_check['is_log_transformed'] and scale_check['stats']['max_abs_value'] > 100:
            X_clean = log1p_transform(X_clean)
            norm_info['transforms_applied']['MPX'] = 'log1p'
            norm_info['notes'].append("MPX: Applied log1p transform.")
        else:
            norm_info['transforms_applied']['MPX'] = 'none'
        
        X_views['MPX'] = X_clean
        feature_names_out['MPX'] = features_clean
        norm_info['MPX_scale'] = scale_check
    
    # MBX - metabolomics
    if 'X_MBX' in bundle:
        X = bundle['X_MBX']
        features = bundle.get('features_MBX', np.arange(X.shape[1]).astype(str))
        
        X_clean, features_clean, miss_report = handle_missingness(
            X, features, 'MBX', max_missing_frac=0.30, impute_method='zero'
        )
        norm_info['missingness_reports']['MBX'] = miss_report
        
        scale_check = check_scale(X_clean, 'MBX')
        
        # Always apply log1p to metabolomics (heavy-tailed intensities)
        if not scale_check['is_log_transformed']:
            X_clean = log1p_transform(X_clean)
            norm_info['transforms_applied']['MBX'] = 'log1p'
            norm_info['notes'].append("MBX: Applied log1p to stabilize heavy tails.")
        else:
            norm_info['transforms_applied']['MBX'] = 'none (already log-scale)'
        
        X_views['MBX'] = X_clean
        feature_names_out['MBX'] = features_clean
        norm_info['MBX_scale'] = scale_check
    
    # Build output bundle
    out_bundle = {
        'y': bundle['y'],
        'sample_ids': bundle['sample_ids'],
    }
    
    for vname, X in X_views.items():
        out_bundle[f'X_{vname}'] = X
        out_bundle[f'features_{vname}'] = feature_names_out[vname]
    
    # Update info
    info = json.loads(str(bundle['info']))
    info['normalization'] = norm_info
    info['n_views'] = len(X_views)
    info['views'] = {k: X_views[k].shape[1] for k in X_views}
    out_bundle['info'] = json.dumps(info)
    
    return out_bundle, norm_info


def normalize_ccle(bundle: Dict) -> Tuple[Dict, Dict[str, Any]]:
    """
    CCLE normalization.
    
    - mRNA: Already TPM+log1p (from filename)
    - CNV: Copy number values, no transform needed
    - proteomics: Already normalized, but has ~28% missing - handle carefully
    
    All views: Handle missingness first (drop >30%, impute median)
    """
    norm_info = {
        'dataset': 'ccle',
        'transforms_applied': {},
        'variance_approach': 'marginal',
        'missingness_reports': {},
        'notes': [],
    }
    
    X_views = {}
    feature_names_out = {}
    
    for view_name in ['mRNA', 'CNV', 'proteomics']:
        key = f'X_{view_name}'
        feat_key = f'features_{view_name}'
        if key not in bundle:
            continue
        
        X = bundle[key]
        features = bundle.get(feat_key, np.arange(X.shape[1]).astype(str))
        
        # Handle missingness - proteomics has ~28% NaN
        impute_method = 'median' if view_name == 'proteomics' else 'zero'
        X_clean, features_clean, miss_report = handle_missingness(
            X, features, view_name, max_missing_frac=0.30, impute_method=impute_method
        )
        norm_info['missingness_reports'][view_name] = miss_report
        
        if miss_report['features_dropped_missing'] > 0:
            norm_info['notes'].append(
                f"{view_name}: Dropped {miss_report['features_dropped_missing']} features ({miss_report['features_dropped_missing_pct']:.1f}%) with >30% missing"
            )
        if miss_report.get('features_dropped_zerovar', 0) > 0:
            norm_info['notes'].append(
                f"{view_name}: Dropped {miss_report['features_dropped_zerovar']} zero-variance features"
            )
        if miss_report['nans_imputed'] > 0:
            norm_info['notes'].append(
                f"{view_name}: Imputed {miss_report['nans_imputed']} NaNs ({miss_report['nans_imputed_pct']:.1f}%) with {impute_method}"
            )
        
        scale_check = check_scale(X_clean, view_name)
        
        # Check if already log-transformed
        if view_name == 'mRNA':
            # Filename: OmicsExpressionProteinCodingGenesTPMLogp1.csv
            norm_info['transforms_applied'][view_name] = 'none (already TPM+log1p)'
        elif view_name == 'CNV':
            norm_info['transforms_applied'][view_name] = 'none (copy number)'
        elif view_name == 'proteomics':
            # Filename: protein_quant_current_normalized.csv
            norm_info['transforms_applied'][view_name] = 'none (already normalized)'
        
        X_views[view_name] = X_clean
        feature_names_out[view_name] = features_clean
        norm_info[f'{view_name}_scale'] = scale_check
    
    # Build output bundle
    out_bundle = {
        'y': bundle['y'],
        'sample_ids': bundle['sample_ids'],
    }
    
    for vname, X in X_views.items():
        out_bundle[f'X_{vname}'] = X
        out_bundle[f'features_{vname}'] = feature_names_out[vname]
    
    # Update info
    info = json.loads(str(bundle['info']))
    info['normalization'] = norm_info
    info['n_views'] = len(X_views)
    info['views'] = {k: X_views[k].shape[1] for k in X_views}
    out_bundle['info'] = json.dumps(info)
    
    return out_bundle, norm_info


def normalize_tcga_gbm(bundle: Dict) -> Tuple[Dict, Dict[str, Any]]:
    """
    TCGA-GBM normalization.
    
    - mRNA: log1p if not already transformed (RSEM/FPKM)
    - methylation: Keep beta values + optional M-value sensitivity
    - CNV: GISTIC thresholded values, no transform needed
    
    All views: Handle missingness first (drop >30%, impute median)
    """
    norm_info = {
        'dataset': 'tcga_gbm',
        'transforms_applied': {},
        'variance_approach': 'marginal',
        'missingness_reports': {},
        'notes': [],
    }
    
    X_views = {}
    feature_names_out = {}
    
    # mRNA
    if 'X_mRNA' in bundle:
        X = bundle['X_mRNA']
        features = bundle.get('features_mRNA', np.arange(X.shape[1]).astype(str))
        
        X_clean, features_clean, miss_report = handle_missingness(
            X, features, 'mRNA', max_missing_frac=0.30, impute_method='median'
        )
        norm_info['missingness_reports']['mRNA'] = miss_report
        
        scale_check = check_scale(X_clean, 'mRNA')
        
        if not scale_check['is_log_transformed'] and scale_check['stats']['max_abs_value'] > 100:
            X_clean = log1p_transform(X_clean)
            norm_info['transforms_applied']['mRNA'] = 'log1p'
            norm_info['notes'].append("mRNA: Applied log1p to expression values.")
        else:
            norm_info['transforms_applied']['mRNA'] = 'none (already log-scale)'
        
        X_views['mRNA'] = X_clean
        feature_names_out['mRNA'] = features_clean
        norm_info['mRNA_scale'] = scale_check
    
    # Methylation
    if 'X_methylation' in bundle:
        X = bundle['X_methylation']
        features = bundle.get('features_methylation', np.arange(X.shape[1]).astype(str))
        
        X_clean, features_clean, miss_report = handle_missingness(
            X, features, 'methylation', max_missing_frac=0.30, impute_method='median'
        )
        norm_info['missingness_reports']['methylation'] = miss_report
        
        scale_check = check_scale(X_clean, 'methylation')
        
        # Check if beta values (0-1 range)
        is_beta = scale_check['stats']['min'] >= 0 and scale_check['stats']['max'] <= 1
        
        X_views['methylation'] = X_clean
        feature_names_out['methylation'] = features_clean
        norm_info['transforms_applied']['methylation'] = 'none (beta values)'
        norm_info['methylation_scale'] = scale_check
        
        # Add M-value as sensitivity view
        if is_beta:
            X_mval = beta_to_mvalue(X_clean)
            X_views['methylation_Mval'] = X_mval
            feature_names_out['methylation_Mval'] = features_clean
            norm_info['transforms_applied']['methylation_Mval'] = 'M-value (sensitivity)'
            norm_info['notes'].append("methylation: Added M-value sensitivity view.")
    
    # CNV
    if 'X_CNV' in bundle:
        X = bundle['X_CNV']
        features = bundle.get('features_CNV', np.arange(X.shape[1]).astype(str))
        
        X_clean, features_clean, miss_report = handle_missingness(
            X, features, 'CNV', max_missing_frac=0.30, impute_method='zero'
        )
        norm_info['missingness_reports']['CNV'] = miss_report
        
        scale_check = check_scale(X_clean, 'CNV')
        
        X_views['CNV'] = X_clean
        feature_names_out['CNV'] = features_clean
        norm_info['transforms_applied']['CNV'] = 'none (GISTIC values)'
        norm_info['CNV_scale'] = scale_check
    
    # Build output bundle
    out_bundle = {
        'y': bundle['y'],
        'sample_ids': bundle['sample_ids'],
    }
    
    for vname, X in X_views.items():
        out_bundle[f'X_{vname}'] = X
        out_bundle[f'features_{vname}'] = feature_names_out[vname]
    
    # Update info
    info = json.loads(str(bundle['info']))
    info['normalization'] = norm_info
    info['n_views'] = len(X_views)
    info['views'] = {k: X_views[k].shape[1] for k in X_views}
    out_bundle['info'] = json.dumps(info)
    
    return out_bundle, norm_info


# =============================================================================
# Main
# =============================================================================

NORMALIZERS = {
    'mlomics': normalize_mlomics,
    'ibdmdb': normalize_ibdmdb,
    'ccle': normalize_ccle,
    'tcga_gbm': normalize_tcga_gbm,
}


def main():
    parser = argparse.ArgumentParser(description="Normalize bundles for variance-prediction analysis")
    parser.add_argument('--bundles-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--dataset', choices=['all'] + list(NORMALIZERS.keys()), default='all')
    
    args = parser.parse_args()
    
    # Determine directories
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if not (project_root / 'outputs').exists():
        project_root = Path.cwd()
    
    bundles_dir = args.bundles_dir or project_root / 'outputs' / 'bundles'
    output_dir = args.output_dir or bundles_dir  # Same directory by default
    
    print("=" * 70)
    print("BUNDLE NORMALIZATION (v2)")
    print("=" * 70)
    print(f"Input dir:  {bundles_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)
    
    # Select datasets
    if args.dataset == 'all':
        datasets = list(NORMALIZERS.keys())
    else:
        datasets = [args.dataset]
    
    results = []
    
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Normalizing: {ds_name.upper()}")
        print('='*60)
        
        # Load bundle
        input_path = bundles_dir / f'{ds_name}_bundle.npz'
        if not input_path.exists():
            print(f"  [!] Bundle not found: {input_path}")
            results.append({'dataset': ds_name, 'status': 'not_found'})
            continue
        
        try:
            bundle = dict(np.load(input_path, allow_pickle=False))
            
            # Normalize
            norm_bundle, norm_info = NORMALIZERS[ds_name](bundle)
            
            # Save with safe non-pickled format
            output_path = output_dir / f'{ds_name}_bundle_normalized.npz'
            save_normalized_bundle(norm_bundle, output_path)
            
            # Report
            print(f"\n  Transforms applied:")
            for view, transform in norm_info['transforms_applied'].items():
                print(f"    {view}: {transform}")
            
            if norm_info.get('notes'):
                print(f"\n  Notes:")
                for note in norm_info['notes']:
                    print(f"    - {note}")
            
            print(f"\n  Variance approach: {norm_info['variance_approach']}")
            print(f"\n  [OK] Saved: {output_path.name}")
            
            results.append({
                'dataset': ds_name,
                'status': 'success',
                'transforms': norm_info['transforms_applied'],
                'variance_approach': norm_info['variance_approach'],
            })
            
        except Exception as e:
            print(f"  [X] Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'dataset': ds_name, 'status': 'error', 'error': str(e)})
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for r in results:
        status_icon = {'success': '[OK]', 'error': '[X]', 'not_found': '[!]'}.get(r['status'], '?')
        print(f"  {status_icon} {r['dataset']}: {r['status']}")
        if r['status'] == 'success':
            print(f"      Variance approach: {r['variance_approach']}")
    
    print("=" * 70)
    
    return 0 if all(r['status'] == 'success' for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())