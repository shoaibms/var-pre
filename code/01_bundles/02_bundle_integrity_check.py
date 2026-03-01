#!/usr/bin/env python3
"""
02_bundle_integrity_check.py (v3 - Updated for .npz format)
============================================================

Contract-style integrity checks for prepared bundles in outputs/bundles/*.npz

Goals
-----
1) Confirm bundles match the expected analysis contract
2) Detect silent alignment/dtype/scale problems early
3) Produce audit trail (json/md/txt) for downstream scripts

Bundle Schema (.npz format from 01_prepare_all_bundles_v2.py)
-------------------------------------------------------------
- X_{view_name}: np.ndarray - samples × features matrix
- y: np.ndarray - classification labels
- sample_ids: np.ndarray - sample identifiers  
- features_{view_name}: np.ndarray - feature names per view
- info: JSON string - dataset metadata

Outputs
-------
Writes to outputs/qc/:
  - bundle_integrity.json
  - bundle_integrity.md
  - bundle_integrity.txt

Usage
-----
  python 02_bundle_integrity_check.py
  python 02_bundle_integrity_check.py --deep
  python 02_bundle_integrity_check.py --dataset mlomics
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# =============================================================================
# Expected Bundle Specifications
# =============================================================================

class BundleSpec:
    """Expected structure for a bundle."""
    def __init__(
        self,
        name: str,
        filename: str,
        expected_views: List[str],
        min_samples: int,
        min_features_per_view: Dict[str, int],
        n_classes: int,
        domain: str,
    ):
        self.name = name
        self.filename = filename
        self.expected_views = expected_views
        self.min_samples = min_samples
        self.min_features_per_view = min_features_per_view
        self.n_classes = n_classes
        self.domain = domain


EXPECTED_BUNDLES = {
    'mlomics': BundleSpec(
        name='MLOmics BRCA',
        filename='mlomics_bundle.npz',
        expected_views=['mRNA', 'miRNA', 'methylation', 'CNV'],
        min_samples=600,
        min_features_per_view={'mRNA': 10000, 'miRNA': 200, 'methylation': 10000, 'CNV': 10000},
        n_classes=5,  # PAM50 subtypes
        domain='Human Cancer (Breast)',
    ),
    'ibdmdb': BundleSpec(
        name='IBDMDB/HMP2',
        filename='ibdmdb_bundle.npz',
        expected_views=['MGX', 'MGX_func', 'MPX', 'MBX'],
        min_samples=100,
        min_features_per_view={'MGX': 100, 'MGX_func': 1000, 'MPX': 100, 'MBX': 1000},
        n_classes=3,  # nonIBD, UC, CD
        domain='Gut Microbiome',
    ),
    'ccle': BundleSpec(
        name='CCLE/DepMap',
        filename='ccle_bundle.npz',
        expected_views=['mRNA', 'CNV', 'proteomics'],
        min_samples=300,
        min_features_per_view={'mRNA': 15000, 'CNV': 20000, 'proteomics': 5000},  # proteomics: 36% dropped for missingness
        n_classes=10,  # tissue types (at least)
        domain='Cell Lines',
    ),
    'tcga_gbm': BundleSpec(
        name='TCGA-GBM',
        filename='tcga_gbm_bundle.npz',
        expected_views=['mRNA', 'methylation', 'CNV'],
        min_samples=30,
        min_features_per_view={'mRNA': 15000, 'methylation': 15000, 'CNV': 20000},
        n_classes=2,  # subtypes or tumor/normal
        domain='Human Cancer (Brain)',
    ),
}


# =============================================================================
# Validation Functions
# =============================================================================

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_bundle(
    bundle_path: Path,
    spec: BundleSpec,
    deep: bool = False,
    sample_rows: int = 100,
    sample_cols: int = 1000,
) -> Dict[str, Any]:
    """
    Validate a single bundle against its specification.
    
    Returns a report dict with status, errors, warnings, and view details.
    """
    report = {
        'dataset': spec.name,
        'bundle_path': str(bundle_path),
        'status': 'UNKNOWN',
        'errors': [],
        'warnings': [],
        'views': {},
        'n_errors': 0,
        'n_warnings': 0,
    }
    
    # Check file exists
    if not bundle_path.exists():
        report['errors'].append(f"Bundle file not found: {bundle_path}")
        report['status'] = 'FAIL'
        report['n_errors'] = 1
        return report
    
    # File metadata
    report['file'] = {
        'size_mb': bundle_path.stat().st_size / (1024 * 1024),
        'sha256': compute_file_hash(bundle_path),
    }
    
    # Load bundle (no pickle needed with proper save format)
    try:
        bundle = np.load(bundle_path, allow_pickle=False)
    except Exception as e:
        # Fallback for legacy bundles with pickle
        try:
            bundle = np.load(bundle_path, allow_pickle=True)
            report['warnings'].append("Bundle loaded with allow_pickle=True (legacy format)")
        except Exception as e2:
            report['errors'].append(f"Failed to load bundle: {e2}")
            report['status'] = 'FAIL'
            report['n_errors'] = 1
            return report
    
    # Check required keys
    required_keys = ['y', 'sample_ids', 'info']
    for key in required_keys:
        if key not in bundle.files:
            report['errors'].append(f"Missing required key: {key}")
    
    # Load basic arrays
    try:
        y = bundle['y']
        sample_ids = bundle['sample_ids']
        info_str = str(bundle['info'])
        info = json.loads(info_str)
    except Exception as e:
        report['errors'].append(f"Failed to load basic arrays: {e}")
        report['status'] = 'FAIL'
        report['n_errors'] = len(report['errors'])
        return report
    
    n_samples = len(sample_ids)
    report['n_samples'] = n_samples
    report['n_classes'] = len(np.unique(y))
    report['info'] = info
    
    # Check sample count
    if n_samples < spec.min_samples:
        report['warnings'].append(
            f"Sample count {n_samples} below expected minimum {spec.min_samples}"
        )
    
    # Check label alignment
    if len(y) != n_samples:
        report['errors'].append(
            f"Label count ({len(y)}) != sample count ({n_samples})"
        )
    
    # Check class count
    if report['n_classes'] < spec.n_classes:
        report['warnings'].append(
            f"Class count {report['n_classes']} below expected {spec.n_classes}"
        )
    
    # Find and validate views
    view_keys = [k for k in bundle.files if k.startswith('X_')]
    found_views = [k[2:] for k in view_keys]  # Remove 'X_' prefix
    
    report['found_views'] = found_views
    
    # Check expected views - allow extra sensitivity views (e.g., MGX_CLR, methylation_Mval)
    # These are added by normalization and are acceptable
    sensitivity_view_suffixes = ['_CLR', '_Mval', '_log']
    
    for expected_view in spec.expected_views:
        if expected_view not in found_views:
            report['warnings'].append(f"Expected view '{expected_view}' not found")
    
    # Note extra views (not an error, just informational)
    extra_views = [v for v in found_views if v not in spec.expected_views]
    if extra_views:
        # Check if they're sensitivity views (acceptable)
        sensitivity_extras = [v for v in extra_views 
                            if any(v.endswith(s) for s in sensitivity_view_suffixes)]
        unexpected_extras = [v for v in extra_views if v not in sensitivity_extras]
        
        if sensitivity_extras:
            report['sensitivity_views'] = sensitivity_extras
        if unexpected_extras:
            report['warnings'].append(f"Unexpected extra views: {unexpected_extras}")
    
    # Validate each view
    for view_name in found_views:
        view_report = {'name': view_name}
        
        try:
            X = bundle[f'X_{view_name}']
            view_report['shape'] = X.shape
            view_report['dtype'] = str(X.dtype)
            
            # Check alignment
            if X.shape[0] != n_samples:
                report['errors'].append(
                    f"View '{view_name}' has {X.shape[0]} samples, expected {n_samples}"
                )
            
            # Check feature count
            min_features = spec.min_features_per_view.get(view_name, 0)
            if X.shape[1] < min_features:
                report['warnings'].append(
                    f"View '{view_name}' has {X.shape[1]} features, expected >= {min_features}"
                )
            
            # Check for NaN/Inf
            if np.any(np.isnan(X)):
                nan_count = np.sum(np.isnan(X))
                nan_frac = nan_count / X.size
                view_report['nan_fraction'] = nan_frac
                if nan_frac > 0.5:
                    report['errors'].append(
                        f"View '{view_name}' has {nan_frac:.1%} NaN values"
                    )
                elif nan_frac > 0.01:
                    report['warnings'].append(
                        f"View '{view_name}' has {nan_frac:.1%} NaN values"
                    )
            
            if np.any(np.isinf(X)):
                inf_count = np.sum(np.isinf(X))
                report['errors'].append(
                    f"View '{view_name}' has {inf_count} Inf values"
                )
            
            # Deep checks: statistics
            if deep:
                # Sample subset for large arrays
                row_idx = np.random.choice(X.shape[0], min(sample_rows, X.shape[0]), replace=False)
                col_idx = np.random.choice(X.shape[1], min(sample_cols, X.shape[1]), replace=False)
                X_sample = X[np.ix_(row_idx, col_idx)]
                
                # Replace NaN for stats
                X_clean = np.nan_to_num(X_sample, nan=0.0)
                
                view_report['stats'] = {
                    'mean': float(np.mean(X_clean)),
                    'std': float(np.std(X_clean)),
                    'min': float(np.min(X_clean)),
                    'max': float(np.max(X_clean)),
                    'q05': float(np.percentile(X_clean, 5)),
                    'q50': float(np.percentile(X_clean, 50)),
                    'q95': float(np.percentile(X_clean, 95)),
                }
                
                # Check for constant features
                col_vars = np.var(X_clean, axis=0)
                n_constant = np.sum(col_vars == 0)
                if n_constant > X_sample.shape[1] * 0.5:
                    report['warnings'].append(
                        f"View '{view_name}' has {n_constant}/{X_sample.shape[1]} constant features in sample"
                    )
                view_report['n_constant_features_sample'] = int(n_constant)
            
            # Check feature names
            feature_key = f'features_{view_name}'
            if feature_key in bundle.files:
                features = bundle[feature_key]
                view_report['n_features_named'] = len(features)
                if len(features) != X.shape[1]:
                    report['errors'].append(
                        f"View '{view_name}' feature names ({len(features)}) != feature count ({X.shape[1]})"
                    )
            else:
                report['warnings'].append(f"No feature names for view '{view_name}'")
        
        except Exception as e:
            report['errors'].append(f"Error validating view '{view_name}': {e}")
            view_report['error'] = str(e)
        
        report['views'][view_name] = view_report
    
    # Determine final status
    report['n_errors'] = len(report['errors'])
    report['n_warnings'] = len(report['warnings'])
    
    if report['n_errors'] > 0:
        report['status'] = 'FAIL'
    elif report['n_warnings'] > 0:
        report['status'] = 'WARN'
    else:
        report['status'] = 'PASS'
    
    return report


# =============================================================================
# Report Writing
# =============================================================================

def write_reports(reports: List[Dict], out_dir: Path) -> None:
    """Write validation reports in multiple formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    payload = {
        'generated_at': timestamp,
        'reports': reports,
    }
    
    # JSON
    with open(out_dir / 'bundle_integrity.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    
    # Markdown
    md = []
    md.append("# Bundle Integrity Report")
    md.append("")
    md.append(f"Generated: `{timestamp}`")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("| Dataset | Status | Samples | Views | Errors | Warnings |")
    md.append("|---------|--------|---------|-------|--------|----------|")
    
    for rep in reports:
        status_icon = {'PASS': 'PASS', 'WARN': 'WARN', 'FAIL': 'FAIL'}.get(rep['status'], '?')
        n_samples = rep.get('n_samples', '?')
        n_views = len(rep.get('views', {}))
        md.append(
            f"| {rep['dataset']} | {status_icon} | {n_samples} | {n_views} | "
            f"{rep['n_errors']} | {rep['n_warnings']} |"
        )
    
    md.append("")
    
    for rep in reports:
        md.append(f"## {rep['dataset']}")
        md.append("")
        md.append(f"- **Status**: {rep['status']}")
        md.append(f"- **Bundle**: `{rep['bundle_path']}`")
        
        if 'file' in rep:
            md.append(f"- **Size**: {rep['file'].get('size_mb', 0):.2f} MB")
            md.append(f"- **SHA256**: `{rep['file'].get('sha256', '')[:16]}...`")
        
        if rep.get('n_samples'):
            md.append(f"- **Samples**: {rep['n_samples']}")
            md.append(f"- **Classes**: {rep.get('n_classes', '?')}")
        
        md.append("")
        
        if rep.get('errors'):
            md.append("### Errors")
            for e in rep['errors']:
                md.append(f"- ERROR: {e}")
            md.append("")
        
        if rep.get('warnings'):
            md.append("### Warnings")
            for w in rep['warnings']:
                md.append(f"- WARNING: {w}")
            md.append("")
        
        if rep.get('views'):
            md.append("### Views")
            md.append("")
            md.append("| View | Shape | Dtype |")
            md.append("|------|-------|-------|")
            for vname, vinfo in rep['views'].items():
                shape = vinfo.get('shape', '?')
                dtype = vinfo.get('dtype', '?')
                md.append(f"| {vname} | {shape} | {dtype} |")
            md.append("")
        
        md.append("---")
        md.append("")
    
    with open(out_dir / 'bundle_integrity.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    
    # Plain text
    txt = []
    txt.append("=" * 70)
    txt.append("BUNDLE INTEGRITY REPORT")
    txt.append("=" * 70)
    txt.append(f"Generated: {timestamp}")
    txt.append("")
    
    for rep in reports:
        txt.append(f"{rep['dataset']}: {rep['status']}")
        txt.append(f"  Bundle: {rep['bundle_path']}")
        txt.append(f"  Samples: {rep.get('n_samples', '?')}, Classes: {rep.get('n_classes', '?')}")
        txt.append(f"  Errors: {rep['n_errors']}, Warnings: {rep['n_warnings']}")
        
        if rep.get('errors'):
            for e in rep['errors']:
                txt.append(f"    ERROR: {e}")
        if rep.get('warnings'):
            for w in rep['warnings']:
                txt.append(f"    WARNING: {w}")
        
        if rep.get('views'):
            txt.append("  Views:")
            for vname, vinfo in rep['views'].items():
                txt.append(f"    {vname}: {vinfo.get('shape', '?')}")
        
        txt.append("")
    
    with open(out_dir / 'bundle_integrity.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt))


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate bundle integrity for variance-prediction analysis"
    )
    parser.add_argument(
        '--bundles-dir', type=Path, default=None,
        help='Bundles directory (default: outputs/bundles)'
    )
    parser.add_argument(
        '--out-dir', type=Path, default=None,
        help='Output directory for reports (default: outputs/qc)'
    )
    parser.add_argument(
        '--dataset', choices=['all'] + list(EXPECTED_BUNDLES.keys()), default='all',
        help='Which dataset to check'
    )
    parser.add_argument(
        '--deep', action='store_true',
        help='Enable deeper statistical checks'
    )
    parser.add_argument(
        '--strict', action='store_true',
        help='Treat warnings as failures'
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='Check *_bundle_normalized.npz instead of raw bundles'
    )
    
    args = parser.parse_args()
    
    # Determine directories
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if not (project_root / 'outputs').exists():
        project_root = Path.cwd()
    
    bundles_dir = args.bundles_dir or project_root / 'outputs' / 'bundles'
    out_dir = args.out_dir or project_root / 'outputs' / 'qc'
    
    print("=" * 70)
    print("BUNDLE INTEGRITY CHECK (v3 - .npz format)")
    print("=" * 70)
    print(f"Bundles dir: {bundles_dir}")
    print(f"Output dir:  {out_dir}")
    print(f"Deep checks: {args.deep}")
    print(f"Strict mode: {args.strict}")
    print(f"Normalized:  {args.normalized}")
    print("=" * 70)
    
    # Select datasets
    if args.dataset == 'all':
        datasets = list(EXPECTED_BUNDLES.keys())
    else:
        datasets = [args.dataset]
    
    # Validate each
    reports = []
    any_fail = False
    
    for ds_name in datasets:
        spec = EXPECTED_BUNDLES[ds_name]
        bundle_path = bundles_dir / spec.filename
        missing_normalized = False
        
        # Handle --normalized flag
        if args.normalized:
            norm_name = spec.filename.replace('_bundle.npz', '_bundle_normalized.npz')
            norm_path = bundles_dir / norm_name
            if norm_path.exists():
                bundle_path = norm_path
            else:
                missing_normalized = True
        
        print(f"\n--- Checking {ds_name} ({bundle_path.name}) ---")
        
        report = validate_bundle(
            bundle_path=bundle_path,
            spec=spec,
            deep=args.deep,
        )
        
        # Handle missing normalized warning
        if missing_normalized:
            report['warnings'].append(
                f"Normalized bundle not found. Checked raw bundle instead: {bundle_path.name}"
            )
            report['n_warnings'] = len(report['warnings'])
            if args.strict:
                report['status'] = 'FAIL'
            elif report['status'] == 'PASS':
                report['status'] = 'WARN'
        
        # Mark if checking normalized bundle (for report clarity)
        report['is_normalized'] = args.normalized and not missing_normalized
        
        reports.append(report)
        
        status_icon = {'PASS': '[OK]', 'WARN': '[!]', 'FAIL': '[X]'}.get(report['status'], '?')
        print(f"Status: {status_icon} {report['status']}")
        print(f"  Samples: {report.get('n_samples', '?')}, Views: {len(report.get('views', {}))}")
        print(f"  Errors: {report['n_errors']}, Warnings: {report['n_warnings']}")
        
        if report['status'] == 'FAIL':
            any_fail = True
        elif report['status'] == 'WARN' and args.strict:
            any_fail = True
    
    # Write reports
    write_reports(reports, out_dir)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Reports written to: {out_dir}")
    print("  - bundle_integrity.json")
    print("  - bundle_integrity.md")
    print("  - bundle_integrity.txt")
    
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

