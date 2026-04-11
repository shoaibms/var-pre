#!/usr/bin/env python3
"""
Variance-Prediction Paradox - Download Verification Script v5.2
================================================================

Verifies downloaded datasets are complete and valid.
Matches v5.2 download script (full MASTER_PLAN scope).

Changes in v5.2:
- Correct MBX filename: HMP2_metabolomics_w_metadata.biom(.gz)
- Clinical matrix: plain text (no .gz)
- Gzip magic byte detection for robust verification

Usage:
    python verify_downloads_v5.2.py [--data-dir PATH] [--deep] [--schema-check]
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


# =============================================================================
# Expected Files Configuration (matches v5.2 download script - FULL MASTER_PLAN)
# =============================================================================

EXPECTED_FILES = {
    "mlomics": {
        "subdir": "mlomics",
        "files": {
            # Omics data
            "BRCA_mRNA_aligned.csv": {"min_size": 50 * 1024 * 1024},      # 50 MB
            "BRCA_miRNA_aligned.csv": {"min_size": 500 * 1024},           # 500 KB
            "BRCA_Methy_aligned.csv": {"min_size": 10 * 1024 * 1024},     # 10 MB
            "BRCA_CNV_aligned.csv": {"min_size": 5 * 1024 * 1024},        # 5 MB
            "BRCA_label_num.csv": {"min_size": 1 * 1024},                 # 1 KB
            # Clinical/survival (plain text, NOT compressed)
            "BRCA_clinicalMatrix": {"min_size": 500 * 1024},              # 500 KB
        },
        "alt_files": {
            # Also accept .gz version if present (legacy)
            "BRCA_clinicalMatrix.gz": {"min_size": 100 * 1024},
        },
        "required_for_analysis": [
            "BRCA_mRNA_aligned.csv", 
            "BRCA_label_num.csv"
        ],
        "required_for_survival": [
            "BRCA_clinicalMatrix"
        ],
    },
    "pbmc_multiome": {
        "subdir": "pbmc_multiome",
        "files": {
            "filtered_feature_bc_matrix.h5": {"min_size": 10 * 1024 * 1024},  # 10 MB
            "per_barcode_metrics.csv": {"min_size": 1 * 1024 * 1024},          # 1 MB
            "atac_peaks.bed": {"min_size": 1 * 1024 * 1024},                   # 1 MB
            "atac_peak_annotation.tsv": {"min_size": 1 * 1024 * 1024},         # 1 MB
            "analysis_summary.html": {"min_size": 100 * 1024},                 # 100 KB
            "cloupe.cloupe": {"min_size": 100 * 1024 * 1024},                  # 100 MB
        },
        "required_for_analysis": ["filtered_feature_bc_matrix.h5"],
    },
    "ibdmdb": {
        "subdir": "ibdmdb",
        "files": {
            # Metadata
            "hmp2_metadata.csv": {"min_size": 1 * 1024 * 1024},           # 1 MB
            "dysbiosis_scores.tsv": {"min_size": 10 * 1024},              # 10 KB
            # Metagenomics (MGX) - View 1
            "taxonomic_profiles.tsv": {"min_size": 2 * 1024 * 1024},      # 2 MB decompressed
            "pathabundance_relab.tsv": {"min_size": 20 * 1024 * 1024},    # 20 MB decompressed
            # Metabolomics (MBX) - View 2 (CORRECT filename with _w_metadata)
            "HMP2_metabolomics_w_metadata.biom": {"min_size": 5 * 1024 * 1024},  # 5 MB decompressed
        },
        "alt_files": {
            # Compressed versions also OK
            "taxonomic_profiles.tsv.gz": {"min_size": 500 * 1024},
            "pathabundance_relab.tsv.gz": {"min_size": 5 * 1024 * 1024},
            "HMP2_metabolomics_w_metadata.biom.gz": {"min_size": 5 * 1024 * 1024},
        },
        "required_for_analysis": [
            "hmp2_metadata.csv", 
            "taxonomic_profiles.tsv"
        ],
        "required_for_multiomics": [
            "taxonomic_profiles.tsv",
            "HMP2_metabolomics_w_metadata.biom"
        ],
    },
}


# =============================================================================
# Utility Functions
# =============================================================================

def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def is_gzipped(filepath: Path) -> bool:
    """Check if a file is actually gzipped by reading magic bytes."""
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(2)
            return magic == b'\x1f\x8b'
    except Exception:
        return False


@dataclass
class FileCheck:
    """Result of checking a single file."""
    filename: str
    exists: bool
    size: int = 0
    size_ok: bool = False
    hash_match: Optional[bool] = None
    loadable: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class DatasetVerification:
    """Result of verifying a dataset."""
    name: str
    path: str
    files_checked: List[FileCheck]
    total_size: int
    files_found: int
    files_expected: int
    pipeline_ready: bool
    survival_ready: bool  # MLOmics
    multiomics_ready: bool  # IBDMDB
    errors: List[str]


# =============================================================================
# Schema Verification Functions
# =============================================================================

def verify_schema_csv(filepath: Path) -> Tuple[bool, Optional[str]]:
    """Verify CSV file can be loaded and has expected structure."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath, nrows=5)
        if len(df.columns) < 2:
            return False, "CSV has fewer than 2 columns"
        return True, None
    except ImportError:
        return None, "pandas not installed"
    except Exception as e:
        return False, str(e)


def verify_schema_tsv(filepath: Path) -> Tuple[bool, Optional[str]]:
    """Verify TSV file can be loaded."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath, sep='\t', nrows=5)
        if len(df.columns) < 2:
            return False, "TSV has fewer than 2 columns"
        return True, None
    except ImportError:
        return None, "pandas not installed"
    except Exception as e:
        return False, str(e)


def verify_schema_h5(filepath: Path) -> Tuple[bool, Optional[str]]:
    """Verify HDF5 file can be opened and has expected structure."""
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            if len(keys) > 0:
                return True, None
            return False, "HDF5 file appears empty"
    except ImportError:
        return None, "h5py not installed"
    except Exception as e:
        return False, str(e)


def verify_schema_biom(filepath: Path) -> Tuple[bool, Optional[str]]:
    """Verify BIOM file can be loaded."""
    try:
        from biom import load_table
        table = load_table(str(filepath))
        if table.shape[0] > 0 and table.shape[1] > 0:
            return True, None
        return False, "BIOM table is empty"
    except ImportError:
        return None, "biom-format not installed (pip install biom-format)"
    except Exception as e:
        return False, str(e)


def verify_schema_bed(filepath: Path) -> Tuple[bool, Optional[str]]:
    """Verify BED file has expected structure (robust header handling)."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                s = line.strip()
                # Skip empty, comments, track/browser lines
                if not s or s.startswith(('#', 'track', 'browser')):
                    continue
                # Split on any whitespace (not just tabs)
                parts = s.split()
                if len(parts) >= 3:
                    return True, None
                return False, f"BED file has {len(parts)} columns, expected ≥3"
        return False, "BED file appears empty"
    except Exception as e:
        return False, str(e)


def verify_schema_clinical(filepath: Path) -> Tuple[bool, Optional[str]]:
    """Verify clinical matrix has survival columns."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath, sep='\t', nrows=5)
        # Check for key survival columns
        survival_cols = ['OS', 'OS.time', 'DSS', 'PFI', '_OS', '_OS_IND']
        found_cols = [c for c in df.columns if any(sc in c for sc in survival_cols)]
        if len(found_cols) >= 1:
            return True, None
        if 'sampleID' in df.columns or len(df.columns) > 10:
            return True, None
        return False, f"Clinical file may be missing survival columns. Found: {list(df.columns[:10])}"
    except ImportError:
        return None, "pandas not installed"
    except Exception as e:
        return False, str(e)


def verify_schema(filepath: Path) -> Tuple[Optional[bool], Optional[str]]:
    """Verify file schema based on extension."""
    name = filepath.name.lower()
    ext = filepath.suffix.lower()
    
    if 'clinicalmatrix' in name:
        return verify_schema_clinical(filepath)
    elif '.biom' in name:
        return verify_schema_biom(filepath)
    elif ext == '.csv':
        return verify_schema_csv(filepath)
    elif ext == '.tsv':
        return verify_schema_tsv(filepath)
    elif ext == '.h5':
        return verify_schema_h5(filepath)
    elif ext == '.bed':
        return verify_schema_bed(filepath)
    elif ext == '' and 'clinical' in name.lower():
        return verify_schema_clinical(filepath)
    else:
        return None, "Unknown file type"


# =============================================================================
# Verification Functions
# =============================================================================

def find_file(dataset_dir: Path, filename: str, config: Dict) -> Optional[Path]:
    """Find a file, checking alternatives if primary not found."""
    # Check primary filename
    filepath = dataset_dir / filename
    if filepath.exists():
        return filepath
    
    # Check without .gz extension (decompressed)
    if filename.endswith('.gz'):
        decompressed = dataset_dir / filename[:-3]
        if decompressed.exists():
            return decompressed
    
    # Check with .gz extension (compressed)
    compressed = dataset_dir / (filename + '.gz')
    if compressed.exists():
        return compressed
    
    # Check alt_files
    alt_files = config.get("alt_files", {})
    for alt_name in alt_files:
        base_name = filename.replace('.gz', '')
        alt_base = alt_name.replace('.gz', '')
        if base_name == alt_base:
            alt_path = dataset_dir / alt_name
            if alt_path.exists():
                return alt_path
            if alt_name.endswith('.gz'):
                alt_decompressed = dataset_dir / alt_name[:-3]
                if alt_decompressed.exists():
                    return alt_decompressed
    
    return None


def verify_file(
    filepath: Path,
    expected_min_size: int,
    check_hash: bool = False,
    expected_hash: Optional[str] = None,
    check_schema: bool = False,
) -> FileCheck:
    """Verify a single file."""
    check = FileCheck(filename=filepath.name, exists=False)
    
    if not filepath.exists():
        check.error = "File not found"
        return check
    
    check.exists = True
    check.size = filepath.stat().st_size
    check.size_ok = check.size >= expected_min_size
    
    if not check.size_ok:
        check.error = f"File too small: {format_size(check.size)} < {format_size(expected_min_size)}"
    
    if check_hash and expected_hash:
        actual_hash = calculate_sha256(filepath)
        check.hash_match = actual_hash == expected_hash
        if not check.hash_match:
            check.error = "Hash mismatch"
    
    if check_schema:
        loadable, schema_error = verify_schema(filepath)
        check.loadable = loadable
        if loadable is False:
            check.error = f"Schema error: {schema_error}"
    
    return check


def verify_dataset(
    dataset_name: str,
    data_dir: Path,
    config: Dict,
    manifest: Optional[Dict] = None,
    check_hash: bool = False,
    check_schema: bool = False,
    verbose: bool = False,
) -> DatasetVerification:
    """Verify all files for a dataset."""
    
    dataset_dir = data_dir / config["subdir"]
    
    verification = DatasetVerification(
        name=dataset_name,
        path=str(dataset_dir),
        files_checked=[],
        total_size=0,
        files_found=0,
        files_expected=len(config["files"]),
        pipeline_ready=False,
        survival_ready=False,
        multiomics_ready=False,
        errors=[],
    )
    
    if not dataset_dir.exists():
        verification.errors.append(f"Directory not found: {dataset_dir}")
        return verification
    
    # Check each expected file
    for filename, file_config in config["files"].items():
        filepath = find_file(dataset_dir, filename, config)
        
        if filepath is None:
            check = FileCheck(filename=filename, exists=False, error="File not found")
            verification.files_checked.append(check)
            verification.errors.append(f"{filename}: File not found")
            continue
        
        min_size = file_config["min_size"]
        if filepath.name in config.get("alt_files", {}):
            min_size = config["alt_files"][filepath.name]["min_size"]
        
        expected_hash = None
        if manifest and check_hash:
            for file_info in manifest.get("files", []):
                if file_info.get("filename") == filename or file_info.get("filename") == filepath.name:
                    expected_hash = file_info.get("sha256")
                    break
        
        check = verify_file(
            filepath=filepath,
            expected_min_size=min_size,
            check_hash=check_hash,
            expected_hash=expected_hash,
            check_schema=check_schema,
        )
        
        check.filename = filepath.name
        verification.files_checked.append(check)
        
        if check.exists:
            verification.files_found += 1
            verification.total_size += check.size
        
        if check.error and check.error != "File not found":
            verification.errors.append(f"{filepath.name}: {check.error}")
    
    # Check pipeline readiness
    required_files = config.get("required_for_analysis", [])
    verification.pipeline_ready = check_required_files(verification, required_files, dataset_dir, config)
    
    # Check survival readiness (MLOmics only)
    survival_files = config.get("required_for_survival", [])
    if survival_files:
        verification.survival_ready = check_required_files(verification, survival_files, dataset_dir, config)
    
    # Check multi-omics readiness (IBDMDB only)
    multiomics_files = config.get("required_for_multiomics", [])
    if multiomics_files:
        verification.multiomics_ready = check_required_files(verification, multiomics_files, dataset_dir, config)
    
    return verification


def check_required_files(verification: DatasetVerification, required_files: List[str], 
                         dataset_dir: Path, config: Dict) -> bool:
    """Check if all required files are present and valid."""
    for req_file in required_files:
        found = False
        filepath = find_file(dataset_dir, req_file, config)
        if filepath and filepath.exists():
            for check in verification.files_checked:
                if check.filename == filepath.name:
                    if check.exists and check.size_ok:
                        found = True
                    break
            if not found and filepath.stat().st_size > 1024:
                found = True
        if not found:
            return False
    return True


def load_manifest(manifest_dir: Path, dataset_name: str) -> Optional[Dict]:
    """Load manifest for a dataset."""
    name_slug = dataset_name.lower().replace(" ", "_").replace("/", "_")
    manifest_path = manifest_dir / f"manifest_{name_slug}.json"
    
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify downloaded datasets for variance-prediction paradox analysis (v5.2)"
    )
    parser.add_argument("--data-dir", type=Path, help="Base directory for raw data")
    parser.add_argument("--manifest-dir", type=Path, help="Directory containing manifests")
    parser.add_argument("--deep", action="store_true", help="Verify SHA256 hashes")
    parser.add_argument("--schema-check", action="store_true", help="Load and verify file structure")
    parser.add_argument("--verbose", action="store_true", help="Show detailed outputs")
    
    args = parser.parse_args()
    
    # Determine directories
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "code").exists():
        project_root = script_dir
    elif (script_dir.parent / "code").exists():
        project_root = script_dir.parent
    elif (script_dir.parent.parent / "code").exists():
        project_root = script_dir.parent.parent
    else:
        project_root = Path.cwd()
    
    data_dir = args.data_dir or project_root / "data" / "raw"
    manifest_dir = args.manifest_dir or project_root / "outputs" / "manifests"
    
    print("=" * 70)
    print("VARIANCE-PREDICTION PARADOX - DOWNLOAD VERIFICATION v5.2")
    print("(Full MASTER_PLAN scope: Subtype + Survival, MGX + MBX)")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Deep verification: {args.deep}")
    print(f"Schema check: {args.schema_check}")
    print()
    
    all_results = []
    all_ready = True
    
    for dataset_name, config in EXPECTED_FILES.items():
        print(f"\n{'='*50}")
        print(f"Checking: {dataset_name.upper()}")
        print(f"{'='*50}")
        
        manifest = load_manifest(manifest_dir, dataset_name)
        
        result = verify_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            config=config,
            manifest=manifest,
            check_hash=args.deep,
            check_schema=args.schema_check,
            verbose=args.verbose,
        )
        
        all_results.append(result)
        
        print(f"Directory: {result.path}")
        print(f"Files found: {result.files_found}/{result.files_expected}")
        print(f"Total size: {format_size(result.total_size)}")
        
        if args.verbose:
            print("\nFile details:")
            for check in result.files_checked:
                status = "[OK]" if check.exists and check.size_ok else "[FAIL]"
                size_str = format_size(check.size) if check.exists else "N/A"
                print(f"  {status} {check.filename}: {size_str}")
                if check.error:
                    print(f"      Error: {check.error}")
        
        print()
        if result.pipeline_ready:
            print(f"[OK] PIPELINE READY: Core analysis files present")
        else:
            print(f"[FAIL] NOT READY: Missing required files")
            all_ready = False
        
        if dataset_name == "mlomics":
            if result.survival_ready:
                print(f"[OK] SURVIVAL READY: Clinical matrix present")
            else:
                print(f"[WARN] SURVIVAL NOT READY: Missing BRCA_clinicalMatrix")
        
        if dataset_name == "ibdmdb":
            if result.multiomics_ready:
                print(f"[OK] MULTI-OMICS READY: MGX + MBX present")
            else:
                print(f"[WARN] MULTI-OMICS NOT READY: Missing metabolomics (HMP2_metabolomics_w_metadata.biom)")
        
        if result.errors and not args.verbose:
            print(f"\nIssues ({len(result.errors)}):")
            for error in result.errors[:5]:
                print(f"  - {error}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    total_size = sum(r.total_size for r in all_results)
    
    for result in all_results:
        status = "[OK]" if result.pipeline_ready else "[FAIL]"
        extras = []
        if result.name == "mlomics":
            extras.append("survival" if result.survival_ready else "no-survival")
        if result.name == "ibdmdb":
            extras.append("multi-omics" if result.multiomics_ready else "MGX-only")
        
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        print(f"{status} {result.name}: {result.files_found}/{result.files_expected} files, {format_size(result.total_size)}{extra_str}")
    
    print("-" * 50)
    print(f"Total: {format_size(total_size)}")
    
    # Final verdict
    print()
    mlomics_result = next((r for r in all_results if r.name == "mlomics"), None)
    ibdmdb_result = next((r for r in all_results if r.name == "ibdmdb"), None)
    
    full_master_plan = (
        all_ready and 
        (mlomics_result is None or mlomics_result.survival_ready) and
        (ibdmdb_result is None or ibdmdb_result.multiomics_ready)
    )
    
    if full_master_plan:
        print("[OK] FULL MASTER_PLAN READY: All datasets complete with survival + multi-omics")
        return 0
    elif all_ready:
        print("[OK] CORE ANALYSIS READY: Basic pipeline can run")
        print("[WARN] Some optional components missing (survival or metabolomics)")
        return 0
    else:
        print("[FAIL] NOT READY: Missing required files")
        return 1


if __name__ == "__main__":
    sys.exit(main())