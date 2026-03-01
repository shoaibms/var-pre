#!/usr/bin/env python3
"""\
Variance-Prediction Paradox - Multi-omics Data Download Script v11.2

Key changes vs v11.1:
- IBDMDB/HMP2: fixed MBX metabolomics URL (the MBX products are served from /products/HMP2/MBX/ rather than
  /products/HMP2/Metabolites/...).
- IBDMDB/HMP2: relaxed the minimum-size threshold for the proteomics ECS table (compressed files can be small)
  and enabled post-decompression text validation to catch portal error payloads.
- Arabidopsis: added Expression Atlas RNA matrix (TPM) from EBI FTP to complete the transcriptome view.
- TCGA-GBM: tighter minimum-size threshold + basic HTML/error-page detection for miRNA downloads.
- Added --force to overwrite existing files.

Usage:
  python download_all_data_v11_2.py --datasets mlomics,ibdmdb,ccle,nci60,arabidopsis,tcga_gbm --data-dir data/raw
  python download_all_data_v11_2.py --list-datasets

Notes:
- Some sources (e.g., IBDMDB Globus-backed URLs, DepMap signed URLs) can change over time. This script
  is designed to be resilient (retries, alt URLs, validation) but you should still keep the generated
  manifest JSON for provenance.
"""

import argparse
import csv
import gzip
import io
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class DatasetFile:
    """Configuration for a single downloadable file."""
    filename: str
    url: str
    alt_url: Optional[str] = None
    expected_min_size: int = 1024
    save_as: Optional[str] = None
    decompress: bool = False
    unzip: bool = False
    required: bool = True
    depmap_file: Optional[str] = None  # For dynamic DepMap URL resolution
    validate_text: bool = False        # For basic HTML/error-page detection


@dataclass
class DatasetConfig:
    """Configuration for a complete dataset."""
    name: str
    description: str
    destination_subdir: str
    domain: str
    modalities: List[str]
    files: List[DatasetFile] = field(default_factory=list)
    readme_content: str = ""
    requires_auth: bool = False
    auth_instructions: str = ""
    use_depmap_api: bool = False


# -----------------------------------------------------------------------------
# Dataset configurations
# -----------------------------------------------------------------------------

# MLOmics BRCA
MLOMICS_CONFIG = DatasetConfig(
    name="MLOmics BRCA",
    description="MLOmics Breast Cancer dataset",
    destination_subdir="mlomics",
    domain="Human Cancer (Breast)",
    modalities=["mRNA", "miRNA", "methylation", "CNV"],
    files=[
        DatasetFile(
            "BRCA_mRNA_aligned.csv",
            "https://huggingface.co/datasets/AIBIC/MLOmics/resolve/main/Main_Dataset/Classification_datasets/GS-BRCA/Aligned/BRCA_mRNA_aligned.csv?download=true",
            expected_min_size=50 * 1024 * 1024,
        ),
        DatasetFile(
            "BRCA_miRNA_aligned.csv",
            "https://huggingface.co/datasets/AIBIC/MLOmics/resolve/main/Main_Dataset/Classification_datasets/GS-BRCA/Aligned/BRCA_miRNA_aligned.csv?download=true",
            expected_min_size=500 * 1024,
        ),
        DatasetFile(
            "BRCA_Methy_aligned.csv",
            "https://huggingface.co/datasets/AIBIC/MLOmics/resolve/main/Main_Dataset/Classification_datasets/GS-BRCA/Aligned/BRCA_Methy_aligned.csv?download=true",
            expected_min_size=10 * 1024 * 1024,
        ),
        DatasetFile(
            "BRCA_CNV_aligned.csv",
            "https://huggingface.co/datasets/AIBIC/MLOmics/resolve/main/Main_Dataset/Classification_datasets/GS-BRCA/Aligned/BRCA_CNV_aligned.csv?download=true",
            expected_min_size=5 * 1024 * 1024,
        ),
        DatasetFile(
            "BRCA_label_num.csv",
            "https://huggingface.co/datasets/AIBIC/MLOmics/resolve/main/Main_Dataset/Classification_datasets/GS-BRCA/Aligned/BRCA_label_num.csv?download=true",
            expected_min_size=1024,
        ),
        DatasetFile(
            "BRCA_clinicalMatrix",
            "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
            expected_min_size=500 * 1024,
            validate_text=True,
        ),
    ],
    readme_content=(
        "# MLOmics BRCA\n"
        "Domain: Human Cancer (Breast)\n"
        "Modalities: mRNA, miRNA, Methylation, CNV\n"
        "Samples: ~671\n"
        "Task: PAM50 subtype\n"
    ),
)


# IBDMDB/HMP2 (4-modality)
# Sources are Globus-backed links surfaced by ibdmdb.org.
IBDMDB_CONFIG = DatasetConfig(
    name="IBDMDB",
    description="HMP2 IBD Multi-omics",
    destination_subdir="ibdmdb",
    domain="Gut Microbiome",
    modalities=["MGX", "MTX", "MBX", "MPX", "metadata"],
    files=[
        DatasetFile(
            "hmp2_metadata.csv",
            "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/metadata/hmp2_metadata_2018-08-20.csv",
            expected_min_size=100 * 1024,
            validate_text=True,
        ),
        # MGX (metagenomics)
        DatasetFile(
            "taxonomic_profiles_3.tsv.gz",
            "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MGX/2018-05-04/taxonomic_profiles_3.tsv.gz",
            expected_min_size=100 * 1024,
            decompress=True,
        ),
        DatasetFile(
            "pathabundances_3.tsv.gz",
            "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MGX/2018-05-04/pathabundances_3.tsv.gz",
            expected_min_size=200 * 1024,
            decompress=True,
            required=False,  # optional: not all analyses need functional MGX
        ),
        # MTX (metatranscriptomics) – merged tables
        DatasetFile(
            "mtx_genefamilies.tsv.gz",
            "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MTX/2017-12-14/genefamilies.tsv.gz",
            expected_min_size=200 * 1024,
            decompress=True,
            save_as="genefamilies.tsv.gz",
        ),
        DatasetFile(
            "mtx_pathabundance_relab.tsv.gz",
            "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MTX/2017-12-14/pathabundance_relab.tsv.gz",
            expected_min_size=200 * 1024,
            decompress=True,
            save_as="pathabundance_relab.tsv.gz",
            required=False,
        ),
        # MBX (metabolomics)
        DatasetFile(
            "HMP2_metabolomics_w_metadata.biom.gz",
            # MBX run products are served directly from /products/HMP2/MBX/
            # (see ibdmdb.org run-products page for MBX).
            "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MBX/HMP2_metabolomics_w_metadata.biom.gz",
            alt_url="https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MBX/2017-03-01/HMP2_metabolomics_w_metadata.biom.gz",
            expected_min_size=200 * 1024,
            decompress=True,
            required=True,
        ),
        # MPX (proteomics)
        DatasetFile(
            "HMP2_proteomics_ecs.tsv.gz",
            "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MPX/2017-03-20/HMP2_proteomics_ecs.tsv.gz",
            # Note: the compressed ECS table can legitimately be small; validate AFTER decompression.
            expected_min_size=50 * 1024,
            decompress=True,
            required=True,
            validate_text=True,
        ),
    ],
    readme_content=(
        "# IBDMDB/HMP2\n"
        "Domain: Gut Microbiome\n"
        "Modalities: MGX (taxa + optional functional), MTX (functional), MBX (metabolomics), MPX (proteomics), metadata\n"
        "Design: longitudinal, IBD vs nonIBD\n"
        "Source: https://ibdmdb.org/\n"
        "Citation: Lloyd-Price et al. Nature 2019\n"
    ),
)


# CCLE/DepMap (dynamic signed URLs)
CCLE_CONFIG = DatasetConfig(
    name="CCLE/DepMap",
    description="Cancer Cell Line Encyclopedia (DepMap 24Q2)",
    destination_subdir="ccle",
    domain="Cell Lines (22 tissues)",
    modalities=["mRNA", "proteomics", "copy_number", "metadata"],
    use_depmap_api=True,
    files=[
        DatasetFile(
            "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
            "",
            expected_min_size=50 * 1024 * 1024,
            depmap_file="OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        ),
        DatasetFile(
            "protein_quant_current_normalized.csv.gz",
            "https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz",
            expected_min_size=50 * 1024 * 1024,
            decompress=True,
        ),
        DatasetFile(
            "OmicsCNGene.csv",
            "",
            expected_min_size=50 * 1024 * 1024,
            depmap_file="OmicsCNGene.csv",
        ),
        DatasetFile(
            "Model.csv",
            "",
            expected_min_size=200 * 1024,
            depmap_file="Model.csv",
        ),
    ],
    readme_content=(
        "# CCLE/DepMap 24Q2\n"
        "Domain: Cell Lines (22 tissue types)\n"
        "Modalities: mRNA (TPM), Proteomics (MS), Copy Number, metadata\n"
        "Samples: ~1900 cell lines (varies by modality)\n"
        "Source: https://depmap.org/portal/\n"
        "Citation: Ghandi et al. Nature 2019; Nusinow et al. Cell 2020\n"
    ),
)


# NCI-60
NCI60_CONFIG = DatasetConfig(
    name="NCI-60",
    description="NCI-60 Cell Lines",
    destination_subdir="nci60",
    domain="Cell Lines (9 tissues)",
    modalities=["mRNA", "miRNA", "proteomics", "drug_response"],
    files=[
        DatasetFile(
            "nci60_RNA__RNA_seq_composite_expression.zip",
            "https://discover.nci.nih.gov/cellminer/download/processeddataset/nci60_RNA__RNA_seq_composite_expression.zip",
            expected_min_size=100 * 1024,
            unzip=True,
        ),
        DatasetFile(
            "nci60_RNA__microRNA_OSU_V3_chip_log2.zip",
            "https://discover.nci.nih.gov/cellminer/download/processeddataset/nci60_RNA__microRNA_OSU_V3_chip_log2.zip",
            expected_min_size=50 * 1024,
            unzip=True,
        ),
        DatasetFile(
            "nci60_Protein__SWATH_(Mass_spectrometry)_Protein.zip",
            "https://discover.nci.nih.gov/cellminer/download/processeddataset/nci60_Protein__SWATH_(Mass_spectrometry)_Protein.zip",
            expected_min_size=100 * 1024,
            unzip=True,
        ),
        DatasetFile(
            "DTP_NCI60_ZSCORE.zip",
            "https://discover.nci.nih.gov/cellminer/download/processeddataset/DTP_NCI60_ZSCORE.zip",
            expected_min_size=1 * 1024 * 1024,
            unzip=True,
        ),
        DatasetFile(
            "NCI60_Fingerprinting.zip",
            "https://discover.nci.nih.gov/cellminer/download/rawdataset/NCI60_Fingerprinting.zip",
            expected_min_size=5 * 1024,
            unzip=True,
        ),
    ],
    readme_content=(
        "# NCI-60\n"
        "Domain: Cell Lines (9 tissue types)\n"
        "Modalities: mRNA (RNA-seq), miRNA, Proteomics (SWATH-MS), Drug response\n"
        "Samples: 60 cell lines\n"
        "Source: CellMiner (https://discover.nci.nih.gov/cellminer/)\n"
    ),
)


# Arabidopsis: add Expression Atlas RNA matrix
ARABIDOPSIS_CONFIG = DatasetConfig(
    name="Arabidopsis",
    description="Arabidopsis 30-tissue atlas (Nature 2020)",
    destination_subdir="arabidopsis",
    domain="Plant Development",
    modalities=["transcriptome", "proteome", "phosphoproteome"],
    files=[
        DatasetFile(
            "E-MTAB-7978-tpms.tsv",
            "https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/E-MTAB-7978/E-MTAB-7978-tpms.tsv",
            expected_min_size=5 * 1024 * 1024,
            validate_text=True,
        ),
        DatasetFile(
            "E-MTAB-7978.condensed-sdrf.tsv",
            "https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/E-MTAB-7978/E-MTAB-7978.condensed-sdrf.tsv",
            expected_min_size=48 * 1024,
            required=False,
            validate_text=True,
        ),
        # SDRF (ArrayExpress) – useful for mapping
        DatasetFile(
            "E-MTAB-7978.sdrf.txt",
            "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7978/E-MTAB-7978.sdrf.txt",
            alt_url="https://ftp.ebi.ac.uk/pub/databases/arrayexpress/data/experiment/MTAB/E-MTAB-7978/E-MTAB-7978.sdrf.txt",
            expected_min_size=10 * 1024,
            required=False,
            validate_text=True,
        ),
        # Supplementary tables from Mergner et al. (contain proteome & phosphoproteome matrices)
        DatasetFile(
            "41586_2020_2094_MOESM3_ESM.xlsx",
            "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2094-2/MediaObjects/41586_2020_2094_MOESM3_ESM.xlsx",
            expected_min_size=5 * 1024,
        ),
        DatasetFile(
            "41586_2020_2094_MOESM4_ESM.xlsx",
            "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2094-2/MediaObjects/41586_2020_2094_MOESM4_ESM.xlsx",
            expected_min_size=50 * 1024,
        ),
        DatasetFile(
            "41586_2020_2094_MOESM5_ESM.xlsx",
            "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2094-2/MediaObjects/41586_2020_2094_MOESM5_ESM.xlsx",
            expected_min_size=50 * 1024,
        ),
    ],
    readme_content=(
        "# Arabidopsis Multi-omics Atlas\n"
        "Domain: Plant Development\n"
        "Modalities: Transcriptome (TPM), Proteome, Phosphoproteome\n"
        "Samples: 30 tissues/organs\n"
        "Source: Mergner et al. Nature 2020 (doi:10.1038/s41586-020-2094-2)\n"
    ),
)


# TCGA-GBM
TCGA_GBM_CONFIG = DatasetConfig(
    name="TCGA-GBM",
    description="TCGA Glioblastoma",
    destination_subdir="tcga_gbm",
    domain="Human Cancer (Brain)",
    modalities=["mRNA", "miRNA", "methylation", "CNV", "clinical"],
    files=[
        DatasetFile(
            "HiSeqV2.gz",
            "https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/HiSeqV2.gz",
            expected_min_size=5 * 1024 * 1024,
            decompress=True,
            validate_text=True,
        ),
        DatasetFile(
            "miRNA_HiSeq_gene.gz",
            "https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/miRNA_HiSeq_gene.gz",
            expected_min_size=200 * 1024,  # tighter than v10 to avoid accepting tiny HTML/error payloads
            decompress=True,
            validate_text=True,
        ),
        DatasetFile(
            "HumanMethylation450.gz",
            "https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/HumanMethylation450.gz",
            expected_min_size=50 * 1024 * 1024,
            decompress=True,
            validate_text=True,
        ),
        DatasetFile(
            "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz",
            "https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz",
            expected_min_size=100 * 1024,
            decompress=True,
            validate_text=True,
        ),
        DatasetFile(
            "GBM_clinicalMatrix",
            "https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/GBM_clinicalMatrix",
            expected_min_size=100 * 1024,
            validate_text=True,
        ),
    ],
    readme_content=(
        "# TCGA-GBM\n"
        "Domain: Human Cancer (Brain/Glioblastoma)\n"
        "Modalities: mRNA (RNA-seq), miRNA, DNA Methylation (450k), Copy Number (GISTIC2), clinical\n"
        "Source: UCSC Xena (https://xenabrowser.net/)\n"
    ),
)


ALL_DATASETS: Dict[str, DatasetConfig] = {
    "mlomics": MLOMICS_CONFIG,
    "ibdmdb": IBDMDB_CONFIG,
    "ccle": CCLE_CONFIG,
    "nci60": NCI60_CONFIG,
    "arabidopsis": ARABIDOPSIS_CONFIG,
    "tcga_gbm": TCGA_GBM_CONFIG,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def setup_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def get_depmap_urls(session: requests.Session, release: str = "DepMap Public 24Q2") -> Dict[str, str]:
    """Fetch DepMap API and return mapping of filename -> signed download URL."""
    print("  Fetching DepMap file catalog...")
    api_url = "https://depmap.org/portal/api/download/files"

    try:
        resp = session.get(api_url, timeout=60)
        resp.raise_for_status()
        url_map: Dict[str, str] = {}
        reader = csv.DictReader(io.StringIO(resp.text))
        for row in reader:
            if row.get("release") == release and row.get("filename") and row.get("url"):
                url_map[row["filename"]] = row["url"]
        print(f"    Found {len(url_map)} files for {release}")
        return url_map
    except Exception as e:
        print(f"    Warning: failed to fetch DepMap catalog: {e}")
        return {}


def looks_like_html_or_error(path: Path, max_bytes: int = 4096) -> bool:
    """Heuristic: detect common HTML/error payloads that masquerade as data files."""
    try:
        with open(path, "rb") as f:
            head = f.read(max_bytes)
        # Allow gzip binary, but we only run this on plain files after download
        text = head.decode("utf-8", errors="ignore").lower()
        if "<html" in text or "<!doctype html" in text:
            return True
        if "access denied" in text or "error" in text and "\t" not in text and "," not in text:
            return True
        # some portals return JSON errors
        if text.strip().startswith("{") and "error" in text and "message" in text:
            return True
        return False
    except Exception:
        return False


def download_file(
    session: requests.Session, url: str, dest_path: Path, expected_min_size: int
) -> Tuple[bool, str]:
    """Download a file with progress bar and size validation."""
    try:
        resp = session.get(url, stream=True, timeout=180, allow_redirects=True)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name, ncols=110) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        file_size = dest_path.stat().st_size
        if file_size < expected_min_size:
            return False, f"Too small: {file_size} < {expected_min_size}"

        return True, str(file_size)

    except requests.exceptions.RequestException as e:
        return False, str(e)


def decompress_gzip(src_path: Path, dest_path: Path) -> bool:
    try:
        with gzip.open(src_path, "rb") as f_in:
            with open(dest_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        src_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"    Decompress error: {e}")
        return False


def extract_zip(src_path: Path, dest_dir: Path) -> bool:
    try:
        with zipfile.ZipFile(src_path, "r") as z:
            z.extractall(dest_dir)
        src_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"    Extract error: {e}")
        return False


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def download_dataset(
    session: requests.Session,
    config: DatasetConfig,
    data_dir: Path,
    depmap_urls: Optional[Dict[str, str]] = None,
    force: bool = False,
) -> dict:
    dest_dir = data_dir / config.destination_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "name": config.name,
        "domain": config.domain,
        "modalities": config.modalities,
        "downloaded": [],
        "skipped": [],
        "failed": [],
        "total_size": 0,
    }

    if config.requires_auth:
        print(config.auth_instructions)
        (dest_dir / "README_AUTH.txt").write_text(config.auth_instructions)
        return results

    for fcfg in config.files:
        filename = fcfg.save_as or fcfg.filename
        dest_path = dest_dir / filename
        decompressed_path = None
        if fcfg.decompress and filename.endswith('.gz'):
            decompressed_path = dest_dir / filename[:-3]

        print(f"  Downloading: {filename}")

        # Fast skip logic:
        # - Prefer the original file if it exists.
        # - If we previously decompressed and removed the .gz, skip if the decompressed file exists.
        if not force and dest_path.exists() and dest_path.stat().st_size >= fcfg.expected_min_size:
            # optional additional validation for text-like files
            if fcfg.validate_text and looks_like_html_or_error(dest_path):
                print("    Warning: existing file looks like HTML/error payload; re-downloading...")
            else:
                print("    Already exists")
                # If this is a zip that needs extraction, extract even when skipping download
                if fcfg.unzip and filename.endswith(".zip"):
                    if extract_zip(dest_path, dest_dir):
                        print(f"    Extracted to {config.destination_subdir}/ (existing zip)")
                results["skipped"].append(filename)
                results["total_size"] += dest_path.stat().st_size
                continue

        if (
            not force
            and (not dest_path.exists())
            and decompressed_path is not None
            and decompressed_path.exists()
            and decompressed_path.stat().st_size > 0
        ):
            # optional additional validation for text-like files
            if fcfg.validate_text and looks_like_html_or_error(decompressed_path):
                print("    Warning: existing decompressed file looks like HTML/error payload; re-downloading...")
            else:
                print(f"    Already exists (decompressed: {decompressed_path.name})")
                results["skipped"].append(decompressed_path.name)
                results["total_size"] += decompressed_path.stat().st_size
                continue

        # Resolve URL via DepMap catalog
        url = fcfg.url
        if fcfg.depmap_file and depmap_urls is not None:
            if fcfg.depmap_file in depmap_urls:
                url = depmap_urls[fcfg.depmap_file]
                print("    Using DepMap API URL")
            else:
                msg = "File not found in DepMap catalog"
                print(f"    Warning: {msg}")
                if fcfg.required:
                    results["failed"].append(filename)
                else:
                    results["skipped"].append(filename)
                continue

        if not url:
            msg = "No URL available"
            print(f"    Error: {msg}")
            if fcfg.required:
                results["failed"].append(filename)
            else:
                results["skipped"].append(filename)
            continue

        ok, msg = download_file(session, url, dest_path, fcfg.expected_min_size)
        if not ok and fcfg.alt_url:
            print("    Trying alternate URL...")
            ok, msg = download_file(session, fcfg.alt_url, dest_path, fcfg.expected_min_size)

        if not ok:
            req = "" if fcfg.required else " (optional)"
            print(f"    Error: {msg}{req}")
            if fcfg.required:
                results["failed"].append(filename)
            else:
                results["skipped"].append(filename)
            # remove partial
            try:
                if dest_path.exists():
                    dest_path.unlink()
            except Exception:
                pass
            continue

        # Basic content validation (after download, before decompress)
        if fcfg.validate_text and looks_like_html_or_error(dest_path):
            req = "" if fcfg.required else " (optional)"
            print(f"    Error: downloaded payload looks like HTML/error{req}")
            if fcfg.required:
                results["failed"].append(filename)
            else:
                results["skipped"].append(filename)
            try:
                dest_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # Decompress
        final_path = dest_path
        if fcfg.decompress and filename.endswith(".gz"):
            out_name = filename[:-3]
            final_path = dest_dir / out_name
            if decompress_gzip(dest_path, final_path):
                print(f"    Decompressed: {out_name}")

            # Validate decompressed content as text if requested
            if fcfg.validate_text and looks_like_html_or_error(final_path):
                req = "" if fcfg.required else " (optional)"
                print(f"    Error: decompressed payload looks like HTML/error{req}")
                if fcfg.required:
                    results["failed"].append(filename)
                else:
                    results["skipped"].append(filename)
                try:
                    final_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

        # Unzip
        if fcfg.unzip and filename.endswith(".zip"):
            if extract_zip(final_path, dest_dir):
                print(f"    Extracted to {config.destination_subdir}/")

        print(f"    Downloaded {int(msg):,} bytes")
        results["downloaded"].append(filename)
        results["total_size"] += int(msg)

    if config.readme_content:
        (dest_dir / "README.md").write_text(config.readme_content)

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download multi-omics datasets for variance-prediction paradox analysis"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets to download",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Base directory for downloaded data",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--depmap-release",
        type=str,
        default="DepMap Public 24Q2",
        help="DepMap release to use (default: DepMap Public 24Q2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file already exists",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("\nAvailable datasets:")
        print("=" * 80)
        for key, cfg in ALL_DATASETS.items():
            print(f"  {key:12} | {cfg.name:20} | {cfg.domain}")
        return

    if not args.datasets:
        parser.print_help()
        return

    data_dir = Path(args.data_dir).resolve()
    selected = [d.strip() for d in args.datasets.split(",") if d.strip()]

    invalid = [d for d in selected if d not in ALL_DATASETS]
    if invalid:
        print(f"Unknown datasets: {invalid}")
        print(f"Available: {list(ALL_DATASETS.keys())}")
        return

    print("\n" + "=" * 80)
    print("VARIANCE-PREDICTION PARADOX - DATA DOWNLOAD v11.2")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Selected: {selected}")
    print()

    session = setup_session()

    depmap_urls: Dict[str, str] = {}
    if any(ALL_DATASETS[d].use_depmap_api for d in selected):
        depmap_urls = get_depmap_urls(session, args.depmap_release)

    all_results = []

    for key in selected:
        cfg = ALL_DATASETS[key]
        print("\n" + "=" * 70)
        print(f"{cfg.name} | {cfg.domain}")
        print("=" * 70)
        res = download_dataset(
            session,
            cfg,
            data_dir,
            depmap_urls=depmap_urls,
            force=args.force,
        )
        all_results.append(res)

        print(
            f"\n  {len(res['downloaded'])} downloaded, {len(res['skipped'])} skipped, {len(res['failed'])} failed"
        )
        if res["total_size"] > 0:
            print(f"  Size: {format_size(res['total_size'])}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_size = 0
    for res in all_results:
        status = "OK" if not res["failed"] else "WARNING"
        size_str = format_size(res["total_size"]) if res["total_size"] else "0.0 B"
        print(f"{status} {res['name']:15} | {res['domain']:25} | {size_str}")
        total_size += res["total_size"]

    print(f"\nTotal: {format_size(total_size)}")

    # Save manifest
    manifest_dir = data_dir.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "version": "11.0",
        "data_dir": str(data_dir),
        "depmap_release": args.depmap_release,
        "datasets": all_results,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
