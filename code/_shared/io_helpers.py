#!/usr/bin/env python3
"""
io_helpers.py

Shared I/O utilities for Phase 8 (Biology) and Phase 9 (Simulation).

Provides:
  - VIEW_REGISTRY: canonical dataset/view definitions
  - Column name candidates with fallback logic (pick_col)
  - Structured file discovery (VPRecord)
  - Phase manifest writer

This module eliminates schema drift across scripts by centralizing:
  - Which files to read
  - Which columns to expect
  - How to parse dataset/view from paths

Usage (from any script in 08_biology/ or 09_simulation/):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _shared.io_helpers import discover_vp_files, pick_col, IMPORTANCE_COL_CANDIDATES

Part of the variance-prediction paradox pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd


# =============================================================================
# VIEW REGISTRY - Single source of truth for datasets and views
# =============================================================================

VIEW_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlomics": {
        "core_views": ["mRNA", "miRNA", "methylation", "CNV"],
        "sensitivity_views": [],
        "analysis_role": "primary",
        "description": "Multi-omics breast cancer (BRCA subtypes)",
    },
    "ibdmdb": {
        "core_views": ["MGX", "MGX_func", "MPX", "MBX"],
        "sensitivity_views": ["MGX_CLR"],
        "analysis_role": "primary",
        "description": "Gut microbiome IBD (taxonomic + functional)",
    },
    "ccle": {
        "core_views": ["mRNA", "CNV", "proteomics"],
        "sensitivity_views": [],
        "analysis_role": "primary",
        "description": "Cancer cell lines (DepMap)",
    },
    "tcga_gbm": {
        "core_views": ["mRNA", "methylation", "CNV"],
        "sensitivity_views": ["methylation_Mval"],
        "analysis_role": "sensitivity",
        "description": "TCGA glioblastoma (small N, sensitivity only)",
    },
}

# Representative views for quick validation runs (one per regime)
HERO_VIEWS: List[Tuple[str, str]] = [
    ("mlomics", "methylation"),  # ANTI_ALIGNED
    ("ibdmdb", "MGX"),           # COUPLED
    ("ccle", "mRNA"),            # MIXED
]

# Default K% thresholds for overlap analysis
DEFAULT_K_PCTS: List[float] = [1.0, 5.0, 10.0, 20.0]


def resolve_views(dataset: str, which: str = "core") -> List[str]:
    """
    Resolve view list for a dataset.
    
    Args:
        dataset: Dataset name (must be in VIEW_REGISTRY)
        which: One of "core", "all", "sensitivity"
    
    Returns:
        List of view names
    """
    if dataset not in VIEW_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. Options: {sorted(VIEW_REGISTRY.keys())}")
    
    info = VIEW_REGISTRY[dataset]
    core = info["core_views"]
    sens = info.get("sensitivity_views", [])
    
    if which == "core":
        return list(core)
    elif which == "all":
        return list(core) + list(sens)
    elif which == "sensitivity":
        return list(sens)
    else:
        raise ValueError(f"which must be 'core', 'all', or 'sensitivity', got '{which}'")


def get_all_dataset_view_pairs(which: str = "core") -> List[Tuple[str, str]]:
    """Return all (dataset, view) pairs across registry."""
    pairs = []
    for dataset in VIEW_REGISTRY:
        for view in resolve_views(dataset, which):
            pairs.append((dataset, view))
    return pairs


def is_hero_view(dataset: str, view: str) -> bool:
    """Check if (dataset, view) is a hero view."""
    return (dataset, view) in HERO_VIEWS


# =============================================================================
# COLUMN NAME CANDIDATES - Robust column selection with fallbacks
# =============================================================================

# Importance/prediction columns (from various model outputs)
IMPORTANCE_COL_CANDIDATES: List[str] = [
    "importance",
    "importance_mean", 
    "mean_abs_shap",
    "mean_abs",
    "shap_importance",
    "feature_importance",
    "p_score",
    "p_xgb_bal_score",   # VP summary
    "p_rf_score",        # VP summary
    "p_xgb_score",       # (optional legacy)
    "gain",
    "weight",
]

# Variance score columns
VARIANCE_COL_CANDIDATES: List[str] = [
    "variance",
    "var",
    "score",
    "v_marginal_score",  # VP summary
    "v_score",
    "variance_score",
    "var_score",
]

# Rank columns (variance)
V_RANK_COL_CANDIDATES: List[str] = [
    "v_rank",
    "variance_rank",
    "var_rank",
]

# Rank columns (importance)
P_RANK_COL_CANDIDATES: List[str] = [
    "p_rank",
    "importance_rank",
    "pred_rank",
]

# Feature name columns
FEATURE_COL_CANDIDATES: List[str] = [
    "feature",
    "feature_name",
    "gene",
    "probe",
    "variable",
    "name",
]


def pick_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Pick the first matching column from candidates.
    
    Args:
        df: DataFrame to search
        candidates: Ordered list of column name candidates
        required: If True, raise error when no match found
        default: Return this if no match and not required
    
    Returns:
        Matched column name, or default/None
    
    Raises:
        KeyError: If required=True and no candidate found
    """
    cols = set(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    
    if required:
        raise KeyError(
            f"None of the candidate columns found in DataFrame. "
            f"Candidates: {list(candidates)}, Available: {sorted(df.columns)[:20]}..."
        )
    return default


def pick_importance_col(df: pd.DataFrame, model: Optional[str] = None) -> str:
    pref: List[str] = []
    if model:
        pref += [f"p_{model}_score", f"{model}_score"]
    return pick_col(df, pref + IMPORTANCE_COL_CANDIDATES, required=True)


def pick_variance_col(df: pd.DataFrame) -> str:
    pref = ["v_marginal_score"]
    return pick_col(df, pref + VARIANCE_COL_CANDIDATES, required=True)


def pick_cols_multi(
    df: pd.DataFrame,
    candidate_groups: Dict[str, Sequence[str]],
    required_keys: Optional[Set[str]] = None,
) -> Dict[str, Optional[str]]:
    """
    Pick columns for multiple candidate groups at once.
    
    Args:
        df: DataFrame to search
        candidate_groups: Dict mapping logical name -> candidate list
        required_keys: Which keys must be found (raises if missing)
    
    Returns:
        Dict mapping logical name -> matched column (or None)
    """
    required_keys = required_keys or set()
    result = {}
    for key, candidates in candidate_groups.items():
        is_req = key in required_keys
        result[key] = pick_col(df, candidates, required=is_req)
    return result


# =============================================================================
# STRUCTURED FILE DISCOVERY - VPRecord dataclass
# =============================================================================

@dataclass
class VPRecord:
    """
    Structured record for a variance-prediction file pair.
    
    Contains all metadata needed by downstream scripts without re-parsing paths.
    """
    dataset: str
    view: str
    model: str
    
    # Paths
    variance_path: Optional[Path] = None
    importance_path: Optional[Path] = None
    joined_path: Optional[Path] = None
    
    # Metadata (populated on load)
    n_features: Optional[int] = None
    has_variance: bool = False
    has_importance: bool = False
    has_joined: bool = False
    
    # Analysis role
    is_hero: bool = False
    analysis_role: str = "primary"
    
    def __post_init__(self):
        self.is_hero = is_hero_view(self.dataset, self.view)
        if self.dataset in VIEW_REGISTRY:
            self.analysis_role = VIEW_REGISTRY[self.dataset].get("analysis_role", "primary")
    
    @property
    def key(self) -> str:
        """Unique key for this record."""
        return f"{self.dataset}__{self.view}__{self.model}"
    
    @property
    def short_key(self) -> str:
        """Short key without model."""
        return f"{self.dataset}/{self.view}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        # Convert Path objects to strings
        for k in ["variance_path", "importance_path", "joined_path"]:
            if d[k] is not None:
                d[k] = str(d[k])
        return d


def discover_vp_files(
    outputs_dir: Union[str, Path],
    model: str = "xgb_bal",
    datasets: Optional[Sequence[str]] = None,
    views_which: str = "core",
    require_joined: bool = True,
) -> List[VPRecord]:
    """
    Discover variance-prediction file pairs and return structured records.
    
    Args:
        outputs_dir: Root outputs directory
        model: Model name (xgb_bal, rf, xgb)
        datasets: Subset of datasets to scan (None = all)
        views_which: Which views to include ("core", "all", "sensitivity")
        require_joined: If True, only return records with joined VP files
    
    Returns:
        List of VPRecord with populated paths and metadata
    
    File locations searched:
        - Variance: outputs/02_unsupervised/variance_scores/variance_scores__{dataset}__{view}.csv.gz
        - Importance: outputs/04_importance/per_model/importance__{dataset}__{view}__{model}.csv.gz
        - Joined: outputs/04_importance/joined_vp/vp_joined__{dataset}__{view}.csv.gz
    """
    outputs_dir = Path(outputs_dir)
    
    # Determine datasets to scan
    if datasets is None:
        datasets = list(VIEW_REGISTRY.keys())
    
    records: List[VPRecord] = []
    
    for dataset in datasets:
        if dataset not in VIEW_REGISTRY:
            continue
        
        for view in resolve_views(dataset, views_which):
            rec = VPRecord(dataset=dataset, view=view, model=model)
            
            # Check variance file
            var_path = outputs_dir / "02_unsupervised" / "variance_scores" / f"variance_scores__{dataset}__{view}.csv.gz"
            if var_path.exists():
                rec.variance_path = var_path
                rec.has_variance = True
            
            # Check importance file
            imp_path = outputs_dir / "04_importance" / "per_model" / f"importance__{dataset}__{view}__{model}.csv.gz"
            if imp_path.exists():
                rec.importance_path = imp_path
                rec.has_importance = True
            
            # Check joined file
            joined_path = outputs_dir / "04_importance" / "joined_vp" / f"vp_joined__{dataset}__{view}.csv.gz"
            if joined_path.exists():
                rec.joined_path = joined_path
                rec.has_joined = True
                
                # Get feature count from joined file with lightweight line count
                try:
                    import gzip
                    with gzip.open(joined_path, 'rt') as gz:
                        rec.n_features = sum(1 for _ in gz) - 1  # -1 for header
                except Exception:
                    pass
            
            # Apply filter
            if require_joined and not rec.has_joined:
                continue
            
            records.append(rec)
    
    return records


def discover_hero_views(
    outputs_dir: Union[str, Path],
    model: str = "xgb_bal",
    require_joined: bool = True,
) -> List[VPRecord]:
    """
    Discover only hero view files.
    
    Returns:
        List of VPRecord for hero views only
    """
    all_records = discover_vp_files(
        outputs_dir=outputs_dir,
        model=model,
        datasets=None,
        views_which="all",
        require_joined=require_joined,
    )
    return [r for r in all_records if r.is_hero]


def load_vp_joined(
    record: VPRecord,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load joined VP file for a record with column validation.
    
    Args:
        record: VPRecord with joined_path set
        validate: If True, verify expected columns exist
    
    Returns:
        DataFrame with standardized column access
    """
    if record.joined_path is None or not record.has_joined:
        raise FileNotFoundError(f"No joined file for {record.short_key}")
    
    df = pd.read_csv(record.joined_path)
    
    if validate:
        # Verify we can find required columns
        _ = pick_col(df, FEATURE_COL_CANDIDATES, required=True)
        _ = pick_variance_col(df)
        _ = pick_importance_col(df, model=record.model)
    
    return df


# =============================================================================
# PHASE MANIFEST WRITER
# =============================================================================

@dataclass
class PhaseManifest:
    """Manifest for a completed phase run."""
    phase: str
    phase_dir: str
    timestamp: str
    duration_seconds: float
    
    steps_completed: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    
    outputs_written: List[str] = field(default_factory=list)
    records_processed: int = 0
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_phase_manifest(
    manifest: PhaseManifest,
    outputs_dir: Union[str, Path],
) -> Path:
    """
    Write phase manifest to JSON file.
    
    Args:
        manifest: PhaseManifest object
        outputs_dir: Root outputs directory
    
    Returns:
        Path to written manifest file
    """
    outputs_dir = Path(outputs_dir)
    manifest_dir = outputs_dir / manifest.phase_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # Use timestamp in filename for history
    ts = manifest.timestamp.replace(":", "-").replace("T", "_")
    filename = f"manifest__{manifest.phase}__{ts}.json"
    filepath = manifest_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, default=str)
    
    # Also write a "latest" manifest copy
    latest_path = manifest_dir / f"manifest__{manifest.phase}__latest.json"
    with open(latest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, default=str)
    
    return filepath


def now_iso() -> str:
    """Current timestamp in ISO format."""
    return datetime.now().isoformat(timespec="seconds")


# =============================================================================
# PATH UTILITIES
# =============================================================================

def ensure_dir(p: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_vp_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse dataset, view, model from VP filename.
    
    Handles patterns like:
        vp_joined__mlomics__methylation.csv.gz
        importance__mlomics__methylation__xgb_bal.csv.gz
    
    Returns:
        Dict with 'dataset', 'view', optionally 'model', or None if no match
    """
    patterns = [
        r"^(?P<prefix>\w+)__(?P<dataset>\w+)__(?P<view>\w+)__(?P<model>\w+)\.csv(?:\.gz)?$",
        r"^(?P<prefix>\w+)__(?P<dataset>\w+)__(?P<view>\w+)\.csv(?:\.gz)?$",
    ]
    
    for pattern in patterns:
        m = re.match(pattern, filename)
        if m:
            return m.groupdict()
    return None


# =============================================================================
# REGIME CLASSIFICATION HELPERS
# =============================================================================

REGIME_THRESHOLDS = {
    "coupled_max_di": 0.85,
    "antialigned_min_di": 1.05,
}


def classify_regime(di_value: float) -> str:
    """
    Classify regime based on DI value.
    
    Returns:
        One of: "COUPLED", "DECOUPLED", "ANTI_ALIGNED"
    """
    if di_value < REGIME_THRESHOLDS["coupled_max_di"]:
        return "COUPLED"
    elif di_value > REGIME_THRESHOLDS["antialigned_min_di"]:
        return "ANTI_ALIGNED"
    else:
        return "DECOUPLED"


# =============================================================================
# FEATURE TYPE CLASSIFICATION (for biology phase)
# =============================================================================

FEATURE_TYPE_PATTERNS = {
    "mRNA": {"type": "gene_expression", "mappable_to_gene": True, "mapping_method": "direct"},
    "miRNA": {"type": "mirna", "mappable_to_gene": False, "mapping_method": None},
    "methylation": {"type": "methylation_cpg", "mappable_to_gene": True, "mapping_method": "probe_annotation"},
    "CNV": {"type": "copy_number", "mappable_to_gene": True, "mapping_method": "direct"},
    "proteomics": {"type": "protein", "mappable_to_gene": True, "mapping_method": "direct"},
    "MGX": {"type": "taxonomy", "mappable_to_gene": False, "mapping_method": None},
    "MGX_func": {"type": "functional", "mappable_to_gene": False, "mapping_method": None},
    "MPX": {"type": "metaproteomics", "mappable_to_gene": False, "mapping_method": None},
    "MBX": {"type": "metabolomics", "mappable_to_gene": False, "mapping_method": None},
}


def get_feature_type_info(view: str) -> Dict[str, Any]:
    """Get feature type information for a view."""
    base_view = view.split("_")[0] if "_" in view else view
    return FEATURE_TYPE_PATTERNS.get(base_view, {
        "type": "unknown", "mappable_to_gene": False, "mapping_method": None,
    })


def is_gene_mappable(view: str) -> bool:
    """Check if view features can be mapped to genes."""
    return get_feature_type_info(view).get("mappable_to_gene", False)
