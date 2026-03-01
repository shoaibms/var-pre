#!/usr/bin/env python3
"""01_define_tasks_and_splits.py

Phase 3 (Supervised): Define tasks and CV splits.

This script creates a reviewer-proof, deterministic splitting artefact per dataset.
Splits are defined at the *dataset* level (shared across views) and saved as:

  outputs/splits/splits__{dataset}.npz
  outputs/splits/splits__{dataset}.json

Key design choices
------------------
- Primary datasets: repeated stratified K-fold.
- Longitudinal / repeated-measures cohorts: group-aware stratified folds.
- Splits are stored as fold assignments (fold_ids) with shape (n_repeats, n_samples)
  so the artefact is fixed-size and pickle-free.

Bundle contract
---------------
Loads normalized bundles from:
  outputs/bundles/{dataset}_bundle_normalized.npz

Expected bundle keys:
  y (n,), sample_ids (n,), info (json string), X_{view}, features_{view}

For IBDMDB/HMP2, this script attempts to recover participant/subject IDs from the
raw metadata file to enable group-aware CV. If group IDs cannot be recovered,
it falls back to sample-level CV with a loud warning.

Usage
-----
  python 01_define_tasks_and_splits.py --dataset all
  python 01_define_tasks_and_splits.py --dataset mlomics --n-splits 5 --n-repeats 5 --seed 42

"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Dataset registry (kept intentionally small and explicit)
# =============================================================================

VIEW_REGISTRY: Dict[str, Dict] = {
    "mlomics": {
        "core_views": ["mRNA", "miRNA", "methylation", "CNV"],
        "sensitivity_views": [],
        "analysis_role": "primary",
        "task": "PAM50 subtype classification",
        "n_classes": 5,
        "variance_approach": "latent_axis",
        "cv_policy": {"n_splits": 5, "n_repeats": 5, "group_aware": False},
    },
    "ibdmdb": {
        "core_views": ["MGX", "MGX_func", "MPX", "MBX"],
        "sensitivity_views": ["MGX_CLR"],
        "analysis_role": "primary",
        "task": "IBD diagnosis classification (nonIBD/UC/CD)",
        "n_classes": 3,
        "variance_approach": "marginal",
        "cv_policy": {"n_splits": 5, "n_repeats": 10, "group_aware": True},
    },
    "ccle": {
        "core_views": ["mRNA", "CNV", "proteomics"],
        "sensitivity_views": [],
        "analysis_role": "primary",
        "task": "Tissue (OncotreeLineage) classification",
        "n_classes": 22,
        "variance_approach": "marginal",
        "cv_policy": {"n_splits": 5, "n_repeats": 5, "group_aware": False},
    },
    "tcga_gbm": {
        "core_views": ["mRNA", "methylation", "CNV"],
        "sensitivity_views": ["methylation_Mval"],
        "analysis_role": "sensitivity",
        "task": "GBM subtype classification",
        "n_classes": 4,
        "variance_approach": "marginal",
        "cv_policy": {"n_splits": 5, "n_repeats": 20, "group_aware": False},
    },
}


# =============================================================================
# Helpers
# =============================================================================


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_normalized_bundle(bundle_path: Path) -> Dict:
    """Load normalized bundle and parse info."""
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    bundle = np.load(bundle_path, allow_pickle=False)
    info = json.loads(str(bundle["info"]))

    X_views = {}
    feature_names = {}
    for key in bundle.files:
        if key.startswith("X_"):
            view = key[2:]
            X_views[view] = bundle[key]
        elif key.startswith("features_"):
            view = key[9:]
            feature_names[view] = bundle[key]

    return {
        "y": bundle["y"].astype(np.int32),
        "sample_ids": bundle["sample_ids"].astype(str),
        "info": info,
        "X_views": X_views,
        "feature_names": feature_names,
    }


def infer_effective_n_splits(y: np.ndarray, requested: int) -> int:
    """Ensure n_splits is feasible given the smallest class."""
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    if min_count < 2:
        raise ValueError(
            f"Smallest class has <2 samples (min_count={min_count}); "
            "cannot construct any meaningful CV split."
        )
    return max(2, min(int(requested), min_count))


def make_stratified_kfold_assignments(
    y: np.ndarray, n_splits: int, seed: int
) -> np.ndarray:
    """Return fold assignments (n_samples,) using stratified K-fold."""
    try:
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_id = np.full(len(y), -1, dtype=np.int16)
        for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold_id[te] = k
        if np.any(fold_id < 0):
            raise RuntimeError("Internal error: some samples were not assigned to any fold")
        return fold_id
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for stratified splits. Please install scikit-learn."
        ) from e


def is_fold_assignment_valid(y: "np.ndarray", fold_id: "np.ndarray", n_splits: int) -> tuple[bool, str]:
    """Check fold assignment validity for multiclass training.

    Requirements:
      - every fold non-empty
      - for every held-out fold, the training split contains all classes
    """
    classes = set(np.unique(y).tolist())
    for k in range(n_splits):
        if not np.any(fold_id == k):
            return False, f"fold {k} is empty"
        train_y = y[fold_id != k]
        train_classes = set(np.unique(train_y).tolist())
        if train_classes != classes:
            missing = sorted(list(classes - train_classes))
            return False, f"fold {k} yields missing training classes {missing}"
    return True, "ok"


# ---- Group-aware stratification (greedy assignment of groups to folds) ----




def choose_feasible_group_n_splits(
    y: np.ndarray,
    groups: np.ndarray,
    requested_n_splits: int,
    seed: int,
    max_tries: int = 50,
) -> Optional[int]:
    """Find the largest feasible n_splits for stratified group CV.

    Group constraints can make some folds miss one or more classes in the training
    split (fatal for multiclass training). This helper searches downward from
    requested_n_splits to 2 and tests multiple random seeds.

    Returns None if no feasible split is found.
    """
    if requested_n_splits < 2:
        return None

    n_groups = int(len(np.unique(groups)))
    upper = min(int(requested_n_splits), n_groups)
    if upper < 2:
        return None

    for k in range(upper, 1, -1):
        for t in range(int(max_tries)):
            fold = make_stratified_group_kfold_assignments(y, groups, k, seed + 10007 * t)
            if is_fold_assignment_valid(y, fold, k):
                return k
    return None

def _group_label_counts(y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique_groups, group_sizes, group_class_counts (G x C)."""
    ug, inv = np.unique(groups, return_inverse=True)
    classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    G = len(ug)
    C = len(classes)
    group_sizes = np.zeros(G, dtype=np.int32)
    group_class_counts = np.zeros((G, C), dtype=np.int32)

    for i in range(len(y)):
        g = inv[i]
        group_sizes[g] += 1
        group_class_counts[g, class_to_idx[y[i]]] += 1

    return ug, group_sizes, group_class_counts


def make_stratified_group_kfold_assignments(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Return fold assignments (n_samples,) with group integrity preserved.

    Greedy heuristic: assign groups to folds to match global class proportions.
    """
    rng = np.random.default_rng(seed)

    # Basic checks
    if len(y) != len(groups):
        raise ValueError("y and groups must have same length")

    # Compute group x class counts
    ug, group_sizes, group_class_counts = _group_label_counts(y, groups)
    G, C = group_class_counts.shape

    if G < n_splits:
        # Cannot have more folds than groups; reduce folds
        n_splits = max(2, G)

    # Desired per-fold class counts
    total_class_counts = group_class_counts.sum(axis=0).astype(np.float64)
    total = total_class_counts.sum()
    target_props = total_class_counts / max(total, 1.0)

    # Shuffle groups, then sort by size (desc) to place large groups first
    order = np.arange(G)
    rng.shuffle(order)
    order = order[np.argsort(-group_sizes[order], kind="mergesort")]

    fold_class_counts = np.zeros((n_splits, C), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.float64)
    group_to_fold = np.full(G, -1, dtype=np.int16)

    def fold_cost(f: int) -> float:
        # L2 distance between fold class proportions and target proportions
        if fold_sizes[f] <= 0:
            return float(np.sum((0.0 - target_props) ** 2))
        props = fold_class_counts[f] / fold_sizes[f]
        return float(np.sum((props - target_props) ** 2))

    for g in order:
        best_f = None
        best_score = None
        g_counts = group_class_counts[g].astype(np.float64)
        g_size = float(group_sizes[g])

        for f in range(n_splits):
            # Evaluate marginal cost if we add group g to fold f
            old_cost = fold_cost(f)
            fold_class_counts[f] += g_counts
            fold_sizes[f] += g_size
            new_cost = fold_cost(f)
            fold_class_counts[f] -= g_counts
            fold_sizes[f] -= g_size

            score = new_cost - old_cost
            if (best_score is None) or (score < best_score):
                best_score = score
                best_f = f

        group_to_fold[g] = int(best_f)
        fold_class_counts[best_f] += g_counts
        fold_sizes[best_f] += g_size

    # Build sample-level fold assignments
    ug_to_idx = {g: i for i, g in enumerate(ug)}
    fold_id = np.full(len(y), -1, dtype=np.int16)
    for i in range(len(y)):
        fold_id[i] = group_to_fold[ug_to_idx[groups[i]]]

    if np.any(fold_id < 0):
        raise RuntimeError("Internal error: some samples were not assigned to any fold")

    return fold_id


def _choose_best_matching_column(df, values: List[str]) -> Optional[str]:
    """Return the df column name with best overlap against 'values'."""
    values_set = set(values)
    best_col, best_hit = None, -1
    for col in df.columns:
        s = df[col].astype(str)
        hit = int(s.isin(values_set).sum())
        if hit > best_hit:
            best_hit = hit
            best_col = col
    if best_hit <= 0:
        return None
    return best_col


def load_ibdmdb_groups(sample_ids: np.ndarray, repo_root: Path) -> Tuple[np.ndarray, Dict[str, str]]:
    """Try to recover participant/subject IDs for IBDMDB/HMP2.

    Returns
    -------
    groups : (n_samples,) array of group IDs
    meta   : dict with provenance and diagnostics
    """
    meta: Dict[str, str] = {}

    # Candidate metadata paths (keep flexible)
    candidates = [
        repo_root / "data" / "raw" / "ibdmdb" / "hmp2_metadata.csv",
        repo_root / "data" / "raw" / "ibdmdb" / "HMP2_metadata.csv",
        repo_root / "data" / "raw" / "ibdmdb" / "metadata.csv",
        repo_root / "data" / "raw" / "ibdmdb" / "hmp2_metadata.tsv",
    ]
    meta_path = None
    for p in candidates:
        if p.exists():
            meta_path = p
            break

    if meta_path is None:
        meta["status"] = "missing_metadata_file"
        meta["note"] = "Could not find HMP2 metadata file; falling back to sample-level CV."
        return sample_ids.copy(), meta

    try:
        import pandas as pd

        if meta_path.suffix.lower() == ".tsv":
            df = pd.read_csv(meta_path, sep="\t", low_memory=False)
        else:
            df = pd.read_csv(meta_path, low_memory=False)
    except Exception as e:
        meta["status"] = "metadata_read_error"
        meta["note"] = f"Failed to read metadata ({meta_path.name}): {e}; falling back to sample-level CV."
        return sample_ids.copy(), meta

    # Identify sample-id column by overlap
    sid_col = _choose_best_matching_column(df, list(sample_ids))
    if sid_col is None:
        meta["status"] = "sample_id_column_not_found"
        meta["note"] = (
            "Could not identify a sample-id column in metadata; "
            "falling back to sample-level CV."
        )
        return sample_ids.copy(), meta

    # Identify participant/subject column
    lower_cols = {c.lower(): c for c in df.columns}
    preferred = [
        "participant",
        "participant_id",
        "subject",
        "subject_id",
        "hostsubjectid",
        "host_subject_id",
        "individual",
        "individual_id",
    ]
    grp_col = None
    for key in preferred:
        if key in lower_cols:
            grp_col = lower_cols[key]
            break

    if grp_col is None:
        # Heuristic: choose a column with few unique values relative to rows
        best = None
        best_score = None
        for col in df.columns:
            nunq = df[col].nunique(dropna=True)
            if nunq <= 1:
                continue
            # prefer columns with moderate uniqueness, not almost-all-unique
            score = abs((nunq / max(len(df), 1)) - 0.05)
            if (best_score is None) or (score < best_score):
                best_score = score
                best = col
        grp_col = best

    if grp_col is None:
        meta["status"] = "group_column_not_found"
        meta["note"] = "Could not identify participant/subject column; falling back to sample-level CV."
        return sample_ids.copy(), meta

    # Build mapping sample_id -> group_id
    df2 = df[[sid_col, grp_col]].copy()
    df2[sid_col] = df2[sid_col].astype(str)
    df2[grp_col] = df2[grp_col].astype(str)

    mapping = dict(zip(df2[sid_col], df2[grp_col]))
    groups = np.array([mapping.get(sid, sid) for sid in sample_ids], dtype=str)

    n_unmapped = int(np.sum(groups == sample_ids))

    meta.update(
        {
            "status": "ok" if n_unmapped == 0 else "partial_map",
            "metadata_path": str(meta_path),
            "sample_id_column": sid_col,
            "group_column": grp_col,
            "n_samples": str(len(sample_ids)),
            "n_groups": str(len(np.unique(groups))),
            "n_unmapped_samples": str(n_unmapped),
        }
    )

    return groups, meta


@dataclass
class SplitArtefact:
    dataset: str
    task_type: str
    n_samples: int
    n_classes: int
    y: np.ndarray
    sample_ids: np.ndarray
    groups: np.ndarray
    fold_ids: np.ndarray  # (n_repeats, n_samples)
    n_splits: int
    n_repeats: int
    seed: int
    bundle_path: str
    bundle_sha256: str
    created_at: str
    notes: List[str]
    group_meta: Dict[str, str]


def build_splits_for_dataset(
    dataset: str,
    bundle_dir: Path,
    repo_root: Path,
    seed: int,
    n_splits: Optional[int],
    n_repeats: Optional[int],
) -> SplitArtefact:
    if dataset not in VIEW_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {sorted(VIEW_REGISTRY)}")

    registry = VIEW_REGISTRY[dataset]
    policy = dict(registry.get("cv_policy", {}))

    if n_splits is None:
        n_splits = int(policy.get("n_splits", 5))
    if n_repeats is None:
        n_repeats = int(policy.get("n_repeats", 5))

    # Load data
    bundle_path = bundle_dir / f"{dataset}_bundle_normalized.npz"
    data = load_normalized_bundle(bundle_path)

    y = data["y"]
    sample_ids = data["sample_ids"]

    # Determine effective n_splits based on class counts
    eff_n_splits = infer_effective_n_splits(y, int(n_splits))

    notes: List[str] = []
    if eff_n_splits != int(n_splits):
        notes.append(
            f"Requested n_splits={n_splits} reduced to eff_n_splits={eff_n_splits} "
            "due to smallest-class constraint."
        )

    # Group handling
    group_meta: Dict[str, str] = {}
    if bool(policy.get("group_aware", False)):
        if dataset == "ibdmdb":
            groups, group_meta = load_ibdmdb_groups(sample_ids, repo_root)
            if group_meta.get("status", "") != "ok":
                notes.append(
                    "IBDMDB group-aware splitting requested, but participant mapping was incomplete. "
                    "Proceeding with best-effort groups (unmapped samples use their own sample_id as group)."
                )
        else:
            groups = sample_ids.copy()
            group_meta = {"status": "not_applicable"}
    else:
        groups = sample_ids.copy()
        group_meta = {"status": "not_used"}

    # Decide whether we can actually do group-aware CV (may fail due to class-by-group structure)
    use_group_cv = bool(policy.get("group_aware", False)) and dataset == "ibdmdb"
    if use_group_cv and group_meta.get("status") in {
        "missing_metadata_file",
        "metadata_read_error",
        "sample_id_column_not_found",
        "group_column_not_found",
    }:
        notes.append(
            "Participant/subject IDs unavailable for IBDMDB; using sample-level stratified CV."
        )
        use_group_cv = False
    if use_group_cv:
        feasible = choose_feasible_group_n_splits(y, groups, eff_n_splits, int(seed), max_tries=50)
        if feasible is None:
            notes.append(
                "Group-aware CV requested, but no feasible stratified group split was found. "
                "Falling back to sample-level stratified CV (reviewer note: ensure participant IDs are correct)."
            )
            use_group_cv = False
        elif feasible != eff_n_splits:
            notes.append(
                f"Group constraints required reducing n_splits from {eff_n_splits} to {feasible}."
            )
            eff_n_splits = int(feasible)

    # Build fold assignments
    fold_ids = np.zeros((int(n_repeats), len(y)), dtype=np.int16)

    for r in range(int(n_repeats)):
        base = int(seed) + 1000 * r
        if use_group_cv:
            # Try multiple seeds per repeat to avoid pathological assignments
            found = False
            for t in range(50):
                r_seed = base + 10007 * t
                fold = make_stratified_group_kfold_assignments(y, groups, eff_n_splits, r_seed)
                ok, _ = is_fold_assignment_valid(y, fold, eff_n_splits)
                if ok:
                    fold_ids[r] = fold
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"Repeat {r}: unable to construct a valid group-aware split after 50 attempts. "
                    "Reduce n_splits, verify participant mapping, or consider class collapsing for rare labels."
                )
        else:
            fold_ids[r] = make_stratified_kfold_assignments(y, eff_n_splits, base)

    artefact = SplitArtefact(
        dataset=dataset,
        task_type="multiclass_classification",
        n_samples=len(y),
        n_classes=int(len(np.unique(y))),
        y=y,
        sample_ids=sample_ids,
        groups=groups,
        fold_ids=fold_ids,
        n_splits=int(eff_n_splits),
        n_repeats=int(n_repeats),
        seed=int(seed),
        bundle_path=str(bundle_path),
        bundle_sha256=sha256_file(bundle_path),
        created_at=datetime.now().isoformat(timespec="seconds"),
        notes=notes,
        group_meta=group_meta,
    )

    # Sanity checks
    _validate_split_artefact(artefact)

    return artefact


def _validate_split_artefact(artefact: SplitArtefact) -> None:
    y = artefact.y
    fold_ids = artefact.fold_ids

    # Each repeat should assign every sample to a fold
    if np.any(fold_ids < 0):
        bad = int(np.sum(fold_ids < 0))
        raise RuntimeError(f"Found {bad} unassigned samples in fold_ids")

    # Each fold should have at least one sample
    for r in range(artefact.n_repeats):
        for k in range(artefact.n_splits):
            if not np.any(fold_ids[r] == k):
                raise RuntimeError(f"Repeat {r}: fold {k} is empty")

    # Training folds should cover all classes (best-effort)
    # This is a hard requirement for stable multiclass training.
    classes = set(np.unique(y).tolist())
    for r in range(artefact.n_repeats):
        for k in range(artefact.n_splits):
            train_y = y[fold_ids[r] != k]
            train_classes = set(np.unique(train_y).tolist())
            if train_classes != classes:
                missing = sorted(list(classes - train_classes))
                raise RuntimeError(
                    f"Repeat {r}, fold {k}: training split is missing classes {missing}. "
                    "Reduce n_splits or revise task definition (e.g., collapse rare classes)."
                )


def save_split_artefact(artefact: SplitArtefact, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # NPZ (pickle-free)
    npz_path = out_dir / f"splits__{artefact.dataset}.npz"
    info = {
        "dataset": artefact.dataset,
        "task_type": artefact.task_type,
        "n_samples": artefact.n_samples,
        "n_classes": artefact.n_classes,
        "n_splits": artefact.n_splits,
        "n_repeats": artefact.n_repeats,
        "seed": artefact.seed,
        "bundle_path": artefact.bundle_path,
        "bundle_sha256": artefact.bundle_sha256,
        "created_at": artefact.created_at,
        "notes": artefact.notes,
        "group_meta": artefact.group_meta,
    }

    np.savez_compressed(
        npz_path,
        y=artefact.y.astype(np.int32),
        sample_ids=artefact.sample_ids.astype(str),
        groups=artefact.groups.astype(str),
        fold_ids=artefact.fold_ids.astype(np.int16),
        info=json.dumps(info),
    )

    # JSON (human-readable, light)
    json_path = out_dir / f"splits__{artefact.dataset}.json"
    json_payload = {
        "info": info,
        "fold_ids": artefact.fold_ids.tolist(),
        "sample_ids": artefact.sample_ids.tolist(),
        "groups": artefact.groups.tolist(),
        "y": artefact.y.tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    return npz_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Define tasks and save deterministic CV splits.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=f"Dataset name or 'all'. Options: {', '.join(sorted(VIEW_REGISTRY))}",
    )
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default="outputs/bundles",
        help="Directory containing *_bundle_normalized.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/splits",
        help="Output directory for split artefacts",
    )
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root (for raw metadata lookup)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--n-splits", type=int, default=None, help="Override number of folds")
    parser.add_argument("--n-repeats", type=int, default=None, help="Override number of CV repeats")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    bundle_dir = (repo_root / args.bundle_dir).resolve() if not Path(args.bundle_dir).is_absolute() else Path(args.bundle_dir)
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    if args.dataset == "all":
        datasets = sorted(VIEW_REGISTRY.keys())
    else:
        datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "bundle_dir": str(bundle_dir),
        "datasets": {},
    }

    print("=" * 80)
    print("PHASE 3 — DEFINE TASKS AND SPLITS")
    print("=" * 80)
    print(f"Repo root:  {repo_root}")
    print(f"Bundle dir: {bundle_dir}")
    print(f"Out dir:    {out_dir}")
    print("")

    any_fail = False

    for ds in datasets:
        print("-" * 80)
        print(f"Dataset: {ds}")
        try:
            artefact = build_splits_for_dataset(
                dataset=ds,
                bundle_dir=bundle_dir,
                repo_root=repo_root,
                seed=args.seed,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
            )

            npz_path, json_path = save_split_artefact(artefact, out_dir)

            # Summarise
            unique_groups = len(np.unique(artefact.groups))
            print(f"  n_samples: {artefact.n_samples}")
            print(f"  n_classes: {artefact.n_classes}")
            print(f"  n_splits:  {artefact.n_splits}")
            print(f"  n_repeats: {artefact.n_repeats}")
            print(f"  groups:    {unique_groups} (group_mode={artefact.group_meta.get('status','')})")
            print(f"  bundle:    {artefact.bundle_path}")
            print(f"  sha256:    {artefact.bundle_sha256[:16]}…")
            print(f"  saved:     {npz_path}")
            print(f"  saved:     {json_path}")
            if artefact.notes:
                for n in artefact.notes:
                    print(f"  NOTE: {n}")

            manifest["datasets"][ds] = {
                "npz": str(npz_path),
                "json": str(json_path),
                "n_samples": artefact.n_samples,
                "n_classes": artefact.n_classes,
                "n_splits": artefact.n_splits,
                "n_repeats": artefact.n_repeats,
                "bundle_sha256": artefact.bundle_sha256,
                "group_meta": artefact.group_meta,
                "notes": artefact.notes,
            }

        except Exception as e:
            any_fail = True
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            manifest["datasets"][ds] = {"status": "failed", "error": str(e)}

    manifest_path = out_dir / "splits_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print(f"Manifest written: {manifest_path}")

    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
