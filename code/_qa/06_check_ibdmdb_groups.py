#!/usr/bin/env python3
"""
06_check_ibdmdb_groups.py (FINAL)

Verify IBDMDB repeated measures and group-aware CV status.

ANSWERS TWO EXPLICIT QUESTIONS:
1. Do repeated measures exist? (groups with >1 sample)
2. Were groups respected in splitting? (no group in both train AND test)

If (1) yes and (2) no → flags "leakage risk" and recommends rerunning with GroupKFold.

Outputs:
    outputs/05_mechanistic/ibdmdb_group_check.json

Usage:
    python 06_check_ibdmdb_groups.py --outputs-dir outputs
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Optional
import numpy as np


def find_splits_file(outputs_dir: Path, dataset: str = "ibdmdb") -> Optional[Path]:
    """Find splits file with robust search."""
    candidates = [
        outputs_dir / "splits" / f"splits__{dataset}.json",
        outputs_dir / "splits" / f"splits__{dataset}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    for m in outputs_dir.rglob(f"*splits*{dataset}*.json"):
        return m
    for m in outputs_dir.rglob(f"*splits*{dataset}*.npz"):
        return m
    return None


def check_leakage_in_folds(groups: np.ndarray, fold_ids: np.ndarray) -> Dict[str, Any]:
    """
    Check if any group appears in both train and test within any fold.
    
    fold_ids: (n_repeats, n_samples) - fold assignment for each sample
    
    Returns dict with leakage status and details.
    """
    n_repeats, n_samples = fold_ids.shape
    leakage_cases = []
    
    for r in range(n_repeats):
        folds_r = fold_ids[r]
        unique_folds = np.unique(folds_r)
        
        for fold in unique_folds:
            test_mask = (folds_r == fold)
            train_mask = ~test_mask
            
            test_groups = set(groups[test_mask])
            train_groups = set(groups[train_mask])
            
            overlap = test_groups & train_groups
            if overlap:
                leakage_cases.append({
                    "repeat": int(r),
                    "fold": int(fold),
                    "n_overlapping_groups": len(overlap),
                    "example_groups": list(overlap)[:3],  # First 3 examples
                })
    
    return {
        "leakage_detected": len(leakage_cases) > 0,
        "n_leakage_cases": len(leakage_cases),
        "leakage_cases": leakage_cases[:10],  # First 10 examples
    }


def check_ibdmdb_groups(outputs_dir: Path) -> Dict[str, Any]:
    """
    Main check function.
    
    Returns dict with:
    - Q1: Do repeated measures exist?
    - Q2: Were groups respected in splitting?
    - Recommendation based on answers
    """
    result = {
        "dataset": "ibdmdb",
        "splits_file": None,
        "status": "UNKNOWN",
        
        # Q1: Repeated measures
        "q1_repeated_measures_exist": None,
        "n_samples": None,
        "n_groups": None,
        "samples_per_group_mean": None,
        "samples_per_group_max": None,
        
        # Q2: Group-aware splitting
        "q2_groups_respected": None,
        "leakage_detected": None,
        "n_leakage_cases": None,
        
        # Metadata
        "group_meta_status": None,
        "notes": [],
        "recommendation": None,
    }
    
    # Find splits file
    splits_path = find_splits_file(outputs_dir)
    if splits_path is None:
        result["status"] = "ERROR"
        result["error"] = "Splits file not found"
        result["recommendation"] = "Run 01_define_tasks_and_splits.py first"
        return result
    
    result["splits_file"] = str(splits_path)
    
    # Load based on file type
    if splits_path.suffix == ".json":
        _load_json(splits_path, result)
    else:
        _load_npz(splits_path, result)
    
    # Generate final status and recommendation
    _finalize_result(result)
    
    return result


def _load_json(path: Path, result: Dict) -> None:
    with open(path) as f:
        data = json.load(f)
    
    info = data.get("info", {})
    group_meta = info.get("group_meta", {})
    
    groups = np.array(data.get("groups", []))
    fold_ids = np.array(data.get("fold_ids", []))
    
    if len(groups) == 0:
        result["status"] = "ERROR"
        result["error"] = "No 'groups' in splits file"
        return
    
    result["n_samples"] = len(groups)
    result["n_groups"] = len(np.unique(groups))
    result["n_splits"] = info.get("n_splits")
    result["n_repeats"] = info.get("n_repeats")
    result["group_meta_status"] = group_meta.get("status", "unknown")
    result["notes"] = info.get("notes", [])
    
    # Q1: Repeated measures?
    group_counts = Counter(groups)
    result["samples_per_group_mean"] = sum(group_counts.values()) / len(group_counts)
    result["samples_per_group_max"] = max(group_counts.values())
    result["samples_per_group_min"] = min(group_counts.values())
    result["q1_repeated_measures_exist"] = result["n_groups"] < result["n_samples"]
    
    # Q2: Groups respected in splitting?
    if len(fold_ids) > 0 and fold_ids.ndim == 2:
        leakage = check_leakage_in_folds(groups, fold_ids)
        result["q2_groups_respected"] = not leakage["leakage_detected"]
        result["leakage_detected"] = leakage["leakage_detected"]
        result["n_leakage_cases"] = leakage["n_leakage_cases"]
        if leakage["leakage_cases"]:
            result["leakage_examples"] = leakage["leakage_cases"]
    else:
        result["q2_groups_respected"] = None
        result["notes"].append("Could not verify leakage: fold_ids missing or malformed")


def _load_npz(path: Path, result: Dict) -> None:
    z = np.load(path, allow_pickle=True)
    
    if "groups" not in z.files:
        result["status"] = "ERROR"
        result["error"] = "No 'groups' in NPZ file"
        result["available_keys"] = list(z.files)
        return
    
    groups = z["groups"].astype(str)
    fold_ids = z["fold_ids"] if "fold_ids" in z.files else None
    
    result["n_samples"] = len(groups)
    result["n_groups"] = len(np.unique(groups))
    
    # Q1
    unique, counts = np.unique(groups, return_counts=True)
    result["samples_per_group_mean"] = float(counts.mean())
    result["samples_per_group_max"] = int(counts.max())
    result["samples_per_group_min"] = int(counts.min())
    result["q1_repeated_measures_exist"] = result["n_groups"] < result["n_samples"]
    
    # Q2
    if fold_ids is not None and fold_ids.ndim == 2:
        leakage = check_leakage_in_folds(groups, fold_ids)
        result["q2_groups_respected"] = not leakage["leakage_detected"]
        result["leakage_detected"] = leakage["leakage_detected"]
        result["n_leakage_cases"] = leakage["n_leakage_cases"]
    
    # Parse info if available
    if "info" in z.files:
        try:
            info = json.loads(str(z["info"]))
            result["group_meta_status"] = info.get("group_meta", {}).get("status", "unknown")
            result["notes"] = info.get("notes", [])
            result["n_splits"] = info.get("n_splits")
            result["n_repeats"] = info.get("n_repeats")
        except Exception:
            pass


def _finalize_result(result: Dict) -> None:
    """Determine final status and recommendation."""
    q1 = result.get("q1_repeated_measures_exist")
    q2 = result.get("q2_groups_respected")
    
    # Decision matrix
    if q1 is None:
        result["status"] = "ERROR"
        result["recommendation"] = "Could not determine repeated measures status"
    elif not q1:
        # No repeated measures - standard CV is fine
        result["status"] = "OK"
        result["recommendation"] = (
            "No repeated measures (1 sample per participant). "
            "Standard stratified CV is appropriate."
        )
    elif q1 and q2 is True:
        # Repeated measures AND groups respected
        result["status"] = "OK"
        avg = result.get("samples_per_group_mean", 1)
        result["recommendation"] = (
            f"Repeated measures exist ({avg:.1f} samples/participant avg) "
            f"AND groups are respected in CV splits. No leakage risk."
        )
    elif q1 and q2 is False:
        # Repeated measures BUT groups NOT respected = LEAKAGE!
        result["status"] = "LEAKAGE_RISK"
        avg = result.get("samples_per_group_mean", 1)
        result["recommendation"] = (
            f"LEAKAGE RISK: Repeated measures exist ({avg:.1f} samples/participant avg) "
            f"but groups cross fold boundaries ({result.get('n_leakage_cases', '?')} cases). "
            f"Rerun 01_define_tasks_and_splits.py with GroupKFold for IBDMDB."
        )
    elif q1 and q2 is None:
        # Repeated measures but couldn't verify leakage
        result["status"] = "UNCERTAIN"
        result["recommendation"] = (
            "Repeated measures exist but could not verify group-aware splitting. "
            "Check splits manually or rerun with explicit GroupKFold."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    
    print("=" * 70)
    print("IBDMDB REPEATED MEASURES & LEAKAGE CHECK")
    print("=" * 70)
    print()
    
    result = check_ibdmdb_groups(outputs_dir)
    
    # Print results
    print(f"Splits file: {result.get('splits_file', 'NOT FOUND')}")
    print()
    
    print("Q1: DO REPEATED MEASURES EXIST?")
    print(f"  n_samples: {result.get('n_samples', '?')}")
    print(f"  n_groups (participants): {result.get('n_groups', '?')}")
    print(f"  samples/group: {result.get('samples_per_group_mean', '?'):.1f} avg "
          f"(range: {result.get('samples_per_group_min', '?')}-{result.get('samples_per_group_max', '?')})")
    print(f"  → Answer: {'YES' if result.get('q1_repeated_measures_exist') else 'NO'}")
    print()
    
    print("Q2: WERE GROUPS RESPECTED IN SPLITTING?")
    print(f"  Leakage detected: {result.get('leakage_detected', '?')}")
    print(f"  Leakage cases: {result.get('n_leakage_cases', '?')}")
    q2_answer = "YES (no leakage)" if result.get('q2_groups_respected') else (
        "NO (leakage!)" if result.get('q2_groups_respected') is False else "UNKNOWN")
    print(f"  → Answer: {q2_answer}")
    print()
    
    if result.get("notes"):
        print("NOTES:")
        for note in result["notes"]:
            print(f"  • {note}")
        print()
    
    # Final verdict
    print("=" * 70)
    status = result["status"]
    
    if status == "OK":
        print("VERIFICATION PASSED")
    elif status == "LEAKAGE_RISK":
        print("LEAKAGE RISK DETECTED")
    elif status == "UNCERTAIN":
        print("VERIFICATION UNCERTAIN")
    else:
        print(f"VERIFICATION FAILED: {result.get('error', status)}")
    
    print()
    print(f"RECOMMENDATION: {result.get('recommendation', 'N/A')}")
    print("=" * 70)
    
    # Save results
    out_dir = outputs_dir / "05_mechanistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ibdmdb_group_check.json"
    
    # Clean for JSON
    clean_result = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            clean_result[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            clean_result[k] = float(v)
        else:
            clean_result[k] = v
    
    with open(out_path, 'w') as f:
        json.dump(clean_result, f, indent=2)
    
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
