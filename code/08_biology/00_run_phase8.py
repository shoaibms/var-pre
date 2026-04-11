#!/usr/bin/env python3
"""
PHASE 8 — 00_run_phase8.py

Orchestrator script that runs all Phase 8 (Biology) steps in sequence
and produces a unified manifest.

Steps:
    1. 01_gene_mapping_sensitivity.py - Map features to genes
    2. 02_pathway_enrichment.py - Run pathway enrichment on TopV/TopP
    3. 03_module_overlap.py - Analyze gene vs pathway convergence
    4. 04_exemplar_panels_data.py - Extract exemplar features for figures

Usage:
    python 00_run_phase8.py --outputs-dir outputs
    python 00_run_phase8.py --outputs-dir outputs --steps 1,4  # Skip enrichment
    python 00_run_phase8.py --outputs-dir outputs --views hero  # Hero views only

Author: variance-prediction paradox pipeline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import (
    ensure_dir, now_iso,
    PhaseManifest, write_phase_manifest,
)


# =============================================================================
# STEP DEFINITIONS
# =============================================================================

STEPS = {
    1: {
        "name": "gene_mapping",
        "script": "01_gene_mapping_sensitivity.py",
        "description": "Map features to genes for pathway analysis",
    },
    2: {
        "name": "pathway_enrichment", 
        "script": "02_pathway_enrichment.py",
        "description": "Run pathway enrichment on TopV/TopP gene lists",
        "requires_network": True,  # g:Profiler API
    },
    3: {
        "name": "module_overlap",
        "script": "03_module_overlap.py", 
        "description": "Analyze gene vs pathway overlap (convergence)",
    },
    4: {
        "name": "exemplar_panels",
        "script": "04_exemplar_panels_data.py",
        "description": "Extract exemplar features for manuscript figures",
    },
}


def run_script(script_path: Path, argv: List[str]) -> int:
    """Run a script as subprocess and return exit code."""
    cmd = [sys.executable, str(script_path)] + argv
    result = subprocess.run(cmd)
    return result.returncode


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run all Phase 8 (Biology) steps"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        required=True,
        help="Root outputs directory",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="1,2,3,4",
        help="Comma-separated step numbers to run (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="all",
        help="Views to process: 'all', 'hero', or comma-separated dataset/view",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgb_bal",
        help="Model for importance files",
    )
    parser.add_argument(
        "--k-pct",
        type=int,
        default=10,
        help="K percentage used by enrichment/overlap steps (default: 10)",
    )
    parser.add_argument(
        "--k-pcts",
        type=str,
        default="5,10,20",
        help="Comma-separated K% list used by gene-mapping to precompute multiple gene lists (default: 5,10,20)",
    )
    parser.add_argument(
        "--skip-enrichment-api",
        action="store_true",
        help="Skip g:Profiler API calls (step 2)",
    )
    
    args = parser.parse_args(argv)
    
    # Parse steps
    steps_to_run = [int(s.strip()) for s in args.steps.split(",")]
    for s in steps_to_run:
        if s not in STEPS:
            raise ValueError(f"Unknown step {s}. Valid steps: {list(STEPS.keys())}")
    
    # Skip step 2 if requested
    if args.skip_enrichment_api and 2 in steps_to_run:
        steps_to_run.remove(2)
    
    # Initialize manifest
    manifest = PhaseManifest(
        phase="08_biology",
        phase_dir="08_biology",
        timestamp=now_iso(),
        duration_seconds=0.0,
        parameters={
            "views": args.views,
            "model": args.model,
            "k_pct": args.k_pct,
            "k_pcts": args.k_pcts,
        },
    )
    
    script_dir = Path(__file__).resolve().parent
    start_time = time.time()
    
    print(f"{'='*70}")
    print(f"PHASE 8: BIOLOGY")
    print(f"{'='*70}")
    print(f"[{now_iso()}] Starting Phase 8 orchestration")
    print(f"  Steps to run: {steps_to_run}")
    print(f"  Views: {args.views}")
    print(f"  Model: {args.model}")
    print(f"  Outputs: {args.outputs_dir}")
    
    # Run each step
    for step_num in steps_to_run:
        step_info = STEPS[step_num]
        step_name = step_info["name"]
        script_name = step_info["script"]
        script_path = script_dir / script_name
        
        print(f"\n{'-'*70}")
        print(f"STEP {step_num}: {step_info['description']}")
        print(f"{'-'*70}")
        
        if not script_path.exists():
            msg = f"Script not found: {script_path}"
            print(f"  ERROR: {msg}")
            manifest.steps_failed.append(step_name)
            manifest.errors.append(msg)
            continue
        
        # Build arguments for this step
        step_argv = [
            "--outputs-dir", str(args.outputs_dir),
            "--views", args.views,
            "--model", args.model,
        ]
        
        # Step-specific K args (CLIs differ across scripts)
        if step_num == 1:
            # gene mapping expects a CSV list
            step_argv += ["--k-pcts", args.k_pcts]
        elif step_num in (2, 3):
            # enrichment + overlap expect a single integer
            step_argv += ["--k-pct", str(args.k_pct)]
        
        # Run the step
        try:
            step_start = time.time()
            exit_code = run_script(script_path, step_argv)
            step_duration = time.time() - step_start
            
            if exit_code == 0:
                manifest.steps_completed.append(step_name)
                print(f"\n  [OK] Step {step_num} completed in {step_duration:.1f}s")
            else:
                manifest.steps_failed.append(step_name)
                manifest.errors.append(f"Step {step_num} ({step_name}): exit code {exit_code}")
                print(f"\n  [FAIL] Step {step_num} FAILED: exit code {exit_code}")
                
        except Exception as e:
            manifest.steps_failed.append(step_name)
            manifest.errors.append(f"Step {step_num} ({step_name}): {str(e)}")
            print(f"\n  [FAIL] Step {step_num} FAILED: {e}")
    
    # Finalize manifest
    manifest.duration_seconds = time.time() - start_time
    
    # Collect outputs
    output_base = args.outputs_dir / "08_biology"
    if output_base.exists():
        for subdir in ["gene_mapping", "pathway_enrichment", "module_overlap", "exemplar_panels"]:
            subpath = output_base / subdir
            if subpath.exists():
                for f in subpath.glob("*"):
                    if f.is_file():
                        manifest.outputs_written.append(str(f.relative_to(args.outputs_dir)))
    
    # Write manifest
    manifest_path = write_phase_manifest(manifest, args.outputs_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PHASE 8 COMPLETE")
    print(f"{'='*70}")
    print(f"  Duration: {manifest.duration_seconds:.1f}s")
    print(f"  Steps completed: {manifest.steps_completed}")
    print(f"  Steps failed: {manifest.steps_failed}")
    print(f"  Outputs: {len(manifest.outputs_written)} files")
    print(f"  Manifest: {manifest_path}")
    
    if manifest.errors:
        print(f"\n  ERRORS:")
        for err in manifest.errors:
            print(f"    - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
