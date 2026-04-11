#!/usr/bin/env python3
"""
PHASE 9 — 00_run_phase9.py

Orchestrator script that runs all Phase 9 (Simulation) steps in sequence.

Steps:
    1. 01_generate_synthetic.py - Generate synthetic datasets
    2. 02_sim_compute_decoupling.py - Compute DI on synthetic data
    3. 03_sim_param_sweeps.py - Parameter sweeps for phase diagram

Usage:
    python 00_run_phase9.py --outputs-dir outputs
    python 00_run_phase9.py --outputs-dir outputs --quick

Author: variance-prediction paradox pipeline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import (
    ensure_dir, now_iso,
    PhaseManifest, write_phase_manifest,
)


STEPS = {
    1: {"name": "generate_synthetic", "script": "01_generate_synthetic.py",
        "description": "Generate synthetic datasets with controlled variance"},
    2: {"name": "compute_decoupling", "script": "02_sim_compute_decoupling.py",
        "description": "Compute DI metrics on synthetic datasets"},
    3: {"name": "param_sweeps", "script": "03_sim_param_sweeps.py",
        "description": "Parameter sweeps for phase diagram"},
}


def run_script(script_path: Path, argv: List[str]) -> int:
    cmd = [sys.executable, str(script_path)] + argv
    return subprocess.run(cmd).returncode


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run all Phase 9 (Simulation) steps")
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--steps", type=str, default="1,2,3")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-datasets", type=int, default=5)
    parser.add_argument("--n-features", type=int, default=1000)
    parser.add_argument("--scenarios", type=str, default="all",
                        help="Comma-separated scenario names or 'all'")
    parser.add_argument("--k-pcts", type=str, default="1,5,10,20",
                        help="K% list for DI curve computation")
    
    args = parser.parse_args(argv)
    
    steps_to_run = [int(s.strip()) for s in args.steps.split(",")]
    if args.quick:
        args.n_datasets = 2
        args.n_features = min(args.n_features, 500)
    
    manifest = PhaseManifest(
        phase="09_simulation", phase_dir="09_simulation",
        timestamp=now_iso(), duration_seconds=0.0,
        parameters={
            "quick": args.quick,
            "n_datasets": args.n_datasets,
            "n_features": args.n_features,
            "scenarios": args.scenarios,
            "k_pcts": args.k_pcts,
        },
    )
    
    script_dir = Path(__file__).resolve().parent
    start_time = time.time()
    
    print(f"{'='*70}")
    print(f"PHASE 9: SIMULATION")
    print(f"{'='*70}")
    print(f"[{now_iso()}] Starting Phase 9 orchestration")
    print(f"  Steps: {steps_to_run}, Quick: {args.quick}")
    
    for step_num in steps_to_run:
        step_info = STEPS[step_num]
        script_path = script_dir / step_info["script"]
        
        print(f"\n{'-'*70}")
        print(f"STEP {step_num}: {step_info['description']}")
        print(f"{'-'*70}")
        
        if not script_path.exists():
            manifest.steps_failed.append(step_info["name"])
            manifest.errors.append(f"Script not found: {script_path}")
            continue
        
        step_argv = ["--outputs-dir", str(args.outputs_dir)]
        
        if step_num == 1:
            step_argv.extend(["--n-datasets", str(args.n_datasets)])
            step_argv.extend(["--n-features", str(args.n_features)])
            step_argv.extend(["--scenarios", args.scenarios])
        elif step_num == 2:
            step_argv.extend(["--k-pcts", args.k_pcts])
        elif step_num == 3:
            if args.quick:
                step_argv.append("--quick")
        
        try:
            step_start = time.time()
            exit_code = run_script(script_path, step_argv)
            step_duration = time.time() - step_start
            
            if exit_code == 0:
                manifest.steps_completed.append(step_info["name"])
                print(f"\n  [OK] Step {step_num} completed in {step_duration:.1f}s")
            else:
                manifest.steps_failed.append(step_info["name"])
                manifest.errors.append(f"Step {step_num}: exit code {exit_code}")
                print(f"\n  [FAIL] Step {step_num} FAILED: exit code {exit_code}")
                
        except Exception as e:
            manifest.steps_failed.append(step_info["name"])
            manifest.errors.append(f"Step {step_num}: {str(e)}")
            print(f"\n  [FAIL] Step {step_num} FAILED: {e}")
    
    manifest.duration_seconds = time.time() - start_time
    manifest_path = write_phase_manifest(manifest, args.outputs_dir)
    
    print(f"\n{'='*70}")
    print(f"PHASE 9 COMPLETE")
    print(f"{'='*70}")
    print(f"  Duration: {manifest.duration_seconds:.1f}s")
    print(f"  Completed: {manifest.steps_completed}")
    print(f"  Failed: {manifest.steps_failed}")
    print(f"  Manifest: {manifest_path}")
    
    if manifest.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
