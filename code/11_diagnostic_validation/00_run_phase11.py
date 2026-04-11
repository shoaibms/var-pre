#!/usr/bin/env python3
r"""00_run_phase11.py

Phase 11 orchestrator: Diagnostic validation.

Creates a run folder under outputs/<out_dirname>/runs/run__YYYYMMDD_HHMMSS and executes
Phase 11 steps in order.

Usage (Windows PowerShell):
  python .\code\compute\11_diagnostic_validation\00_run_phase11.py `
    --outputs-dir "<path-to-outputs>" `
    --steps 1,2,3

Steps:
  1 -> 01_calibration_thresholds_zones.py
  2 -> 02_validation.py
  3 -> 03_decision_assets_and_manifests.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True, help="Path to outputs/")
    ap.add_argument("--out-dirname", default="11_diagnostic_validation", help="Folder name under outputs/")
    ap.add_argument("--steps", default="1,2,3", help="Comma-separated: 1,2,3")
    ap.add_argument("--primary-k", type=int, default=10)
    ap.add_argument("--primary-metric", default="balanced_accuracy")
    ap.add_argument("--metrics", default="balanced_accuracy,auroc_ovr_macro")
    ap.add_argument("--ablation-xgb-dirname", default="07_ablation")
    ap.add_argument("--ablation-rf-dirname", default="07_ablation_rf")
    ap.add_argument("--label-perm-dirname", default="06_robustness_100\\label_perm")
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-ratios", default="1,5,10", help="Comma-separated FN:FP cost ratios")
    ap.add_argument("--primary-cost-ratio", type=float, default=5.0, help="Which cost ratio to use for t_opt")
    ap.add_argument("--target-sens", type=float, default=0.90)
    ap.add_argument("--target-spec", type=float, default=0.90)
    ap.add_argument("--material-harm", type=float, default=0.0, help="Harm threshold: delta < -material_harm")
    ap.add_argument("--epsilon", type=float, default=0.005, help="Deadzone for sign comparisons")
    ap.add_argument("--paper-dirname", default="09_paper\\v1", help="Optional: export figure inputs")
    ap.add_argument("--no-paper-export", action="store_true")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_root = outputs_dir / args.out_dirname
    run_dir = out_root / "runs" / f"run__{_ts()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]

    here = Path(__file__).resolve().parent
    step_map = {
        1: here / "01_calibration_thresholds_zones.py",
        2: here / "02_validation.py",
        3: here / "03_decision_assets_and_manifests.py",
    }

    argv_step1 = [
        "--outputs-dir", str(outputs_dir),
        "--run-dir", str(run_dir),
        "--primary-k", str(args.primary_k),
        "--primary-metric", str(args.primary_metric),
        "--ablation-xgb-dirname", str(args.ablation_xgb_dirname),
        "--bootstrap-n", str(args.bootstrap_n),
        "--seed", str(args.seed),
        "--cost-ratios", str(args.cost_ratios),
        "--primary-cost-ratio", str(args.primary_cost_ratio),
        "--target-sens", str(args.target_sens),
        "--target-spec", str(args.target_spec),
        "--material-harm", str(args.material_harm),
    ]
    argv_step2 = [
        "--outputs-dir", str(outputs_dir),
        "--run-dir", str(run_dir),
        "--primary-k", str(args.primary_k),
        "--primary-metric", str(args.primary_metric),
        "--metrics", str(args.metrics),
        "--ablation-xgb-dirname", str(args.ablation_xgb_dirname),
        "--ablation-rf-dirname", str(args.ablation_rf_dirname),
        "--label-perm-dirname", str(args.label_perm_dirname),
        "--material-harm", str(args.material_harm),
        "--epsilon", str(args.epsilon),
    ]
    argv_step3 = [
        "--outputs-dir", str(outputs_dir),
        "--run-dir", str(run_dir),
        "--primary-k", str(args.primary_k),
        "--primary-metric", str(args.primary_metric),
    ]
    if args.no_paper_export:
        argv_step3 += ["--no-paper-export"]
    else:
        argv_step3 += ["--paper-dirname", str(args.paper_dirname)]

    print(f"Phase11 run_dir: {run_dir}")

    for step in steps:
        if step not in step_map:
            raise SystemExit(f"Unknown step: {step}")
        script = step_map[step]
        if not script.exists():
            raise SystemExit(f"Missing script: {script}")
        if step == 1:
            cmd = [sys.executable, str(script)] + argv_step1
        elif step == 2:
            cmd = [sys.executable, str(script)] + argv_step2
        elif step == 3:
            cmd = [sys.executable, str(script)] + argv_step3
        else:
            cmd = [sys.executable, str(script)]
        print("\n" + "=" * 88)
        print(f"STEP {step}: {script.name}")
        print("=" * 88)
        print("Command:")
        print("  " + " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(f"Step {step} failed with exit code {rc}")

    (out_root / "latest.txt").write_text(str(run_dir), encoding="utf-8")
    print("\nDONE. Latest run pointer updated:")
    print(f"  {out_root / 'latest.txt'}")


if __name__ == "__main__":
    main()
