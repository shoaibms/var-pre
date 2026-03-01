#!/usr/bin/env python3
"""
00_run_all_results.py

Orchestrator that runs all Phase 13 results-compilation scripts in order.
"""
import argparse, subprocess, sys
from pathlib import Path

SCRIPTS = [
    "section_1_result.py",
    "section_2_result.py",
    "section_3_result.py",
    "section_4_result.py",
    "section_5_result.py",
    "01_compile_supplementary_tables.py",
    "compile_supplementary_data.py",
    "# 01_build_paper_numbers.py  # moved to _qa/ (QA tool, not core pipeline)",
]

def run(py: Path, script: Path, outputs_dir: str) -> None:
    cmd = [str(py), str(script), "--outputs-dir", outputs_dir]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True, help="Path to outputs/")
    ap.add_argument("--only", default="", help="Comma list of basenames to run (optional)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    py = Path(sys.executable)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    for s in SCRIPTS:
        if only and s not in only:
            continue
        run(py, here / s, args.outputs_dir)

if __name__ == "__main__":
    main()
