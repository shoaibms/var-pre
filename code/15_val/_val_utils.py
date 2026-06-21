#!/usr/bin/env python3
"""
Shared helpers for the external-validation refactor.

This module is intentionally small and behaviour-neutral.  It centralises only
path discovery and legacy-script dispatch used by the refactored wrappers.

Expected layout
---------------
code/compute/15_val/
    _val_utils.py
    01_build_external_validation_bundles.py
    02_run_external_validation_audit.py
    03_explain_cd4_boundary_case.py
    archive/
        10_download_external_validation_batch.py
        11_diagnose_cd4_vad_boundary.py
        29_download_depmap_drug_response_mechanism_first_v6.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def here() -> Path:
    """Return the directory containing this helper module."""
    return Path(__file__).resolve().parent


def archive_dir() -> Path:
    """Return the expected archive directory inside code/compute/15_val."""
    return here() / "archive"


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the var-pre project root from a script location or current working directory."""
    start = (start or Path.cwd()).resolve()
    for p in [start] + list(start.parents):
        if (p / "data" / "raw" / "val").exists() and (p / "outputs").exists():
            return p
        if (p / "README.md").exists() and (p / "code").exists():
            return p
    return start


def legacy_script_path(script_name: str) -> Path:
    """Return a required script path from the archive folder, with a clear error if absent."""
    path = archive_dir() / script_name
    if not path.exists():
        raise FileNotFoundError(
            f"Required archived script not found:\n"
            f"  {path}\n\n"
            f"Place the original working script in:\n"
            f"  {archive_dir()}"
        )
    return path


def run_python_script(script_path: Path, args: Iterable[str], cwd: Optional[Path] = None) -> int:
    """Run a Python script with the current interpreter and return its exit code."""
    cmd = [sys.executable, str(script_path), *list(args)]
    print("\n" + "=" * 90)
    print("RUNNING")
    print(" ".join(cmd))
    print("=" * 90)
    return subprocess.call(cmd, cwd=str(cwd or find_project_root()))
