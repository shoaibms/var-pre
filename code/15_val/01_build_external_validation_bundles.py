#!/usr/bin/env python3
r"""
01_build_external_validation_bundles.py

Clean entry point for building external-validation bundles.

This is a behaviour-preserving transition wrapper around the validated
bundle-builder scripts used during external validation. If the builder
backends are kept outside the public repository, place them in:

    code\compute\15_val\archive

Why this wrapper exists
-----------------------
This script provides a stable public interface for external-validation
bundle construction while keeping the validated backend code unchanged.

Sources
-------
depmap_prism
    Runs:
      archive\29_download_depmap_drug_response_mechanism_first_v6.py

geo_treatment
    Runs:
      archive\10_download_external_validation_batch.py

Examples
--------
From the project root:

    python code\compute\15_val\01_build_external_validation_bundles.py `
      --source depmap_prism `
      --max-expression-features 0

    python code\compute\15_val\01_build_external_validation_bundles.py `
      --source geo_treatment `
      --datasets batch1_fast

Notes
-----
- Raw data location is unchanged: data\raw\val
- Output location is unchanged: outputs\15_val
- The original scripts are not modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from _val_utils import find_project_root, legacy_script_path, run_python_script


SOURCE_TO_SCRIPT = {
    "depmap_prism": "29_download_depmap_drug_response_mechanism_first_v6.py",
    "geo_treatment": "10_download_external_validation_batch.py",
}


def infer_source(args: List[str]) -> str | None:
    """Infer source from legacy-style arguments when --source is omitted."""
    joined = " ".join(args).lower()
    if "--max-expression-features" in joined or "depmap" in joined or "prism" in joined:
        return "depmap_prism"
    if "--datasets" in joined or "--targets" in joined or "gse" in joined or "batch" in joined:
        return "geo_treatment"
    return None


def parse_source_and_passthrough(argv: List[str]) -> Tuple[str, List[str]]:
    """Parse only --source/--list-sources and pass all other args unchanged."""
    if "--list-sources" in argv:
        print("Available sources:")
        for source, script in SOURCE_TO_SCRIPT.items():
            print(f"  {source:<14s} -> archive\\{script}")
        raise SystemExit(0)

    source = None
    passthrough: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--source":
            if i + 1 >= len(argv):
                raise SystemExit("ERROR: --source requires one of: " + ", ".join(SOURCE_TO_SCRIPT))
            source = argv[i + 1]
            i += 2
            continue
        passthrough.append(arg)
        i += 1

    if source is None:
        source = infer_source(passthrough)

    if source not in SOURCE_TO_SCRIPT:
        msg = [
            "ERROR: Could not determine source.",
            "",
            "Use one of:",
            "  --source depmap_prism",
            "  --source geo_treatment",
            "",
            "Then add the original script arguments after --source.",
            "",
            "Examples:",
            "  python code\\compute\\15_val\\01_build_external_validation_bundles.py --source depmap_prism --max-expression-features 0",
            "  python code\\compute\\15_val\\01_build_external_validation_bundles.py --source geo_treatment --datasets batch1_fast",
        ]
        raise SystemExit("\n".join(msg))

    return source, passthrough


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    source, passthrough = parse_source_and_passthrough(argv)
    script = legacy_script_path(SOURCE_TO_SCRIPT[source])
    root = find_project_root(Path.cwd())

    print(f"Project root: {root}")
    print(f"Source      : {source}")
    print(f"Legacy code : {script}")
    print("Mode        : behaviour-preserving dispatch")

    return run_python_script(script, passthrough, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
