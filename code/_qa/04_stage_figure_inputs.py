"""
04_stage_figure_inputs.py

Stage and verify figure-input files for the manuscript.
Copies required CSVs to a central figure-inputs directory with integrity checks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _expand_location(outputs_dir: Path, loc: str) -> List[Path]:
    """
    Expand patterns in Location column.
    Supports:
      - wildcards: *.csv, di_per_repeat__*.csv
      - placeholders: vp_joined__{dataset}__{view}.csv.gz -> vp_joined__*__*.csv.gz
    """
    loc = loc.replace("\\", "/")
    if "{" in loc and "}" in loc:
        # Replace any {...} token with *
        loc_glob = ""
        i = 0
        while i < len(loc):
            if loc[i] == "{":
                j = loc.find("}", i)
                if j == -1:
                    loc_glob += loc[i]
                    i += 1
                else:
                    loc_glob += "*"
                    i = j + 1
            else:
                loc_glob += loc[i]
                i += 1
        return sorted((outputs_dir).glob(loc_glob))
    if "*" in loc or "?" in loc or "[" in loc:
        return sorted((outputs_dir).glob(loc))
    return [outputs_dir / loc]


def stage_figure_inputs(
    outputs_dir: Path,
    input_csv: Path,
    stage_root: Optional[Path] = None,
    mode: str = "copy",
    only_outside_results: bool = True,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Stage figure-required inputs into outputs/results/figure_inputs/...
    and write a manifest with hashes.

    mode:
      - copy     : physical copy (safest snapshot)
      - hardlink : os.link (fast, no extra disk, but NOT an immutable snapshot)
      - symlink  : os.symlink (may require permissions on Windows)
    """
    notes: List[str] = []
    outputs_dir = outputs_dir.resolve()

    if stage_root is None:
        stage_root = outputs_dir / "results" / "figure_inputs"
    stage_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    req_cols = {"File Name", "Location (relative to outputs_dir)", "Used By"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"input_csv missing columns: {sorted(missing)}")

    rows: List[Dict[str, object]] = []
    staged_count = 0
    missing_count = 0

    for _, r in df.iterrows():
        loc = str(r["Location (relative to outputs_dir)"])
        used_by = str(r.get("Used By", ""))
        fname = str(r.get("File Name", ""))

        # Optionally stage only things NOT already under results/
        if only_outside_results and loc.replace("\\", "/").startswith("results/"):
            continue

        src_paths = _expand_location(outputs_dir, loc)
        if not src_paths:
            missing_count += 1
            rows.append({
                "spec_file_name": fname,
                "spec_location": loc,
                "used_by": used_by,
                "src_path": "",
                "dest_path": "",
                "status": "MISSING",
                "sha256_src": "",
                "sha256_dest": "",
                "size_bytes": "",
            })
            continue

        for src in src_paths:
            if not src.exists() or src.is_dir():
                continue

            rel = Path(loc.replace("\\", "/"))
            # If loc was a glob/pattern, preserve directory but use real filename
            if "*" in rel.name or "{" in rel.name:
                rel = rel.parent / src.name

            dest = stage_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() and not overwrite:
                sha_src = sha256_file(src)
                sha_dst = sha256_file(dest)
                rows.append({
                    "spec_file_name": fname,
                    "spec_location": loc,
                    "used_by": used_by,
                    "src_path": str(src),
                    "dest_path": str(dest),
                    "status": "SKIPPED_EXISTS",
                    "sha256_src": sha_src,
                    "sha256_dest": sha_dst,
                    "size_bytes": src.stat().st_size,
                })
                continue

            # Stage
            if mode == "copy":
                shutil.copy2(src, dest)
            elif mode == "hardlink":
                if dest.exists():
                    dest.unlink()
                os.link(src, dest)
            elif mode == "symlink":
                if dest.exists():
                    dest.unlink()
                os.symlink(src, dest)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            staged_count += 1
            sha_src = sha256_file(src)
            sha_dst = sha256_file(dest)
            status = "OK" if sha_src == sha_dst else "HASH_MISMATCH"

            rows.append({
                "spec_file_name": fname,
                "spec_location": loc,
                "used_by": used_by,
                "src_path": str(src),
                "dest_path": str(dest),
                "status": status,
                "sha256_src": sha_src,
                "sha256_dest": sha_dst,
                "size_bytes": src.stat().st_size,
            })

    manifest = pd.DataFrame(rows)

    notes.append(f"Staged files: {staged_count}")
    notes.append(f"Missing specs (no matches): {missing_count}")

    return manifest, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True, type=Path)
    ap.add_argument("--input-csv", required=True, type=Path)
    ap.add_argument("--mode", default="copy", choices=["copy", "hardlink", "symlink"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include-already-in-results", action="store_true",
                    help="Also stage inputs whose Location already starts with results/")
    args = ap.parse_args()

    only_outside = not args.include_already_in_results

    manifest, notes = stage_figure_inputs(
        outputs_dir=args.outputs_dir,
        input_csv=args.input_csv,
        mode=args.mode,
        overwrite=args.overwrite,
        only_outside_results=only_outside,
    )

    out_dir = args.outputs_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "MANIFEST_FIGURE_INPUTS.csv"

    manifest.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")
    for n in notes:
        print(n)

    # Fail fast on hash mismatch
    bad = manifest[manifest["status"] == "HASH_MISMATCH"]
    if len(bad) > 0:
        raise SystemExit(f"ERROR: {len(bad)} staged files have hash mismatch.")


if __name__ == "__main__":
    main()
