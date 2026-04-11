"""
03_verify_paper_consistency.py
Run end-to-end verification:
1) build paper_numbers.json
2) extract/compare Results.txt claims
3) scan figure code for bypassed data sources (raw outputs usage)

Features:
- Scans code/figures/** for python figure scripts (figure_*.py etc).
- Normalizes Windows backslashes before pattern matching.
- Fails CI-style if any claim is UNRESOLVED / NOT_FOUND / MISMATCH.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Match both / and \ in code strings by normalizing text first.
BYPASS_PATTERNS = [
    r"outputs/04_importance/",
    r"outputs/05_mechanistic/",
    r"outputs/06_robustness/",
    r"outputs/07_ablation/",
    r"outputs/08_biology/",
    r"outputs/11_diagnostic_validation/",
]

ALLOWED_PATTERNS = [
    r"outputs/results/main_results/",
    r"outputs/results/sup_table/",
    r"outputs/results/sup_data/",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--repo-root", default=None, help="Project root; defaults to outputs-dir/..")
    ap.add_argument("--results-txt", default=None)
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--figure-code-dir", action="append", default=None,
                    help="Repeatable; defaults to <repo>/code/figures")
    return ap.parse_args()


def run(cmd: List[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_claims_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def any_fail(rows: List[Dict[str, str]]) -> Tuple[bool, Dict[str, int]]:
    bad = {"UNRESOLVED", "NOT_FOUND", "MISMATCH"}
    counts: Dict[str, int] = {}
    fail = False
    for r in rows:
        st = r.get("status", "")
        counts[st] = counts.get(st, 0) + 1
        if st in bad:
            fail = True
    return fail, counts


def _is_archived_figure(path: Path) -> bool:
    p = str(path).replace("\\", "/").lower()
    return "/archive/" in p or "/deprecated/" in p or "/_old/" in p


def iter_figure_files(roots: List[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            name = p.name.lower()
            if name.startswith(("figure_", "fig_", "plot_")) or "figure" in name or "fig" in name:
                files.append(p)
    # de-dup
    files = sorted(list(dict.fromkeys(files)))
    files = [p for p in files if not _is_archived_figure(p)]
    return files


def scan_bypass(fig_py: Path) -> List[str]:
    txt = fig_py.read_text(encoding="utf-8", errors="ignore")
    txt = txt.replace("\\", "/")  # normalize
    hits: List[str] = []

    # allow-list overrides bypass
    for pat in ALLOWED_PATTERNS:
        if re.search(pat, txt):
            # allowed usage does not remove bypass hits elsewhere; it just isn't itself a hit
            pass

    for pat in BYPASS_PATTERNS:
        if re.search(pat, txt):
            # if the match is only in allowed contexts, skip; otherwise flag
            # (simple approach: still flag; you'll remove these progressively)
            hits.append(pat)

    return hits


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    repo_root = Path(args.repo_root) if args.repo_root else outputs_dir.parent

    script_dir = repo_root / "code" / "compute" / "13_results"
    s01 = script_dir / "01_build_paper_numbers.py"
    s02 = script_dir / "02_extract_results_claims.py"

    # 1) build paper numbers
    run([sys.executable, str(s01), "--outputs-dir", str(outputs_dir)])

    # 2) compare Results.txt claims
    cmd = [sys.executable, str(s02), "--outputs-dir", str(outputs_dir)]
    if args.results_txt:
        cmd += ["--results-txt", args.results_txt]
    run(cmd)

    claims_csv = outputs_dir / "results" / "results_claims_check.csv"
    if not claims_csv.exists():
        raise FileNotFoundError(f"Missing: {claims_csv}")

    rows = load_claims_csv(claims_csv)
    fail, counts = any_fail(rows)

    print("Claims status counts:", counts)
    if fail:
        # Print top failures
        print("\nTop failing claims:")
        for r in rows:
            if r.get("status") in {"UNRESOLVED", "NOT_FOUND", "MISMATCH"}:
                print(f"- {r.get('claim_id')} {r.get('key')} [{r.get('status')}] "
                      f"paper={r.get('paper_value')} text={r.get('text_value')} line={r.get('line_no_1based')}")
        raise SystemExit(2)

    # 3) figure bypass scan
    if not args.skip_figures:
        fig_dirs: List[Path] = []
        if args.figure_code_dir:
            fig_dirs = [Path(d) for d in args.figure_code_dir]
        else:
            fig_dirs = [repo_root / "code" / "figures"]

        fig_files = iter_figure_files(fig_dirs)
        flagged: List[Tuple[Path, List[str]]] = []
        for fp in fig_files:
            hits = scan_bypass(fp)
            if hits:
                flagged.append((fp, hits))

        if flagged:
            print("\nFigure bypass flagged (raw outputs referenced):")
            for fp, hits in flagged:
                print(f"- {fp} -> {sorted(set(hits))}")
            raise SystemExit(3)

    print("\nOK: paper_numbers + Results.txt claims are consistent.")


if __name__ == "__main__":
    main()
