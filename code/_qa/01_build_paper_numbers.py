"""
01_build_paper_numbers.py
Aggregate all section key_numbers.json into a single paper_numbers.json.

Features:
- Normalises fraction strings like "10/14" into derived keys:
    direction_agreement_count_n = 10
    direction_agreement_count_d = 14
This makes verification robust and numeric.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

FRACTION_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(obj: Any, p: Path) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def collect_section_key_numbers(outputs_dir: Path) -> Dict[str, Any]:
    base = outputs_dir / "results" / "main_results"
    if not base.exists():
        raise FileNotFoundError(f"Missing: {base}")

    section_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("section_")])
    out: Dict[str, Any] = {
        "_generated": now_iso(),
        "_source": str(base),
        "_sections": [d.name for d in section_dirs],
    }

    for sd in section_dirs:
        kpath = sd / "key_numbers.json"
        if not kpath.exists():
            continue
        sec = read_json(kpath)

        # Merge keys; prefix collisions (rare)
        for k, v in sec.items():
            if k in out:
                out[f"{sd.name}.{k}"] = v
            else:
                out[k] = v

    # Normalise fraction strings "X/Y" -> add derived numeric keys
    for k, v in list(out.items()):
        if isinstance(v, str):
            m = FRACTION_RE.match(v)
            if m:
                out[f"{k}_n"] = int(m.group(1))
                out[f"{k}_d"] = int(m.group(2))

    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--out-json", default=None, help="Override output path (json)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)

    paper_numbers = collect_section_key_numbers(outputs_dir)

    out_json = Path(args.out_json) if args.out_json else (outputs_dir / "results" / "paper_numbers.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(paper_numbers, out_json)

    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
