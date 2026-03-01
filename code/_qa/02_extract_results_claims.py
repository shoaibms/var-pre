"""
02_extract_results_claims.py
Build a claims ledger from Results.txt and compare against paper_numbers.json.

Features:
- Locates claims in Results.txt and compares values (not just key existence checks).
- Context patterns are made robust by replacing embedded numeric literals with capture groups.
- Handles:
  - percent ("97.9%")
  - pp ("-16.2 pp")
  - ranges ("0.57–6.24" / "from X to Y")
  - counts ("6 of 14", "27/28")
  - fraction strings in paper_numbers ("10/14") via *_n derived keys (from 01_build_paper_numbers.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _to_float_num(s: str) -> float:
    """Handle unicode minus (−) before converting to float."""
    return float(s.replace("−", "-"))


def extract_rho_for_tag(line: str, tag: str) -> Optional[float]:
    """Extract ρ = <num> (<tag>) from a line, matching the view-tagged pattern."""
    rx = re.compile(
        rf"ρ\s*=\s*([−-]?[0-9]*\.?[0-9]+)\s*\([^)]*{re.escape(tag)}[^)]*\)",
        re.IGNORECASE,
    )
    m = rx.search(line)
    return _to_float_num(m.group(1)) if m else None


# -----------------------------
# Claim map: maps paper numbers to expected context in Results.txt.
# -----------------------------
CLAIM_MAP = [
    # Section 1
    {"id": "S1.01", "key": "rho_min", "context": r"ranged from ρ = ", "tolerance": 0.02},
    {"id": "S1.02", "key": "rho_max", "context": r"to ρ = ", "tolerance": 0.02},
    {"id": "S1.03", "key": "rho_mean", "context": r"cross-view mean", "tolerance": 0.02},
    {"id": "S1.04", "key": "DI_mean_min_at_k",
     "context": r"DI\)\s*ranged\s*from\s*([0-9.]+)\s*\(", "tolerance": 0.02},
    {"id": "S1.05", "key": "DI_mean_max_at_k",
     "context": r"DI\)\s*ranged\s*from\s*[0-9.]+\s*\([^)]*\)\s*to\s*([0-9.]+)", "tolerance": 0.02},
    {"id": "S1.06", "key": "n_anti_aligned_DI_gt_1", "context": r"exceeded 1\.0 in", "tolerance": 0.0},

    # MGX (appears as IBDMDB:MGX and IBD/MGX)
    {"id": "S1.07", "key": "exemplar_ibdmdb_MGX_DI",
     "context": r"(?:IBDMDB:MGX|IBD/MGX).*?DI\s*=\s*([0-9.]+)", "tolerance": 0.02},
    {"id": "S1.08", "key": "exemplar_ibdmdb_MGX_rho",
     "context": r"(?:IBDMDB:MGX|IBD/MGX).*?ρ\s*=\s*([−-]?[0-9.]+)", "tolerance": 0.02},
    # methylation (appears as MLOmics:methylation and MLO/methylation)
    {"id": "S1.09", "key": "exemplar_mlomics_methylation_DI",
     "context": r"(?:MLOmics:methylation|MLO/methylation).*?DI\s*=\s*([0-9.]+)", "tolerance": 0.02},
    {"id": "S1.10", "key": "exemplar_mlomics_methylation_rho",
     "context": r"(?:mlomics[:/]\s*methylation|MLO/?methylation).*?ρ\s*=\s*([−-]?[0-9.]+)", "tolerance": 0.02},

    # Section 2 examples
    {"id": "S2.01", "key": "cross_model_pearson_r",
     "context": r"Pearson correlation.*?r\s*=\s*([0-9.]+)", "tolerance": 0.01},
    {"id": "S2.02", "key": "cross_model_spearman_rho",
     "context": r"Spearman\s*ρ\s*=\s*([0-9.]+)", "tolerance": 0.01},
    {"id": "S2.03", "key": "direction_agreement_count_n", "context": r"direction-of-effect agreement", "tolerance": 0.0},
    {"id": "S2.04", "key": "shap_beats_var_count_n",
     "context": r"outperformed variance selection in\s*([0-9]+)\s*of\s*([0-9]+)", "tolerance": 0},

    # Section 4
    {"id": "S4.01", "key": "ablation_worst_topshap_ba", "context": r"TopSHAP", "tolerance": 0.01},
    {"id": "S4.02", "key": "meth_deficit_k1_pp", "context": r"K=1%|K = 1%|K=1 percent", "tolerance": 0.6},
    {"id": "S4.03", "key": "gene_divergence_gt_null_count",
     "context": r"more divergent than expected.*?\(([0-9]+)\s*/\s*8\)", "tolerance": 0},
    {"id": "S4.04", "key": "obs_null_CR_mean", "context": r"obs/null", "tolerance": 0.05},
]


NUM_TOKEN = r"([−-]?[0-9]+(?:\.[0-9]+)?)"
RE_NUMERIC_LITERAL = re.compile(r"[−-]?\d+(?:\.\d+)?")  # finds numerics inside contexts
RE_PP = re.compile(r"([−-]?\d+(?:\.\d+)?)\s*pp", re.IGNORECASE)
RE_PCT = re.compile(r"([−-]?\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
RE_OF = re.compile(r"(\d+)\s*(?:of|/)\s*(\d+)", re.IGNORECASE)
RE_PAREN = {
    "random": re.compile(r"([0-9]*\.?[0-9]+)\s*\(Random\)", re.IGNORECASE),
    "topvar": re.compile(r"([0-9]*\.?[0-9]+)\s*\(TopVar\)", re.IGNORECASE),
    "topshap": re.compile(r"([0-9]*\.?[0-9]+)\s*\(TopSHAP\)", re.IGNORECASE),
}


def _norm_minus(s: str) -> str:
    return s.replace("−", "-").replace("–", "-").replace("—", "-")


def _norm_ws(s: str) -> str:
    """Collapse all whitespace (incl. thin / non-breaking spaces) to single regular space."""
    return re.sub(r"\s+", " ", s).strip()


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(str(x).strip())
    except Exception:
        return None


def coerce_paper_value(v: Any) -> Union[float, int, List[float], str, None]:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        a = _to_float(v[0])
        b = _to_float(v[1])
        if a is not None and b is not None:
            return [a, b]
        return None
    if isinstance(v, str):
        s = v.strip()
        # already normalised fractions should be handled via *_n keys, but keep fallback
        m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", s)
        if m:
            return int(m.group(1))
        f = _to_float(_norm_minus(s))
        if f is not None:
            return f
        return s
    return None


def _norm_spaces(s: str) -> str:
    return s.replace("\u2009"," ").replace("\u202f"," ").replace("\xa0"," ")


def make_loose_context_regex(context: str) -> re.Pattern:
    cx = _norm_spaces(context)
    return re.compile(cx, re.IGNORECASE | re.DOTALL)


def find_first_matching_line(lines: List[str], rx: re.Pattern) -> Tuple[Optional[int], Optional[re.Match]]:
    for i, line in enumerate(lines):
        # Normalize thin / non-breaking spaces so they never block a match
        line = line.replace("\u2009", " ").replace("\u202f", " ").replace("\xa0", " ")
        line_norm = _norm_ws(_norm_minus(line))
        m = rx.search(line_norm)
        if m:
            return i, m
    return None, None


def extract_value_for_key(key: str, line: str, match: Optional[re.Match]) -> Union[float, int, List[float], None]:
    if match and match.groups():
        g = match.group(1)
        if g is None:
            return None
        return float(g) if "." in str(g) else int(g)

    s = _norm_ws(_norm_minus(line))

    # --- portion of the line AFTER the context-match (where the value lives) ---
    after = s[match.end():] if match else s

    # ---- helper: last non-None capture group from the context regex ----
    def _first_group_number() -> Optional[str]:
        if match is None:
            return None
        # Use captured groups first (context-anchored).
        # If multiple groups, take the last numeric-looking one.
        if match.groups():
            for g in reversed(match.groups()):
                if g is not None:
                    return g
        return None

    # If key is a range, extract two numbers after match
    if "range" in key.lower():
        nums = [float(n) for n in RE_NUMERIC_LITERAL.findall(after)]
        if len(nums) >= 2:
            return [nums[0], nums[-1]]
        return None

    # counts like "6 of 14" or "27/28"
    if key.lower().startswith("n_") or "count" in key.lower() or key.lower().endswith("_n"):
        m_of = RE_OF.search(after)
        if m_of:
            return int(m_of.group(1))
        # fallback: first integer-looking token after match
        nums = RE_NUMERIC_LITERAL.findall(after)
        for n in nums:
            if re.match(r"^\d+$", _norm_minus(n)):
                return int(n)
        # last resort: capture group heuristics
        if match and match.groups():
            groups = [g for g in match.groups() if g is not None]
            for g in groups:
                gg = _norm_minus(g)
                if re.match(r"^\d+$", gg):
                    return int(gg)
            g0 = _to_float(_norm_minus(groups[-1]))
            if g0 is not None:
                return int(round(g0))
        return None

    # percent points / deficit
    if "pp" in key.lower() or "delta" in key.lower() or "deficit" in key.lower() or "adv" in key.lower():
        mpp = RE_PP.search(after)
        if mpp:
            return float(mpp.group(1))
        # fallback: first numeric after match position
        nums = RE_NUMERIC_LITERAL.findall(after)
        if nums:
            return float(_norm_minus(nums[0]))
        # last resort: capture groups (value was embedded in context itself)
        if match and match.groups():
            groups = [g for g in match.groups() if g is not None]
            if groups:
                g = _to_float(_norm_minus(groups[-1]))
                if g is not None:
                    return float(g)
        return None

    # parenthesized selectors (search full normalised line — the tag is specific enough)
    lk = key.lower()
    for tag, rx in RE_PAREN.items():
        if tag in lk:
            mm = rx.search(s)
            if mm:
                return float(mm.group(1))

    # percents — search after match
    mp = RE_PCT.search(after)
    if mp and ("pct" in lk or "percent" in lk or "confidence" in lk):
        return float(mp.group(1))

    # default: prefer capture groups from context regex, else first number after match
    gn = _first_group_number()
    if gn is not None:
        v = _to_float(_norm_minus(gn))
        if v is not None:
            return float(v)
    nums = RE_NUMERIC_LITERAL.findall(after)
    return float(_norm_minus(nums[0])) if nums else None


def compare_values(paper_v: Any, text_v: Any, tol: float) -> Tuple[str, str]:
    pv = coerce_paper_value(paper_v)

    if pv is None or text_v is None:
        return "NOT_FOUND", "could not coerce value(s)"

    # range compare
    if isinstance(pv, list) and isinstance(text_v, list) and len(pv) == 2 and len(text_v) == 2:
        d0 = abs(pv[0] - text_v[0])
        d1 = abs(pv[1] - text_v[1])
        ok = (d0 <= tol) and (d1 <= tol)
        return ("MATCH" if ok else "MISMATCH", f"range_diffs=({d0:.6g},{d1:.6g})")

    # numeric compare
    if isinstance(pv, (int, float)) and isinstance(text_v, (int, float)):
        d = abs(float(pv) - float(text_v))
        return ("MATCH" if d <= tol else "MISMATCH", f"abs_diff={d:.6g}")

    # string compare (rare) — do not fail hard
    return ("INFO", "non-numeric compare (skipped)")


def extract_number_inventory(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Extract every numeric literal from Results.txt with line context.
    Uses RE_NUMERIC_LITERAL already defined in this script.
    """
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        for m in RE_NUMERIC_LITERAL.finditer(line):
            raw = m.group(0)
            v = _to_float(_norm_minus(raw))
            out.append({
                "line_no_1based": i + 1,
                "raw": raw,
                "value": v,
                "text_line": line,
            })
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--results-txt", default=None)
    ap.add_argument("--paper-numbers", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--audit-all-numbers", action="store_true",
                    help="Also write an inventory of all numeric literals found in Results.txt.")
    ap.add_argument("--out-inventory-csv", default=None,
                    help="Optional path for the numeric inventory CSV.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)

    results_txt = Path(args.results_txt) if args.results_txt else (outputs_dir / "results" / "Results.txt")
    paper_json = Path(args.paper_numbers) if args.paper_numbers else (outputs_dir / "results" / "paper_numbers.json")
    out_csv = Path(args.out_csv) if args.out_csv else (outputs_dir / "results" / "results_claims_check.csv")

    if not paper_json.exists():
        raise FileNotFoundError(f"Missing paper_numbers.json: {paper_json}")
    paper = json.loads(paper_json.read_text(encoding="utf-8"))

    lines: List[str] = []
    if results_txt.exists():
        lines = results_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

    rows: List[Dict[str, Any]] = []

    claim_hits_by_line = defaultdict(list)  # line_no (0-based) -> [claim_id, ...]

    for c in CLAIM_MAP:
        cid = c["id"]
        key = c["key"]
        ctx = c["context"]
        tol = float(c.get("tolerance", 0.0))

        paper_has = key in paper
        paper_val = paper.get(key, None)

        rx = make_loose_context_regex(ctx)
        line_no, m = (None, None)
        text_val = None
        text_line = None

        if lines:
            line_no, m = find_first_matching_line(lines, rx)
            if line_no is not None:
                text_line = lines[line_no]
                text_val = extract_value_for_key(key, text_line, m)

            if line_no is not None:
                claim_hits_by_line[line_no].append(cid)

                # Special-case: extract ρ using view-tagged pattern for methylation
                if key == "exemplar_mlomics_methylation_rho":
                    v = extract_rho_for_tag(text_line, "mlomics:methylation")
                    if v is not None:
                        text_val = v

        if not paper_has:
            status = "UNRESOLVED"
            detail = "missing key in paper_numbers"
        elif not lines:
            status = "NO_RESULTS_TXT"
            detail = "Results.txt not provided/found; only key coverage checked"
        elif line_no is None:
            status = "NOT_FOUND"
            detail = "context not found in Results.txt"
        else:
            status, detail = compare_values(paper_val, text_val, tol)

        rows.append({
            "claim_id": cid,
            "key": key,
            "status": status,
            "tolerance": tol,
            "paper_value": paper_val,
            "text_value": text_val,
            "line_no_1based": (line_no + 1) if line_no is not None else None,
            "text_line": text_line,
            "detail": detail,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_csv}")

    if args.audit_all_numbers and lines:
        inv_csv = Path(args.out_inventory_csv) if args.out_inventory_csv else (
            outputs_dir / "results" / "results_numbers_inventory.csv"
        )
        inv_rows = extract_number_inventory(lines)

        # annotate: whether this number is on a line that is already covered by a registered claim
        for r in inv_rows:
            ln0 = (r["line_no_1based"] - 1)
            hits = claim_hits_by_line.get(ln0, [])
            r["covered_by_claim_ids"] = ";".join(hits) if hits else ""

        inv_csv.parent.mkdir(parents=True, exist_ok=True)
        with inv_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(inv_rows[0].keys()) if inv_rows else [
                "line_no_1based", "raw", "value", "text_line", "covered_by_claim_ids"
            ])
            w.writeheader()
            if inv_rows:
                w.writerows(inv_rows)

        print(f"Wrote: {inv_csv}")
        print(f"Inventory: {len(inv_rows)} numeric literals found in Results.txt")


if __name__ == "__main__":
    main()
