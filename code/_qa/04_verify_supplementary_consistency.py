#!/usr/bin/env python3
"""
04_verify_supplementary_consistency.py

Audit supplementary table/data citations against produced artifacts.
Checks that every 'Supplementary Table/Data' mention in the manuscript
maps to an existing file, and flags orphaned files.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -------------------------
# Utilities
# -------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def split_main_vs_supp(text: str) -> Tuple[str, str]:
    # Heuristic split: everything before the last "Supplementary Information" is main.
    m = re.search(r"(?im)^\s*Supplementary\s+Information\s*$", text)
    if not m:
        return text, ""
    return text[: m.start()], text[m.start() :]


def _uniq_ints(xs: List[str]) -> List[int]:
    out = sorted({int(x) for x in xs})
    return out


def extract_supp_fig_nums(text: str) -> List[int]:
    # Matches: "Supplementary Fig. S4" or "Supplementary Fig S4" or "Fig. S4"
    pats = [
        r"\bSupplementary\s+Fig\.?\s*S(\d+)\b",
        r"\bFig\.?\s*S(\d+)\b",
        r"\bSupplementary\s+Figure\s*S(\d+)\b",
    ]
    hits: List[str] = []
    for p in pats:
        hits += re.findall(p, text, flags=re.IGNORECASE)
    return _uniq_ints(hits)


def extract_supp_table_nums(text: str) -> List[int]:
    pats = [
        r"\bSupplementary\s+Table\s+(\d+)\b",
        r"\bTable\s*S(\d+)\b",
    ]
    hits: List[str] = []
    for p in pats:
        hits += re.findall(p, text, flags=re.IGNORECASE)
    return _uniq_ints(hits)


def extract_supp_data_nums(text: str) -> List[int]:
    pats = [
        r"\bSupplementary\s+Data\s+(\d+)\b",
        r"\bData\s*S(\d+)\b",
    ]
    hits: List[str] = []
    for p in pats:
        hits += re.findall(p, text, flags=re.IGNORECASE)
    return _uniq_ints(hits)


def parse_num_from_filename(path: Path, kind: str) -> Optional[int]:
    name = path.name
    if kind == "ST":
        m = re.match(r"^ST(\d+)_", name, flags=re.IGNORECASE)
        return int(m.group(1)) if m else None
    if kind == "SD":
        # Try common patterns:
        #   SD3_*.csv
        #   Supplementary_Data_3_*.csv
        m = re.match(r"^SD(\d+)_", name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.match(r"^Supplementary[_\s-]*Data[_\s-]*(\d+)(?=[^0-9]|$)", name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None
    return None


def read_table_any(path: Path) -> pd.DataFrame:
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(path)
    if suf.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if suf.endswith(".csv"):
        return pd.read_csv(path)
    # Unknown tabular type: return empty
    return pd.DataFrame()


def column_profile(df: pd.DataFrame) -> List[Dict]:
    prof = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        row = {
            "column": c,
            "dtype": dtype,
            "n_missing": int(s.isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(s):
            # numeric scale (min/max) ignoring NA
            sn = pd.to_numeric(s, errors="coerce")
            row["min"] = float(sn.min()) if sn.notna().any() else None
            row["max"] = float(sn.max()) if sn.notna().any() else None
        else:
            row["min"] = None
            row["max"] = None
        prof.append(row)
    return prof


def audit_dir_schema(dir_path: Path, out_dir: Path, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_files = []
    rows_cols = []
    for p in sorted(dir_path.glob("*")):
        if not p.is_file():
            continue
        suf = "".join(p.suffixes).lower()
        if not (suf.endswith(".csv") or suf.endswith(".csv.gz") or suf.endswith(".parquet")):
            continue

        try:
            df = read_table_any(p)
            n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
            cols = list(df.columns)
        except Exception as e:
            n_rows, n_cols, cols = 0, 0, []
            df = pd.DataFrame()
            rows_files.append({
                "set": label, "file": p.name, "path": str(p), "bytes": p.stat().st_size,
                "sha256": sha256_file(p), "n_rows": None, "n_cols": None,
                "error": f"READ_FAIL: {e}"
            })
            continue

        rows_files.append({
            "set": label,
            "file": p.name,
            "path": str(p),
            "bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": "|".join(cols),
            "error": None,
        })

        for r in column_profile(df):
            rows_cols.append({
                "set": label,
                "file": p.name,
                **r,
            })

    df_files = pd.DataFrame(rows_files)
    df_cols = pd.DataFrame(rows_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    df_files.to_csv(out_dir / f"SCHEMA__{label}__files.csv", index=False)
    df_cols.to_csv(out_dir / f"SCHEMA__{label}__columns.csv", index=False)

    return df_files, df_cols


def main():
    ap = argparse.ArgumentParser(description="Audit supplementary citations vs produced sup_table/sup_data artifacts.")
    ap.add_argument("--outputs-dir", required=True, help="Path to outputs/")
    ap.add_argument("--manuscript", required=True, help="Path to manuscript txt (the one containing main + supplementary info)")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    ms_path = Path(args.manuscript)

    sup_table_dir = outputs_dir / "results" / "sup_table"
    sup_data_dir  = outputs_dir / "results" / "sup_data"
    out_audit_dir = outputs_dir / "results" / "supp_audit"

    text = read_text(ms_path)
    main_text, supp_text = split_main_vs_supp(text)

    # --- Extract citations (main vs supplementary section) ---
    cited_fig_main  = extract_supp_fig_nums(main_text)
    cited_fig_any   = extract_supp_fig_nums(text)
    defined_fig_supp = extract_supp_fig_nums(supp_text) if supp_text else []

    cited_tab_any = extract_supp_table_nums(text)
    cited_dat_any = extract_supp_data_nums(text)

    # --- Inventory produced artifacts ---
    st_files = []
    if sup_table_dir.exists():
        for p in sorted(sup_table_dir.glob("*")):
            if p.is_file() and re.match(r"^ST\d+_", p.name, flags=re.IGNORECASE):
                st_files.append((parse_num_from_filename(p, "ST"), p))
    sd_files = []
    if sup_data_dir.exists():
        for p in sorted(sup_data_dir.glob("*")):
            if p.is_file():
                n = parse_num_from_filename(p, "SD")
                if n is not None:
                    sd_files.append((n, p))

    st_nums_prod = sorted({n for n, _ in st_files if n is not None})
    sd_nums_prod = sorted({n for n, _ in sd_files if n is not None})

    # --- Build audit table ---
    rows = []

    def add_row(kind: str, num: int, cited: bool, produced: bool, paths: List[str], note: str = ""):
        rows.append({
            "kind": kind,
            "num": num,
            "cited_in_manuscript": cited,
            "produced_on_disk": produced,
            "paths": "|".join(paths),
            "note": note,
        })

    # Figures (defined in supplementary vs cited in main)
    fig_all = sorted(set(cited_fig_any) | set(defined_fig_supp))
    for n in fig_all:
        add_row(
            "SuppFig",
            n,
            cited=(n in cited_fig_main),
            produced=True,  # "produced" here means "defined in Supplementary Information block"
            paths=[],
            note=("DEFINED_IN_SUPP" if n in defined_fig_supp else "CITED_BUT_NOT_DEFINED_IN_SUPP"),
        )

    # Tables (map Supplementary Table N <-> STN_* file)
    tab_all = sorted(set(cited_tab_any) | set(st_nums_prod))
    for n in tab_all:
        paths = [str(p) for nn, p in st_files if nn == n]
        add_row(
            "SuppTable",
            n,
            cited=(n in cited_tab_any),
            produced=(n in st_nums_prod),
            paths=paths,
            note=("OK" if (n in cited_tab_any and n in st_nums_prod) else ""),
        )

    # Data (Supplementary Data N <-> SDN_*/Supplementary_Data_N* file)
    dat_all = sorted(set(cited_dat_any) | set(sd_nums_prod))
    for n in dat_all:
        paths = [str(p) for nn, p in sd_files if nn == n]
        add_row(
            "SuppData",
            n,
            cited=(n in cited_dat_any),
            produced=(n in sd_nums_prod),
            paths=paths,
            note=("OK" if (n in cited_dat_any and n in sd_nums_prod) else ""),
        )

    df_audit = pd.DataFrame(rows).sort_values(["kind", "num"]).reset_index(drop=True)
    out_audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_audit_dir / "SUPP_CITATION_AUDIT.csv"
    df_audit.to_csv(audit_path, index=False)

    # --- Schema summaries (variable names, types, scales) ---
    if sup_table_dir.exists():
        audit_dir_schema(sup_table_dir, out_audit_dir, "sup_table")
    if sup_data_dir.exists():
        audit_dir_schema(sup_data_dir, out_audit_dir, "sup_data")

    # --- Console summary + hard errors ---
    missing_tables = df_audit[(df_audit["kind"] == "SuppTable") & (df_audit["cited_in_manuscript"]) & (~df_audit["produced_on_disk"])]
    missing_data   = df_audit[(df_audit["kind"] == "SuppData")  & (df_audit["cited_in_manuscript"]) & (~df_audit["produced_on_disk"])]
    fig_undefined  = df_audit[(df_audit["kind"] == "SuppFig") & (df_audit["note"] == "CITED_BUT_NOT_DEFINED_IN_SUPP")]

    print("=" * 80)
    print("SUPPLEMENTARY AUDIT SUMMARY")
    print(f"Manuscript: {ms_path}")
    print(f"sup_table:  {sup_table_dir}  (exists={sup_table_dir.exists()})")
    print(f"sup_data:   {sup_data_dir}   (exists={sup_data_dir.exists()})")
    print(f"Audit CSV:  {audit_path}")
    print("-" * 80)
    print(f"Figures defined in Supplementary Information: {sorted(defined_fig_supp)}")
    print(f"Figures cited in MAIN TEXT:                 {sorted(cited_fig_main)}")
    print(f"Supplementary Tables cited (anywhere):      {sorted(cited_tab_any)}")
    print(f"Supplementary Tables produced (ST*):        {st_nums_prod}")
    print(f"Supplementary Data cited (anywhere):        {sorted(cited_dat_any)}")
    print(f"Supplementary Data produced (SD*):          {sd_nums_prod}")
    print("-" * 80)

    hard_fail = False
    if len(fig_undefined) > 0:
        hard_fail = True
        print("ERROR: Supplementary figure(s) cited in main text but NOT defined in Supplementary Information:")
        print(fig_undefined[["kind", "num", "note"]].to_string(index=False))
        print("-" * 80)
    if len(missing_tables) > 0:
        hard_fail = True
        print("ERROR: Supplementary table(s) cited but missing from outputs/results/sup_table:")
        print(missing_tables[["kind", "num"]].to_string(index=False))
        print("-" * 80)
    if len(missing_data) > 0:
        hard_fail = True
        print("ERROR: Supplementary data item(s) cited but missing from outputs/results/sup_data:")
        print(missing_data[["kind", "num"]].to_string(index=False))
        print("-" * 80)

    if hard_fail:
        raise SystemExit(2)

    print("OK: No missing cited supplementary artifacts detected.")
    print("=" * 80)


if __name__ == "__main__":
    main()
