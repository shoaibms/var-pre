#!/usr/bin/env python3
r"""
audit_supplementary_assets.py

Terminal audit of supplementary tables/data:
- headers, inferred variable types, missingness, basic stats
- optional manuscript scan for "Supplementary Table/Data" mentions
- suggested manuscript citation anchors per file (heuristic)

Usage (Windows PowerShell):
  python code\compute\audit_supplementary_assets_v3.py `
    --sup-table-dir "<path-to-outputs>\results\sup_table" `
    --sup-data-dir  "<path-to-outputs>\results\sup_data" `
    --manuscript    "<path-to-manuscript-txt>" `
    --out-md        "<path-to-outputs>\results\sup_audit.md"

Notes:
- For very large files, quantiles/medians may be computed from a sample for speed.
- Script is read-only; it does not modify any files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers: file handling
# -----------------------------

TABULAR_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".feather"}
COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zip"}  # pandas can read some directly

SUP_LABEL_RE = re.compile(r"(?:supp(?:lementary)?[_\-\s]*)?(?:table|data)[_\-\s]*(\d+)", re.IGNORECASE)
MANUS_SUP_RE = re.compile(r"\bSupplementary\s+(Table|Data)\s*([0-9]+)\b", re.IGNORECASE)

def sizeof_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except Exception:
        return float("nan")

def is_probably_tabular(p: Path) -> bool:
    suf = "".join(p.suffixes).lower()
    # handle e.g. .csv.gz
    base = p.suffixes[-2].lower() if len(p.suffixes) >= 2 and p.suffixes[-1].lower() in COMPRESSED_SUFFIXES else p.suffix.lower()
    return base in TABULAR_EXTS

def detect_delim_from_name(p: Path) -> str:
    name = p.name.lower()
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return "\t"
    # default
    return ","

def read_manuscript_text(path: Optional[Path]) -> str:
    if path is None:
        return ""
    if not path.exists():
        return ""
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".docx":
        try:
            from docx import Document  # python-docx
        except Exception:
            print("[WARN] python-docx not installed; cannot parse .docx manuscript.", file=sys.stderr)
            return ""
        doc = Document(str(path))
        parts = []
        for para in doc.paragraphs:
            t = (para.text or "").strip()
            if t:
                parts.append(t)
        return "\n".join(parts)
    # fallback
    return path.read_text(encoding="utf-8", errors="ignore")


# -----------------------------
# Column typing + stats
# -----------------------------

def infer_kind(series: pd.Series) -> str:
    s = series
    dt = s.dtype

    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime"
    if pd.api.types.is_numeric_dtype(dt):
        # split int/float for readability
        return "int" if pd.api.types.is_integer_dtype(dt) else "float"

    # object / string: decide categorical vs text via cardinality + avg length
    nonnull = s.dropna()
    if nonnull.empty:
        return "empty"
    # attempt datetime coercion (common in exported CSVs)
    if nonnull.sample(min(len(nonnull), 2000), random_state=0).astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.8:
        return "datetime_like"

    nunique = nonnull.nunique(dropna=True)
    n = len(nonnull)
    # heuristic thresholds
    if nunique <= min(50, max(10, int(0.05 * n))):
        return "categorical"
    # text length heuristic
    try:
        lens = nonnull.astype(str).str.len()
        if float(lens.mean()) >= 30:
            return "text"
    except Exception:
        pass
    return "categorical_highcard"


def numeric_summary(series: pd.Series, quantiles: Iterable[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[str, Optional[float]]:
    s = pd.to_numeric(series, errors="coerce")
    out = {}
    if s.dropna().empty:
        for k in ["min", "max", "mean", "std"] + [f"q{int(q*100):02d}" for q in quantiles]:
            out[k] = None
        return out
    out["min"] = float(np.nanmin(s.values))
    out["max"] = float(np.nanmax(s.values))
    out["mean"] = float(np.nanmean(s.values))
    out["std"] = float(np.nanstd(s.values, ddof=1)) if np.sum(np.isfinite(s.values)) > 1 else 0.0
    try:
        qs = s.quantile(list(quantiles), interpolation="linear")
        for q, v in qs.items():
            out[f"q{int(q*100):02d}"] = float(v) if pd.notna(v) else None
    except Exception:
        for q in quantiles:
            out[f"q{int(q*100):02d}"] = None
    return out


def categorical_summary(series: pd.Series, top_k: int = 5) -> Dict[str, object]:
    s = series.dropna()
    out: Dict[str, object] = {}
    out["nunique"] = int(s.nunique(dropna=True)) if not s.empty else 0
    if s.empty:
        out["top"] = []
        return out
    vc = s.astype(str).value_counts(dropna=True).head(top_k)
    out["top"] = [(idx, int(val)) for idx, val in vc.items()]
    return out


@dataclass
class ColumnProfile:
    name: str
    pandas_dtype: str
    kind: str
    n: int
    missing_n: int
    missing_pct: float
    nonmissing_n: int
    nunique: Optional[int] = None
    num_min: Optional[float] = None
    num_max: Optional[float] = None
    num_mean: Optional[float] = None
    num_std: Optional[float] = None
    num_q05: Optional[float] = None
    num_q50: Optional[float] = None
    num_q95: Optional[float] = None
    top_levels: Optional[str] = None  # compact string


@dataclass
class FileRecord:
    """Accumulated per-file stats for post-audit summaries."""
    bucket: str              # "sup_table" or "sup_data"
    path: Path
    label: Optional[str]     # e.g. "Supplementary Table 1"
    theme: Optional[str]
    rows: int = 0
    cols: int = 0
    total_cells: int = 0
    total_missing: int = 0
    missing_pct: float = 0.0
    size_mb: float = 0.0
    profiles: Optional[List[ColumnProfile]] = None
    view_values: Optional[set] = None       # unique view identifiers found
    view_col_name: Optional[str] = None     # which column held views
    error: Optional[str] = None


def profile_dataframe(
    df: pd.DataFrame,
    compute_quantiles: bool = True,
    top_k: int = 4,
) -> List[ColumnProfile]:
    profiles: List[ColumnProfile] = []
    n = int(len(df))

    for col in df.columns:
        s = df[col]
        missing_n = int(s.isna().sum())
        nonmissing_n = n - missing_n
        missing_pct = (missing_n / n * 100.0) if n > 0 else 0.0

        kind = infer_kind(s)
        pd_dtype = str(s.dtype)

        cp = ColumnProfile(
            name=str(col),
            pandas_dtype=pd_dtype,
            kind=kind,
            n=n,
            missing_n=missing_n,
            missing_pct=float(missing_pct),
            nonmissing_n=int(nonmissing_n),
        )

        if kind in {"int", "float"}:
            q = (0.05, 0.5, 0.95) if compute_quantiles else ()
            summ = numeric_summary(s, quantiles=q if q else (0.5,))
            cp.num_min = summ.get("min")
            cp.num_max = summ.get("max")
            cp.num_mean = summ.get("mean")
            cp.num_std = summ.get("std")
            cp.num_q05 = summ.get("q05") if compute_quantiles else None
            cp.num_q50 = summ.get("q50") if compute_quantiles else summ.get("q50")
            cp.num_q95 = summ.get("q95") if compute_quantiles else None

        elif kind in {"categorical", "categorical_highcard", "text", "datetime_like", "bool"}:
            cs = categorical_summary(s, top_k=top_k)
            cp.nunique = int(cs.get("nunique", 0))
            top = cs.get("top", [])
            if top:
                cp.top_levels = "; ".join([f"{k} ({v})" for k, v in top])
        elif kind == "datetime":
            try:
                ss = pd.to_datetime(s, errors="coerce")
                cp.num_min = None
                cp.num_max = None
                cp.top_levels = f"min={ss.min()} max={ss.max()}"
            except Exception:
                pass

        profiles.append(cp)

    return profiles


# -----------------------------
# Big-file streaming (numeric stats + missingness)
# -----------------------------

@dataclass
class RunningNum:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    minv: float = float("inf")
    maxv: float = float("-inf")

    def update(self, x: np.ndarray) -> None:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        # update min/max
        self.minv = float(min(self.minv, float(np.min(x))))
        self.maxv = float(max(self.maxv, float(np.max(x))))

        # Welford batch update
        batch_n = int(x.size)
        batch_mean = float(np.mean(x))
        batch_M2 = float(np.sum((x - batch_mean) ** 2))

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_M2
            return

        delta = batch_mean - self.mean
        new_n = self.n + batch_n
        self.mean = self.mean + delta * (batch_n / new_n)
        self.M2 = self.M2 + batch_M2 + delta**2 * (self.n * batch_n / new_n)
        self.n = new_n

    @property
    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        return float(math.sqrt(self.M2 / (self.n - 1)))


def stream_profile_csv(
    path: Path,
    sep: str,
    chunksize: int = 200_000,
    max_cat_levels: int = 200,
    sample_rows_for_types: int = 20_000,
) -> Tuple[Dict[str, Dict[str, object]], int]:
    """
    Stream through a CSV/TSV to compute:
      - missing counts for all columns
      - numeric min/max/mean/std for numeric-ish columns
      - categorical top counts only for low-card columns (based on sample)

    Returns:
      col_stats dict + total_rows
    """
    # sample for type inference
    sample = pd.read_csv(path, sep=sep, nrows=sample_rows_for_types, low_memory=False)
    kinds = {c: infer_kind(sample[c]) for c in sample.columns}

    missing = {c: 0 for c in sample.columns}
    running = {c: RunningNum() for c in sample.columns if kinds[c] in {"int", "float"}}
    cat_counters: Dict[str, Counter] = {c: Counter() for c in sample.columns if kinds[c] in {"categorical", "bool"}}

    total_rows = 0
    for chunk in pd.read_csv(path, sep=sep, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        for c in chunk.columns:
            missing[c] += int(chunk[c].isna().sum())

        # numeric updates
        for c, rn in running.items():
            x = pd.to_numeric(chunk[c], errors="coerce").to_numpy(dtype=float, copy=False)
            rn.update(x)

        # categorical updates (low-card only)
        for c in list(cat_counters.keys()):
            vc = chunk[c].dropna().astype(str).value_counts()
            # stop tracking if explodes
            if len(vc) > max_cat_levels:
                del cat_counters[c]
                continue
            cat_counters[c].update({k: int(v) for k, v in vc.items()})

    # assemble stats
    out: Dict[str, Dict[str, object]] = {}
    for c in sample.columns:
        out[c] = {
            "kind": kinds[c],
            "missing_n": missing[c],
            "nonmissing_n": total_rows - missing[c],
        }
        if c in running and running[c].n > 0:
            out[c].update({
                "min": running[c].minv if np.isfinite(running[c].minv) else None,
                "max": running[c].maxv if np.isfinite(running[c].maxv) else None,
                "mean": running[c].mean,
                "std": running[c].std,
            })
        if c in cat_counters:
            top = cat_counters[c].most_common(5)
            out[c]["top"] = top

    return out, total_rows


# -----------------------------
# Manuscript mention scan + citation suggestions
# -----------------------------

def extract_manuscript_mentions(text: str) -> Dict[str, List[str]]:
    """
    Returns dict: "Supplementary Table 1" -> [snippets...]
    """
    out: Dict[str, List[str]] = defaultdict(list)
    if not text:
        return out
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        for m in MANUS_SUP_RE.finditer(ln):
            kind = m.group(1).title()  # Table/Data
            num = m.group(2)
            key = f"Supplementary {kind} {num}"
            # store local context
            ctx = ln.strip()
            if len(ctx) < 10:
                # take next non-empty
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip():
                        ctx = (ctx + " " + lines[j].strip()).strip()
                        break
            out[key].append(ctx[:240])
    return out


def guess_theme_from_columns(cols: List[str]) -> str:
    C = {c.lower() for c in cols}
    if {"dataset", "view"}.issubset(C) and ({"n_features", "features", "n_samples", "samples"} & C):
        return "datasets/views inventory (likely Supplementary Table 1; cite in Methods: Datasets)"
    if any("di" == c or c.startswith("di_") for c in C) or "spearman" in C or "rho" in C:
        return "alignment/regime table (cite in Results around Fig. 1–2; regime_map/DI summaries)"
    if any("topvar" in c or "topshap" in c or "random" in c for c in C) and ("delta" in "".join(C) or "pp" in "".join(C)):
        return "ablation/performance effects (cite in Results: Fig. 4 and/or Supp Fig. S3)"
    if any(k in C for k in ["vsa", "eta_es", "pcla", "sas"]):
        return "VAD metrics/validation (cite in Results: VAD section; Fig. 6 + Supp Fig. S4)"
    if any("jaccard" in c for c in C) or any("pathway" in c for c in C) or any("gprofiler" in c for c in C):
        return "biology overlap/enrichment (cite in Results: biological interpretation section; Fig. 5)"
    if any("perm" in c for c in C) or any("shuffle" in c for c in C) or any("null" in c for c in C):
        return "permutation/null controls (cite in Methods: permutation-based null controls; Supp panels as relevant)"
    return "general supplementary output (cite where first referenced; check filenames + figure/table callouts)"


def extract_sup_number_from_filename(p: Path) -> Optional[int]:
    m = SUP_LABEL_RE.search(p.stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # also try "S1", "S2" patterns
    m2 = re.search(r"(?:^|[_\-\s])S(\d+)(?:[_\-\s]|$)", p.stem, flags=re.IGNORECASE)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    # handle ST1, ST2, ..., SD1, SD2, ... filenames
    m3 = re.search(r"^(?:ST|SD)(\d+)(?:[_\-\s]|$)", p.stem, flags=re.IGNORECASE)
    if m3:
        try:
            return int(m3.group(1))
        except Exception:
            return None
    return None


# -----------------------------
# Main audit logic
# -----------------------------

def iter_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            # skip obvious junk
            if p.name.startswith("~$"):
                continue
            files.append(p)
    return sorted(files)


def print_file_header(p: Path) -> None:
    print("\n" + "=" * 110)
    print(f"FILE: {p}")
    print(f"SIZE: {sizeof_mb(p):.2f} MB")
    print("=" * 110)


def format_profiles_table(profiles: List[ColumnProfile], max_rows: int = 200) -> str:
    # compact DataFrame
    rows = []
    for cp in profiles[:max_rows]:
        rows.append({
            "col": cp.name,
            "dtype": cp.pandas_dtype,
            "kind": cp.kind,
            "missing_n": cp.missing_n,
            "missing_%": f"{cp.missing_pct:.2f}",
            "nunique": cp.nunique if cp.nunique is not None else "",
            "min": f"{cp.num_min:.4g}" if cp.num_min is not None else "",
            "max": f"{cp.num_max:.4g}" if cp.num_max is not None else "",
            "mean": f"{cp.num_mean:.4g}" if cp.num_mean is not None else "",
            "std": f"{cp.num_std:.4g}" if cp.num_std is not None else "",
            "q05": f"{cp.num_q05:.4g}" if cp.num_q05 is not None else "",
            "q50": f"{cp.num_q50:.4g}" if cp.num_q50 is not None else "",
            "q95": f"{cp.num_q95:.4g}" if cp.num_q95 is not None else "",
            "top": (cp.top_levels[:80] + "…") if cp.top_levels and len(cp.top_levels) > 80 else (cp.top_levels or ""),
        })
    df = pd.DataFrame(rows)
    # avoid overly-wide prints
    with pd.option_context("display.max_colwidth", 120, "display.width", 220):
        return df.to_string(index=False)


def df_shape_safe(df: pd.DataFrame) -> Tuple[int, int]:
    try:
        return int(df.shape[0]), int(df.shape[1])
    except Exception:
        return (0, 0)


def audit_tabular_file(
    p: Path,
    big_file_mb: float,
    chunksize: int,
    sample_rows: int,
) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    """
    Returns (df_preview_or_none, meta)
    """
    meta: Dict[str, object] = {}
    suf = "".join(p.suffixes).lower()
    base_ext = p.suffixes[-2].lower() if len(p.suffixes) >= 2 and p.suffixes[-1].lower() in COMPRESSED_SUFFIXES else p.suffix.lower()

    # choose reader
    try:
        if base_ext in {".csv", ".tsv", ".txt"}:
            sep = detect_delim_from_name(p) if base_ext != ".tsv" else "\t"
            meta["reader"] = f"read_csv(sep={repr(sep)})"

            if sizeof_mb(p) >= big_file_mb:
                # stream for missing + numeric stats (exact) + limited top categories
                col_stats, total_rows = stream_profile_csv(p, sep=sep, chunksize=chunksize, sample_rows_for_types=min(sample_rows, 20000))
                meta["rows"] = total_rows
                meta["cols"] = len(col_stats)
                meta["streamed"] = True
                meta["stream_col_stats"] = col_stats

                # preview sample
                df_prev = pd.read_csv(p, sep=sep, nrows=min(sample_rows, 2000), low_memory=False)
                return df_prev, meta

            # small enough: load full (or modest cap)
            df = pd.read_csv(p, sep=sep, low_memory=False)
            r, c = df_shape_safe(df)
            meta["rows"] = r
            meta["cols"] = c
            meta["streamed"] = False

            # preview = either full or head
            if r > sample_rows:
                return df.head(sample_rows).copy(), meta
            return df, meta

        if base_ext in {".xlsx", ".xls"}:
            meta["reader"] = "read_excel"
            df = pd.read_excel(p)
            r, c = df_shape_safe(df)
            meta["rows"] = r
            meta["cols"] = c
            meta["streamed"] = False
            return (df.head(sample_rows).copy() if r > sample_rows else df), meta

        if base_ext == ".parquet":
            meta["reader"] = "read_parquet"
            df = pd.read_parquet(p)
            r, c = df_shape_safe(df)
            meta["rows"] = r
            meta["cols"] = c
            meta["streamed"] = False
            return (df.head(sample_rows).copy() if r > sample_rows else df), meta

        if base_ext == ".feather":
            meta["reader"] = "read_feather"
            df = pd.read_feather(p)
            r, c = df_shape_safe(df)
            meta["rows"] = r
            meta["cols"] = c
            meta["streamed"] = False
            return (df.head(sample_rows).copy() if r > sample_rows else df), meta

    except Exception as e:
        meta["error"] = str(e)
        return None, meta

    meta["error"] = f"Unsupported tabular extension: {base_ext}"
    return None, meta


def audit_non_tabular(p: Path) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    suf = "".join(p.suffixes).lower()

    try:
        if suf.endswith(".json"):
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            meta["type"] = "json"
            if isinstance(obj, dict):
                meta["keys"] = list(obj.keys())[:50]
            elif isinstance(obj, list):
                meta["list_len"] = len(obj)
            return meta

        if suf.endswith(".npz"):
            meta["type"] = "npz"
            with np.load(p, allow_pickle=False) as z:
                meta["arrays"] = {k: {"shape": z[k].shape, "dtype": str(z[k].dtype)} for k in z.files}
            return meta

        # add more handlers if needed (e.g., .pkl is intentionally not supported)
        meta["type"] = "unknown"
        return meta

    except Exception as e:
        meta["error"] = str(e)
        return meta


# -----------------------------
# Post-audit aggregate analyses
# -----------------------------

def _extract_view_column(df: pd.DataFrame) -> Tuple[Optional[str], Optional[set]]:
    """Find the view-identifier column and return (col_name, unique_values)."""
    candidates = ["view", "view_key", "view_id", "dataset_view"]
    for c in df.columns:
        if c.lower() in candidates:
            return c, set(df[c].dropna().unique())
    return None, None


def print_grand_summary(records: List[FileRecord], label: str) -> List[str]:
    """Print + return md lines for an aggregated summary table."""
    ok = [r for r in records if r.error is None]
    err = [r for r in records if r.error is not None]
    lines: List[str] = []

    hdr = f"\n{'#' * 110}\n  {label} — GRAND SUMMARY\n{'#' * 110}"
    print(hdr)

    print(f"\n  {'Label':<30s} {'File':<42s} {'Rows':>8s} {'Cols':>5s} {'Miss%':>7s} {'Size':>9s}")
    print(f"  {'—' * 30} {'—' * 42} {'—' * 8} {'—' * 5} {'—' * 7} {'—' * 9}")

    total_rows = total_cells = total_miss = 0
    for r in ok:
        fn = r.path.name[:40] + ".." if len(r.path.name) > 42 else r.path.name
        lbl = (r.label or "")[:28] + ".." if r.label and len(r.label) > 30 else (r.label or "—")
        print(f"  {lbl:<30s} {fn:<42s} {r.rows:>8,d} {r.cols:>5d} {r.missing_pct:>6.1f}% {r.size_mb:>8.2f}M")
        total_rows += r.rows
        total_cells += r.total_cells
        total_miss += r.total_missing

    pct = 100.0 * total_miss / total_cells if total_cells else 0
    print(f"  {'—' * 30} {'—' * 42} {'—' * 8} {'—' * 5} {'—' * 7}")
    print(f"  {'TOTAL':<30s} {'':42s} {total_rows:>8,d} {'':>5s} {pct:>6.1f}%")

    if err:
        print(f"\n  [WARN] NOT FOUND / ERROR ({len(err)}):")
        for r in err:
            print(f"    - {r.path.name}: {r.error}")

    # md
    lines.append(f"## {label} — Grand Summary\n")
    lines.append(f"| Label | File | Rows | Cols | Miss% | Size |")
    lines.append(f"|-------|------|-----:|-----:|------:|-----:|")
    for r in ok:
        lines.append(f"| {r.label or '—'} | `{r.path.name}` | {r.rows:,} | {r.cols} | {r.missing_pct:.1f}% | {r.size_mb:.2f} MB |")
    lines.append(f"| **TOTAL** | | **{total_rows:,}** | | **{pct:.1f}%** | |")
    lines.append("")
    return lines


def print_missing_data_heatmap(records: List[FileRecord], top_n: int = 20) -> List[str]:
    """Print + return md lines for worst missing-data columns across all files."""
    lines: List[str] = []
    hdr = f"\n{'#' * 110}\n  COLUMNS WITH MOST MISSING DATA (top {top_n})\n{'#' * 110}"
    print(hdr)

    rows = []
    for r in records:
        if r.error is not None or r.profiles is None:
            continue
        for cp in r.profiles:
            if cp.missing_n > 0:
                rows.append({
                    "file": r.label or r.path.name,
                    "column": cp.name,
                    "missing": cp.missing_n,
                    "total": cp.n,
                    "pct": cp.missing_pct,
                })

    if not rows:
        print("  [OK] No missing data found in any file!")
        return lines

    rows.sort(key=lambda x: x["pct"], reverse=True)
    print(f"\n  {'File':<30s} {'Column':<40s} {'Missing':>8s} {'Total':>8s} {'%':>7s}")
    print(f"  {'—' * 30} {'—' * 40} {'—' * 8} {'—' * 8} {'—' * 7}")
    for row in rows[:top_n]:
        col_d = row["column"][:38] + ".." if len(row["column"]) > 40 else row["column"]
        file_d = row["file"][:28] + ".." if len(row["file"]) > 30 else row["file"]
        print(f"  {file_d:<30s} {col_d:<40s} {row['missing']:>8,d} {row['total']:>8,d} {row['pct']:>6.1f}%")

    lines.append("## Columns with most missing data\n")
    lines.append("| File | Column | Missing | Total | % |")
    lines.append("|------|--------|--------:|------:|--:|")
    for row in rows[:top_n]:
        lines.append(f"| {row['file']} | `{row['column']}` | {row['missing']:,} | {row['total']:,} | {row['pct']:.1f}% |")
    lines.append("")
    return lines


def print_dtype_distribution(records: List[FileRecord]) -> List[str]:
    """Print + return md lines for column data-type distribution."""
    lines: List[str] = []
    hdr = f"\n{'#' * 110}\n  DATA TYPE DISTRIBUTION (all columns)\n{'#' * 110}"
    print(hdr)

    counts: Counter = Counter()
    for r in records:
        if r.profiles is None:
            continue
        for cp in r.profiles:
            counts[cp.kind] += 1

    total = sum(counts.values())
    print(f"\n  {'Kind':<22s} {'Count':>6s} {'%':>7s}")
    print(f"  {'—' * 22} {'—' * 6} {'—' * 7}")
    for kind, cnt in counts.most_common():
        pct = 100.0 * cnt / total if total else 0
        print(f"  {kind:<22s} {cnt:>6d} {pct:>6.1f}%")
    print(f"  {'—' * 22} {'—' * 6}")
    print(f"  {'TOTAL':<22s} {total:>6d}")

    lines.append("## Data type distribution\n")
    lines.append("| Kind | Count | % |")
    lines.append("|------|------:|--:|")
    for kind, cnt in counts.most_common():
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"| {kind} | {cnt} | {pct:.1f}% |")
    lines.append(f"| **TOTAL** | **{total}** | |")
    lines.append("")
    return lines


def print_crossfile_view_consistency(records: List[FileRecord]) -> List[str]:
    """Print + return md lines for cross-file view identifier consistency."""
    lines: List[str] = []
    hdr = f"\n{'#' * 110}\n  CROSS-FILE VIEW CONSISTENCY\n{'#' * 110}"
    print(hdr)

    have_views = [(r.label or r.path.name, r.view_col_name, r.view_values)
                  for r in records
                  if r.view_values is not None and len(r.view_values) > 0]

    if len(have_views) < 2:
        print("  (Not enough files with view-like columns to compare)")
        return lines

    all_views: set = set()
    for _, _, vs in have_views:
        all_views |= vs

    print(f"\n  Files with view identifiers: {len(have_views)}")
    print(f"  Union of all view values   : {len(all_views)} unique")

    # find mismatches (only show pairs where something differs)
    mismatches_found = 0
    for i, (lbl1, col1, vs1) in enumerate(have_views):
        for lbl2, col2, vs2 in have_views[i + 1:]:
            only1 = vs1 - vs2
            only2 = vs2 - vs1
            if only1 or only2:
                mismatches_found += 1
                if mismatches_found <= 15:   # cap output
                    shared = vs1 & vs2
                    print(f"\n  {lbl1} ({col1}) vs {lbl2} ({col2}): {len(shared)} shared")
                    if only1:
                        print(f"    Only in {lbl1}: {sorted(only1)[:10]}"
                              f"{'...' if len(only1) > 10 else ''}")
                    if only2:
                        print(f"    Only in {lbl2}: {sorted(only2)[:10]}"
                              f"{'...' if len(only2) > 10 else ''}")

    if mismatches_found == 0:
        print("  [OK] All files with view columns share identical view sets.")
    elif mismatches_found > 15:
        print(f"\n  ... ({mismatches_found - 15} more mismatched pairs omitted)")

    lines.append("## Cross-file view consistency\n")
    lines.append(f"- Files with view identifiers: {len(have_views)}")
    lines.append(f"- Union of all view values: {len(all_views)} unique")
    if mismatches_found == 0:
        lines.append("- All files share identical view sets.")
    else:
        lines.append(f"- WARNING: {mismatches_found} mismatched file pairs found (see terminal output)")
    lines.append("")
    return lines


def print_citation_roadmap(
    records: List[FileRecord],
    mentions: Dict[str, List[str]],
) -> List[str]:
    """
    Reverse-indexed roadmap: for each manuscript mention → which files support it.
    Also flags files that the manuscript never references.
    """
    lines: List[str] = []
    hdr = f"\n{'#' * 110}\n  CITATION ROADMAP  (manuscript section → supplementary file)\n{'#' * 110}"
    print(hdr)

    if not mentions:
        print("  (No manuscript provided or no mentions detected; skipping roadmap)")
        return lines

    # Build: mention_key → list of matching file labels
    mention_to_files: Dict[str, List[str]] = defaultdict(list)
    labelled = {r.label: r for r in records if r.label}

    for key in mentions:
        # key is like "Supplementary Table 3"
        if key in labelled:
            mention_to_files[key].append(labelled[key].path.name)
        # Also try matching by number to the file records
        for r in records:
            if r.label == key and r.path.name not in mention_to_files[key]:
                mention_to_files[key].append(r.path.name)

    # Print roadmap sorted by type then number
    sorted_keys = sorted(mentions.keys(),
                         key=lambda x: (0 if "Table" in x else 1, int(x.split()[-1])))

    print(f"\n  {'Manuscript Mention':<35s} {'#Refs':>5s}   {'Matched File(s)'}")
    print(f"  {'—' * 35} {'—' * 5}   {'—' * 55}")
    for key in sorted_keys:
        n = len(mentions[key])
        files = mention_to_files.get(key, ["(no file matched)"])
        print(f"  {key:<35s} {n:>5d}   {', '.join(files)}")

    # Flag unreferenced files
    referenced_labels = set(mentions.keys())
    unreferenced = [r for r in records if r.label and r.label not in referenced_labels and r.error is None]
    if unreferenced:
        print(f"\n  [WARN] FILES NEVER REFERENCED in manuscript ({len(unreferenced)}):")
        for r in unreferenced:
            print(f"    - {r.label}  ({r.path.name})")
            print(f"      Theme: {r.theme or '—'}")
    else:
        print("\n  [OK] All found supplementary files are referenced in the manuscript.")

    lines.append("## Citation Roadmap\n")
    lines.append("| Manuscript Mention | #Refs | File(s) |")
    lines.append("|-------------------|------:|---------|")
    for key in sorted_keys:
        n = len(mentions[key])
        files = mention_to_files.get(key, ["—"])
        lines.append(f"| {key} | {n} | `{'`, `'.join(files)}` |")
    if unreferenced:
        lines.append(f"\n**WARNING: Unreferenced files ({len(unreferenced)}):**")
        for r in unreferenced:
            lines.append(f"- `{r.path.name}` ({r.label}) — Theme: {r.theme or '—'}")
    lines.append("")
    return lines


def generate_llm_report(
    records: List[FileRecord],
    mentions: Dict[str, List[str]],
    manuscript_path: Optional[Path],
) -> str:
    """
    Generate a compact, token-efficient audit report designed for LLM context windows.

    Design principles:
    - No decorative borders, no repeated headers, no wide padding
    - Structured sections with minimal markup
    - All information from the verbose output preserved, just compressed
    - Column-level detail in compact tabular format
    - Actionable items (missing data, unreferenced files) surfaced first
    """
    L: List[str] = []

    # ── Header ──
    table_recs = [r for r in records if r.bucket == "sup_table" and r.error is None]
    data_recs = [r for r in records if r.bucket == "sup_data" and r.error is None]
    err_recs = [r for r in records if r.error is not None]

    L.append("# Supplementary Materials Audit — LLM Reference")
    L.append(f"Sup Tables: {len(table_recs)} files | Sup Data: {len(data_recs)} files"
             f"{f' | Errors: {len(err_recs)}' if err_recs else ''}")
    L.append("")

    # ── 1. ACTIONABLE ISSUES (most important for LLM to see first) ──
    L.append("## 1. Actionable Issues")
    L.append("")

    # 1a. Missing data hotspots
    miss_rows = []
    for r in records:
        if r.profiles is None:
            continue
        for cp in r.profiles:
            if cp.missing_pct >= 10.0:  # only flag >=10%
                miss_rows.append((r.label or r.path.name, cp.name, cp.missing_n, cp.n, cp.missing_pct))
    if miss_rows:
        miss_rows.sort(key=lambda x: -x[4])
        L.append("### Missing data (≥10% threshold)")
        L.append("File | Column | Missing/Total | %")
        L.append("---|---|---|---")
        for f, col, m, t, pct in miss_rows:
            L.append(f"{f} | {col} | {m:,}/{t:,} | {pct:.1f}%")
        L.append("")
    else:
        L.append("No columns with ≥10% missing data.")
        L.append("")

    # 1b. Errored/missing files
    if err_recs:
        L.append("### Files not found or errored")
        for r in err_recs:
            L.append(f"- {r.label or r.path.name}: {r.error}")
        L.append("")

    # 1c. Unreferenced files (if manuscript provided)
    if mentions:
        referenced = set(mentions.keys())
        unreferenced = [r for r in records if r.label and r.label not in referenced and r.error is None]
        if unreferenced:
            L.append("### Files never cited in manuscript")
            for r in unreferenced:
                L.append(f"- {r.label} ({r.path.name}) — Theme: {r.theme or '?'}")
            L.append("")

    # ── 2. FILE INVENTORY (compact) ──
    L.append("## 2. File Inventory")
    L.append("")
    L.append("ID | File | Rows | Cols | Miss% | Size | Theme")
    L.append("---|---|---:|---:|---:|---|---")
    for r in records:
        if r.error is not None:
            L.append(f"{r.label or '—'} | {r.path.name} | — | — | — | {r.size_mb:.2f}MB | ERROR: {r.error}")
        else:
            L.append(f"{r.label or '—'} | {r.path.name} | {r.rows:,} | {r.cols} | "
                     f"{r.missing_pct:.1f}% | {r.size_mb:.2f}MB | {r.theme or '—'}")
    L.append("")

    # ── 3. PER-FILE COLUMN DETAIL ──
    L.append("## 3. Column Detail per File")
    L.append("")

    for r in records:
        if r.error is not None or r.profiles is None:
            continue

        L.append(f"### {r.label or r.path.name} ({r.rows:,}×{r.cols})")

        # Compact column table: one line per column, CSV-like
        L.append("Column | Kind | Miss | Unique | Range/Values")
        L.append("---|---|---|---|---")
        for cp in r.profiles:
            miss_s = f"{cp.missing_n}" if cp.missing_n == 0 else f"**{cp.missing_n}** ({cp.missing_pct:.0f}%)"
            uniq_s = str(cp.nunique) if cp.nunique is not None else ""

            if cp.kind in ("int", "float") and cp.num_min is not None:
                range_s = f"[{cp.num_min:.4g}–{cp.num_max:.4g}] μ={cp.num_mean:.4g} σ={cp.num_std:.4g}"
                if cp.num_q50 is not None:
                    range_s += f" med={cp.num_q50:.4g}"
            elif cp.top_levels:
                range_s = cp.top_levels[:100]
            else:
                range_s = "—"

            L.append(f"{cp.name} | {cp.kind} | {miss_s} | {uniq_s} | {range_s}")
        L.append("")

    # ── 4. CROSS-FILE CONSISTENCY ──
    have_views = [(r.label or r.path.name, r.view_col_name, r.view_values)
                  for r in records if r.view_values and len(r.view_values) > 0]
    if len(have_views) >= 2:
        L.append("## 4. Cross-file View Consistency")
        all_v: set = set()
        for _, _, vs in have_views:
            all_v |= vs
        L.append(f"{len(have_views)} files have view columns, {len(all_v)} unique views total.")
        L.append("")

        # Only show mismatches
        mismatches = []
        for i, (l1, c1, v1) in enumerate(have_views):
            for l2, c2, v2 in have_views[i + 1:]:
                o1, o2 = v1 - v2, v2 - v1
                if o1 or o2:
                    mismatches.append((l1, c1, l2, c2, v1 & v2, o1, o2))

        if mismatches:
            L.append(f"{len(mismatches)} mismatched pairs:")
            for l1, c1, l2, c2, shared, o1, o2 in mismatches[:10]:
                L.append(f"- {l1}.{c1} vs {l2}.{c2}: {len(shared)} shared"
                         f"{f', only-left={sorted(o1)[:5]}' if o1 else ''}"
                         f"{f', only-right={sorted(o2)[:5]}' if o2 else ''}")
            if len(mismatches) > 10:
                L.append(f"- ...({len(mismatches) - 10} more)")
        else:
            L.append("All files with view columns share identical view sets.")
        L.append("")

    # ── 5. CITATION ROADMAP ──
    if mentions:
        L.append("## 5. Citation Roadmap (manuscript mention → file)")
        L.append("")
        labelled = {r.label: r for r in records if r.label}
        sorted_keys = sorted(mentions.keys(),
                             key=lambda x: (0 if "Table" in x else 1, int(x.split()[-1])))
        L.append("Mention | #Refs | File | Snippets")
        L.append("---|---:|---|---")
        for key in sorted_keys:
            n = len(mentions[key])
            fmatch = labelled.get(key)
            fname = fmatch.path.name if fmatch else "(no file)"
            snip = mentions[key][0][:120].replace("|", "/") if mentions[key] else ""
            L.append(f"{key} | {n} | {fname} | {snip}")
        L.append("")

    # ── 6. DATA TYPE SUMMARY ──
    counts: Counter = Counter()
    for r in records:
        if r.profiles:
            for cp in r.profiles:
                counts[cp.kind] += 1
    if counts:
        total_c = sum(counts.values())
        L.append("## 6. Column Type Distribution")
        type_parts = [f"{k}={v} ({100*v/total_c:.0f}%)" for k, v in counts.most_common()]
        L.append(f"Total columns: {total_c}. " + ", ".join(type_parts))
        L.append("")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sup-table-dir", type=str, required=True)
    ap.add_argument("--sup-data-dir", type=str, required=True)
    ap.add_argument("--manuscript", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")
    ap.add_argument("--llm-report", type=str, default="",
                    help="Write a compact, token-efficient report optimised for LLM context windows.")
    ap.add_argument("--big-file-mb", type=float, default=300.0, help="Above this, stream CSV/TSV stats.")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--sample-rows", type=int, default=20_000, help="Rows to preview/sample for stats.")
    args = ap.parse_args()

    sup_table_dir = Path(args.sup_table_dir)
    sup_data_dir = Path(args.sup_data_dir)
    manuscript_path = Path(args.manuscript) if args.manuscript else None

    manuscript_text = read_manuscript_text(manuscript_path)
    mentions = extract_manuscript_mentions(manuscript_text)

    # Gather files
    table_files = iter_files(sup_table_dir)
    data_files = iter_files(sup_data_dir)
    all_files = [("sup_table", p) for p in table_files] + [("sup_data", p) for p in data_files]

    print("\n" + "#" * 110)
    print("SUPPLEMENTARY ASSET AUDIT")
    print("#" * 110)
    print(f"sup_table_dir: {sup_table_dir} ({len(table_files)} files)")
    print(f"sup_data_dir : {sup_data_dir} ({len(data_files)} files)")
    if manuscript_path:
        print(f"manuscript   : {manuscript_path} (parsed={bool(manuscript_text)})")
        if mentions:
            keys = sorted(mentions.keys(), key=lambda x: (x.split()[-2], int(x.split()[-1])))
            print("\nMentions found in manuscript:")
            for k in keys:
                print(f"  - {k}: {len(mentions[k])} mention(s)")
        else:
            print("\nNo 'Supplementary Table/Data X' mentions detected in manuscript text.")
    else:
        print("manuscript   : (none)")

    md_lines: List[str] = []
    if args.out_md:
        md_lines.append("# Supplementary asset audit\n")
        md_lines.append(f"- sup_table_dir: `{sup_table_dir}` ({len(table_files)} files)")
        md_lines.append(f"- sup_data_dir: `{sup_data_dir}` ({len(data_files)} files)")
        if manuscript_path:
            md_lines.append(f"- manuscript: `{manuscript_path}`")
        md_lines.append("")

    # Audit loop — collect FileRecord for each file
    file_records: List[FileRecord] = []

    for bucket, p in all_files:
        print_file_header(p)
        sup_num = extract_sup_number_from_filename(p)
        label_guess = None
        if sup_num is not None:
            label_guess = f"Supplementary {'Table' if bucket=='sup_table' else 'Data'} {sup_num}"

        # Tabular
        if is_probably_tabular(p):
            df_prev, meta = audit_tabular_file(
                p, big_file_mb=float(args.big_file_mb), chunksize=int(args.chunksize), sample_rows=int(args.sample_rows)
            )
            if "error" in meta:
                print(f"[ERROR] {meta['error']}")
                file_records.append(FileRecord(
                    bucket=bucket, path=p, label=label_guess, theme=None,
                    size_mb=sizeof_mb(p), error=str(meta["error"]),
                ))
                continue

            print(f"READER: {meta.get('reader')}")
            print(f"ROWS/COLS: {meta.get('rows')} × {meta.get('cols')}   (streamed={meta.get('streamed', False)})")

            # Column profiling: if streamed, we print streaming stats + sample-derived kinds
            if meta.get("streamed", False):
                col_stats = meta["stream_col_stats"]
                cols = list(col_stats.keys())
                theme = guess_theme_from_columns(cols)

                # build profiles from streaming stats (quantiles not available)
                profs = []
                nrows = int(meta["rows"])
                for c in cols:
                    st = col_stats[c]
                    miss = int(st["missing_n"])
                    miss_pct = (miss / nrows * 100.0) if nrows > 0 else 0.0
                    kind = str(st.get("kind", "unknown"))
                    pd_dtype = "streamed"
                    cp = ColumnProfile(
                        name=c,
                        pandas_dtype=pd_dtype,
                        kind=kind,
                        n=nrows,
                        missing_n=miss,
                        missing_pct=float(miss_pct),
                        nonmissing_n=int(st["nonmissing_n"]),
                        nunique=None,
                        num_min=st.get("min"),
                        num_max=st.get("max"),
                        num_mean=st.get("mean"),
                        num_std=st.get("std"),
                        top_levels="; ".join([f"{k} ({v})" for k, v in st.get("top", [])]) if st.get("top") else None
                    )
                    profs.append(cp)

                print("\nCOLUMNS (streamed; quantiles/medians not computed):")
                print(format_profiles_table(profs))

                if df_prev is not None:
                    print("\nPREVIEW (head):")
                    print(df_prev.head(5).to_string(index=False))

                # Manuscript citation suggestion
                print("\nMANUSCRIPT LINKING:")
                print(f"  Theme guess: {theme}")
                if label_guess and label_guess in mentions:
                    print(f"  Manuscript mentions {label_guess}:")
                    for snip in mentions[label_guess][:3]:
                        print(f"    - {snip}")
                else:
                    if label_guess:
                        print(f"  No exact mention found for {label_guess}.")
                    print(f"  Suggested cite anchor: {theme}")

                # Accumulate for post-audit summaries
                nrows_s = int(meta["rows"])
                ncols_s = int(meta["cols"])
                total_cells_s = nrows_s * ncols_s
                total_miss_s = sum(int(col_stats[c]["missing_n"]) for c in cols)
                rec = FileRecord(
                    bucket=bucket, path=p, label=label_guess, theme=theme,
                    rows=nrows_s, cols=ncols_s,
                    total_cells=total_cells_s, total_missing=total_miss_s,
                    missing_pct=(100.0 * total_miss_s / total_cells_s) if total_cells_s else 0,
                    size_mb=sizeof_mb(p), profiles=profs,
                )
                if df_prev is not None:
                    vc, vv = _extract_view_column(df_prev)
                    rec.view_col_name, rec.view_values = vc, vv
                file_records.append(rec)

                # Markdown
                if args.out_md:
                    md_lines.append(f"## {bucket}: `{p.name}`\n")
                    md_lines.append(f"- Path: `{p}`")
                    md_lines.append(f"- Size: {sizeof_mb(p):.2f} MB")
                    md_lines.append(f"- Shape: {meta.get('rows')} × {meta.get('cols')} (streamed)")
                    md_lines.append(f"- Theme guess: {theme}")
                    if label_guess and label_guess in mentions:
                        md_lines.append(f"- Manuscript mentions: **{label_guess}**")
                        for snip in mentions[label_guess][:3]:
                            md_lines.append(f"  - {snip}")
                    md_lines.append("")
                continue

            # Non-streamed: we have a DataFrame (full or preview)
            if df_prev is None:
                print("[ERROR] could not load dataframe preview.")
                continue

            # profiles on preview; quantiles are OK
            theme = guess_theme_from_columns([str(c) for c in df_prev.columns])
            profs = profile_dataframe(df_prev, compute_quantiles=True)

            print("\nCOLUMNS (computed on loaded table or preview subset):")
            print(format_profiles_table(profs))

            # Missingness totals on preview only; warn if previewed
            if int(meta.get("rows", 0)) > len(df_prev):
                print(f"\n[NOTE] Loaded preview ({len(df_prev):,} rows) from a {int(meta['rows']):,}-row table; "
                      "missingness/quantiles reflect preview, not full table.")

            print("\nMANUSCRIPT LINKING:")
            print(f"  Theme guess: {theme}")
            if label_guess and label_guess in mentions:
                print(f"  Manuscript mentions {label_guess}:")
                for snip in mentions[label_guess][:3]:
                    print(f"    - {snip}")
            else:
                if label_guess:
                    print(f"  No exact mention found for {label_guess}.")
                print(f"  Suggested cite anchor: {theme}")

            # Accumulate for post-audit summaries
            nrows_ns = int(meta.get("rows", len(df_prev)))
            ncols_ns = int(meta.get("cols", df_prev.shape[1]))
            total_cells_ns = nrows_ns * ncols_ns
            total_miss_ns = sum(cp.missing_n for cp in profs)
            rec = FileRecord(
                bucket=bucket, path=p, label=label_guess, theme=theme,
                rows=nrows_ns, cols=ncols_ns,
                total_cells=total_cells_ns, total_missing=total_miss_ns,
                missing_pct=(100.0 * total_miss_ns / total_cells_ns) if total_cells_ns else 0,
                size_mb=sizeof_mb(p), profiles=profs,
            )
            vc, vv = _extract_view_column(df_prev)
            rec.view_col_name, rec.view_values = vc, vv
            file_records.append(rec)

            # Markdown
            if args.out_md:
                md_lines.append(f"## {bucket}: `{p.name}`\n")
                md_lines.append(f"- Path: `{p}`")
                md_lines.append(f"- Size: {sizeof_mb(p):.2f} MB")
                md_lines.append(f"- Preview shape: {len(df_prev)} × {df_prev.shape[1]} (full rows reported={meta.get('rows')})")
                md_lines.append(f"- Theme guess: {theme}")
                if label_guess and label_guess in mentions:
                    md_lines.append(f"- Manuscript mentions: **{label_guess}**")
                    for snip in mentions[label_guess][:3]:
                        md_lines.append(f"  - {snip}")
                md_lines.append("")
            continue

        # Non-tabular
        meta = audit_non_tabular(p)
        file_records.append(FileRecord(
            bucket=bucket, path=p, label=label_guess, theme="non-tabular",
            size_mb=sizeof_mb(p),
        ))
        print(f"TYPE: {meta.get('type', 'unknown')}")
        if "keys" in meta:
            print(f"KEYS: {meta['keys']}")
        if "arrays" in meta:
            print("NPZ ARRAYS:")
            for k, v in meta["arrays"].items():
                print(f"  - {k}: shape={v['shape']} dtype={v['dtype']}")
        if "error" in meta:
            print(f"[ERROR] {meta['error']}")

        if args.out_md:
            md_lines.append(f"## {bucket}: `{p.name}`\n")
            md_lines.append(f"- Path: `{p}`")
            md_lines.append(f"- Size: {sizeof_mb(p):.2f} MB")
            md_lines.append(f"- Type: {meta.get('type', 'unknown')}")
            if "keys" in meta:
                md_lines.append(f"- Keys: {meta['keys']}")
            if "arrays" in meta:
                md_lines.append("- Arrays:")
                for k, v in meta["arrays"].items():
                    md_lines.append(f"  - {k}: shape={v['shape']} dtype={v['dtype']}")
            if "error" in meta:
                md_lines.append(f"- Error: {meta['error']}")
            md_lines.append("")

    # =====================================================================
    # POST-AUDIT AGGREGATE ANALYSES
    # =====================================================================
    table_records = [r for r in file_records if r.bucket == "sup_table"]
    data_records = [r for r in file_records if r.bucket == "sup_data"]

    md_lines += print_grand_summary(table_records, "SUPPLEMENTARY TABLES")
    md_lines += print_grand_summary(data_records, "SUPPLEMENTARY DATA")
    md_lines += print_missing_data_heatmap(file_records)
    md_lines += print_dtype_distribution(file_records)
    md_lines += print_crossfile_view_consistency(file_records)
    md_lines += print_citation_roadmap(file_records, mentions)

    # Write markdown report if requested
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md_lines), encoding="utf-8")
        print("\n" + "-" * 110)
        print(f"WROTE MARKDOWN REPORT: {out_md}")
        print("-" * 110)

    # Write LLM-optimised report if requested
    if args.llm_report:
        llm_text = generate_llm_report(file_records, mentions, manuscript_path)
        llm_path = Path(args.llm_report)
        llm_path.parent.mkdir(parents=True, exist_ok=True)
        llm_path.write_text(llm_text, encoding="utf-8")
        # Show token estimate
        approx_tokens = len(llm_text.split()) * 1.3  # rough tokenizer estimate
        print("\n" + "-" * 110)
        print(f"WROTE LLM REPORT: {llm_path}")
        print(f"  Size: {len(llm_text):,} chars ≈ {approx_tokens:,.0f} tokens")
        print(f"  (vs terminal output which would be ~4-5× larger)")
        print("-" * 110)


if __name__ == "__main__":
    main()