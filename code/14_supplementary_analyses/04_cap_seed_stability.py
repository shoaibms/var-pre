#!/usr/bin/env python3
"""
Cap-seed stability: test whether variance-neutral feature capping
affects VAD zone assignments.

For views where features were randomly subsampled to a cap (e.g., 20,000),
this script re-caps with multiple seeds and checks zone stability.

Capped views (from bundle prep):
  ibdmdb / MGX_func : cap = 10,000 (gene families)
  ibdmdb / MBX      : cap = 20,000 (metabolomics, requires biom package)
  tcga_gbm / methylation : cap = 20,000 (CpG probes)

Usage:
  python 14_supplementary_analyses/05_cap_seed_stability.py --outputs-dir outputs --data-dir data/raw

Outputs:
  outputs/14_supplementary_analyses/cap_seed_stability/cap_seed_stability.csv
  outputs/14_supplementary_analyses/cap_seed_stability/cap_seed_stability_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.vad_metrics import (
    eta2_features, eta_enrichment, vsa_mannwhitney, alpha_prime,
    f_di, classify_zone,
)


# ── Capped views registry ──
CAPPED_VIEWS = [
    {"dataset": "ibdmdb", "view": "MGX_func", "cap": 10_000,
     "raw_loader": "ibdmdb_mgx_func"},
    {"dataset": "ibdmdb", "view": "MBX", "cap": 20_000,
     "raw_loader": "ibdmdb_mbx"},
    {"dataset": "tcga_gbm", "view": "methylation", "cap": 20_000,
     "raw_loader": "tcga_gbm_methylation"},
]

SEEDS = [1, 2, 3, 4, 5]
K_PCT = 10


# ── Minimal QC (matches bundle prep exactly) ──
def drop_all_nan_and_zerovar_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[~df.isna().all(axis=1)]
    try:
        v = df.var(axis=1, skipna=True)
    except TypeError:
        v = df.apply(lambda x: np.nanvar(pd.to_numeric(x, errors="coerce").values), axis=1)
    df = df.loc[v > 0]
    return df


def cap_rows_neutral(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if df.shape[0] <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy()
    chosen = rng.choice(idx, size=max_rows, replace=False)
    return df.loc[chosen].sort_index()


def extract_hmp2_sample_id(col: str) -> str:
    """Extract core HMP2 sample ID from column name."""
    col = str(col).strip()
    if "_" in col:
        parts = col.split("_")
        for p in parts:
            if p.startswith("CSM") or p.startswith("MSM") or p.startswith("HSM") or p.startswith("PSM"):
                return p
    return col


# ── Raw data loaders (one per capped view) ──
def load_ibdmdb_mgx_func(data_dir: Path, sample_ids: np.ndarray) -> Optional[pd.DataFrame]:
    """Load gene families, filter to gene-level, subset to samples."""
    func_file = data_dir / "ibdmdb" / "genefamilies.tsv"
    if not func_file.exists():
        return None
    df = pd.read_csv(func_file, sep="\t", index_col=0)
    df.columns = [extract_hmp2_sample_id(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    # Gene-level only (matches bundle prep)
    gene_mask = ~df.index.astype(str).str.contains("\\|")
    df = df[gene_mask]
    # Subset to available samples
    avail = [s for s in sample_ids if s in df.columns]
    if len(avail) < 10:
        return None
    return df[avail]


def load_ibdmdb_mbx(data_dir: Path, sample_ids: np.ndarray) -> Optional[pd.DataFrame]:
    """Load metabolomics from biom format."""
    biom_file = data_dir / "ibdmdb" / "HMP2_metabolomics_w_metadata.biom"
    if not biom_file.exists():
        return None
    try:
        import biom
        table = biom.load_table(str(biom_file))
        data = table.to_dataframe()
        data.columns = [extract_hmp2_sample_id(str(c)) for c in data.columns]
        data = data.loc[:, ~data.columns.duplicated()]
        if hasattr(data, "sparse"):
            data = data.sparse.to_dense()
        avail = [s for s in sample_ids if s in data.columns]
        if len(avail) < 10:
            return None
        return data[avail]
    except ImportError:
        print("    [SKIP] biom package not installed")
        return None
    except Exception as e:
        print(f"    [SKIP] MBX load error: {e}")
        return None


def load_tcga_gbm_methylation(data_dir: Path, sample_ids: np.ndarray) -> Optional[pd.DataFrame]:
    """Load TCGA GBM methylation probes."""
    methy_file = data_dir / "tcga_gbm" / "HumanMethylation450"
    if not methy_file.exists():
        return None
    df = pd.read_csv(methy_file, sep="\t", index_col=0)
    # Normalize TCGA IDs
    df.columns = [str(c).replace(".", "-")[:15] if str(c).startswith("TCGA") else str(c)
                  for c in df.columns]
    avail = [s for s in sample_ids if s in df.columns]
    if len(avail) < 10:
        return None
    return df[avail]


RAW_LOADERS = {
    "ibdmdb_mgx_func": load_ibdmdb_mgx_func,
    "ibdmdb_mbx": load_ibdmdb_mbx,
    "tcga_gbm_methylation": load_tcga_gbm_methylation,
}


def load_splits(outputs_dir: Path, dataset: str):
    p = outputs_dir / "01_bundles" / "splits" / f"splits__{dataset}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    y = np.asarray(d["y"])
    sample_ids = np.asarray(d["sample_ids"]).astype(str)
    return y, sample_ids


def compute_vad_for_view(X: np.ndarray, y: np.ndarray, k_pct: int = 10) -> dict:
    """Compute VAD metrics for a single (capped) view."""
    v_total, v_between, eta2 = eta2_features(X, y)
    eta_es, _, _ = eta_enrichment(eta2, v_total, k_pct=k_pct)
    vsa = vsa_mannwhitney(eta2, v_total, k_pct=k_pct)
    ap = alpha_prime(v_total, eta2)
    fdi = f_di(v_total, eta2, k_pct=k_pct)
    zone = classify_zone(eta_es, vsa, fdi)

    return {
        "eta_es": float(eta_es),
        "vsa": float(vsa),
        "alpha_prime": float(ap),
        "f_di": float(fdi),
        "zone": zone,
    }


# ── Main ──
def main():
    ap = argparse.ArgumentParser(description="Cap-seed stability test")
    ap.add_argument("--outputs-dir", type=str, required=True)
    ap.add_argument("--data-dir", type=str, required=True,
                    help="Path to raw data directory (e.g., data/raw)")
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5",
                    help="Comma-separated seeds to test")
    ap.add_argument("--k-pct", type=int, default=10)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    data_dir = Path(args.data_dir)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    out_dir = outputs_dir / "14_supplementary_analyses" / "cap_seed_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cap-Seed Stability Test")
    print(f"Seeds: {seeds}")
    print(f"K: {args.k_pct}%")
    print("=" * 60)

    rows: List[dict] = []

    for spec in CAPPED_VIEWS:
        dataset = spec["dataset"]
        view = spec["view"]
        cap = spec["cap"]
        loader_name = spec["raw_loader"]
        loader = RAW_LOADERS[loader_name]

        print(f"\n── {dataset} / {view} (cap={cap:,}) ──")

        # Load labels
        try:
            y, sample_ids = load_splits(outputs_dir, dataset)
        except Exception as e:
            print(f"  [SKIP] Cannot load splits: {e}")
            continue

        # Load raw data
        print(f"  Loading raw data...", end=" ", flush=True)
        df_raw = loader(data_dir, sample_ids)
        if df_raw is None:
            print("NOT FOUND or insufficient samples")
            continue

        # QC (matches bundle prep)
        df_qc = drop_all_nan_and_zerovar_rows(df_raw)
        n_pre_cap = df_qc.shape[0]
        n_samples = df_qc.shape[1]
        print(f"{n_pre_cap:,} features x {n_samples} samples after QC")

        if n_pre_cap <= cap:
            print(f"  [INFO] Features ({n_pre_cap:,}) <= cap ({cap:,}) — cap never fires. Stable by definition.")
            for seed in seeds:
                rows.append({
                    "dataset": dataset, "view": view,
                    "cap": cap, "seed": seed,
                    "n_features_pre_cap": n_pre_cap,
                    "n_features_post_cap": n_pre_cap,
                    "capped": False,
                    "eta_es": None, "vsa": None, "alpha_prime": None,
                    "f_di": None, "zone": "N/A (not capped)",
                })
            continue

        # Align samples: subset y to match available columns
        id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
        avail_cols = [c for c in df_qc.columns if c in id_to_idx]
        col_indices = [id_to_idx[c] for c in avail_cols]
        y_aligned = y[col_indices]
        df_qc = df_qc[avail_cols]

        print(f"  Testing {len(seeds)} seeds...")
        zones_seen = set()

        for seed in seeds:
            df_capped = cap_rows_neutral(df_qc, cap, seed)
            n_post = df_capped.shape[0]
            X = df_capped.values.T.astype(np.float32)  # samples x features

            vad = compute_vad_for_view(X, y_aligned, k_pct=args.k_pct)
            zones_seen.add(vad["zone"])

            print(f"    seed={seed}: {n_post:,} features -> "
                  f"eta_ES={vad['eta_es']:.3f}  VSA={vad['vsa']:.3f}  "
                  f"f_DI={vad['f_di']:.3f}  zone={vad['zone']}")

            rows.append({
                "dataset": dataset, "view": view,
                "cap": cap, "seed": seed,
                "n_features_pre_cap": n_pre_cap,
                "n_features_post_cap": n_post,
                "capped": True,
                "eta_es": vad["eta_es"],
                "vsa": vad["vsa"],
                "alpha_prime": vad["alpha_prime"],
                "f_di": vad["f_di"],
                "zone": vad["zone"],
            })

        if len(zones_seen) == 1:
            print(f"  --> STABLE: zone = {zones_seen.pop()} across all seeds")
        else:
            print(f"  --> UNSTABLE: zones = {sorted(zones_seen)}")

    # Write outputs
    df = pd.DataFrame(rows)
    csv_path = out_dir / "cap_seed_stability.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    capped = df[df["capped"] == True]
    summary = {
        "n_views_tested": int(capped["view"].nunique()) if not capped.empty else 0,
        "n_seeds": len(seeds),
        "k_pct": args.k_pct,
        "all_stable": bool(
            all(g["zone"].nunique() == 1 for _, g in capped.groupby(["dataset", "view"]))
        ) if not capped.empty else True,
        "per_view": [],
    }
    for (ds, vw), g in df.groupby(["dataset", "view"]):
        entry = {
            "dataset": ds, "view": vw,
            "capped": bool(g["capped"].iloc[0]),
            "zones_seen": sorted(g["zone"].unique().tolist()),
            "stable": len(g["zone"].unique()) == 1,
        }
        if g["capped"].iloc[0] and g["eta_es"].notna().any():
            entry["eta_es_range"] = [
                round(float(g["eta_es"].min()), 4),
                round(float(g["eta_es"].max()), 4),
            ]
            entry["f_di_range"] = [
                round(float(g["f_di"].min()), 4),
                round(float(g["f_di"].max()), 4),
            ]
        summary["per_view"].append(entry)

    json_path = out_dir / "cap_seed_stability_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    print("\n" + "=" * 60)
    if summary["all_stable"]:
        print("VERDICT: All capped views are zone-stable across seeds")
    else:
        unstable = [v for v in summary["per_view"] if not v["stable"]]
        print(f"VERDICT: {len(unstable)} view(s) show zone instability")
    print("=" * 60)


if __name__ == "__main__":
    main()
