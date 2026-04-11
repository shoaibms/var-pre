#!/usr/bin/env python
"""
Cross-model SHAP agreement analysis.

Compares top-K feature sets between models (e.g., RF vs XGBoost) using
Jaccard similarity on per-model importance tables and per-repeat SHAP values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


HERO_VIEWS = [
    ("mlomics", "methylation"),
    ("ibdmdb", "MGX"),
    ("ccle", "mRNA"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union)


def k_to_n(n_features: int, k_pct: int) -> int:
    # match your DI files: floor behavior
    return max(1, int(n_features * (k_pct / 100.0)))


def expected_jaccard_random(n: int, k_a: int, k_b: int) -> float:
    # E[|A∩B|] = k_a*k_b/n; E[J] ≈ E[inter]/(k_a+k_b-E[inter])
    if n <= 0:
        return float("nan")
    e_inter = (k_a * k_b) / float(n)
    denom = (k_a + k_b - e_inter)
    return float(e_inter / denom) if denom > 0 else float("nan")


@dataclass(frozen=True)
class ImportanceTable:
    dataset: str
    view: str
    model: str
    path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-model SHAP agreement using per_model tables and/or SHAP NPZ.")
    p.add_argument("--outputs-dir", required=True, help="Root outputs directory")
    p.add_argument("--models", default="rf,xgb_bal", help="Comma-separated models to compare (default: rf,xgb_bal)")
    p.add_argument("--k-pcts", default="1,5,10,20", help="Comma-separated K percentages")
    p.add_argument("--hero-views", action="store_true", help="Restrict to HERO_VIEWS only")
    p.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples for CI (repeat-based rows only)")
    return p.parse_args()


def discover_importance_tables(per_model_dir: Path) -> List[ImportanceTable]:
    out: List[ImportanceTable] = []
    rgx = re.compile(r"^importance__(.+?)__(.+?)__(.+?)\.csv(\.gz)?$")
    for p in sorted(per_model_dir.glob("importance__*.csv*")):
        if p.name.endswith(".bak"):
            continue
        m = rgx.match(p.name)
        if not m:
            continue
        out.append(ImportanceTable(dataset=m.group(1), view=m.group(2), model=m.group(3), path=p))
    return out


def load_importance_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"dataset", "view", "model", "feature", "p_score", "p_rank", "p_rank_pct"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def topk_from_importance_df(df: pd.DataFrame, k_pct: int) -> Tuple[set, int, int]:
    n_features = int(df.shape[0])
    k_n = k_to_n(n_features, k_pct)
    sub = df.nsmallest(k_n, "p_rank")
    return set(sub["feature"].astype(str).tolist()), k_n, n_features


def find_shap_npz(outputs_dir: Path, dataset: str, view: str, model: str) -> Optional[Path]:
    # filename token aliases
    token_list = ["rf"] if model == "rf" else (["xgb"] if model in ("xgb", "xgb_bal") else [model])
    dir_list = ["rf"] if model == "rf" else (["xgb_bal", "xgb"] if model in ("xgb", "xgb_bal") else [model])

    for d in dir_list:
        for tok in token_list:
            p = outputs_dir / "03_supervised" / f"tree_models_{d}" / "importance" / f"shap__{dataset}__{view}__{tok}.npz"
            if p.exists():
                return p

    return None


def load_shap_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    feature_names = z["feature_names.npy"]
    mean_abs_per_repeat = z["mean_abs_shap_per_repeat.npy"]
    return feature_names, mean_abs_per_repeat


def topk_from_shap_repeat(feature_names: np.ndarray, shap_row: np.ndarray, k_pct: int) -> Tuple[set, int, int]:
    n_features = int(feature_names.shape[0])
    k_n = k_to_n(n_features, k_pct)
    # top-k by shap magnitude
    idx = np.argpartition(shap_row, -k_n)[-k_n:]
    top = feature_names[idx]
    return set(map(str, top.tolist())), k_n, n_features


def bootstrap_ci(values: np.ndarray, n_boot: int, alpha: float = 0.05) -> Tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        v = float(values[0])
        return (v, v)
    rng = np.random.default_rng(12345)
    boots = []
    n = values.size
    for _ in range(n_boot):
        samp = values[rng.integers(0, n, size=n)]
        boots.append(float(np.mean(samp)))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def main() -> None:
    args = parse_args()
    t0 = time.time()

    outputs_dir = Path(args.outputs_dir)
    per_model_dir = outputs_dir / "04_importance" / "per_model"
    out_dir = outputs_dir / "06_robustness" / "agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    runlog = out_dir / "RUNLOG__01_cross_model_shap_agreement.txt"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    k_pcts = [int(x.strip()) for x in args.k_pcts.split(",") if x.strip()]
    hero_set = set(HERO_VIEWS)

    tables = discover_importance_tables(per_model_dir)
    # index: (dataset,view) -> model -> path
    idx: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for t in tables:
        idx.setdefault((t.dataset, t.view), {})[t.model] = t.path

    rows = []

    for (dataset, view), mp in sorted(idx.items()):
        if args.hero_views and (dataset, view) not in hero_set:
            continue
        # only compare when both models exist
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                if a not in mp or b not in mp:
                    continue

                df_a = load_importance_df(mp[a])
                df_b = load_importance_df(mp[b])

                for k_pct in k_pcts:
                    set_a, k_a, n_a = topk_from_importance_df(df_a, k_pct)
                    set_b, k_b, n_b = topk_from_importance_df(df_b, k_pct)

                    # use common feature universe for null approx if possible
                    universe = set(df_a["feature"].astype(str)) & set(df_b["feature"].astype(str))
                    n_u = len(universe) if universe else min(n_a, n_b)
                    j_null = expected_jaccard_random(n_u, k_a, k_b)

                    rows.append({
                        "dataset": dataset, "view": view,
                        "model_a": a, "model_b": b,
                        "method": "per_model_rank",
                        "repeat": -1,
                        "k_pct": k_pct,
                        "k_n": min(k_a, k_b),
                        "n_features": n_u,
                        "overlap_n": len(set_a & set_b),
                        "jaccard": jaccard(set_a, set_b),
                        "jaccard_null": j_null,
                    })

                # per-repeat agreement (if both SHAP npz exist)
                npz_a = find_shap_npz(outputs_dir, dataset, view, a)
                npz_b = find_shap_npz(outputs_dir, dataset, view, b)
                if npz_a and npz_b:
                    fn_a, per_a = load_shap_npz(npz_a)
                    fn_b, per_b = load_shap_npz(npz_b)

                    # Align by feature name if needed
                    map_a = {str(f): ix for ix, f in enumerate(fn_a.tolist())}
                    map_b = {str(f): ix for ix, f in enumerate(fn_b.tolist())}
                    common = sorted(set(map_a) & set(map_b))
                    if len(common) > 0:
                        ia = np.array([map_a[f] for f in common], dtype=int)
                        ib = np.array([map_b[f] for f in common], dtype=int)
                        fn = np.array(common, dtype=object)
                        per_a_al = per_a[:, ia]
                        per_b_al = per_b[:, ib]
                    else:
                        fn = fn_a
                        per_a_al = per_a
                        per_b_al = per_b

                    n_rep = min(per_a_al.shape[0], per_b_al.shape[0])
                    for k_pct in k_pcts:
                        for r in range(n_rep):
                            set_a_r, k_a, n_u = topk_from_shap_repeat(fn, per_a_al[r], k_pct)
                            set_b_r, k_b, _ = topk_from_shap_repeat(fn, per_b_al[r], k_pct)
                            j_null = expected_jaccard_random(n_u, k_a, k_b)

                            rows.append({
                                "dataset": dataset, "view": view,
                                "model_a": a, "model_b": b,
                                "method": "per_repeat_shap",
                                "repeat": int(r),
                                "k_pct": k_pct,
                                "k_n": min(k_a, k_b),
                                "n_features": int(n_u),
                                "overlap_n": len(set_a_r & set_b_r),
                                "jaccard": jaccard(set_a_r, set_b_r),
                                "jaccard_null": j_null,
                            })

    long_df = pd.DataFrame(rows)
    long_path = out_dir / "shap_agreement_long.csv.gz"
    long_df.to_csv(long_path, index=False, compression="gzip")

    # Summary (only meaningful for per_repeat_shap rows; per_model_rank is single row)
    summ_rows = []
    for (dataset, view, a, b, method, k_pct), g in long_df.groupby(["dataset", "view", "model_a", "model_b", "method", "k_pct"]):
        vals = g["jaccard"].to_numpy(dtype=float)
        k_n = int(np.median(g["k_n"]))
        n_feat = int(np.median(g["n_features"]))
        j_null = float(np.median(g["jaccard_null"]))
        if method == "per_repeat_shap":
            lo, hi = bootstrap_ci(vals, args.n_bootstrap)
        else:
            lo, hi = float(vals[0]), float(vals[0])
        summ_rows.append({
            "dataset": dataset, "view": view,
            "model_a": a, "model_b": b,
            "method": method,
            "k_pct": int(k_pct),
            "k_n": k_n,
            "n_features": n_feat,
            "n_rows": int(vals.size),
            "jaccard_mean": float(np.mean(vals)),
            "jaccard_ci_lo": lo,
            "jaccard_ci_hi": hi,
            "jaccard_null": j_null,
        })

    summ_df = pd.DataFrame(summ_rows).sort_values(["dataset", "view", "method", "k_pct"])
    summ_path = out_dir / "shap_agreement_summary.csv"
    summ_df.to_csv(summ_path, index=False)

    manifest = {
        "script": "01_cross_model_shap_agreement.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "inputs": {
            "per_model_dir": str(per_model_dir),
        },
        "outputs": {
            "long": str(long_path),
            "summary": str(summ_path),
        },
        "hashes": {
            "shap_agreement_long.csv.gz": sha256_file(long_path),
            "shap_agreement_summary.csv": sha256_file(summ_path),
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    (out_dir / "MANIFEST__agreement.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runlog.write_text(
        f"OK\ncreated={manifest['created_utc']}\nlong={long_path}\nsummary={summ_path}\nruntime_sec={manifest['runtime_sec']}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
