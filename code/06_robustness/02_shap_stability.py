#!/usr/bin/env python
"""
Within-model SHAP stability analysis.

Measures pairwise Jaccard similarity of top-K feature sets across CV repeats
for each model, quantifying how consistently features are selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    return len(a & b) / float(len(a | b))


def k_to_n(n_features: int, k_pct: int) -> int:
    return max(1, int(n_features * (k_pct / 100.0)))


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Within-model SHAP stability across repeats (Jaccard of top-K sets).")
    p.add_argument("--outputs-dir", required=True, help="Root outputs directory")
    p.add_argument("--models", default="rf,xgb_bal", help="Comma-separated models to analyse")
    p.add_argument("--k-pcts", default="1,5,10,20", help="Comma-separated K percentages")
    p.add_argument("--hero-views", action="store_true", help="Restrict to HERO_VIEWS only")
    p.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples for CI over pairwise values")
    return p.parse_args()


def discover_shap_npz(outputs_dir: Path, models: List[str]) -> List[Tuple[str, str, str, Path]]:
    out = []
    rgx = re.compile(r"^shap__(.+?)__(.+?)__(.+?)\.npz$")
    base = outputs_dir / "03_supervised"
    for model in models:
        if model == "rf":
            dir_list = ["rf"]
        elif model == "xgb_bal":
            dir_list = ["xgb_bal"]
        elif model == "xgb":
            dir_list = ["xgb"]
        else:
            dir_list = [model]
        for d in dir_list:
            imp_dir = base / f"tree_models_{d}" / "importance"
            if not imp_dir.exists():
                continue
            for p in sorted(imp_dir.glob("shap__*.npz")):
                m = rgx.match(p.name)
                if not m:
                    continue
                dataset, view, token = m.group(1), m.group(2), m.group(3)
                if model == "rf" and token != "rf":
                    continue
                if model in ("xgb", "xgb_bal") and token != "xgb":
                    continue
                out.append((dataset, view, model, p))
    return out


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    feature_names = z["feature_names.npy"]
    per_repeat = z["mean_abs_shap_per_repeat.npy"]  # (n_repeats, n_features)
    mean_abs = z["mean_abs_shap.npy"]
    return feature_names, per_repeat, mean_abs


def topk_set(feature_names: np.ndarray, shap_row: np.ndarray, k_pct: int) -> Tuple[set, int, int]:
    n_features = int(feature_names.shape[0])
    k_n = k_to_n(n_features, k_pct)
    idx = np.argpartition(shap_row, -k_n)[-k_n:]
    return set(map(str, feature_names[idx].tolist())), k_n, n_features


def main() -> None:
    args = parse_args()
    t0 = time.time()

    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / "06_robustness" / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    k_pcts = [int(x.strip()) for x in args.k_pcts.split(",") if x.strip()]
    hero_set = set(HERO_VIEWS)

    discovered = discover_shap_npz(outputs_dir, models)

    pair_rows = []
    summary_rows = []

    for dataset, view, model, npz_path in discovered:
        if args.hero_views and (dataset, view) not in hero_set:
            continue

        fn, per_rep, mean_abs = load_npz(npz_path)
        n_rep = int(per_rep.shape[0])
        n_feat = int(per_rep.shape[1])

        # Precompute topK sets for all repeats & k
        top_sets: Dict[Tuple[int, int], set] = {}
        k_ns: Dict[int, int] = {}

        for k_pct in k_pcts:
            for r in range(n_rep):
                s, k_n, _ = topk_set(fn, per_rep[r], k_pct)
                top_sets[(k_pct, r)] = s
                k_ns[k_pct] = k_n

            # pairwise jaccard across repeats
            vals = []
            for i in range(n_rep):
                for j in range(i + 1, n_rep):
                    ji = jaccard(top_sets[(k_pct, i)], top_sets[(k_pct, j)])
                    vals.append(ji)
                    pair_rows.append({
                        "dataset": dataset, "view": view, "model": model,
                        "k_pct": k_pct,
                        "k_n": int(k_ns[k_pct]),
                        "n_features": n_feat,
                        "repeat_i": i, "repeat_j": j,
                        "jaccard": float(ji),
                    })
            vals_arr = np.array(vals, dtype=float)
            lo, hi = bootstrap_ci(vals_arr, args.n_bootstrap)
            summary_rows.append({
                "dataset": dataset, "view": view, "model": model,
                "k_pct": k_pct,
                "k_n": int(k_ns[k_pct]),
                "n_features": n_feat,
                "n_pairs": int(vals_arr.size),
                "jaccard_mean": float(np.mean(vals_arr)) if vals_arr.size else float("nan"),
                "ci_lo": lo, "ci_hi": hi,
            })

        # feature selection probabilities (per k)
        df_sel = pd.DataFrame({"feature": fn.astype(str), "mean_abs_shap": mean_abs.astype(float)})
        for k_pct in k_pcts:
            k_n = int(k_ns[k_pct])
            hit = np.zeros(n_feat, dtype=float)
            for r in range(n_rep):
                s = top_sets[(k_pct, r)]
                # mark hits by name (fast map)
            name_to_idx = {str(fn[i]): i for i in range(n_feat)}
            for r in range(n_rep):
                for f in top_sets[(k_pct, r)]:
                    hit[name_to_idx[f]] += 1.0
            df_sel[f"selprob_k{k_pct}"] = hit / float(n_rep)

        out_sel = out_dir / f"shap_selection_prob__{dataset}__{view}__{model}.csv.gz"
        df_sel.to_csv(out_sel, index=False, compression="gzip")

    pair_df = pd.DataFrame(pair_rows)
    pair_path = out_dir / "shap_stability_pairwise.csv.gz"
    pair_df.to_csv(pair_path, index=False, compression="gzip")

    summ_df = pd.DataFrame(summary_rows).sort_values(["dataset", "view", "model", "k_pct"])
    summ_path = out_dir / "shap_stability_summary.csv"
    summ_df.to_csv(summ_path, index=False)

    manifest = {
        "script": "02_shap_stability.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "outputs": {
            "pairwise": str(pair_path),
            "summary": str(summ_path),
        },
        "hashes": {
            "shap_stability_pairwise.csv.gz": sha256_file(pair_path),
            "shap_stability_summary.csv": sha256_file(summ_path),
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    (out_dir / "MANIFEST__stability.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
