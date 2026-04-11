#!/usr/bin/env python3
"""
PHASE 4 — 01_compute_shap_cv.py

Produce canonical per-model importance tables under:
- outputs/04_importance/per_model/importance__{dataset}__{view}__{model}.csv.gz
- outputs/04_importance/per_model/stability__{dataset}__{view}__{model}.json

Default mode: IMPORT + CANONICALISE Phase-3 importance files
Reads (Phase 3)
- outputs/03_supervised/{tree_dir}/importance/prediction_importance__{dataset}__{view}__{suffix}.csv.gz
- outputs/03_supervised/{tree_dir}/importance/shap__{dataset}__{view}__{suffix}.npz  (optional, for stability)

Key contracts (reviewer-proof)
- Model identity is defined by directory, NOT filename suffix:
    tree_models_xgb_bal -> model=xgb_bal
    tree_models_rf      -> model=rf
    tree_models_xgb      -> model=xgb (if used)
- Incoming "percentile" columns are non-authoritative.
- Authoritative percentile is always:
    p_rank_pct = p_rank / n_features    (in (0,1])

This script deliberately does NOT join with V or compute DI/J.
Those are handled by Phase 4 productiser: 02_aggregate_shap.py

Author: variance-prediction paradox pipeline
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Registry (keep consistent with Phase 3)
# -----------------------------
VIEW_REGISTRY = {
    "mlomics": {"core_views": ["mRNA", "miRNA", "methylation", "CNV"], "sensitivity_views": [], "analysis_role": "primary"},
    "ibdmdb":  {"core_views": ["MGX", "MGX_func", "MPX", "MBX"],       "sensitivity_views": ["MGX_CLR"], "analysis_role": "primary"},
    "ccle":    {"core_views": ["mRNA", "CNV", "proteomics"],          "sensitivity_views": [], "analysis_role": "primary"},
    "tcga_gbm":{"core_views": ["mRNA", "methylation", "CNV"],         "sensitivity_views": ["methylation_Mval"], "analysis_role": "sensitivity"},
}

DEFAULT_K_PCTS = [1, 5, 10, 20]


# -----------------------------
# Helpers (pattern-matched to Phase 3)
# -----------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_views(dataset: str, which: str) -> List[str]:
    if dataset not in VIEW_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. Options: {', '.join(sorted(VIEW_REGISTRY))}")
    core = VIEW_REGISTRY[dataset]["core_views"]
    sens = VIEW_REGISTRY[dataset]["sensitivity_views"]
    if which == "core":
        return list(core)
    if which == "all":
        return list(core) + list(sens)
    if which == "sensitivity":
        return list(sens)
    raise ValueError("views must be one of: core, all, sensitivity")


def rank_desc(values: np.ndarray) -> np.ndarray:
    """
    Return 0-based rank positions (0 = largest value).
    Uses stable mergesort for determinism.
    """
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(values.size, dtype=np.int32)
    return ranks


def spearman_via_ranks(a: np.ndarray, b: np.ndarray) -> float:
    ra = rank_desc(a).astype(np.float64)
    rb = rank_desc(b).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum()) * np.sqrt((rb * rb).sum()))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def topk_set_from_scores(scores: np.ndarray, k_frac: float) -> set:
    n = scores.size
    k = max(1, int(math.ceil(k_frac * n)))
    idx = np.argsort(-scores, kind="mergesort")[:k]
    return set(idx.tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union > 0 else float("nan")


def mean_pairwise(values: List[float]) -> float:
    return float(np.nanmean(values)) if values else float("nan")


# -----------------------------
# Discovery
# -----------------------------
MODEL_DIR_MAP = {
    "tree_models_xgb_bal": "xgb_bal",
    "tree_models_xgb": "xgb",
    "tree_models_rf": "rf",
}

PRED_IMPORT_RE = re.compile(
    r"^prediction_importance__(?P<dataset>mlomics|ibdmdb|ccle|tcga_gbm)__(?P<view>.+)__(?P<suffix>[^_]+)\.csv\.gz$"
)
SHAP_NPZ_RE = re.compile(
    r"^shap__(?P<dataset>mlomics|ibdmdb|ccle|tcga_gbm)__(?P<view>.+)__(?P<suffix>[^_]+)\.npz$"
)


@dataclass(frozen=True)
class PredFile:
    path: Path
    dataset: str
    view: str
    model: str  # canonical model name (xgb_bal, rf, xgb)


def infer_model_from_path(p: Path) -> Optional[str]:
    parts = {x.lower() for x in p.parts}
    for dname, model in MODEL_DIR_MAP.items():
        if dname.lower() in parts:
            return model
    return None


def discover_prediction_importance_files(
    phase3_root: Path,
    allowed_pairs: Optional[Set[Tuple[str, str]]],
    models: Optional[Set[str]],
    ignore_archive: bool,
) -> List[PredFile]:
    files: List[PredFile] = []
    for p in phase3_root.rglob("prediction_importance__*.csv.gz"):
        if ignore_archive and any("archive" == x.lower() for x in p.parts):
            continue
        m = PRED_IMPORT_RE.match(p.name)
        if not m:
            continue

        model = infer_model_from_path(p)
        if model is None:
            continue
        if models and model not in models:
            continue

        ds = m.group("dataset")
        view = m.group("view")
        if allowed_pairs is not None and (ds, view) not in allowed_pairs:
            continue

        files.append(PredFile(path=p, dataset=ds, view=view, model=model))

    files.sort(key=lambda x: (x.dataset, x.view, x.model, str(x.path)))
    return files


def find_matching_shap_npz(pred: PredFile, phase3_root: Path) -> Optional[Path]:
    # Restrict search to the corresponding family directory for determinism
    model_dir = None
    for dname, m in MODEL_DIR_MAP.items():
        if m == pred.model:
            model_dir = dname
            break
    if model_dir is None:
        return None

    cand_dir = phase3_root / model_dir / "importance"
    if not cand_dir.exists():
        return None

    # Directory disambiguates suffix (xgb_bal still uses __xgb suffix sometimes)
    for p in cand_dir.glob(f"shap__{pred.dataset}__{pred.view}__*.npz"):
        if SHAP_NPZ_RE.match(p.name):
            return p
    return None


# -----------------------------
# Canonicalisation
# -----------------------------
def canonicalise_prediction_importance(df: pd.DataFrame, *, dataset: str, view: str, model: str) -> pd.DataFrame:
    required = {"feature", "importance", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction_importance for {dataset}/{view}/{model}: {sorted(missing)}")

    out = df[["feature", "importance", "rank"]].copy()
    out = out.rename(columns={"importance": "p_score", "rank": "p_rank"})

    out["feature"] = out["feature"].astype(str)
    out["p_score"] = pd.to_numeric(out["p_score"], errors="coerce")
    out["p_rank"] = pd.to_numeric(out["p_rank"], errors="coerce")

    if out["p_rank"].isna().any():
        raise ValueError(f"NaN ranks detected in {dataset}/{view}/{model}")

    n = int(out.shape[0])
    if n == 0:
        raise ValueError(f"Empty importance table in {dataset}/{view}/{model}")

    # Authoritative percentile semantics (ignore upstream percentile entirely)
    out["p_rank_pct"] = out["p_rank"] / float(n)

    # Enforce 1..n rank convention to avoid silent off-by-one
    if (out["p_rank"] < 1).any() or (out["p_rank"] > n).any():
        raise ValueError(
            f"Rank outside [1, n_features] detected for {dataset}/{view}/{model}. "
            f"Expected rank in 1..{n}."
        )

    out.insert(0, "model", model)
    out.insert(0, "view", view)
    out.insert(0, "dataset", dataset)

    out = out.sort_values(["p_rank", "feature"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    return out


# -----------------------------
# Stability (cheap; repeats-aware via SHAP NPZ)
# -----------------------------
def compute_stability_from_shap_npz(shap_path: Path, k_pcts: List[int]) -> Dict[str, object]:
    z = np.load(str(shap_path), allow_pickle=False)
    keys = set(z.files)
    if "feature_names" not in keys:
        raise ValueError(f"Missing 'feature_names' in {shap_path}")

    if "mean_abs_shap_per_repeat" not in keys:
        return {
            "available": False,
            "reason": "mean_abs_shap_per_repeat missing",
            "source": str(shap_path),
            "n_repeats": None,
            "topk_jaccard_across_repeats": {},
            "pairwise_spearman_mean": None,
            "pairwise_spearman_min": None,
            "rank_sd_median": None,
            "rank_sd_mean": None,
        }

    per_rep = z["mean_abs_shap_per_repeat"].astype(np.float32)  # (n_repeats, n_features)
    if per_rep.ndim != 2:
        raise ValueError(f"mean_abs_shap_per_repeat expected 2D, got shape={per_rep.shape} in {shap_path}")

    n_rep, n_feat = per_rep.shape

    # Pairwise repeat stability (same logic style as Phase 3 eval)
    sp_vals: List[float] = []
    jac_vals: Dict[int, List[float]] = {K: [] for K in k_pcts}

    # Rank SD across repeats (0-based ranks; 0=best)
    ranks = np.zeros((n_rep, n_feat), dtype=np.float32)
    for r in range(n_rep):
        ranks[r, :] = rank_desc(per_rep[r, :]).astype(np.float32)
    rank_sd = np.std(ranks, axis=0, ddof=0)
    rank_sd_median = float(np.median(rank_sd))
    rank_sd_mean = float(np.mean(rank_sd))

    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            sp_vals.append(spearman_via_ranks(per_rep[i], per_rep[j]))
            for K in k_pcts:
                q = K / 100.0
                A = topk_set_from_scores(per_rep[i], q)
                B = topk_set_from_scores(per_rep[j], q)
                jac_vals[K].append(jaccard(A, B))

    topk_j = {str(K): mean_pairwise(jac_vals[K]) for K in k_pcts}

    return {
        "available": True,
        "source": str(shap_path),
        "n_repeats": int(n_rep),
        "n_features": int(n_feat),
        "topk_jaccard_across_repeats": topk_j,
        "pairwise_spearman_mean": mean_pairwise(sp_vals),
        "pairwise_spearman_min": float(np.nanmin(sp_vals)) if sp_vals else float("nan"),
        "rank_sd_median": rank_sd_median,
        "rank_sd_mean": rank_sd_mean,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: canonical per-model importance tables (import+canonicalise from Phase 3)."
    )
    parser.add_argument("--dataset", type=str, default="all",
                        help=f"Dataset name or 'all'. Options: {', '.join(sorted(VIEW_REGISTRY))}")
    parser.add_argument("--views", type=str, default="core", help="Which views: core | all | sensitivity")
    parser.add_argument("--models", type=str, default="xgb_bal,rf",
                        help="Comma-separated models to include (default: xgb_bal,rf).")
    parser.add_argument("--k-pcts", type=str, default="1,5,10,20",
                        help="Comma-separated K%% values for stability (default: 1,5,10,20).")

    parser.add_argument("--repo-root", type=str, default=".", help="Repository root.")
    parser.add_argument("--phase3-root", type=str, default="outputs/03_supervised", help="Phase 3 root (relative to repo).")
    parser.add_argument("--out-root", type=str, default="outputs/04_importance", help="Phase 4 root (relative to repo).")

    parser.add_argument("--copy-shap-tensors", action="store_true",
                        help="If set, copy shap__*.npz into outputs/04_importance/per_model/. Default OFF.")
    parser.add_argument("--ignore-archive", action="store_true",
                        help="If set, ignore any paths containing \\archive.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover and validate only; do not write outputs.")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    phase3_root = (repo_root / args.phase3_root).resolve()
    out_root = (repo_root / args.out_root).resolve()
    per_model_dir = out_root / "per_model"
    ensure_dir(per_model_dir)

    runlog = out_root / "RUNLOG__04_importance.txt"
    manifest_path = out_root / "MANIFEST__04_importance.json"

    k_pcts = [int(x.strip()) for x in args.k_pcts.split(",") if x.strip()]
    models = {x.strip() for x in args.models.split(",") if x.strip()} or None

    # Allowed dataset×view pairs
    datasets = sorted(VIEW_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    allowed_pairs: Set[Tuple[str, str]] = set()
    for ds in datasets:
        for v in resolve_views(ds, args.views):
            allowed_pairs.add((ds, v))

    # Start log
    ensure_dir(out_root)
    with runlog.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_iso()}] START 01_compute_shap_cv.py\n")
        f.write(f"repo_root={repo_root}\n")
        f.write(f"phase3_root={phase3_root}\n")
        f.write(f"out_root={out_root}\n")
        f.write(f"dataset={args.dataset}, views={args.views}, models={models}\n")
        f.write(f"k_pcts={k_pcts}\n")
        f.write(f"copy_shap_tensors={args.copy_shap_tensors}, dry_run={args.dry_run}\n")

    if not phase3_root.exists():
        raise FileNotFoundError(f"Phase 3 root not found: {phase3_root}")

    pred_files = discover_prediction_importance_files(
        phase3_root=phase3_root,
        allowed_pairs=allowed_pairs,
        models=models,
        ignore_archive=args.ignore_archive,
    )

    if not pred_files:
        with runlog.open("a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] No prediction_importance files found. Exiting.\n")
        return

    # Load or init manifest
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"created_at": now_iso(), "records": []}

    n_ok = 0
    for pf in pred_files:
        try:
            df = pd.read_csv(pf.path, compression="gzip")
            canon = canonicalise_prediction_importance(df, dataset=pf.dataset, view=pf.view, model=pf.model)

            out_csv = per_model_dir / f"importance__{pf.dataset}__{pf.view}__{pf.model}.csv.gz"
            stability_path = per_model_dir / f"stability__{pf.dataset}__{pf.view}__{pf.model}.json"

            shap_src = find_matching_shap_npz(pf, phase3_root=phase3_root)
            stability = None
            copied_shap = None

            if shap_src and shap_src.exists():
                try:
                    stability = compute_stability_from_shap_npz(shap_src, k_pcts=k_pcts)
                except Exception as e:
                    stability = {
                        "available": False,
                        "reason": f"stability_failed: {type(e).__name__}: {e}",
                        "source": str(shap_src),
                        "n_repeats": None,
                        "topk_jaccard_across_repeats": {},
                        "pairwise_spearman_mean": None,
                        "pairwise_spearman_min": None,
                        "rank_sd_median": None,
                        "rank_sd_mean": None,
                    }

                if args.copy_shap_tensors and not args.dry_run:
                    dst = per_model_dir / f"shap_oof__{pf.dataset}__{pf.view}__{pf.model}.npz"
                    shutil.copy2(shap_src, dst)
                    copied_shap = str(dst)

            if stability is None:
                stability = {
                    "available": False,
                    "reason": "no_shap_npz_found",
                    "source": None,
                    "n_repeats": None,
                    "topk_jaccard_across_repeats": {},
                    "pairwise_spearman_mean": None,
                    "pairwise_spearman_min": None,
                    "rank_sd_median": None,
                    "rank_sd_mean": None,
                }

            if not args.dry_run:
                canon.to_csv(out_csv, index=False, compression="gzip")
                payload = {
                    "dataset": pf.dataset,
                    "view": pf.view,
                    "model": pf.model,
                    "k_pcts": k_pcts,
                    "source_prediction_importance": str(pf.path),
                    "source_shap_npz": str(shap_src) if shap_src else None,
                    "copied_shap_npz": copied_shap,
                    "stability": stability,
                    "generated_at": now_iso(),
                    "script": "04_importance/01_compute_shap_cv.py",
                }
                stability_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

                manifest["records"].append({
                    "type": "per_model_importance",
                    "dataset": pf.dataset,
                    "view": pf.view,
                    "model": pf.model,
                    "output_path": str(out_csv),
                    "stability_path": str(stability_path),
                    "source_sha256": sha256_file(pf.path),
                    "output_sha256": sha256_file(out_csv),
                    "n_features": int(canon.shape[0]),
                    "generated_at": now_iso(),
                })
                manifest["updated_at"] = now_iso()

            with runlog.open("a", encoding="utf-8") as f:
                f.write(f"[{now_iso()}] OK {pf.dataset}/{pf.view}/{pf.model} -> {out_csv.name}\n")
            n_ok += 1

        except Exception as e:
            with runlog.open("a", encoding="utf-8") as f:
                f.write(f"[{now_iso()}] FAIL {pf.dataset}/{pf.view}/{pf.model} :: {type(e).__name__}: {e}\n")

    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with runlog.open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] DONE n_ok={n_ok}/{len(pred_files)}\n")


if __name__ == "__main__":
    main()