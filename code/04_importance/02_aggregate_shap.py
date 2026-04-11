#!/usr/bin/env python
"""\
Phase 4 — 02_aggregate_shap.py

Phase 4 productiser: joins canonical variance (V) with canonical importance (P) and
emits reviewer-proof decoupling artifacts.

CANONICAL INPUTS
- V (Phase 2):
    outputs/02_unsupervised/variance_scores/
      variance_scores__{dataset}__{view}.csv.gz
  Required columns: feature, score, rank
  Optional columns: marginal_variance, marginal_rank, variance_collapsed
  NOTE: any incoming 'percentile' column is ignored as non-authoritative.

- P (Phase 4 per-model; produced by 01_compute_shap_cv.py):
    outputs/04_importance/per_model/
      importance__{dataset}__{view}__{model}.csv.gz
      stability__{dataset}__{view}__{model}.json (optional)
  Required columns: feature, p_score, p_rank, p_rank_pct

OPTIONAL INPUTS
- Normalised bundle NPZ (for between/within diagnostics):
    outputs/bundles/{dataset}_bundle_normalized.npz
  Keys (per all_code_01_bundles): X_{view}, feature_names_{view}, y

OUTPUTS
- outputs/04_importance/joined_vp/
    vp_joined__{dataset}__{view}.csv.gz
      Feature-level V + per-model P + P-consensus + (optional) between/within

    vp_quadrants__{dataset}__{view}.csv.gz
      Feature-level with boolean quadrant flags at k_pct_primary (default 10%)

    vp_kgrid__{dataset}__{view}.csv.gz
      One row per k_pct in grid: J_obs, J_rand, delta_J, tilde_J, DI

    vp_summary__{dataset}__{view}.json
      Deterministic provenance + metrics + regime + risk

- outputs/04_importance/aggregated/
    importance__{dataset}__{view}__P_consensus.csv.gz
    agreement_models__{dataset}__{view}.csv
    topk_overlap_models__{dataset}__{view}.csv
    regime_consensus.csv

DECOUPLING METRICS
- For each K (k_pct):
    J_obs  = Jaccard(TopK(V), TopK(P))
    J_rand = analytic random baseline for two independent top-q selections:
              J_rand = q / (2 - q), where q = k_pct/100
    delta_J = J_obs - J_rand
    tilde_J = (J_obs - J_rand) / (1 - J_rand)   (normalised so random = 0, perfect = 1)
    DI      = 1 - tilde_J
  DI > 1 is allowed (anti-alignment; J_obs < J_rand).

REGIMES (deterministic, minimal arbitrariness)
- At k_pct_primary (default 10%):
    ANTI_ALIGNED if DI > 1.0
    COUPLED      if DI <= 1.0 AND delta_J > 0 AND spearman_rho(V_rank, P_rank) > 0
    DECOUPLED     if |delta_J| <= delta_eps AND |rho| <= rho_eps AND (optional) stability is adequate
    MIXED         otherwise

Consensus regime = majority vote across available models (xgb_bal, rf, ...).
Risk level is derived from consensus regime and agreement/stability.

This script is designed to be the Phase-4 "source of truth" for decoupling artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def deduplicate_by_max(
    df: pd.DataFrame,
    feature_col: str,
    score_col: str,
    rank_col: str = "rank",
    rank_pct_col: str = "rank_pct",
) -> pd.DataFrame:
    """
    Collapse duplicate features by keeping the maximum score per feature.
    This is a practical gene-level resolution strategy for proteomics where
    multiple peptides/isoforms can map to the same gene symbol.
    """
    if feature_col not in df.columns:
        raise KeyError(f"deduplicate_by_max: missing '{feature_col}' column. cols={list(df.columns)}")
    if score_col not in df.columns:
        raise KeyError(f"deduplicate_by_max: missing '{score_col}' column. cols={list(df.columns)}")

    # Fast path: already unique
    if df[feature_col].is_unique:
        return df

    n_before = len(df)
    # Keep row with max score per feature; retain other columns from that row.
    idx = df.groupby(feature_col, sort=False)[score_col].idxmax()
    df2 = df.loc[idx].copy()
    # Re-rank deterministically
    df2[rank_col] = df2[score_col].rank(method="first", ascending=False).astype(int)
    df2[rank_pct_col] = df2[rank_col] / len(df2)
    df2 = df2.sort_values(rank_col, kind="mergesort").reset_index(drop=True)

    print(f" Deduplicated {feature_col}: {n_before} -> {len(df2)} rows ({n_before - len(df2)} collapsed)")
    return df2


# ----------------------------
# Utilities
# ----------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_text_append(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def read_csv_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, compression="gzip")


def write_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ----------------------------
# Manifest handling (append-only)
# ----------------------------

def load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {"created_at": now_iso(), "records": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def add_manifest_record(manifest: Dict, record: Dict) -> None:
    manifest.setdefault("records", []).append(record)
    manifest["updated_at"] = now_iso()


# ----------------------------
# Canonical readers
# ----------------------------

@dataclass(frozen=True)
class VFile:
    path: Path
    dataset: str
    view: str


def discover_v_files(outputs_dir: Path, *, ignore_archive: bool = True) -> List[VFile]:
    root = outputs_dir / "02_unsupervised" / "variance_scores"
    if not root.exists():
        raise FileNotFoundError(f"Cannot find variance_scores directory: {root}")

    out: List[VFile] = []
    for p in root.glob("variance_scores__*.csv.gz"):
        if ignore_archive and "archive" in {x.lower() for x in p.parts}:
            continue
        # name: variance_scores__{dataset}__{view}.csv.gz
        base = p.name.replace("variance_scores__", "").replace(".csv.gz", "")
        parts = base.split("__")
        if len(parts) < 2:
            continue
        dataset = parts[0]
        view = "__".join(parts[1:])
        out.append(VFile(path=p, dataset=dataset, view=view))

    out.sort(key=lambda x: (x.dataset, x.view, str(x.path)))
    return out


def canonicalise_variance_scores(df: pd.DataFrame, *, dataset: str, view: str) -> pd.DataFrame:
    required = {"feature", "score", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in variance_scores for {dataset}/{view}: {sorted(missing)}")

    out = df[["feature", "score", "rank"]].copy()
    out = out.rename(columns={"score": "v_score", "rank": "v_rank"})

    out["feature"] = out["feature"].astype(str)
    out["v_score"] = pd.to_numeric(out["v_score"], errors="coerce")
    out["v_rank"] = pd.to_numeric(out["v_rank"], errors="coerce")

    # Optional fields, if present (defer rank_pct until after deduplication)
    if "marginal_variance" in df.columns:
        out["v_marginal_score"] = pd.to_numeric(df["marginal_variance"], errors="coerce")
    if "marginal_rank" in df.columns:
        out["v_marginal_rank"] = pd.to_numeric(df["marginal_rank"], errors="coerce")
    if "variance_collapsed" in df.columns:
        out["variance_collapsed"] = df["variance_collapsed"].astype(object)

    # Collapse duplicate features (e.g., CCLE proteomics gene symbols)
    out = deduplicate_by_max(
        out,
        feature_col="feature",
        score_col="v_score",
        rank_col="v_rank",
        rank_pct_col="v_rank_pct",
    )

    if out["v_rank"].isna().any():
        raise ValueError(f"NaN ranks detected in V for {dataset}/{view}")

    n = len(out)
    if n == 0:
        raise ValueError(f"Empty V table for {dataset}/{view}")

    # Canonical percentile (re-compute after deduplication to ensure correctness)
    out["v_rank_pct"] = out["v_rank"] / float(n)

    # Compute marginal_rank_pct after deduplication (if column exists)
    if "v_marginal_rank" in out.columns:
        out["v_marginal_rank_pct"] = out["v_marginal_rank"] / float(n)

    # Sort and validate rank range
    out = out.sort_values(["v_rank", "feature"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    if (out["v_rank"] < 1).any() or (out["v_rank"] > n).any():
        raise ValueError(
            f"V rank outside [1, n_features] for {dataset}/{view}. Expected 1..{n}."
        )

    out.insert(0, "view", view)
    out.insert(0, "dataset", dataset)
    return out


def read_p_model(outputs_dir: Path, *, dataset: str, view: str, model: str) -> pd.DataFrame:
    p = outputs_dir / "04_importance" / "per_model" / f"importance__{dataset}__{view}__{model}.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"Missing Phase-4 per-model importance: {p}")

    df = read_csv_gz(p)
    required = {"feature", "p_score", "p_rank", "p_rank_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in P for {dataset}/{view}/{model}: {sorted(missing)}")

    out = df[["feature", "p_score", "p_rank", "p_rank_pct"]].copy()
    out["feature"] = out["feature"].astype(str)
    out["p_score"] = pd.to_numeric(out["p_score"], errors="coerce")
    out["p_rank"] = pd.to_numeric(out["p_rank"], errors="coerce")
    out["p_rank_pct"] = pd.to_numeric(out["p_rank_pct"], errors="coerce")

    # Collapse duplicate features (e.g., CCLE proteomics gene symbols)
    # Expect p_score in per-model importance tables.
    out = deduplicate_by_max(
        out,
        feature_col="feature",
        score_col="p_score",
        rank_col="p_rank",
        rank_pct_col="p_rank_pct",
    )

    n = len(out)
    if n == 0:
        raise ValueError(f"Empty P table for {dataset}/{view}/{model}")
    if out["p_rank"].isna().any():
        raise ValueError(f"NaN ranks detected in P for {dataset}/{view}/{model}")

    # Validate p_rank range
    if (out["p_rank"] < 1).any() or (out["p_rank"] > n).any():
        raise ValueError(
            f"P rank outside [1, n_features] for {dataset}/{view}/{model}. Expected 1..{n}."
        )

    return out


def read_stability(outputs_dir: Path, *, dataset: str, view: str, model: str) -> Dict:
    p = outputs_dir / "04_importance" / "per_model" / f"stability__{dataset}__{view}__{model}.json"
    if not p.exists():
        return {"available": False, "reason": "missing_stability_json", "path": None}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    # Normalise minimal expected fields
    stab = obj.get("stability", {})
    out = {
        "available": bool(stab.get("available", False)),
        "path": str(p),
        "topk_jaccard_across_repeats": stab.get("topk_jaccard_across_repeats", {}) or {},
        "rank_sd_median": stab.get("rank_sd_median", None),
        "rank_sd_mean": stab.get("rank_sd_mean", None),
        "n_repeats": stab.get("n_repeats", None),
    }
    if not out["available"]:
        out["reason"] = stab.get("reason", "stability_unavailable")
    return out


# ----------------------------
# Between/within diagnostic (optional)
# ----------------------------

def find_bundle_npz(outputs_dir: Path, dataset: str) -> Optional[Path]:
    # Prefer canonical location
    p1 = outputs_dir / "bundles" / f"{dataset}_bundle_normalized.npz"
    if p1.exists():
        return p1

    # Fallback search (handles older layouts)
    candidates = list(outputs_dir.rglob(f"{dataset}*_bundle_normalized.npz"))
    return candidates[0] if candidates else None


def compute_between_within_from_bundle(
    bundle_path: Path,
    *,
    dataset: str,
    view: str,
    expected_features: List[str],
    eps: float,
) -> Tuple[pd.DataFrame, Dict]:
    """Return per-feature var_total/var_between/var_within/R_snr/R_frac and a summary dict."""
    d = np.load(bundle_path, allow_pickle=True)

    x_key = f"X_{view}"
    fn_key = f"feature_names_{view}"
    if x_key not in d.files or fn_key not in d.files or "y" not in d.files:
        raise KeyError(
            f"Bundle {bundle_path} missing required keys for view={view}: "
            f"need {x_key}, {fn_key}, y; found {d.files}"
        )

    X = d[x_key]
    y = d["y"]
    feature_names = [str(x) for x in d[fn_key].tolist()]

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape} in {bundle_path}")
    n, p = X.shape

    if len(feature_names) != p:
        raise ValueError(f"feature_names length != n_features: {len(feature_names)} vs {p} in {bundle_path}")

    # Ensure features align with Phase-2/4 tables
    if feature_names != expected_features:
        # Try to align by reindexing (safe only if same set)
        if set(feature_names) != set(expected_features):
            raise ValueError(
                f"Feature set mismatch between bundle and V/P tables for {dataset}/{view}. "
                f"bundle={len(set(feature_names))}, expected={len(set(expected_features))}"
            )
        idx = {f: i for i, f in enumerate(feature_names)}
        order = [idx[f] for f in expected_features]
        X = X[:, order]
        feature_names = expected_features

    # Convert y to integer-coded classes
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n:
        raise ValueError(f"y must be 1D length n_samples; got shape {y.shape} in {bundle_path}")

    classes, y_int = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    # Total variance (population variance; ddof=0)
    mean_all = X.mean(axis=0)
    var_total = X.var(axis=0, ddof=0)

    # Between/within by weighted class decomposition
    var_between = np.zeros(p, dtype=float)
    var_within = np.zeros(p, dtype=float)

    for c in range(n_classes):
        mask = y_int == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        w = n_c / float(n)
        Xc = X[mask, :]
        mean_c = Xc.mean(axis=0)
        var_c = Xc.var(axis=0, ddof=0)
        var_between += w * (mean_c - mean_all) ** 2
        var_within += w * var_c

    # Ratios
    R_snr = var_between / (var_within + eps)
    R_frac = var_between / (var_between + var_within + eps)

    # Decomposition check
    recon = var_between + var_within
    max_abs_err = float(np.max(np.abs(recon - var_total)))
    mean_abs_err = float(np.mean(np.abs(recon - var_total)))

    diag = pd.DataFrame({
        "feature": feature_names,
        "var_total": var_total,
        "var_between": var_between,
        "var_within": var_within,
        "R_snr": R_snr,
        "R_frac": R_frac,
    })

    summary = {
        "bundle_path": str(bundle_path),
        "n_samples": int(n),
        "n_features": int(p),
        "n_classes": int(n_classes),
        "classes": [str(x) for x in classes.tolist()],
        "var_decomp_max_abs_err": max_abs_err,
        "var_decomp_mean_abs_err": mean_abs_err,
        "eps": float(eps),
    }

    return diag, summary


# ----------------------------
# Metrics
# ----------------------------

def topk_set_by_rank(df: pd.DataFrame, rank_col: str, k_n: int) -> set:
    # Returns set of features (strings)
    return set(df.nsmallest(k_n, columns=[rank_col], keep="all")["feature"].tolist()[:k_n])


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    denom = len(a | b)
    return 0.0 if denom == 0 else len(a & b) / denom


def j_rand_analytic(q: float) -> float:
    # Expected Jaccard for two independent size-q selections from large universe
    # J = q/(2-q)
    if q <= 0:
        return 0.0
    if q >= 1:
        return 1.0
    return q / (2.0 - q)


def spearman_rho(rank_a: pd.Series, rank_b: pd.Series) -> float:
    # ranks should be numeric, aligned by index
    return float(rank_a.corr(rank_b, method="spearman"))


def compute_kgrid_metrics(
    joined: pd.DataFrame,
    *,
    p_rank_col: str,
    v_rank_col: str,
    k_pct_grid: List[int],
) -> pd.DataFrame:
    n = len(joined)
    rows = []
    for k_pct in k_pct_grid:
        q = k_pct / 100.0
        k_n = max(1, int(math.floor(q * n)))
        Sv = topk_set_by_rank(joined, v_rank_col, k_n)
        Sp = topk_set_by_rank(joined, p_rank_col, k_n)
        J_obs = jaccard(Sv, Sp)
        J_rand = j_rand_analytic(q)
        delta_J = J_obs - J_rand
        denom = (1.0 - J_rand)
        tilde_J = (delta_J / denom) if denom > 0 else float("nan")
        DI = 1.0 - tilde_J if not math.isnan(tilde_J) else float("nan")
        rows.append({
            "k_pct": int(k_pct),
            "k_n": int(k_n),
            "q": float(q),
            "J_obs": float(J_obs),
            "J_rand": float(J_rand),
            "delta_J": float(delta_J),
            "tilde_J": float(tilde_J),
            "DI": float(DI),
        })
    return pd.DataFrame(rows)


def trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    # assumes x sorted
    if len(x) < 2:
        return float("nan")
    # NumPy 2.0 deprecates np.trapz; prefer np.trapezoid when available.
    auc_fn = getattr(np, "trapezoid", None)
    if auc_fn is None:
        auc_fn = np.trapz
    return float(auc_fn(y=y, x=x))


# ----------------------------
# Regime / risk
# ----------------------------

def classify_regime(
    *,
    DI: float,
    delta_J: float,
    rho: float,
    delta_eps: float,
    rho_eps: float,
    stability_topk_jacc: Optional[float],
    stability_min: float,
) -> str:
    if DI > 1.0:
        return "ANTI_ALIGNED"

    if (DI <= 1.0) and (delta_J > 0.0) and (rho > 0.0):
        return "COUPLED"

    near0 = (abs(delta_J) <= delta_eps) and (abs(rho) <= rho_eps)
    stable = (stability_topk_jacc is not None) and (not math.isnan(stability_topk_jacc)) and (stability_topk_jacc >= stability_min)
    if near0 and stable:
        return "DECOUPLED"

    return "MIXED"


def consensus_vote(labels: Dict[str, str]) -> Tuple[str, float]:
    """Return (consensus_label, agreement_fraction)."""
    if not labels:
        return "MIXED", 0.0
    vals = list(labels.values())
    counts: Dict[str, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    # majority
    best = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0]
    best_label, best_n = best
    if best_n * 2 == len(vals):
        # tie
        return "MIXED", 0.5
    return best_label, best_n / float(len(vals))


def assign_risk(
    *,
    consensus_regime: str,
    model_agreement: float,
    stability_ok: bool,
) -> str:
    if consensus_regime == "ANTI_ALIGNED":
        return "HIGH" if (model_agreement >= 1.0 and stability_ok) else "UNCERTAIN"
    if consensus_regime == "COUPLED":
        return "OK" if (model_agreement >= 0.5 and stability_ok) else "UNCERTAIN"
    if consensus_regime == "DECOUPLED":
        return "NEUTRAL" if stability_ok else "UNCERTAIN"
    return "UNCERTAIN"


# ----------------------------
# Main
# ----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 4: join V with P, compute DI/J and regime/risk outputs.")
    ap.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Project outputs directory.",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="xgb_bal,rf",
        help="Comma-separated models to use (must exist in outputs/04_importance/per_model).",
    )
    ap.add_argument(
        "--k-pct-grid",
        type=str,
        default="1,5,10,20",
        help="K grid in percent for overlap/DI summaries.",
    )
    ap.add_argument(
        "--k-pct-primary",
        type=int,
        default=10,
        help="Primary K percent used for quadrant flags, regime, and diagnostic_rule (default 10).",
    )
    ap.add_argument(
        "--delta-eps",
        type=float,
        default=0.01,
        help="Near-zero threshold for delta_J in DECOUPLED rule (default 0.01).",
    )
    ap.add_argument(
        "--rho-eps",
        type=float,
        default=0.05,
        help="Near-zero threshold for Spearman rho in DECOUPLED rule (default 0.05).",
    )
    ap.add_argument(
        "--stability-min",
        type=float,
        default=0.50,
        help="Minimum top-K Jaccard across repeats to treat ranks as stable (default 0.50).",
    )
    ap.add_argument(
        "--compute-between-within",
        action="store_true",
        help="If set, compute var_total/between/within and R_snr/R_frac from bundle NPZ.",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small epsilon for ratio stability in between/within metrics.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset filter (empty = all V files).",
    )
    ap.add_argument(
        "--views",
        type=str,
        default="",
        help="Comma-separated view filter (empty = all V files).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and compute in-memory only; do not write outputs.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    outputs_dir = Path(args.outputs_dir)
    out_root = outputs_dir / "04_importance"
    joined_dir = out_root / "joined_vp"
    agg_dir = out_root / "aggregated"

    ensure_dir(out_root)
    ensure_dir(joined_dir)
    ensure_dir(agg_dir)

    runlog = out_root / "RUNLOG__04_importance.txt"
    manifest_path = out_root / "MANIFEST__04_importance.json"
    manifest = load_manifest(manifest_path)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    k_pct_grid = [int(x.strip()) for x in args.k_pct_grid.split(",") if x.strip()]

    datasets_filter = set([x.strip() for x in args.datasets.split(",") if x.strip()]) or None
    views_filter = set([x.strip() for x in args.views.split(",") if x.strip()]) or None

    write_text_append(runlog, f"\n[{now_iso()}] START 02_aggregate_shap.py\n")
    write_text_append(runlog, f"outputs_dir={outputs_dir}\n")
    write_text_append(runlog, f"models={models}\n")
    write_text_append(runlog, f"k_pct_grid={k_pct_grid}, k_pct_primary={args.k_pct_primary}\n")
    write_text_append(runlog, f"compute_between_within={args.compute_between_within}\n")
    write_text_append(runlog, f"filters: datasets={datasets_filter}, views={views_filter}\n")

    v_files = discover_v_files(outputs_dir)
    if datasets_filter:
        v_files = [vf for vf in v_files if vf.dataset in datasets_filter]
    if views_filter:
        v_files = [vf for vf in v_files if vf.view in views_filter]

    if not v_files:
        write_text_append(runlog, f"[{now_iso()}] No variance_scores files found (after filters). Exiting.\n")
        return 1

    # Collect per dataset×view summary for regime_consensus.csv
    regime_rows: List[Dict] = []

    n_ok = 0
    for vf in v_files:
        ds, vw = vf.dataset, vf.view
        tag = f"{ds}/{vw}"
        try:
            v_src_sha = sha256_file(vf.path)
            v_raw = read_csv_gz(vf.path)
            V = canonicalise_variance_scores(v_raw, dataset=ds, view=vw)
            n_features = int(len(V))

            # Read P tables (Phase 4 per_model)
            P_models: Dict[str, pd.DataFrame] = {}
            P_sha: Dict[str, str] = {}
            stabilities: Dict[str, Dict] = {}
            for m in models:
                p_path = outputs_dir / "04_importance" / "per_model" / f"importance__{ds}__{vw}__{m}.csv.gz"
                if not p_path.exists():
                    continue
                P_sha[m] = sha256_file(p_path)
                P_models[m] = read_p_model(outputs_dir, dataset=ds, view=vw, model=m)
                stabilities[m] = read_stability(outputs_dir, dataset=ds, view=vw, model=m)

            if not P_models:
                raise FileNotFoundError(f"No Phase-4 per-model P tables found for {tag} under models={models}")

            # Ensure feature sets match across V and all P
            feat_v = V["feature"].tolist()
            feat_set = set(feat_v)
            for m, Pm in P_models.items():
                if set(Pm["feature"].tolist()) != feat_set:
                    raise ValueError(f"Feature set mismatch between V and P for {tag} model={m}")

            # Build wide joined table keyed by feature
            joined = V[[
                "dataset", "view", "feature",
                "v_score", "v_rank", "v_rank_pct",
            ]].copy()

            # Add optional V marginal fields if present
            for c in ["v_marginal_score", "v_marginal_rank", "v_marginal_rank_pct", "variance_collapsed"]:
                if c in V.columns:
                    joined[c] = V[c]

            # Add per-model P columns
            for m, Pm in P_models.items():
                # Align order to V by merging on feature
                tmp = Pm[["feature", "p_score", "p_rank", "p_rank_pct"]].copy()
                joined = joined.merge(tmp, on="feature", how="left", validate="one_to_one", suffixes=("", ""))
                joined = joined.rename(columns={
                    "p_score": f"p_{m}_score",
                    "p_rank": f"p_{m}_rank",
                    "p_rank_pct": f"p_{m}_rank_pct",
                })

            # Build P-consensus rank
            rank_cols = [f"p_{m}_rank" for m in P_models.keys()]
            joined["p_consensus_rank"] = joined[rank_cols].mean(axis=1)
            # Deterministic integer rank: sort by mean-rank then feature
            tmp = joined[["feature", "p_consensus_rank"]].sort_values(["p_consensus_rank", "feature"], ascending=[True, True], kind="mergesort")
            tmp["p_consensus_rank_int"] = np.arange(1, n_features + 1)
            joined = joined.merge(tmp[["feature", "p_consensus_rank_int"]], on="feature", how="left", validate="one_to_one")
            joined["p_consensus_rank_pct"] = joined["p_consensus_rank_int"] / float(n_features)

            # Optional between/within diagnostics
            bw_summary = None
            if args.compute_between_within:
                bundle_path = find_bundle_npz(outputs_dir, ds)
                if not bundle_path:
                    raise FileNotFoundError(f"compute_between_within requested but no bundle NPZ found for dataset={ds}")
                diag_df, bw_summary = compute_between_within_from_bundle(
                    bundle_path,
                    dataset=ds,
                    view=vw,
                    expected_features=feat_v,
                    eps=args.eps,
                )
                joined = joined.merge(diag_df, on="feature", how="left", validate="one_to_one")

            # Compute V vs P metrics
            kgrid_cons = compute_kgrid_metrics(joined, p_rank_col="p_consensus_rank_int", v_rank_col="v_rank", k_pct_grid=k_pct_grid)

            # Per-model V vs P metrics at primary K and rho
            k_primary = int(args.k_pct_primary)
            if k_primary not in set(kgrid_cons["k_pct"].tolist()):
                # Ensure primary K is computed even if not in grid
                kgrid_primary_extra = compute_kgrid_metrics(joined, p_rank_col="p_consensus_rank_int", v_rank_col="v_rank", k_pct_grid=[k_primary])
                kgrid_cons = pd.concat([kgrid_cons, kgrid_primary_extra], ignore_index=True).drop_duplicates(subset=["k_pct"], keep="first")
                kgrid_cons = kgrid_cons.sort_values("k_pct").reset_index(drop=True)

            # DI AUC over the grid (x = q)
            x = kgrid_cons["q"].to_numpy(dtype=float)
            y = kgrid_cons["DI"].to_numpy(dtype=float)
            DI_AUC = trapezoid_auc(x, y)

            # Spearman rho: V vs P ranks (consensus, and per model)
            rho_cons = spearman_rho(joined["v_rank"], joined["p_consensus_rank_int"])

            rho_models: Dict[str, float] = {}
            kgrid_models: Dict[str, pd.DataFrame] = {}
            for m in P_models.keys():
                rho_models[m] = spearman_rho(joined["v_rank"], joined[f"p_{m}_rank"])
                kdf = compute_kgrid_metrics(joined, p_rank_col=f"p_{m}_rank", v_rank_col="v_rank", k_pct_grid=[k_primary])
                kgrid_models[m] = kdf

            # Quadrant flags at primary K
            q = k_primary / 100.0
            k_n = max(1, int(math.floor(q * n_features)))
            Sv = topk_set_by_rank(joined, "v_rank", k_n)
            Sp = topk_set_by_rank(joined, "p_consensus_rank_int", k_n)
            joined["in_topV"] = joined["feature"].isin(Sv)
            joined["in_topP"] = joined["feature"].isin(Sp)
            joined["quad_Vhi_Phi"] = joined["in_topV"] & joined["in_topP"]
            joined["quad_Vhi_Plo"] = joined["in_topV"] & (~joined["in_topP"])
            joined["quad_Vlo_Phi"] = (~joined["in_topV"]) & joined["in_topP"]
            joined["quad_Vlo_Plo"] = (~joined["in_topV"]) & (~joined["in_topP"])

            # Per-model stability (headline at k_primary)
            stability_headline: Dict[str, Dict] = {}
            stability_vals = []
            for m, stab in stabilities.items():
                topk_map = stab.get("topk_jaccard_across_repeats", {}) if stab else {}
                key = str(k_primary)
                topk_j = None
                if isinstance(topk_map, dict) and key in topk_map:
                    try:
                        topk_j = float(topk_map[key])
                    except Exception:
                        topk_j = None
                stability_headline[m] = {
                    "available": bool(stab.get("available", False)),
                    "topk_jaccard_primary": topk_j,
                    "rank_sd_median": stab.get("rank_sd_median", None),
                    "rank_sd_mean": stab.get("rank_sd_mean", None),
                    "n_repeats": stab.get("n_repeats", None),
                    "path": stab.get("path", None),
                }
                if topk_j is not None and (not math.isnan(topk_j)):
                    stability_vals.append(topk_j)

            stability_ok = (len(stability_vals) > 0) and (min(stability_vals) >= args.stability_min)

            # Regime per model + consensus
            krow_cons = kgrid_cons.loc[kgrid_cons["k_pct"] == k_primary].iloc[0].to_dict()
            regimes: Dict[str, str] = {}
            for m in P_models.keys():
                krow_m = kgrid_models[m].iloc[0].to_dict()
                topk_j = stability_headline.get(m, {}).get("topk_jaccard_primary")
                regimes[m] = classify_regime(
                    DI=float(krow_m["DI"]),
                    delta_J=float(krow_m["delta_J"]),
                    rho=float(rho_models[m]),
                    delta_eps=float(args.delta_eps),
                    rho_eps=float(args.rho_eps),
                    stability_topk_jacc=topk_j,
                    stability_min=float(args.stability_min),
                )

            consensus_regime, model_agreement = consensus_vote(regimes)

            # Risk level
            risk_level = assign_risk(
                consensus_regime=consensus_regime,
                model_agreement=float(model_agreement),
                stability_ok=bool(stability_ok),
            )

            diagnostic_rule = {
                "k_pct_primary": int(k_primary),
                "delta_J_10pct": float(krow_cons["delta_J"]) if k_primary == 10 else float(krow_cons["delta_J"]),
                "DI_10pct": float(krow_cons["DI"]) if k_primary == 10 else float(krow_cons["DI"]),
                "spearman_rho": float(rho_cons),
                "triggered": "DI > 1.0" if float(krow_cons["DI"]) > 1.0 else "(see regime rules)",
            }

            # Build vp_kgrid output (consensus)
            vp_kgrid = kgrid_cons.copy()
            vp_kgrid.insert(0, "view", vw)
            vp_kgrid.insert(0, "dataset", ds)

            # Build vp_quadrants (joined + flags) output
            vp_quadrants = joined.copy()
            vp_quadrants.insert(0, "k_pct_primary", int(k_primary))

            # Write per-dataset/view outputs
            out_joined = joined_dir / f"vp_joined__{ds}__{vw}.csv.gz"
            out_quads = joined_dir / f"vp_quadrants__{ds}__{vw}.csv.gz"
            out_kgrid = joined_dir / f"vp_kgrid__{ds}__{vw}.csv.gz"
            out_summary = joined_dir / f"vp_summary__{ds}__{vw}.json"

            # Aggregated outputs
            out_pcons = agg_dir / f"importance__{ds}__{vw}__P_consensus.csv.gz"
            out_agree = agg_dir / f"agreement_models__{ds}__{vw}.csv"
            out_overlap = agg_dir / f"topk_overlap_models__{ds}__{vw}.csv"

            # P-consensus table (minimal)
            pcons_cols = [
                "feature",
                "p_consensus_rank_int",
                "p_consensus_rank_pct",
            ]
            for m in P_models.keys():
                pcons_cols += [f"p_{m}_rank", f"p_{m}_rank_pct"]
            P_cons = joined[pcons_cols].copy()

            # Model agreement and overlap between models
            agree_rows = []
            overlap_rows = []
            if ("xgb_bal" in P_models) and ("rf" in P_models):
                rho_pr = spearman_rho(joined["p_xgb_bal_rank"], joined["p_rf_rank"])
                agree_rows.append({
                    "dataset": ds,
                    "view": vw,
                    "models": "xgb_bal_vs_rf",
                    "spearman_rho": float(rho_pr),
                    "n_features": int(n_features),
                })

                # Top-K overlap between models (not V vs P)
                for k_pct in k_pct_grid:
                    q2 = k_pct / 100.0
                    k_n2 = max(1, int(math.floor(q2 * n_features)))
                    Sa = topk_set_by_rank(joined, "p_xgb_bal_rank", k_n2)
                    Sb = topk_set_by_rank(joined, "p_rf_rank", k_n2)
                    overlap_rows.append({
                        "dataset": ds,
                        "view": vw,
                        "k_pct": int(k_pct),
                        "k_n": int(k_n2),
                        "J_models": float(jaccard(Sa, Sb)),
                    })

            agree_df = pd.DataFrame(agree_rows) if agree_rows else pd.DataFrame(
                [{"dataset": ds, "view": vw, "models": "NA", "spearman_rho": np.nan, "n_features": int(n_features)}]
            )
            overlap_df = pd.DataFrame(overlap_rows) if overlap_rows else pd.DataFrame(
                [{"dataset": ds, "view": vw, "k_pct": k_pct_grid[0], "k_n": max(1, int(math.floor((k_pct_grid[0]/100)*n_features))), "J_models": np.nan}]
            )

            # Summary JSON
            vp_summary = {
                "dataset": ds,
                "view": vw,
                "n_features": int(n_features),
                "generated_at": now_iso(),
                "inputs": {
                    "variance_scores_path": str(vf.path),
                    "variance_scores_sha256": v_src_sha,
                    "per_model_importance": {
                        m: {
                            "path": str(outputs_dir / "04_importance" / "per_model" / f"importance__{ds}__{vw}__{m}.csv.gz"),
                            "sha256": P_sha.get(m),
                        }
                        for m in P_models.keys()
                    },
                },
                "k_pct_grid": [int(x) for x in sorted(set(kgrid_cons["k_pct"].tolist()))],
                "k_pct_primary": int(k_primary),
                "baseline": {
                    "type": "analytic",
                    "J_rand_formula": "q/(2-q)",
                },
                "vp": {
                    "spearman_rho_consensus": float(rho_cons),
                    "spearman_rho_per_model": {m: float(rho_models[m]) for m in rho_models},
                    "DI_AUC": float(DI_AUC),
                    "kgrid": {
                        str(int(row["k_pct"])): {
                            "k_n": int(row["k_n"]),
                            "q": float(row["q"]),
                            "J_obs": float(row["J_obs"]),
                            "J_rand": float(row["J_rand"]),
                            "delta_J": float(row["delta_J"]),
                            "tilde_J": float(row["tilde_J"]),
                            "DI": float(row["DI"]),
                        }
                        for _, row in kgrid_cons.iterrows()
                    },
                },
                "stability": {
                    "k_pct_primary": int(k_primary),
                    "stability_min": float(args.stability_min),
                    "per_model": stability_headline,
                    "stability_ok": bool(stability_ok),
                },
                "regime": {
                    **{m: regimes[m] for m in regimes},
                    "consensus": consensus_regime,
                    "model_agreement": float(model_agreement),
                },
                "risk_level": risk_level,
                "diagnostic_rule": diagnostic_rule,
                "between_within": bw_summary,
                "notes": {
                    "percentile_semantics": "All thresholds use rank_pct = rank / n_features. Incoming 'percentile' columns are ignored.",
                },
            }

            if not args.dry_run:
                write_csv_gz(joined, out_joined)
                write_csv_gz(vp_quadrants, out_quads)
                write_csv_gz(vp_kgrid, out_kgrid)
                write_json(vp_summary, out_summary)

                write_csv_gz(P_cons, out_pcons)
                ensure_dir(out_agree.parent)
                agree_df.to_csv(out_agree, index=False)
                overlap_df.to_csv(out_overlap, index=False)

                # Manifest records
                for typ, outp in [
                    ("vp_joined", out_joined),
                    ("vp_quadrants", out_quads),
                    ("vp_kgrid", out_kgrid),
                    ("vp_summary", out_summary),
                    ("P_consensus", out_pcons),
                    ("agreement_models", out_agree),
                    ("topk_overlap_models", out_overlap),
                ]:
                    add_manifest_record(manifest, {
                        "type": typ,
                        "dataset": ds,
                        "view": vw,
                        "output_path": str(outp),
                        "output_sha256": sha256_file(outp) if outp.exists() else None,
                        "n_features": int(n_features),
                        "generated_at": now_iso(),
                    })

            # Accumulate regime consensus row
            row = {
                "dataset": ds,
                "view": vw,
                "n_features": int(n_features),
                "consensus_regime": consensus_regime,
                "risk_level": risk_level,
                "model_agreement": float(model_agreement),
                "DI_10pct_consensus": float(krow_cons["DI"]),
                "delta_J_10pct_consensus": float(krow_cons["delta_J"]),
                "spearman_rho_consensus": float(rho_cons),
            }
            for m in models:
                if m in regimes:
                    km = kgrid_models[m].iloc[0]
                    row[f"regime_{m}"] = regimes[m]
                    row[f"DI_10pct_{m}"] = float(km["DI"])
                    row[f"spearman_rho_{m}"] = float(rho_models[m])
                    row[f"stability_topk_jaccard_{m}"] = stability_headline.get(m, {}).get("topk_jaccard_primary")
                else:
                    row[f"regime_{m}"] = None
                    row[f"DI_10pct_{m}"] = np.nan
                    row[f"spearman_rho_{m}"] = np.nan
                    row[f"stability_topk_jaccard_{m}"] = np.nan

            regime_rows.append(row)

            write_text_append(runlog, f"[{now_iso()}] OK {tag} -> vp_joined/vp_summary + aggregated\n")
            n_ok += 1

        except Exception as e:
            write_text_append(runlog, f"[{now_iso()}] FAIL {tag} :: {type(e).__name__}: {e}\n")

    # Write cross-view regime consensus table
    if not args.dry_run and regime_rows:
        regime_df = pd.DataFrame(regime_rows).sort_values(["dataset", "view"]).reset_index(drop=True)
        out_regime = agg_dir / "regime_consensus.csv"
        regime_df.to_csv(out_regime, index=False)
        add_manifest_record(manifest, {
            "type": "regime_consensus",
            "output_path": str(out_regime),
            "output_sha256": sha256_file(out_regime) if out_regime.exists() else None,
            "n_rows": int(len(regime_df)),
            "generated_at": now_iso(),
        })

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    write_text_append(runlog, f"[{now_iso()}] DONE n_ok={n_ok}/{len(v_files)}\n")
    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())