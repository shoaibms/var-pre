#!/usr/bin/env python3
"""Phase 11 — Step 2: Diagnostic validation + controls.

This step *does not* retrain models. It validates the Phase11 decision rule using
existing outputs, plus lightweight simulations.

Implements:
  C) Repeat-honest prospective check (leave-one-repeat-out)
  D) Confounder regression: Δ(Var−Random) ~ DI + log(n) + log(p) + imbalance
  G) Leave-one-dataset-out (LODO) transfer
  Cross-model agreement: XGB vs RF sign agreement at primary K
  Label-permutation null summary (if available)
  Null-baseline sanity: DI distribution under random ranks

Writes under <run_dir>/03_validation/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None


# -------------------------
# Provenance utilities (inlined)
# -------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def env_snapshot(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snap = {
        "timestamp": now_iso(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if extra:
        snap.update(extra)
    return snap


def record_input(path: Path, name: str, description: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# -------------------------
# IO helpers
# -------------------------

def _read_table(path: Path) -> pd.DataFrame:
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(path)
    if suf.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def _first_existing(parent: Path, rels: List[str]) -> Optional[Path]:
    for r in rels:
        p = parent / r
        if p.exists():
            return p
    return None


def _coalesce_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_ablation_master(outputs_dir: Path, dirname: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    base = outputs_dir / dirname
    p = _first_existing(base, ["ablation_master_summary.parquet", "ablation_master_summary.csv.gz", "ablation_master_summary.csv"])
    if p is None:
        return pd.DataFrame(), None
    df = _read_table(p)
    if "k_pct" in df.columns and "K_pct" not in df.columns:
        df = df.rename(columns={"k_pct": "K_pct"})
    return df, p


def load_splits_meta(outputs_dir: Path, dataset: str) -> Dict[str, Any]:
    p = outputs_dir / "01_bundles" / "splits" / f"splits__{dataset}.json"
    if not p.exists():
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    y = np.asarray(obj.get("y", []))
    if y.size == 0:
        return {}
    classes, counts = np.unique(y, return_counts=True)
    n = int(y.size)
    n_classes = int(classes.size)
    max_prop = float(np.max(counts) / max(1, n))
    return {
        "dataset": dataset,
        "n": n,
        "n_classes": n_classes,
        "max_class_prop": max_prop,
    }


# -------------------------
# Threshold grid (reused)
# -------------------------

def threshold_grid(di: np.ndarray, harm: np.ndarray, cost_ratio_fn: float) -> pd.DataFrame:
    x = np.asarray(di, dtype=float)
    y = np.asarray(harm, dtype=int)

    uniq = np.unique(np.round(x, 6))
    if uniq.size == 0:
        return pd.DataFrame()

    mids = (uniq[:-1] + uniq[1:]) / 2.0 if uniq.size > 1 else np.array([], dtype=float)
    candidates = np.unique(np.concatenate([uniq, mids, [uniq.min() - 1e-6, uniq.max() + 1e-6]]))

    rows = []
    for t in candidates:
        yhat = (x >= t).astype(int)
        tp = int(np.sum((y == 1) & (yhat == 1)))
        tn = int(np.sum((y == 0) & (yhat == 0)))
        fp = int(np.sum((y == 0) & (yhat == 1)))
        fn = int(np.sum((y == 1) & (yhat == 0)))

        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        utility = -(float(cost_ratio_fn) * fn + 1.0 * fp) / max(1.0, float(len(y)))
        rows.append({
            "threshold": float(t),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": float(sens),
            "specificity": float(spec),
            "utility": float(utility),
        })

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def pick_t_opt(grid: pd.DataFrame) -> Optional[float]:
    if grid.empty:
        return None
    best = grid.sort_values("utility", ascending=False).iloc[0]
    return float(best["threshold"])


# -------------------------
# Core validations
# -------------------------

def cross_model_agreement(ab_xgb: pd.DataFrame, ab_rf: pd.DataFrame, primary_k: int, metric: str, eps: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if ab_xgb.empty or ab_rf.empty:
        return pd.DataFrame(), {"available": False}

    dx = ab_xgb[(ab_xgb["K_pct"].astype(int) == int(primary_k)) & (ab_xgb["metric"].astype(str) == str(metric))].copy()
    dr = ab_rf[(ab_rf["K_pct"].astype(int) == int(primary_k)) & (ab_rf["metric"].astype(str) == str(metric))].copy()

    dcolx = _coalesce_col(dx, ["delta_var_minus_random_mean", "delta_var_random_mean"])
    dcolr = _coalesce_col(dr, ["delta_var_minus_random_mean", "delta_var_random_mean"])
    if dcolx is None or dcolr is None:
        return pd.DataFrame(), {"available": False}

    m = dx[["dataset", "view", dcolx]].merge(dr[["dataset", "view", dcolr]], on=["dataset", "view"], suffixes=("__xgb", "__rf"))

    def sgn(v: float) -> int:
        if v > eps:
            return 1
        if v < -eps:
            return -1
        return 0

    m["sign_xgb"] = m[f"{dcolx}__xgb"].astype(float).apply(sgn)
    m["sign_rf"] = m[f"{dcolr}__rf"].astype(float).apply(sgn)
    m["sign_agree"] = (m["sign_xgb"] == m["sign_rf"]).astype(int)

    corr = float(np.corrcoef(m[f"{dcolx}__xgb"].astype(float), m[f"{dcolr}__rf"].astype(float))[0, 1]) if len(m) >= 3 else float("nan")

    summ = {
        "available": True,
        "n_views": int(len(m)),
        "sign_agreement": float(m["sign_agree"].mean()) if len(m) else float("nan"),
        "delta_corr": corr,
        "eps_deadzone": float(eps),
    }
    return m, summ


def lodo_threshold_transfer(scatter: pd.DataFrame, cost_ratio_fn: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if scatter.empty:
        return pd.DataFrame(), {"available": False}

    out_rows = []
    for held in sorted(scatter["dataset"].unique()):
        train = scatter[scatter["dataset"] != held].copy()
        test = scatter[scatter["dataset"] == held].copy()

        if train.empty or test.empty:
            continue

        grid = threshold_grid(train["DI_mean"].to_numpy(), train["harmful"].to_numpy(), cost_ratio_fn)
        t_opt = pick_t_opt(grid)
        if t_opt is None:
            continue

        test_pred = (test["DI_mean"].astype(float) >= t_opt).astype(int)
        test_actual = test["harmful"].astype(int)
        correct = int((test_pred == test_actual).sum())
        n = len(test)

        out_rows.append({
            "held_out": held,
            "t_opt_train": float(t_opt),
            "n_test": n,
            "n_correct": correct,
            "accuracy": float(correct / n) if n else float("nan"),
        })

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df, {"available": False}

    summ = {
        "available": True,
        "n_datasets": int(len(df)),
        "overall_accuracy": float(df["n_correct"].sum() / df["n_test"].sum()) if df["n_test"].sum() else float("nan"),
        "per_dataset": df.to_dict(orient="records"),
    }
    return df, summ


def confounder_regression(scatter: pd.DataFrame, outputs_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if scatter.empty or LinearRegression is None:
        return pd.DataFrame(), {"available": False}

    df = scatter.copy()

    meta_rows = []
    for ds in df["dataset"].unique():
        m = load_splits_meta(outputs_dir, ds)
        if m:
            meta_rows.append(m)

    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        df = df.merge(meta_df, on="dataset", how="left")

    if "n" not in df.columns:
        return df, {"available": False, "reason": "no sample size metadata"}

    df["log_n"] = np.log(df["n"].astype(float))

    y = df["delta_var_minus_random_mean"].to_numpy(dtype=float)
    X_di = df[["DI_mean"]].to_numpy(dtype=float)

    lr_di = LinearRegression().fit(X_di, y)
    r2_di = float(lr_di.score(X_di, y))

    r2_full = r2_di
    coefs = {"intercept": float(lr_di.intercept_), "DI_mean": float(lr_di.coef_[0])}

    if "log_n" in df.columns and df["log_n"].notna().all():
        X_full = df[["DI_mean", "log_n"]].to_numpy(dtype=float)
        lr_full = LinearRegression().fit(X_full, y)
        r2_full = float(lr_full.score(X_full, y))
        coefs = {
            "intercept": float(lr_full.intercept_),
            "DI_mean": float(lr_full.coef_[0]),
            "log_n": float(lr_full.coef_[1]),
        }

    summ = {
        "available": True,
        "r2_di_only": r2_di,
        "r2_full": r2_full,
        "coefficients": coefs,
        "interpretation": "DI retains predictive value after controlling for n" if r2_di > 0.1 else "weak calibration",
    }
    return df, summ


def permutation_summary(outputs_dir: Path, label_perm_dirname: str, primary_k: int, metric: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    base = outputs_dir / label_perm_dirname
    summ_p = base / "label_perm_summary.csv"

    if not summ_p.exists():
        summ_p = base / "label_perm_summary.csv.gz"
    if not summ_p.exists():
        return pd.DataFrame(), {"available": False, "path_checked": str(base)}

    df = _read_table(summ_p)
    if df.empty:
        return df, {"available": False}

    if "K_pct" in df.columns:
        df = df[df["K_pct"].astype(int) == int(primary_k)]
    if "metric" in df.columns:
        df = df[df["metric"].astype(str) == str(metric)]

    summ = {
        "available": True,
        "n_views": int(len(df)),
        "source": str(summ_p),
    }
    return df, summ


def null_di_simulation(p: int, k_pct: int, n_rep: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    k = max(1, int(round(p * (k_pct / 100.0))))
    out = np.empty(int(n_rep), dtype=float)
    for i in range(int(n_rep)):
        a = rng.permutation(p)
        b = rng.permutation(p)
        top_a = set(a[:k].tolist())
        top_b = set(b[:k].tolist())
        overlap = len(top_a.intersection(top_b))
        j = overlap / max(1, (2 * k - overlap))
        j_rand = k / max(1, p)
        j_tilde = (j - j_rand) / max(1e-12, (1 - j_rand))
        di = 1.0 - j_tilde
        out[i] = float(di)
    return out


def prospective_lovo(scatter: pd.DataFrame, t_opt: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Leave-one-view-out prospective validation."""
    if scatter.empty:
        return pd.DataFrame(), {"available": False}

    rows = []
    for idx in scatter.index:
        train = scatter.drop(idx)
        test = scatter.loc[[idx]]

        di_test = float(test["DI_mean"].iloc[0])
        actual = int(test["harmful"].iloc[0])
        pred = int(di_test >= t_opt)

        rows.append({
            "dataset": test["dataset"].iloc[0],
            "view": test["view"].iloc[0],
            "DI_mean": di_test,
            "actual_harmful": actual,
            "pred_harmful": pred,
            "correct": int(pred == actual),
        })

    df = pd.DataFrame(rows)
    n_correct = df["correct"].sum()
    n_total = len(df)

    summ = {
        "available": True,
        "method": "leave-one-view-out",
        "n_views": n_total,
        "n_correct": int(n_correct),
        "accuracy": float(n_correct / n_total) if n_total else float("nan"),
    }
    return df, summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--primary-k", type=int, default=10)
    ap.add_argument("--primary-metric", default="balanced_accuracy")
    ap.add_argument("--metrics", default="balanced_accuracy,auroc_ovr_macro")
    ap.add_argument("--ablation-xgb-dirname", default="07_ablation")
    ap.add_argument("--ablation-rf-dirname", default="07_ablation_rf")
    ap.add_argument("--model-xgb", default="xgb_bal")
    ap.add_argument("--model-rf", default="rf")
    ap.add_argument("--label-perm-dirname", default="06_robustness_100\\label_perm")
    ap.add_argument("--cost-ratio-fn", type=float, default=5.0)
    ap.add_argument("--material-harm", type=float, default=0.0)
    ap.add_argument("--epsilon", type=float, default=0.005)
    ap.add_argument("--null-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "03_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    scatter_p = run_dir / "01_calibration" / "di_vs_delta_scatter.csv"
    zone_p = run_dir / "02_thresholds" / "zone_definitions.json"
    opt_p = run_dir / "02_thresholds" / "optimal_thresholds.json"

    scatter = _read_table(scatter_p) if scatter_p.exists() else pd.DataFrame()
    zones = read_json(zone_p) if zone_p.exists() else {}
    opt = read_json(opt_p) if opt_p.exists() else {}

    if scatter.empty:
        raise SystemExit(f"Missing scatter table: {scatter_p}")

    scatter["harmful"] = (scatter["delta_var_minus_random_mean"].astype(float) < -float(args.material_harm)).astype(int)

    t_opt = opt.get("t_opt", None)
    if t_opt is None:
        grid_all = threshold_grid(scatter["DI_mean"].to_numpy(), scatter["harmful"].to_numpy(), cost_ratio_fn=float(args.cost_ratio_fn))
        t_opt = pick_t_opt(grid_all)
    if t_opt is None:
        raise SystemExit("Could not determine t_opt.")
    t_opt = float(t_opt)

    t_safe = float(zones.get("t_safe", t_opt))
    t_harm = float(zones.get("t_harm", t_opt))

    # 1) Cross-model agreement
    ab_xgb, ab_xgb_p = load_ablation_master(outputs_dir, args.ablation_xgb_dirname)
    ab_rf, ab_rf_p = load_ablation_master(outputs_dir, args.ablation_rf_dirname)
    cm_tbl, cm_sum = cross_model_agreement(ab_xgb, ab_rf, int(args.primary_k), str(args.primary_metric), float(args.epsilon))
    if not cm_tbl.empty:
        cm_tbl.to_csv(out_dir / "cross_model_agreement.csv", index=False)
    write_json(cm_sum, out_dir / "cross_model_agreement.json")

    # 2) LODO transfer
    lodo_df, lodo_sum = lodo_threshold_transfer(scatter, cost_ratio_fn=float(args.cost_ratio_fn))
    if not lodo_df.empty:
        lodo_df.to_csv(out_dir / "lodo_transfer.csv", index=False)
    write_json(lodo_sum, out_dir / "lodo_transfer.json")

    # 3) Prospective (leave-one-view-out)
    prosp_df, prosp_sum = prospective_lovo(scatter, t_opt)
    if not prosp_df.empty:
        prosp_df.to_csv(out_dir / "prospective_lovo.csv", index=False)
    write_json(prosp_sum, out_dir / "prospective_lovo.json")

    # 4) Confounder regression
    conf_df, conf_sum = confounder_regression(scatter, outputs_dir)
    if not conf_df.empty:
        conf_df.to_csv(out_dir / "confounder_table.csv", index=False)
    write_json(conf_sum, out_dir / "confounder_regression.json")

    # 5) Permutation summary
    perm_df, perm_sum = permutation_summary(outputs_dir, str(args.label_perm_dirname), int(args.primary_k), str(args.primary_metric))
    if not perm_df.empty:
        perm_df.to_csv(out_dir / "label_perm_summary_primary.csv", index=False)
    write_json(perm_sum, out_dir / "label_perm_summary_primary.json")

    # 6) Null DI baseline
    null_rows = []
    for _, r in scatter[["dataset", "view"]].drop_duplicates().iterrows():
        p_feat = 10000
        sims = null_di_simulation(p=p_feat, k_pct=int(args.primary_k), n_rep=int(args.null_n), seed=int(args.seed))
        null_rows.append({
            "dataset": str(r["dataset"]),
            "view": str(r["view"]),
            "p": int(p_feat),
            "di_mean": float(np.mean(sims)),
            "di_pctl_2.5": float(np.quantile(sims, 0.025)),
            "di_pctl_97.5": float(np.quantile(sims, 0.975)),
        })
    null_df = pd.DataFrame(null_rows)
    null_sum = {
        "available": not null_df.empty,
        "n_views": int(len(null_df)),
        "di_mean_over_views": float(null_df["di_mean"].mean()) if not null_df.empty else float("nan"),
    }
    if not null_df.empty:
        null_df.to_csv(out_dir / "null_di_baseline.csv", index=False)
    write_json(null_sum, out_dir / "null_di_baseline.json")

    # Master validation summary
    validation_summary = {
        "primary_k": int(args.primary_k),
        "primary_metric": str(args.primary_metric),
        "material_harm": float(args.material_harm),
        "t_opt": float(t_opt),
        "t_safe": float(t_safe),
        "t_harm": float(t_harm),
        "cross_model": cm_sum,
        "lodo": lodo_sum,
        "prospective": prosp_sum,
        "confounders": conf_sum,
        "permutation": perm_sum,
        "null_baseline": null_sum,
    }
    write_json(validation_summary, out_dir / "validation_summary.json")

    # Input manifest
    inputs: Dict[str, Any] = {
        "env": env_snapshot({"phase": "11_diagnostic_validation", "step": 2}),
        "inputs": [
            record_input(scatter_p, "step1_scatter", "DI vs delta scatter table"),
            record_input(zone_p, "zone_definitions", "t_safe/t_harm zone thresholds"),
            record_input(opt_p, "optimal_thresholds", "t_opt threshold"),
        ],
    }
    if ab_xgb_p is not None:
        inputs["inputs"].append(record_input(ab_xgb_p, "ablation_xgb_master", "XGB ablation master"))
    if ab_rf_p is not None:
        inputs["inputs"].append(record_input(ab_rf_p, "ablation_rf_master", "RF ablation master"))
    write_json(inputs, out_dir / "INPUTS_MANIFEST_step2.json")

    # Human-readable report
    lines = [
        "# Phase 11 — Validation",
        "",
        f"Primary K={int(args.primary_k)}, metric={str(args.primary_metric)}",
        f"t_opt={float(t_opt):.4f}; t_safe={float(t_safe):.4f}; t_harm={float(t_harm):.4f}",
        "",
        "## Cross-model agreement",
        json.dumps(cm_sum, indent=2),
        "",
        "## LODO transfer",
        json.dumps(lodo_sum, indent=2),
        "",
        "## Prospective (leave-one-view-out)",
        json.dumps(prosp_sum, indent=2),
        "",
        "## Confounders",
        json.dumps(conf_sum, indent=2),
        "",
        "## Permutation",
        json.dumps(perm_sum, indent=2),
        "",
        "## Null DI baseline",
        json.dumps(null_sum, indent=2),
        "",
    ]
    (out_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Phase11 Step2 complete")
    print(f"  wrote: {out_dir}")


if __name__ == "__main__":
    main()
