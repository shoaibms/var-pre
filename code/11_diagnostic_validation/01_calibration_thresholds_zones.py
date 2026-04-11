#!/usr/bin/env python3
"""Phase 11 — Step 1: Diagnostic calibration, thresholds, zones.

This step is intentionally *aggregation-only* (fast) and uses existing outputs.

Inputs (expected):
  - outputs/04_importance/uncertainty/di_summary.csv(.gz)
  - outputs/<ablation_xgb_dirname>/ablation_master_summary.(csv|csv.gz|parquet)

Outputs (written under <run_dir>):
  - 01_calibration/
      di_vs_delta_scatter.csv
      calibration_fits.csv
      calibration_bootstrap_fits.csv
      calibration_quantiles.csv
      CALIBRATION_REPORT.md
  - 02_thresholds/
      threshold_grid.csv
      optimal_thresholds.json
      zone_definitions.json
      views_by_zone.csv
      k_stability_summary.csv
      THRESHOLD_REPORT.md

Key definitions:
  - x := DI (anti-alignment index; DI>1 indicates below-random overlap)
  - y := Δ(Var−Random) = performance(var_topk) - performance(random_mean)
  - harmful := y < -material_harm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression, HuberRegressor
except Exception:
    LinearRegression = None
    HuberRegressor = None


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
    for pkg in ["numpy", "pandas", "sklearn", "xgboost"]:
        try:
            m = __import__(pkg)
            snap[f"{pkg}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            snap[f"{pkg}_version"] = None
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


# -------------------------
# IO
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


def load_di_summary(outputs_dir: Path) -> Tuple[pd.DataFrame, Optional[Path]]:
    p = outputs_dir / "04_importance" / "uncertainty" / "di_summary.csv"
    if not p.exists():
        p = outputs_dir / "04_importance" / "uncertainty" / "di_summary.csv.gz"
    if not p.exists():
        return pd.DataFrame(), None
    df = _read_table(p)
    if "k_pct" in df.columns and "K_pct" not in df.columns:
        df = df.rename(columns={"k_pct": "K_pct"})
    return df, p


def load_ablation_master(outputs_dir: Path, dirname: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    root = outputs_dir / dirname
    p = _first_existing(root, [
        "ablation_master_summary.parquet",
        "ablation_master_summary.csv.gz",
        "ablation_master_summary.csv",
    ])
    if p is None:
        return pd.DataFrame(), None
    df = _read_table(p)
    if "k_pct" in df.columns and "K_pct" not in df.columns:
        df = df.rename(columns={"k_pct": "K_pct"})
    return df, p


# -------------------------
# Core: build calibration scatter
# -------------------------

def build_scatter(
    di: pd.DataFrame,
    ab: pd.DataFrame,
    *,
    primary_k: int,
    primary_metric: str,
    model: str,
) -> pd.DataFrame:
    if di.empty or ab.empty:
        return pd.DataFrame()

    di = di.copy()
    ab = ab.copy()

    di_k = di[di["K_pct"].astype(int) == int(primary_k)].copy()
    di_mean = _coalesce_col(di_k, ["DI_mean", "DI"])
    di_lo = _coalesce_col(di_k, ["DI_pctl_2.5", "DI_pctl_2_5", "DI_lo", "DI_p2p5"])
    di_hi = _coalesce_col(di_k, ["DI_pctl_97.5", "DI_pctl_97_5", "DI_hi", "DI_p97p5"])
    if di_mean is None:
        return pd.DataFrame()

    keep = ["dataset", "view", "model", "K_pct", di_mean]
    if di_lo:
        keep.append(di_lo)
    if di_hi:
        keep.append(di_hi)
    di_k = di_k[[c for c in keep if c in di_k.columns]].copy()

    if "model" in ab.columns:
        ab = ab[ab["model"].astype(str) == str(model)].copy()
    if "metric" in ab.columns:
        ab = ab[ab["metric"].astype(str) == str(primary_metric)].copy()
    ab = ab[ab["K_pct"].astype(int) == int(primary_k)].copy()

    y_col = _coalesce_col(ab, ["delta_var_minus_random_mean", "delta_var_random_mean", "delta_var_minus_random"])
    y_lo = _coalesce_col(ab, ["delta_var_minus_random_ci_lo", "delta_var_minus_random_lo", "delta_var_random_ci_lo"])
    y_hi = _coalesce_col(ab, ["delta_var_minus_random_ci_hi", "delta_var_minus_random_hi", "delta_var_random_ci_hi"])
    if y_col is None:
        return pd.DataFrame()

    keep_ab = ["dataset", "view", "K_pct", "metric", y_col]
    if y_lo:
        keep_ab.append(y_lo)
    if y_hi:
        keep_ab.append(y_hi)
    ab = ab[[c for c in keep_ab if c in ab.columns]].copy()

    m = di_k.merge(ab, on=["dataset", "view", "K_pct"], how="inner")

    m = m.rename(columns={di_mean: "DI_mean", y_col: "delta_var_minus_random_mean"})
    if di_lo and di_lo in m.columns:
        m = m.rename(columns={di_lo: "DI_pctl_2.5"})
    if di_hi and di_hi in m.columns:
        m = m.rename(columns={di_hi: "DI_pctl_97.5"})
    if y_lo and y_lo in m.columns:
        m = m.rename(columns={y_lo: "delta_var_minus_random_ci_lo"})
    if y_hi and y_hi in m.columns:
        m = m.rename(columns={y_hi: "delta_var_minus_random_ci_hi"})

    m["DI_mean"] = m["DI_mean"].astype(float)
    m["delta_var_minus_random_mean"] = m["delta_var_minus_random_mean"].astype(float)
    m["model"] = str(model)
    m["metric"] = str(primary_metric)

    return m


# -------------------------
# Calibration fits + bootstrap
# -------------------------

def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    if LinearRegression is None:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = intercept + slope * x
    else:
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        intercept = float(lr.intercept_)
        slope = float(lr.coef_[0])
        yhat = lr.predict(x.reshape(-1, 1))

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return float(intercept), float(slope), float(r2), float(rmse)


def fit_models(scatter: pd.DataFrame) -> pd.DataFrame:
    if scatter.empty:
        return pd.DataFrame()

    x = scatter["DI_mean"].to_numpy(dtype=float)
    y = scatter["delta_var_minus_random_mean"].to_numpy(dtype=float)

    rows: List[Dict[str, Any]] = []

    i, s, r2, rmse = _fit_line(x, y)
    rows.append({"model": "ols", "intercept": i, "slope": s, "r2": r2, "rmse": rmse, "n": int(len(x))})

    if HuberRegressor is not None:
        try:
            hr = HuberRegressor(alpha=0.0, epsilon=1.35)
            hr.fit(x.reshape(-1, 1), y)
            i_h = float(hr.intercept_)
            s_h = float(hr.coef_[0])
            yhat = hr.predict(x.reshape(-1, 1))
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
            r2_h = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse_h = float(np.sqrt(np.mean((y - yhat) ** 2)))
            rows.append({"model": "huber", "intercept": i_h, "slope": s_h, "r2": r2_h, "rmse": rmse_h, "n": int(len(x))})
        except Exception:
            pass

    return pd.DataFrame(rows)


def bootstrap_fits(scatter: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    if scatter.empty:
        return pd.DataFrame()

    x = scatter["DI_mean"].to_numpy(dtype=float)
    y = scatter["delta_var_minus_random_mean"].to_numpy(dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)

    rows = []
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        i, s, r2, rmse = _fit_line(x[idx], y[idx])
        rows.append({"boot": b, "intercept": i, "slope": s, "r2": r2, "rmse": rmse})

    return pd.DataFrame(rows)


def bootstrap_quantiles(boot: pd.DataFrame, x_grid: np.ndarray, qs: List[float] = [0.10, 0.50, 0.90]) -> pd.DataFrame:
    if boot.empty:
        return pd.DataFrame()

    rows = []
    for x_val in x_grid:
        preds = boot["intercept"].to_numpy() + boot["slope"].to_numpy() * x_val
        row = {"DI": float(x_val)}
        for q in qs:
            row[f"y_q{int(q * 100)}"] = float(np.quantile(preds, q))
        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------
# Threshold scan
# -------------------------

def threshold_scan(scatter: pd.DataFrame, *, material_harm: float, cost_ratios: List[float]) -> pd.DataFrame:
    if scatter.empty:
        return pd.DataFrame()

    x = scatter["DI_mean"].to_numpy(dtype=float)
    y = scatter["delta_var_minus_random_mean"].to_numpy(dtype=float)
    harmful = (y < -material_harm).astype(int)

    uniq = np.unique(np.round(x, 6))
    mids = (uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1 else np.array([])
    candidates = np.unique(np.concatenate([uniq, mids, [uniq.min() - 1e-6, uniq.max() + 1e-6]]))

    rows = []
    for t in candidates:
        yhat = (x >= t).astype(int)
        tp = int(np.sum((harmful == 1) & (yhat == 1)))
        tn = int(np.sum((harmful == 0) & (yhat == 0)))
        fp = int(np.sum((harmful == 0) & (yhat == 1)))
        fn = int(np.sum((harmful == 1) & (yhat == 0)))

        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")

        for cr in cost_ratios:
            utility = -(cr * fn + fp) / max(1.0, float(len(y)))
            rows.append({
                "threshold": float(t),
                "cost_ratio": float(cr),
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "sensitivity": float(sens),
                "specificity": float(spec),
                "utility": float(utility),
            })

    return pd.DataFrame(rows)


def pick_thresholds(
    scatter: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    material_harm: float,
    cost_ratios: List[float],
    primary_cost_ratio: float,
    target_sens: float,
    target_spec: float,
) -> Dict[str, Any]:
    if grid.empty:
        return {}

    g = grid[grid["cost_ratio"] == primary_cost_ratio].copy()
    if g.empty:
        g = grid.copy()

    best_idx = g["utility"].idxmax()
    t_opt = float(g.loc[best_idx, "threshold"])

    t_safe = None
    t_harm = None

    # harmful := (DI >= threshold)  i.e., higher DI => higher risk
    # SAFE boundary controls false-safes (FN) => enforce high sensitivity
    safe_cands = g[g["sensitivity"] >= target_sens]
    if not safe_cands.empty:
        t_safe = float(safe_cands["threshold"].max())

    # HARMFUL boundary controls false-harms (FP) => enforce high specificity
    harm_cands = g[g["specificity"] >= target_spec]
    if not harm_cands.empty:
        t_harm = float(harm_cands["threshold"].min())

    # If constraints are incompatible (t_safe > t_harm), collapse to t_opt
    if (t_safe is not None) and (t_harm is not None) and (t_safe > t_harm):
        t_safe = t_opt
        t_harm = t_opt

    return {
        "t_opt": t_opt,
        "t_safe": t_safe if t_safe is not None else t_opt,
        "t_harm": t_harm if t_harm is not None else t_opt,
        "primary_cost_ratio": primary_cost_ratio,
        "target_sens": target_sens,
        "target_spec": target_spec,
        "material_harm": material_harm,
    }


# -------------------------
# Zone assignment
# -------------------------

def assign_zones(df: pd.DataFrame, *, t_safe: float, t_harm: float) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    def z_naive(di: float) -> str:
        if di <= t_safe:
            return "SAFE"
        if di >= t_harm:
            return "HARMFUL"
        return "INCONCLUSIVE"

    def z_ci(lo: float, hi: float) -> str:
        if hi < t_safe:
            return "SAFE"
        if lo > t_harm:
            return "HARMFUL"
        return "INCONCLUSIVE"

    df["zone_naive"] = df["DI_mean"].astype(float).apply(z_naive)

    if "DI_pctl_2.5" in df.columns and "DI_pctl_97.5" in df.columns:
        df["zone_ci"] = [
            z_ci(float(lo), float(hi))
            for lo, hi in zip(df["DI_pctl_2.5"].astype(float), df["DI_pctl_97.5"].astype(float))
        ]
    else:
        df["zone_ci"] = df["zone_naive"]

    return df


# -------------------------
# K-stability
# -------------------------

def k_stability_summary(ablation_master: pd.DataFrame, *, model: str, metric: str, material_harm: float) -> pd.DataFrame:
    if ablation_master.empty:
        return pd.DataFrame()

    ab = ablation_master.copy()
    if "model" in ab.columns:
        ab = ab[ab["model"].astype(str) == str(model)].copy()
    if "metric" in ab.columns:
        ab = ab[ab["metric"].astype(str) == str(metric)].copy()

    y_col = _coalesce_col(ab, ["delta_var_minus_random_mean", "delta_var_random_mean", "delta_var_minus_random"])
    if y_col is None:
        return pd.DataFrame()

    sub = ab[["dataset", "view", "K_pct", y_col]].copy()
    sub["harmful"] = (sub[y_col].astype(float) < -float(material_harm)).astype(int)

    piv = sub.pivot_table(index=["dataset", "view"], columns="K_pct", values="harmful", aggfunc="first")
    piv = piv.reset_index()

    k_cols = [c for c in piv.columns if c not in ("dataset", "view")]

    def _stable(row: pd.Series) -> int:
        vals = [row[c] for c in k_cols if row[c] == row[c]]
        if len(vals) <= 1:
            return 1
        return int(min(vals) == max(vals))

    piv["k_stable"] = [_stable(r) for _, r in piv.iterrows()]
    piv["k_cols"] = ",".join([str(c) for c in k_cols])

    return piv


def _write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--primary-k", type=int, default=10)
    ap.add_argument("--primary-metric", default="balanced_accuracy")
    ap.add_argument("--ablation-xgb-dirname", default="07_ablation")
    ap.add_argument("--model", default="xgb_bal")
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-ratios", default="5")
    ap.add_argument("--primary-cost-ratio", type=float, default=5.0)
    ap.add_argument("--target-sens", type=float, default=0.90)
    ap.add_argument("--target-spec", type=float, default=0.90)
    ap.add_argument("--material-harm", type=float, default=0.0)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    run_dir = Path(args.run_dir)

    out_cal = run_dir / "01_calibration"
    out_thr = run_dir / "02_thresholds"
    out_cal.mkdir(parents=True, exist_ok=True)
    out_thr.mkdir(parents=True, exist_ok=True)

    di_df, di_path = load_di_summary(outputs_dir)
    ab_df, ab_path = load_ablation_master(outputs_dir, args.ablation_xgb_dirname)

    scatter = build_scatter(
        di_df,
        ab_df,
        primary_k=int(args.primary_k),
        primary_metric=str(args.primary_metric),
        model=str(args.model),
    )

    if scatter.empty:
        raise SystemExit(
            "Step1 failed: could not build DI vs Δ(Var−Random) table. "
            "Check that di_summary and ablation_master_summary exist and share dataset/view names."
        )

    scatter.to_csv(out_cal / "di_vs_delta_scatter.csv", index=False)

    fits = fit_models(scatter)
    if not fits.empty:
        fits.to_csv(out_cal / "calibration_fits.csv", index=False)

    boot = bootstrap_fits(scatter, n_boot=int(args.bootstrap_n), seed=int(args.seed))
    if not boot.empty:
        boot.to_csv(out_cal / "calibration_bootstrap_fits.csv", index=False)

        x_grid = np.linspace(float(scatter["DI_mean"].min()), float(scatter["DI_mean"].max()), 200)
        qdf = bootstrap_quantiles(boot, x_grid, qs=[0.10, 0.50, 0.90])
        qdf.to_csv(out_cal / "calibration_quantiles.csv", index=False)

    cost_ratios = [float(x.strip()) for x in str(args.cost_ratios).split(",") if x.strip()]
    if not cost_ratios:
        cost_ratios = [float(args.primary_cost_ratio)]

    grid = threshold_scan(scatter, material_harm=float(args.material_harm), cost_ratios=cost_ratios)
    grid.to_csv(out_thr / "threshold_grid.csv", index=False)

    thr = pick_thresholds(
        scatter,
        grid,
        material_harm=float(args.material_harm),
        cost_ratios=cost_ratios,
        primary_cost_ratio=float(args.primary_cost_ratio),
        target_sens=float(args.target_sens),
        target_spec=float(args.target_spec),
    )
    write_json(thr, out_thr / "optimal_thresholds.json")

    t_safe = float(thr.get("t_safe", thr.get("t_opt", 1.0)))
    t_harm = float(thr.get("t_harm", thr.get("t_opt", 1.0)))

    if t_safe > t_harm:
        t_safe = float(thr.get("t_opt", t_safe))
        t_harm = float(thr.get("t_opt", t_harm))

    zones = assign_zones(scatter, t_safe=t_safe, t_harm=t_harm)
    zones.to_csv(out_thr / "views_by_zone.csv", index=False)

    zone_def = {
        "primary_k": int(args.primary_k),
        "primary_metric": str(args.primary_metric),
        "model": str(args.model),
        "material_harm": float(args.material_harm),
        "t_opt": thr.get("t_opt"),
        "t_safe": float(t_safe),
        "t_harm": float(t_harm),
        "target_sens": float(args.target_sens),
        "target_spec": float(args.target_spec),
        "rule": {
            "SAFE": "CI-aware: DI_pctl_97.5 <= t_safe",
            "HARMFUL": "CI-aware: DI_pctl_2.5 >= t_harm",
            "INCONCLUSIVE": "otherwise",
        },
    }
    write_json(zone_def, out_thr / "zone_definitions.json")

    kstab = k_stability_summary(ab_df, model=str(args.model), metric=str(args.primary_metric), material_harm=float(args.material_harm))
    if not kstab.empty:
        kstab.to_csv(out_thr / "k_stability_summary.csv", index=False)

    inputs: Dict[str, Any] = {
        "env": env_snapshot({"phase": "11_diagnostic_validation", "step": 1}),
        "inputs": [],
        "parameters": {
            "primary_k": int(args.primary_k),
            "primary_metric": str(args.primary_metric),
            "model": str(args.model),
            "bootstrap_n": int(args.bootstrap_n),
            "seed": int(args.seed),
            "cost_ratios": cost_ratios,
            "primary_cost_ratio": float(args.primary_cost_ratio),
            "target_sens": float(args.target_sens),
            "target_spec": float(args.target_spec),
            "material_harm": float(args.material_harm),
        },
    }
    if di_path is not None:
        inputs["inputs"].append(record_input(di_path, "di_summary", "DI uncertainty summary"))
    if ab_path is not None:
        inputs["inputs"].append(record_input(ab_path, "ablation_master", "Ablation master summary"))
    write_json(inputs, out_cal / "INPUTS_MANIFEST_step1.json")

    rep = [
        "# Phase 11 — DI calibration",
        "",
        f"Primary K: {int(args.primary_k)}%",
        f"Primary metric: {str(args.primary_metric)}",
        f"Model: {str(args.model)}",
        f"Views with complete DI+ablation data: {int(len(scatter))}",
        "",
        "## Fits (DI → Δ(Var−Random))",
    ]
    if fits.empty:
        rep.append("(no fits; sklearn not available)")
    else:
        for _, r in fits.iterrows():
            rep.append(f"- {r['model']}: slope={r['slope']:.4f}, intercept={r['intercept']:.4f}, R²={r['r2']:.3f}, RMSE={r['rmse']:.4f}")
    _write_md(out_cal / "CALIBRATION_REPORT.md", rep)

    rep2 = [
        "# Phase 11 — Thresholds and zones",
        "",
        f"Harm definition: Δ(Var−Random) < -{float(args.material_harm):.4g}",
        f"t_opt (utility): {thr.get('t_opt')}",
        f"t_safe (zone): {t_safe:.4f}",
        f"t_harm (zone): {t_harm:.4f}",
        "",
        "## Zone counts",
        f"naive: {zones['zone_naive'].value_counts().to_dict()}",
        f"CI-aware: {zones['zone_ci'].value_counts().to_dict()}",
    ]
    if not kstab.empty:
        rep2.append("")
        rep2.append(f"K-stable harm label across K: {int(kstab['k_stable'].sum())}/{len(kstab)} views")
    _write_md(out_thr / "THRESHOLD_REPORT.md", rep2)

    print("[OK] Phase11 Step1 complete")
    print(f"  wrote: {out_cal}")
    print(f"  wrote: {out_thr}")


if __name__ == "__main__":
    main()
