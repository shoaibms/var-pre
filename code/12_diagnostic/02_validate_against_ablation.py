#!/usr/bin/env python3
"""
PHASE 12 — Validate VAD against existing ablation (no retraining)

This script merges:
  - Phase12 VAD summary (etaES, alpha', SAS/PCLA)
  - Phase7 ablation master summary (Δ(Var−Random))

and reports correlation/regression evidence that VAD predicts variance-filtering harm.

Outputs under: <outputs_dir>/<out_dirname>/
  - vad_vs_ablation.csv
  - VAD_VALIDATION_REPORT.md

Usage (PowerShell):
  python .\\code\\compute\12_diagnostic\02_validate_against_ablation.py `
    --outputs-dir "<path-to-outputs>" `
    --vad-dirname "12_diagnostic" `
    --ablation-dirname "07_ablation" `
    --k 10 `
    --metric balanced_accuracy `
    --model xgb_bal
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
except Exception:
    _stats = None

# allow imports from code/compute/_shared
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    # normalize K column
    if "k_pct" in df.columns and "K_pct" not in df.columns:
        df = df.rename(columns={"k_pct": "K_pct"})
    return df, p


def spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    if _stats is not None:
        r, p = _stats.spearmanr(x[m], y[m])
        return float(r), float(p)
    # fallback: approximate
    rx = x[m].argsort().argsort().astype(float)
    ry = y[m].argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom <= 0:
        return float("nan"), float("nan")
    r = float((rx * ry).sum() / denom)
    return r, float("nan")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--vad-dirname", default="12_diagnostic")
    ap.add_argument("--ablation-dirname", default="07_ablation")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", default="balanced_accuracy")
    ap.add_argument("--model", default="xgb_bal")
    ap.add_argument("--out-dirname", default="12_diagnostic_validation")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    vad_dir = outputs_dir / args.vad_dirname
    out_dir = outputs_dir / args.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    vad_path = vad_dir / "vad_summary.csv"
    if not vad_path.exists():
        raise FileNotFoundError(f"Missing VAD summary: {vad_path}. Run 01_compute_vad.py first.")
    vad = pd.read_csv(vad_path)

    # keep only the requested K
    vad_k = vad[vad["k_pct"].astype(int) == int(args.k)].copy()
    if vad_k.empty:
        raise ValueError(f"No VAD rows for k_pct={args.k} in {vad_path}")

    ab, ab_path = load_ablation_master(outputs_dir, args.ablation_dirname)
    if ab.empty or ab_path is None:
        raise FileNotFoundError(f"Missing ablation master under outputs/{args.ablation_dirname}")

    # filter ablation rows
    ab = ab.copy()
    if "model" in ab.columns:
        ab = ab[ab["model"].astype(str) == str(args.model)]
    if "metric" in ab.columns:
        ab = ab[ab["metric"].astype(str) == str(args.metric)]
    if "K_pct" in ab.columns:
        ab = ab[ab["K_pct"].astype(int) == int(args.k)]
    elif "k_pct" in ab.columns:
        ab = ab[ab["k_pct"].astype(int) == int(args.k)]

    y_col = _coalesce_col(ab, ["delta_var_minus_random_mean", "delta_var_random_mean", "delta_var_minus_random"])
    if y_col is None:
        raise ValueError(f"Cannot find Δ(Var−Random) column in ablation. cols={list(ab.columns)}")

    keep_cols = ["dataset", "view", y_col]
    if "perf_var_mean" in ab.columns:
        keep_cols.append("perf_var_mean")
    if "perf_random_mean" in ab.columns:
        keep_cols.append("perf_random_mean")
    ab_k = ab[keep_cols].drop_duplicates(["dataset", "view"]).copy()
    ab_k = ab_k.rename(columns={y_col: "delta_var_minus_random_mean"})

    # merge
    m = vad_k.merge(ab_k, on=["dataset", "view"], how="inner")
    if m.empty:
        raise ValueError("Merge produced 0 rows. Check dataset/view naming consistency between VAD and ablation.")

    out_csv = out_dir / "vad_vs_ablation.csv"
    m.to_csv(out_csv, index=False)

    # stats
    y = pd.to_numeric(m["delta_var_minus_random_mean"], errors="coerce").to_numpy(dtype=float)
    stats_rows = []

    def add_stat(name: str, xcol: str):
        """Compute Spearman with Δ(Var−Random) plus *optional* sign metrics.

        Sign metrics are only computed when the predictor has a theory-derived
        decision boundary:
          - ηES: < 1 predicts harm
          - F-DI: > 1 predicts harm
          - α'/SAS/VSA: < 0 predicts harm
          - PCLA: no natural boundary → sign metrics set to NaN
        """
        x = pd.to_numeric(m.get(xcol, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        r, pval = spearman(x, y)

        valid = np.isfinite(x) & np.isfinite(y)
        n = int(valid.sum())
        out = {"x": xcol, "label": name, "rho": r, "p": pval, "n": n,
               "sign_acc": float("nan"), "sensitivity": float("nan"), "specificity": float("nan")}

        if n < 3:
            stats_rows.append(out)
            return

        # Only compute sign metrics if a natural boundary exists
        if xcol == "pcla_mean":
            stats_rows.append(out)
            return

        xv, yv = x[valid], y[valid]
        actual_harm = yv < 0

        if xcol == "eta_es_mean":
            predicted_harm = xv < 1.0
        elif xcol == "f_di_mean":
            predicted_harm = xv > 1.0
        else:
            predicted_harm = xv < 0.0

        sign_acc = float(np.mean(predicted_harm == actual_harm))

        n_actual_harm = int(actual_harm.sum())
        n_detected = int((predicted_harm & actual_harm).sum())
        sensitivity = (n_detected / n_actual_harm) if n_actual_harm > 0 else float("nan")

        n_actual_safe = int((~actual_harm).sum())
        n_correct_safe = int((~predicted_harm & ~actual_harm).sum())
        specificity = (n_correct_safe / n_actual_safe) if n_actual_safe > 0 else float("nan")

        out.update({"sign_acc": sign_acc, "sensitivity": float(sensitivity), "specificity": float(specificity)})
        stats_rows.append(out)

    # primary hypotheses
    add_stat("ηES", "eta_es_mean")
    add_stat("α'", "alpha_prime_mean")
    add_stat("PCLA", "pcla_mean")
    add_stat("SAS", "sas_mean")
    add_stat("VSA", "vsa_mean")
    add_stat("F-DI", "f_di_mean")

    stats_df = pd.DataFrame(stats_rows)

    # ── Auto-winner detection ──
    best_metric = "unknown"
    best_rho_abs = -1.0
    for _, row in stats_df.iterrows():
        r = abs(row["rho"]) if np.isfinite(row["rho"]) else 0.0
        if r > best_rho_abs:
            best_rho_abs = r
            best_metric = row["label"]

    # simple linear regression (optional): y ~ etaES + PCLA
    lm = {}
    try:
        X = m[["eta_es_mean", "pcla_mean"]].astype(float).to_numpy()
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask, :]
        yy = y[mask]
        if X.shape[0] >= 5:
            # add intercept
            X1 = np.column_stack([np.ones(X.shape[0]), X])
            beta, *_ = np.linalg.lstsq(X1, yy, rcond=None)
            yhat = X1 @ beta
            ss_res = float(np.sum((yy - yhat) ** 2))
            ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2)) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            lm = {
                "n": int(X.shape[0]),
                "beta_intercept": float(beta[0]),
                "beta_eta_es": float(beta[1]),
                "beta_pcla": float(beta[2]),
                "r2": float(r2),
            }
    except Exception:
        lm = {}

    # LOO-CV R-squared (C3.2)

    # LOO-CV R² for the bivariate model
    loo = {}
    try:
        X = m[["eta_es_mean", "pcla_mean"]].astype(float).to_numpy()
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask, :]
        yy = y[mask]
        n = X.shape[0]
        if n >= 5:
            X1 = np.column_stack([np.ones(n), X])
            preds = np.full(n, np.nan)
            for i in range(n):
                idx = np.ones(n, dtype=bool)
                idx[i] = False
                beta_i, *_ = np.linalg.lstsq(X1[idx], yy[idx], rcond=None)
                preds[i] = float(X1[i] @ beta_i)
            ss_res = float(np.sum((yy - preds) ** 2))
            ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2)) + 1e-12
            r2_loo = 1.0 - ss_res / ss_tot
            loo = {
                "n": int(n),
                "r2_loo": float(r2_loo),
                "r2_insample": lm.get("r2", float("nan")),
                "shrinkage": lm.get("r2", 0.0) - r2_loo,
                "mae_loo": float(np.mean(np.abs(yy - preds))),
            }
            print(f"  LOO-CV R²: {r2_loo:.3f} (in-sample: {lm.get('r2', float('nan')):.3f}, "
                  f"shrinkage: {loo['shrinkage']:.3f})")
    except Exception:
        loo = {}

    # End LOO-CV R-squared

    # report
    lines = []
    lines.append("# Phase 12 — VAD validation vs ablation")
    lines.append("")
    lines.append(f"- VAD summary: `{vad_path}`")
    lines.append(f"- Ablation master: `{ab_path}`")
    lines.append(f"- Filter: model={args.model}, metric={args.metric}, K={args.k}%")
    lines.append(f"- Merge rows: {len(m)}")
    lines.append("")
    lines.append("## Correlation evidence (Spearman)")
    lines.append("")
    if stats_df.empty:
        lines.append("No stats computed.")
    else:
        show = stats_df.copy()
        show["rho"] = show["rho"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "nan")
        show["p"] = show["p"].map(lambda v: f"{v:.3g}" if np.isfinite(v) else "nan")
        show["sign_acc"] = show["sign_acc"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "nan")
        show["sensitivity"] = show["sensitivity"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "nan")
        show["specificity"] = show["specificity"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "nan")
        lines.append(show.to_markdown(index=False))
        lines.append("")
        lines.append(f"**BEST: {best_metric}** (|rho| = {best_rho_abs:.3f})")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("- **ηES**: enrichment < 1 predicts Δ < 0 (harm)")
        lines.append("- **α'**: negative correlation predicts anti-alignment")
        lines.append("- **PCLA/SAS**: multivariate alignment (low = risky)")
        lines.append("- **F-DI**: high = decoupled (supervision-free DI analog)")
        lines.append("")
        lines.append("Note: Run with `--ablation-dirname 07_ablation_rf` for cross-model (RF) validation.")
        lines.append("")

    if lm:
        lines.append("## Simple regression (least squares)")
        lines.append("")
        lines.append("Model: Δ(Var−Random) ~ 1 + etaES + PCLA")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(lm, indent=2))
        lines.append("```")
        lines.append("")

    if loo:
        lines.append("## Leave-one-out cross-validated R²")
        lines.append("")
        lines.append("Model: Δ(Var−Random) ~ 1 + etaES + PCLA  (LOO-CV)")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(loo, indent=2))
        lines.append("```")
        lines.append("")
        r2_loo_val = loo.get("r2_loo", float("nan"))
        if r2_loo_val > 0:
            lines.append(f"> Leave-one-out cross-validation confirmed that the bivariate model")
            lines.append(f"> generalises beyond the training views (R²_LOO = {r2_loo_val:.2f},")
            lines.append(f"> in-sample R² = {loo.get('r2_insample', float('nan')):.2f}).")
        else:
            lines.append(f"> LOO-CV yielded negative R² ({r2_loo_val:.2f}), indicating the bivariate")
            lines.append(f"> model does not generalise at n = {loo.get('n', '?')}. Interpret in-sample")
            lines.append(f"> R² as illustrative rather than predictive.")
        lines.append("")

    rep = out_dir / "VAD_VALIDATION_REPORT.md"
    rep.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Phase12 validation complete")
    print(f"  wrote: {out_csv}")
    print(f"  wrote: {rep}")


if __name__ == "__main__":
    main()
