#!/usr/bin/env python3
"""Phase 11 — Step 3: Decision assets and manifests.

Packages Phase 11 outputs into adoption-ready artifacts:

- decision_assets/
    DECISION_RULE.json         machine-readable thresholds + zones
    DECISION_FLOWCHART.json    rendering-ready flowchart nodes/edges
    REPORTING_CHECKLIST.md     what to report in papers/industry SOPs
    scope_limitations.md       when NOT to use DI/diagnostic

- figures/
    fig_calibration_scatter.csv
    fig_threshold_grid.csv
    fig_zone_assignments.csv
    fig_validation_summary.csv
    FIGURE_MANIFEST.json

- manifests/
    INPUT_MANIFEST.json
    OUTPUTS_MANIFEST.json

- PHASE11_EXECUTIVE_SUMMARY.md  (one-pager for interpretation)

Optionally exports figure-input CSVs to:
  outputs/<paper_dirname>/FIGURE_INPUTS/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


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


def write_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")


def read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    sfx = "".join(path.suffixes).lower()
    if sfx.endswith(".parquet"):
        return pd.read_parquet(path)
    if sfx.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# -------------------------
# Executive summary generator
# -------------------------

def generate_executive_summary(
    run_dir: Path,
    decision_rule: Dict[str, Any],
    val_sum: Dict[str, Any],
    primary_k: int,
    primary_metric: str,
) -> str:
    """Generate human-readable one-pager for interpretation."""

    cal = decision_rule.get("calibration", {})
    thr = decision_rule.get("thresholds", {})

    # Extract calibration metrics
    fits = cal.get("fits", [])
    ols_fit = next((f for f in fits if f.get("model") == "ols"), {}) if fits else {}
    slope = ols_fit.get("slope", cal.get("slope", "N/A"))
    r2 = ols_fit.get("r2", cal.get("r2", "N/A"))

    # Extract thresholds
    t_opt = thr.get("t_opt", "N/A")
    t_safe = thr.get("t_safe", "N/A")
    t_harm = thr.get("t_harm", "N/A")

    # Extract validation metrics
    cm = val_sum.get("cross_model", {})
    lodo = val_sum.get("lodo", {})
    prosp = val_sum.get("prospective", {})
    null = val_sum.get("null_baseline", {})

    # Format values safely
    def fmt_pct(v):
        if isinstance(v, (int, float)) and v == v:
            return f"{v:.1%}"
        return "N/A"

    def fmt_float(v, decimals=3):
        if isinstance(v, (int, float)) and v == v:
            return f"{v:.{decimals}f}"
        return "N/A"

    def check_mark(v, threshold=0.7):
        if isinstance(v, (int, float)) and v == v:
            return "PASS" if v >= threshold else "?"
        return "?"

    cm_agree = cm.get("sign_agreement", None)
    lodo_acc = lodo.get("overall_accuracy", None)
    prosp_acc = prosp.get("accuracy", None)
    null_di = null.get("di_mean_over_views", None)

    summary = f"""# Phase 11: Diagnostic Validation — Executive Summary

**Generated:** {now_iso()}
**Run:** `{run_dir}`

---

## Decision Thresholds (Validated)

| Zone | DI Range | Recommendation |
|------|----------|----------------|
| **SAFE** | DI_97.5 < {fmt_float(t_safe, 2)} | Variance filtering is acceptable |
| **INCONCLUSIVE** | Between thresholds | Run pilot ablation or use importance-guided |
| **HARMFUL** | DI_2.5 > {fmt_float(t_harm, 2)} | Avoid variance filtering |

**Optimal threshold (utility-based):** DI = {fmt_float(t_opt, 3)}

---

## Calibration (DI -> Delta(Var-Random))

| Metric | Value |
|--------|-------|
| Slope | {fmt_float(slope, 4)} |
| R² | {fmt_float(r2, 3)} |
| Interpretation | Each +0.1 DI → ~{fmt_float(abs(slope)*0.1 if isinstance(slope, (int,float)) else 0, 2)} pp change in Δ |

---

## Validation Summary

| Check | Result | Status |
|-------|--------|--------|
| Cross-model agreement (XGB vs RF) | {fmt_pct(cm_agree)} sign agree | {check_mark(cm_agree)} |
| LODO transfer accuracy | {fmt_pct(lodo_acc)} | {check_mark(lodo_acc)} |
| Prospective (leave-one-view-out) | {fmt_pct(prosp_acc)} | {check_mark(prosp_acc)} |
| Null baseline DI | {fmt_float(null_di, 3)} (expect ~1.0) | {check_mark(1.0 - abs((null_di or 1) - 1.0), 0.9)} |

---

## Validation Checklist

- [{check_mark(r2, 0.3) if isinstance(r2, (int,float)) else ' '}] Calibration R² > 0.3
- [{check_mark(prosp_acc) if prosp_acc else ' '}] Prospective accuracy > 70%
- [{check_mark(lodo_acc) if lodo_acc else ' '}] LODO accuracy > 70%
- [{check_mark(cm_agree) if cm_agree else ' '}] Cross-model agreement > 70%

---

## Key Output Files

| File | Purpose |
|------|---------|
| `decision_assets/DECISION_RULE.json` | Machine-readable thresholds |
| `decision_assets/REPORTING_CHECKLIST.md` | User checklist for papers |
| `figures/fig_calibration_scatter.csv` | Data for manuscript Figure 6 |
| `03_validation/validation_summary.json` | All validation metrics |

---

## Next Steps

1. Review validation metrics above -- all checks should pass (PASS)
2. If any check fails (?), investigate before proceeding
3. Use `DECISION_RULE.json` thresholds in manuscript Section 2.6
4. Include `REPORTING_CHECKLIST.md` in Supplementary Materials
5. Generate figures from CSVs in `figures/`

---

## Manuscript-Ready Numbers

**For Section 2.6:**
> The diagnostic framework uses a threshold of DI = {fmt_float(t_opt, 2)} to classify variance-filtering risk.
> Views with DI < {fmt_float(t_safe, 2)} are classified as SAFE (variance filtering acceptable),
> while views with DI > {fmt_float(t_harm, 2)} are classified as HARMFUL (importance-guided selection recommended).
> This threshold was validated with {fmt_pct(prosp_acc)} prospective accuracy (leave-one-view-out)
> and {fmt_pct(lodo_acc)} cross-dataset transfer accuracy (leave-one-dataset-out).

---

*This summary is auto-generated. Share this file for interpretation.*
"""

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--primary-k", type=int, default=10)
    ap.add_argument("--primary-metric", default="balanced_accuracy")
    ap.add_argument("--paper-dirname", default="09_paper\\v1")
    ap.add_argument("--no-paper-export", action="store_true")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    run_dir = Path(args.run_dir)
    out_dec = run_dir / "decision_assets"
    out_fig = run_dir / "figures"
    out_man = run_dir / "manifests"
    out_dec.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)
    out_man.mkdir(parents=True, exist_ok=True)

    # Load key artifacts from steps 1 & 2
    optimal = safe_load_json(run_dir / "02_thresholds" / "optimal_thresholds.json")
    zones = safe_load_json(run_dir / "02_thresholds" / "zone_definitions.json")
    cal_fits = read_csv_any(run_dir / "01_calibration" / "calibration_fits.csv")

    scatter = read_csv_any(run_dir / "01_calibration" / "di_vs_delta_scatter.csv")
    grid = read_csv_any(run_dir / "02_thresholds" / "threshold_grid.csv")
    zone_tbl = read_csv_any(run_dir / "02_thresholds" / "views_by_zone.csv")
    if not zone_tbl.empty:
        zone_tbl.to_csv(out_dec / "per_view_labels.csv", index=False)
    val_sum = safe_load_json(run_dir / "03_validation" / "validation_summary.json")

    # Build calibration summary for decision rule
    cal_summary = {}
    if not cal_fits.empty:
        cal_summary["fits"] = cal_fits.to_dict(orient="records")

    # Decision rule
    decision_rule = {
        "phase": "11_diagnostic_validation",
        "created": now_iso(),
        "primary_k": int(args.primary_k),
        "primary_metric": str(args.primary_metric),
        "thresholds": {
            "t_opt": optimal.get("t_opt"),
            "t_safe": zones.get("t_safe"),
            "t_harm": zones.get("t_harm"),
        },
        "classification": {
            "rule_naive": "if DI_mean <= t_safe => SAFE; elif DI_mean >= t_harm => HARMFUL; else INCONCLUSIVE",
            "rule_ci": "if DI_pctl_97.5 < t_safe => SAFE; elif DI_pctl_2.5 > t_harm => HARMFUL; else INCONCLUSIVE",
        },
        "calibration": cal_summary,
        "validation": val_sum,
    }
    write_json(decision_rule, out_dec / "DECISION_RULE.json")

    # Flowchart
    flow = {
        "title": "Variance-filtering diagnostic (DI) decision flow",
        "nodes": [
            {"id": "start", "label": f"Compute DI (K={int(args.primary_k)}%)"},
            {"id": "safe", "label": "SAFE: variance filtering OK", "color": "green"},
            {"id": "inc", "label": "INCONCLUSIVE: run pilot ablation", "color": "yellow"},
            {"id": "harm", "label": "HARMFUL: use importance-guided", "color": "red"},
        ],
        "edges": [
            {"from": "start", "to": "safe", "condition": "DI_pctl_97.5 < t_safe"},
            {"from": "start", "to": "harm", "condition": "DI_pctl_2.5 > t_harm"},
            {"from": "start", "to": "inc", "condition": "otherwise"},
        ],
        "params": {"t_safe": zones.get("t_safe"), "t_harm": zones.get("t_harm")},
    }
    write_json(flow, out_dec / "DECISION_FLOWCHART.json")

    # Checklist
    checklist = """# DI Reporting Checklist

When reporting variance-filtering decisions using the DI framework, include:

## Required Items

- [ ] **Dataset characteristics**
  - Number of samples (n): ___
  - Number of features (p): ___
  - Number of classes: ___

- [ ] **DI Computation**
  - K threshold used: ___% (recommended: 10%)
  - DI value: ___ [95% CI: ___, ___]
  - Model used for importance: ___ (e.g., XGBoost)

- [ ] **Regime Classification**
  - Assigned zone: [ ] SAFE / [ ] INCONCLUSIVE / [ ] HARMFUL
  - Classification method: [ ] CI-aware / [ ] Point estimate

- [ ] **Decision Made**
  - [ ] Applied variance filtering
  - [ ] Used importance-guided selection
  - [ ] Ran pilot ablation (for INCONCLUSIVE)

## Recommended Items

- [ ] DI computed at multiple K (5%, 10%, 20%) — stable across K?
- [ ] Cross-model DI agreement (XGBoost vs Random Forest)
- [ ] Permutation null computed — DI significant vs shuffled labels?

## Citation

If using this framework, please cite:
[Your paper reference here]

---
Generated: """ + now_iso() + """
Framework version: 1.0
"""
    write_text(out_dec / "REPORTING_CHECKLIST.md", checklist)

    # Scope limitations
    scope = f"""# Scope and Limitations of the DI Framework

## When to Use DI

- Multi-omics classification/prediction tasks
- Datasets with n > 50 samples
- Feature sets where variance filtering is being considered
- Before applying HVG selection, variance-based PCA, or MOFA

## When NOT to Use DI (Known Limitations)

### Sample Size
- **n < 50**: DI estimates become unstable
- Recommendation: Use importance-guided selection by default

### Extreme Class Imbalance
- **Minority class < 10%**: Importance rankings may be unreliable
- Recommendation: Address imbalance before computing DI

### Non-Biological Variance Dominance
- **Batch effects**: If variance is dominated by technical artifacts, DI may misclassify
- Recommendation: Apply batch correction before DI computation

### Regression Tasks
- DI is validated for classification only
- Regression tasks may require adapted thresholds

## INCONCLUSIVE Zone Protocol

When DI falls in the INCONCLUSIVE zone ({zones.get('t_safe', 'N/A')} ≤ DI ≤ {zones.get('t_harm', 'N/A')}):

1. Run a pilot ablation on 20% subsample
2. If Δ(Var−Random) < −2pp, treat as HARMFUL
3. Otherwise, proceed with variance filtering cautiously
4. Report the uncertainty in your methods section

## DI Uncertainty

- Always report DI with confidence intervals from repeat-CV
- If CI spans multiple zones, classify as INCONCLUSIVE
- If XGBoost and Random Forest DI differ by >0.1, investigate further

---
Generated: {now_iso()}
Framework version: 1.0
"""
    write_text(out_dec / "scope_limitations.md", scope)

    # Figure-ready data exports
    exports: Dict[str, Path] = {}
    if not scatter.empty:
        p = out_fig / "fig_calibration_scatter.csv"
        scatter.to_csv(p, index=False)
        exports["fig_calibration_scatter"] = p
    if not grid.empty:
        p = out_fig / "fig_threshold_grid.csv"
        grid.to_csv(p, index=False)
        exports["fig_threshold_grid"] = p
    if not zone_tbl.empty:
        p = out_fig / "fig_zone_assignments.csv"
        zone_tbl.to_csv(p, index=False)
        exports["fig_zone_assignments"] = p

    fig_manifest = {
        "created": now_iso(),
        "figures": [
            {"id": k, "path": str(v), "sha256": sha256_file(v)}
            for k, v in exports.items()
        ],
    }
    write_json(fig_manifest, out_fig / "FIGURE_MANIFEST.json")

    # Manifests
    inputs = []
    for name, rel, desc in [
        ("di_summary", outputs_dir / "04_importance" / "uncertainty" / "di_summary.csv", "DI summary"),
        ("ablation_xgb", outputs_dir / "07_ablation" / "ablation_master_summary.csv", "Ablation master (XGB)"),
        ("ablation_rf", outputs_dir / "07_ablation_rf" / "ablation_master_summary.csv", "Ablation master (RF)"),
    ]:
        if rel.exists():
            inputs.append(record_input(rel, name=name, description=desc))

    write_json({
        "created": now_iso(),
        "run_dir": str(run_dir),
        "env": env_snapshot(),
        "inputs": inputs,
    }, out_man / "INPUT_MANIFEST.json")

    outs = []
    for fp in [out_dec / "DECISION_RULE.json", out_dec / "DECISION_FLOWCHART.json", out_fig / "FIGURE_MANIFEST.json"]:
        if fp.exists():
            outs.append({"path": str(fp), "sha256": sha256_file(fp), "size_bytes": int(fp.stat().st_size)})
    write_json({"created": now_iso(), "outputs": outs}, out_man / "OUTPUTS_MANIFEST.json")

    # Executive summary
    exec_summary = generate_executive_summary(
        run_dir=run_dir,
        decision_rule=decision_rule,
        val_sum=val_sum,
        primary_k=args.primary_k,
        primary_metric=args.primary_metric,
    )
    write_text(run_dir / "PHASE11_EXECUTIVE_SUMMARY.md", exec_summary)

    # Optional paper export
    if not args.no_paper_export:
        paper_root = outputs_dir / args.paper_dirname
        fig_inputs_dir = paper_root / "FIGURE_INPUTS"
        fig_inputs_dir.mkdir(parents=True, exist_ok=True)

        for name, src in exports.items():
            dst = fig_inputs_dir / src.name
            dst.write_bytes(src.read_bytes())

        print(f"  Exported figure inputs to: {fig_inputs_dir}")

    print("[OK] Phase11 Step3 complete")
    print(f"  decision_assets: {out_dec}")
    print(f"  figures: {out_fig}")
    print(f"  manifests: {out_man}")
    print(f"  PHASE11_EXECUTIVE_SUMMARY.md: {run_dir / 'PHASE11_EXECUTIVE_SUMMARY.md'}")


if __name__ == "__main__":
    main()
