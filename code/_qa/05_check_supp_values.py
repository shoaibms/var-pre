"""
05_check_supp_values.py

Spot-check key numeric values in supplementary tables against expected
pipeline outputs. Confirms that compiled CSVs contain the correct data.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def find_file(root: Path, patterns: list[str]) -> Path | None:
    # patterns are substrings (case-insensitive); prefer the first matching file
    files = [p for p in root.rglob("*.csv") if p.is_file()]
    lower = [(p, p.name.lower()) for p in files]
    for pat in patterns:
        pat = pat.lower()
        for p, nm in lower:
            if pat in nm:
                return p
    return None

def approx(x, y, tol): 
    return (x is not None and y is not None and abs(x-y) <= tol)

def report(tag, ok, msg=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f" :: {msg}" if msg else ""))

def warn(tag, msg=""):
    print(f"[WARN] {tag}" + (f" :: {msg}" if msg else ""))

def parse_zone_counts(text: str):
    # looks for "... GREEN ... 7 views ... RED ... 4 views ... YELLOW ... 3 views"
    m = re.search(
        r"GREEN.*?(\d+)\s*views.*?RED.*?(\d+)\s*views.*?YELLOW.*?(\d+)\s*views",
        text, flags=re.IGNORECASE | re.DOTALL
    )
    return tuple(map(int, m.groups())) if m else None

def parse_rho_p(text: str, metric: str):
    # accepts "Spearman ρ = +0.54, p = 0.047" near metric keyword
    m = re.search(
        rf"{metric}.*?Spearman.*?ρ\s*=\s*([+\-]?\d+\.\d+).*?p\s*=\s*([0-9.]+)",
        text, flags=re.IGNORECASE | re.DOTALL
    )
    return (float(m.group(1)), float(m.group(2))) if m else None

def parse_sim_di_examples(text: str):
    # expects something like: coupled (DI=0.51), decoupled (DI=1.02), anti-aligned (DI=1.05)
    m = re.search(
        r"coupled.*?DI\s*=\s*([0-9.]+).*?decoupled.*?DI\s*=\s*([0-9.]+).*?anti.*?DI\s*=\s*([0-9.]+)",
        text, flags=re.IGNORECASE | re.DOTALL
    )
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None

def infer_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fallback: contains
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--manuscript", required=True)
    args = ap.parse_args()

    out = Path(args.outputs_dir)
    ms  = Path(args.manuscript)
    text = read_text(ms)

    # Try to locate the diagnostic source CSVs robustly
    # Priority order: explicit diagnostic CSVs, then the copied provenance source_tables
    root_results = out / "results"
    vad_val = find_file(root_results, ["vad_validation_by_model", "vad_validation"])
    vad_ov  = find_file(root_results, ["vad_metric_overview", "vad_overview", "vad_summary"])
    sim     = find_file(root_results, ["simulation_validation", "fig6_simulation", "simulation"])

    print("="*80)
    print("SUPP vs RESULTS CONSISTENCY CHECK (values)")
    print("="*80)
    print("Manuscript:", ms)
    print("vad_validation:", vad_val)
    print("vad_overview  :", vad_ov)
    print("simulation    :", sim)
    print("-"*80)

    # Load what exists (warn if missing)
    df_val = pd.read_csv(vad_val) if vad_val else None
    df_ov  = pd.read_csv(vad_ov)  if vad_ov  else None
    df_sim = pd.read_csv(sim)     if sim     else None

    # (1) Zone counts at K=10
    exp_zone = parse_zone_counts(text)
    if exp_zone is None:
        warn("Zone counts in manuscript", "Could not parse GREEN/RED/YELLOW view counts from manuscript text (ok if not written yet).")
    elif df_ov is None:
        report("Zone counts at K=10", False, "vad_overview missing")
    else:
        kcol   = infer_col(df_ov, ["k", "k_pct", "budget_k", "K"])
        zcol   = infer_col(df_ov, ["zone"])
        if not kcol or not zcol:
            report("Zone counts at K=10", False, f"Need k+zone columns; found kcol={kcol}, zcol={zcol}")
        else:
            k = pd.to_numeric(df_ov[kcol], errors="coerce")
            # accept 10 or 0.10 representation
            mask = (k == 10) | (k == 0.10) | (k == 0.1)
            d = df_ov[mask].copy()
            z = d[zcol].astype(str).str.upper()
            got = (int((z=="GREEN").sum()), int((z=="RED").sum()), int((z=="YELLOW").sum()))
            report("Zone counts at K=10 match manuscript", got == exp_zone, f"manuscript={exp_zone}, data={got} using {zcol}@{kcol}")

    # (2) PCLA and VSA rho/p from manuscript vs vad_validation file
    exp_pcla = parse_rho_p(text, "PCLA")
    exp_vsa  = parse_rho_p(text, "VSA")
    if df_val is None:
        warn("Rho/p checks", "vad_validation file not found; cannot verify PCLA/VSA rho/p.")
    else:
        metric_col = infer_col(df_val, ["metric"])
        model_col  = infer_col(df_val, ["model"])
        rho_col    = infer_col(df_val, ["rho", "spearman"])
        p_col      = infer_col(df_val, ["p", "p_value", "pval"])
        if not metric_col or not rho_col or not p_col:
            report("Rho/p columns present in vad_validation", False, f"metric={metric_col}, rho={rho_col}, p={p_col}")
        else:
            def pick(metric, model_hint):
                d = df_val.copy()
                d[metric_col] = d[metric_col].astype(str)
                d = d[d[metric_col].str.lower()==metric.lower()]
                if model_col and model_hint:
                    d[model_col] = d[model_col].astype(str)
                    d = d[d[model_col].str.lower().str.contains(model_hint)]
                if len(d)==0: 
                    return None
                r = d.iloc[0]
                return (float(r[rho_col]), float(r[p_col]))

            got_pcla = pick("PCLA", "xgb") or pick("PCLA", "xgboost") or pick("PCLA", None)
            got_vsa  = pick("VSA", "rf") or pick("VSA", "random") or pick("VSA", None)

            if exp_pcla is None:
                warn("PCLA rho/p in manuscript", "Not found (ok if not cited yet).")
            else:
                ok = got_pcla and approx(exp_pcla[0], got_pcla[0], 1e-2) and approx(exp_pcla[1], got_pcla[1], 1e-3)
                report("PCLA rho/p matches", bool(ok), f"manuscript={exp_pcla}, data={got_pcla}")

            if exp_vsa is None:
                warn("VSA rho/p in manuscript", "Not found (ok if not cited yet).")
            else:
                ok = got_vsa and approx(exp_vsa[0], got_vsa[0], 1e-2) and approx(exp_vsa[1], got_vsa[1], 1e-3)
                report("VSA rho/p matches", bool(ok), f"manuscript={exp_vsa}, data={got_vsa}")

    # (3) PCLA–VSA intercorrelation (if columns exist)
    if df_ov is None:
        warn("PCLA–VSA correlation", "vad_overview missing")
    else:
        pcla_col = infer_col(df_ov, ["pcla"])
        vsa_col  = infer_col(df_ov, ["vsa"])
        if not pcla_col or not vsa_col:
            warn("PCLA–VSA correlation", f"Need PCLA+VSA cols; found pcla={pcla_col}, vsa={vsa_col}")
        else:
            rho = pd.Series(df_ov[pcla_col]).corr(pd.Series(df_ov[vsa_col]), method="spearman")
            # just print it (you can compare to caption manually if needed)
            print(f"[INFO] PCLA–VSA Spearman rho (from {vad_ov.name}): {rho:.3f}")

    # (4) Simulation DI examples (if present in manuscript)
    exp_sim = parse_sim_di_examples(text)
    if exp_sim is None:
        warn("Simulation DI examples in manuscript", "Not found (ok if not written yet).")
    elif df_sim is None:
        report("Simulation DI examples", False, "simulation file missing")
    else:
        scen_col = infer_col(df_sim, ["scenario"])
        di_col   = infer_col(df_sim, ["di"])
        if not scen_col or not di_col:
            report("Simulation DI examples", False, f"Need scenario+di cols; scen={scen_col}, di={di_col}")
        else:
            def get(s):
                d = df_sim[df_sim[scen_col].astype(str).str.lower().str.contains(s)]
                return float(d.iloc[0][di_col]) if len(d) else None
            got = (get("coupled"), get("decoupled"), get("anti"))
            ok = all(g is not None for g in got) and all(abs(got[i]-exp_sim[i])<=0.03 for i in range(3))
            report("Simulation DI examples match (±0.03)", ok, f"manuscript={exp_sim}, data={got}")

    print("="*80)

if __name__ == "__main__":
    main()
