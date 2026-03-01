#!/usr/bin/env python3
"""PHASE 9 — 01_generate_synthetic.py: Generate synthetic datasets."""

from __future__ import annotations
import argparse, json, sys
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io_helpers import ensure_dir, now_iso

@dataclass
class ScenarioParams:
    name: str
    description: str
    n_samples: int = 200
    n_features: int = 1000
    n_classes: int = 2
    prop_type1: float = 0.10
    prop_type2: float = 0.30
    prop_type3: float = 0.50
    prop_type4: float = 0.10
    between_var_type1: float = 4.0
    within_var_type1: float = 1.0
    between_var_type2: float = 0.5
    within_var_type2: float = 8.0
    between_var_type3: float = 0.5
    within_var_type3: float = 0.5
    between_var_type4: float = 2.0
    within_var_type4: float = 4.0
    
    @property
    def feature_counts(self) -> Tuple[int, int, int, int]:
        props = [self.prop_type1, self.prop_type2, self.prop_type3, self.prop_type4]
        if any(p < 0 for p in props):
            raise ValueError(f"Negative prop_* in scenario '{self.name}': {props}")
        s = sum(props)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(
                f"prop_* must sum to 1.0 in scenario '{self.name}': sum={s:.6f} props={props}"
            )
        n1 = int(self.n_features * self.prop_type1)
        n2 = int(self.n_features * self.prop_type2)
        n3 = int(self.n_features * self.prop_type3)
        n4 = self.n_features - n1 - n2 - n3
        if n4 < 0:
            raise ValueError(
                f"Feature counts invalid in scenario '{self.name}': n4={n4} "
                f"(n_features={self.n_features}, props={props})"
            )
        return (n1, n2, n3, n4)

SCENARIOS = {
    "coupled": ScenarioParams(
        "coupled",
        "High-var features ARE predictive",
        prop_type1=0.20,
        prop_type2=0.10,
        prop_type3=0.60,
        prop_type4=0.10,
        between_var_type1=16.0,  # High Fisher ratio (16/4=4)
        within_var_type1=4.0,    # High total variance (20)
        between_var_type2=0.2,   # Low Fisher ratio (0.2/1=0.2)
        within_var_type2=1.0,    # Low total variance (1.2)
    ),
    "decoupled": ScenarioParams(
        "decoupled",
        "Variance orthogonal to prediction",
        # Defaults give ~random relationship
    ),
    "anti_aligned": ScenarioParams(
        "anti_aligned",
        "High-var features ANTI-predictive",
        prop_type1=0.05,
        prop_type2=0.45,
        prop_type3=0.40,
        prop_type4=0.10,
        between_var_type1=4.0,   # Predictive but LOW total (4.5)
        within_var_type1=0.5,
        between_var_type2=0.1,   # Not predictive but HIGH total (12.1)
        within_var_type2=12.0,
    ),
}

def generate_synthetic_dataset(params: ScenarioParams, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1, n2, n3, n4 = params.feature_counts
    feature_types = np.concatenate([np.ones(n1)*1, np.ones(n2)*2, np.ones(n3)*3, np.ones(n4)*4]).astype(int)
    between_var = np.concatenate([np.ones(n1)*params.between_var_type1, np.ones(n2)*params.between_var_type2, np.ones(n3)*params.between_var_type3, np.ones(n4)*params.between_var_type4])
    within_var = np.concatenate([np.ones(n1)*params.within_var_type1, np.ones(n2)*params.within_var_type2, np.ones(n3)*params.within_var_type3, np.ones(n4)*params.within_var_type4])
    between_var *= rng.uniform(0.8, 1.2, params.n_features)
    within_var *= rng.uniform(0.8, 1.2, params.n_features)
    # Balanced labels without dropping samples when n_samples is not divisible by n_classes
    base = params.n_samples // params.n_classes
    y = np.concatenate([np.full(base, c, dtype=int) for c in range(params.n_classes)])
    rem = params.n_samples - base * params.n_classes
    if rem > 0:
        y = np.concatenate([y, rng.integers(0, params.n_classes, size=rem, dtype=int)])
    rng.shuffle(y)
    class_means = np.array([rng.normal(0, np.sqrt(between_var)) for _ in range(params.n_classes)])
    class_means -= class_means.mean(axis=0)
    X = np.array([class_means[y[i]] + rng.normal(0, np.sqrt(within_var)) for i in range(params.n_samples)])
    return {"X": X.astype(np.float32), "y": y.astype(np.int32), "feature_types": feature_types, "true_between_var": between_var.astype(np.float32), "true_within_var": within_var.astype(np.float32)}

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument("--n-datasets", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-features", type=int, default=1000)
    args = parser.parse_args(argv)
    
    scenario_names = list(SCENARIOS.keys()) if args.scenarios == "all" else args.scenarios.split(",")
    output_dir = args.outputs_dir / "09_simulation" / "synthetic_data"
    ensure_dir(output_dir)
    
    print(f"[{now_iso()}] Generating synthetic datasets...")
    for sn in scenario_names:
        params = replace(SCENARIOS[sn], n_features=args.n_features)
        for i in range(args.n_datasets):
            seed = args.base_seed + i
            data = generate_synthetic_dataset(params, seed)
            np.savez_compressed(output_dir / f"synthetic__{sn}__{seed}.npz", **data, scenario_name=sn, seed=seed)
            print(f"  {sn} seed={seed}")
    print(f"[{now_iso()}] Done.")

if __name__ == "__main__":
    main()
