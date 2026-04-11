#!/usr/bin/env python3
"""
Shared utilities for Phase 12 (VAD): fast, supervision-light diagnostics for
variance-filtering risk.

Core primitives:
  - Per-feature total variance V_j
  - Per-feature between-class variance B_j
  - Signal fraction eta^2_j = B_j / (V_j + eps)
From these we compute:
  - etaES(K): enrichment of eta^2 within the TopVar(K%) tail
  - VSA(K): AUROC(eta^2; TopVar vs rest) - 0.5  (Mann–Whitney effect)
  - alpha': Spearman(V, eta^2)   (K-free, monotone association)
  - PCA alignment: SAS and/or PCLA

Design goals:
  - deterministic, fast (no model training)
  - robust to NaNs
  - works fold-by-fold on training indices only (no leakage in validation)

This module has no project-specific IO. It operates on (X, y) arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None


_EPS = 1e-12


def _as_float_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x.astype(np.float32, copy=False)
    return x


def _nanmean0(X: np.ndarray, axis: int) -> np.ndarray:
    m = np.nanmean(X, axis=axis)
    m = np.where(np.isfinite(m), m, 0.0)
    return m


def _nanvar0(X: np.ndarray, axis: int) -> np.ndarray:
    v = np.nanvar(X, axis=axis, ddof=0)
    v = np.where(np.isfinite(v), v, 0.0)
    return v


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    if _stats is not None:
        r, _p = _stats.spearmanr(a[m], b[m])
        return float(r) if np.isfinite(r) else float("nan")

    # fallback: rank transform via argsort (ties handled approximately)
    ra = a[m].argsort().argsort().astype(float)
    rb = b[m].argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    if denom <= 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _mean_impute(X: np.ndarray) -> np.ndarray:
    """Mean-impute NaNs (column-wise)."""
    X = _as_float_array(X)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(X.dtype, copy=False)
    out = X.copy()
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = np.take(col_means, np.where(mask)[1])
    return out


def eta2_features(X: np.ndarray, y: np.ndarray, eps: float = _EPS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-feature:
      - V_total (nanvar)
      - V_between (law of total variance, using class means)
      - eta^2 = V_between / (V_total + eps), clipped to [0, 1]
    """
    X = _as_float_array(X)
    y = np.asarray(y)

    n, p = X.shape
    if n == 0 or p == 0:
        return np.zeros(p, dtype=float), np.zeros(p, dtype=float), np.zeros(p, dtype=float)

    # Total variance and global mean (NaN-robust)
    mu = _nanmean0(X, axis=0)
    v_total = _nanvar0(X, axis=0)

    # Between-class variance
    classes, counts = np.unique(y, return_counts=True)
    v_between = np.zeros(p, dtype=np.float64)

    for c, cnt in zip(classes, counts):
        mask = (y == c)
        if cnt <= 0:
            continue
        w = float(cnt) / float(n)
        mu_c = _nanmean0(X[mask, :], axis=0)
        diff = (mu_c - mu).astype(np.float64, copy=False)
        v_between += w * (diff * diff)

    # eta^2
    denom = v_total.astype(np.float64, copy=False) + float(eps)
    eta2 = v_between / denom
    eta2 = np.where(np.isfinite(eta2), eta2, 0.0)
    eta2 = np.clip(eta2, 0.0, 1.0).astype(np.float64, copy=False)

    v_total = v_total.astype(np.float64, copy=False)
    v_between = np.where(np.isfinite(v_between), v_between, 0.0)

    return v_total, v_between, eta2


def eta_enrichment(
    eta2: np.ndarray,
    v_total: np.ndarray,
    k_pct: int = 10,
    eps: float = _EPS,
) -> Tuple[float, float, float]:
    """
    etaES(K) = mean(eta^2 in TopVar(K%)) / mean(eta^2 in all)

    Returns:
      eta_es, eta_topv, eta_all
    """
    eta2 = np.asarray(eta2, dtype=float)
    v_total = np.asarray(v_total, dtype=float)
    p = int(eta2.size)
    if p == 0:
        return float("nan"), float("nan"), float("nan")

    # TopVar indices (argpartition avoids full sort)
    top_n = max(1, int(p * (float(k_pct) / 100.0)))
    v = np.where(np.isfinite(v_total), v_total, -np.inf)
    if top_n >= p:
        top_idx = np.arange(p)
    else:
        top_idx = np.argpartition(-v, top_n - 1)[:top_n]

    eta_all = float(np.mean(eta2[np.isfinite(eta2)])) if np.isfinite(eta2).any() else 0.0
    eta_top = float(np.mean(eta2[top_idx])) if top_idx.size else float("nan")
    eta_es = float(eta_top / (eta_all + float(eps)))

    return eta_es, eta_top, eta_all


def vsa_mannwhitney(
    eta2: np.ndarray,
    v_total: np.ndarray,
    k_pct: int = 10,
) -> float:
    """
    VSA(K) = AUROC(eta^2; TopVar(K) vs Rest) - 0.5
    Computed via Mann–Whitney U.

    Returns signed effect in [-0.5, 0.5].
    """
    eta2 = np.asarray(eta2, dtype=float)
    v_total = np.asarray(v_total, dtype=float)
    p = int(eta2.size)
    if p < 3:
        return float("nan")

    top_n = max(1, int(p * (float(k_pct) / 100.0)))
    v = np.where(np.isfinite(v_total), v_total, -np.inf)
    if top_n >= p:
        return 0.0
    top_idx = np.argpartition(-v, top_n - 1)[:top_n]
    mask_top = np.zeros(p, dtype=bool)
    mask_top[top_idx] = True

    x1 = eta2[mask_top]
    x0 = eta2[~mask_top]
    if x1.size == 0 or x0.size == 0:
        return float("nan")

    # handle non-finite
    x1 = x1[np.isfinite(x1)]
    x0 = x0[np.isfinite(x0)]
    if x1.size == 0 or x0.size == 0:
        return float("nan")

    if _stats is not None:
        u, _p = _stats.mannwhitneyu(x1, x0, alternative="two-sided")
        auroc = float(u) / float(x1.size * x0.size)
        return float(auroc - 0.5)

    # fallback: rank-based U
    x = np.concatenate([x1, x0])
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    r1 = ranks[: x1.size].sum()
    u = r1 - x1.size * (x1.size + 1) / 2.0
    auroc = float(u) / float(x1.size * x0.size)
    return float(auroc - 0.5)


def alpha_prime(v_total: np.ndarray, eta2: np.ndarray) -> float:
    """alpha' = Spearman(Var_total, eta^2)."""
    return _safe_spearman(np.asarray(v_total, dtype=float), np.asarray(eta2, dtype=float))


def eta2_1d(z: np.ndarray, y: np.ndarray, eps: float = _EPS) -> float:
    """eta^2 for a single vector z against class labels y."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y)
    n = z.size
    if n == 0:
        return float("nan")
    mu = float(np.nanmean(z))
    v_total = float(np.nanvar(z, ddof=0))
    if not np.isfinite(v_total) or v_total <= 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    v_between = 0.0
    for c, cnt in zip(classes, counts):
        if cnt <= 0:
            continue
        w = float(cnt) / float(n)
        mu_c = float(np.nanmean(z[y == c]))
        diff = mu_c - mu
        v_between += w * diff * diff
    eta2 = v_between / (v_total + float(eps))
    if not np.isfinite(eta2):
        return 0.0
    return float(min(1.0, max(0.0, eta2)))


def pca_alignment(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 30,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    PCA-based multivariate alignment.

    Returns:
      - sas: Spearman(explained_variance, eta2_pc)
      - pcla: sum(norm_eigval * eta2_pc)   in [0,1]
    """
    if PCA is None:
        return {"sas": float("nan"), "pcla": float("nan")}

    X = _mean_impute(X)
    y = np.asarray(y)

    n, p = X.shape
    m = int(min(max(1, n_components), max(1, n - 1), max(1, p)))
    if m < 2:
        return {"sas": float("nan"), "pcla": float("nan")}

    pca = PCA(n_components=m, svd_solver="randomized", random_state=int(random_state))
    Z = pca.fit_transform(X)  # centers internally

    lambdas = np.asarray(pca.explained_variance_, dtype=float)
    if not np.isfinite(lambdas).all():
        lambdas = np.where(np.isfinite(lambdas), lambdas, 0.0)

    eta2_pcs = np.array([eta2_1d(Z[:, k], y) for k in range(Z.shape[1])], dtype=float)

    sas = _safe_spearman(lambdas, eta2_pcs)
    w = lambdas / (float(lambdas.sum()) + _EPS)
    pcla = float(np.sum(w * eta2_pcs))

    return {"sas": float(sas) if np.isfinite(sas) else float("nan"), "pcla": float(pcla)}


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Stratified bootstrap sample indices (resample within each class).
    Returns array of indices length n.
    """
    y = np.asarray(y)
    n = y.size
    idx_all = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if idx_c.size == 0:
            continue
        draw = rng.choice(idx_c, size=idx_c.size, replace=True)
        idx_all.append(draw)
    if not idx_all:
        return rng.choice(np.arange(n), size=n, replace=True)
    return np.concatenate(idx_all, axis=0)


def f_di(
    eta2: np.ndarray,
    v_total: np.ndarray,
    k_pct: int = 10,
    eps: float = _EPS,
) -> float:
    """
    F-DI(K): Supervision-free DI analog using η²-ranking instead of SHAP.

    Overlap between top-K% by variance and top-K% by η² (signal fraction).
    Mirrors the original DI formula:  F-DI = 1 − (J_obs − J_rand) / (1 − J_rand)

    Returns:
      F-DI value.  High F-DI → low overlap → anti-aligned structure.
      Comparable to original DI but without model training.
    """
    eta2 = np.asarray(eta2, dtype=float)
    v_total = np.asarray(v_total, dtype=float)
    p = int(eta2.size)
    if p < 3:
        return float("nan")

    top_n = max(1, int(p * (float(k_pct) / 100.0)))
    if top_n >= p:
        return 0.0

    v = np.where(np.isfinite(v_total), v_total, -np.inf)
    e = np.where(np.isfinite(eta2), eta2, -np.inf)

    idx_var = set(np.argpartition(-v, top_n - 1)[:top_n].tolist())
    idx_eta = set(np.argpartition(-e, top_n - 1)[:top_n].tolist())

    union = len(idx_var | idx_eta)
    if union == 0:
        return float("nan")
    j_obs = len(idx_var & idx_eta) / union
    q = float(top_n) / float(p)
    j_rand = q / (2.0 - q)
    tilde_j = (j_obs - j_rand) / (1.0 - j_rand + float(eps))
    return float(1.0 - tilde_j)


def classify_zone(
    eta_es: float,
    vsa: float,
    eta_es_lo: float = float("nan"),
    eta_es_hi: float = float("nan"),
    vsa_lo: float = float("nan"),
    vsa_hi: float = float("nan"),
    margin: float = 0.05,
) -> str:
    """
    Classify VAD risk zone using the (VSA, η_ES) decision map.

    Point-rule (default):
      RED:   η_ES < 1 and VSA < 0
      GREEN: η_ES > 1 and VSA > 0
      YELLOW: otherwise

    Conservative rule (if percentile intervals available):
      RED:   η_ES_hi < 1 and VSA_hi < 0
      GREEN: η_ES_lo > 1 and VSA_lo > 0
      YELLOW: otherwise

    If no intervals, uses a 5% margin on η_ES: 0.95 / 1.05, plus VSA sign.
    """
    import math

    has_ci = (
        math.isfinite(eta_es_lo) and math.isfinite(eta_es_hi) and
        math.isfinite(vsa_lo) and math.isfinite(vsa_hi)
    )

    if has_ci:
        if eta_es_hi < 1.0 and vsa_hi < 0.0:
            return "RED_HARMFUL"
        if eta_es_lo > 1.0 and vsa_lo > 0.0:
            return "GREEN_SAFE"
        return "YELLOW_INCONCLUSIVE"

    # point-estimate fallback (conservative margin on η_ES)
    if not (math.isfinite(eta_es) and math.isfinite(vsa)):
        return "YELLOW_INCONCLUSIVE"
    if eta_es < (1.0 - margin) and vsa < 0.0:
        return "RED_HARMFUL"
    if eta_es > (1.0 + margin) and vsa > 0.0:
        return "GREEN_SAFE"
    return "YELLOW_INCONCLUSIVE"
