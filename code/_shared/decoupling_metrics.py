#!/usr/bin/env python3
"""
decoupling_metrics.py

Single source of truth for the manuscript decoupling metrics.

Definitions (for a view v, top-K fraction q = K/N):
  J(K)      = |TopK(V) ∩ TopK(P)| / |TopK(V) ∪ TopK(P)|   (Jaccard overlap)
  J_rand(q) = q / (2 - q)                                 (expected J under random sets)
  ΔJ(K)     = J(K) - J_rand(q)
  J̃(K)     = (J(K) - J_rand(q)) / (1 - J_rand(q))        (normalized excess overlap)
  DI(K)     = 1 - J̃(K)                                   (Decoupling Index)
Interpretation:
  DI ≈ 0   coupled (above-random overlap)
  DI ≈ 1   random-like overlap
  DI > 1   anti-aligned (below random)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


# -----------------------------
# Core overlap / DI definitions
# -----------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard overlap of two sets. Returns 1.0 if both empty."""
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def j_rand(q: float) -> float:
    """Expected Jaccard overlap of two independent random size-q sets from N (limit N large)."""
    q = float(q)
    if q <= 0:
        return 0.0
    if q >= 1:
        return 1.0
    return q / (2.0 - q)


def delta_j(J: float, q: float) -> float:
    """ΔJ = J - J_rand(q)."""
    return float(J) - j_rand(q)


def j_tilde(J: float, q: float) -> float:
    """
    J̃ = (J - J_rand) / (1 - J_rand).
    This rescales J so that random overlap maps to 0, perfect overlap maps to 1.
    """
    Jr = j_rand(q)
    denom = 1.0 - Jr
    return (float(J) - Jr) / denom if denom > 0 else 0.0


def di(J: float, q: float) -> float:
    """DI = 1 - J̃."""
    return 1.0 - j_tilde(J, q)


def di_stable(J: float, q: float) -> float:
    """
    Algebraically equivalent stable form:
      DI = (1 - J) / (1 - J_rand)
    Useful to avoid subtractive cancellation when J ~ 1.
    """
    Jr = j_rand(q)
    denom = 1.0 - Jr
    return (1.0 - float(J)) / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class OverlapRow:
    k_pct: float
    k: int
    q: float
    J: float
    J_rand: float
    dJ: float
    J_tilde: float
    DI: float


def _topk(items_sorted: Sequence[str], k: int) -> Set[str]:
    k = int(k)
    if k <= 0:
        return set()
    k = min(k, len(items_sorted))
    return set(items_sorted[:k])


def compute_overlap_curve(
    ranked_by_variance: Sequence[str],
    ranked_by_importance: Sequence[str],
    k_pcts: Iterable[float],
) -> List[OverlapRow]:
    """
    Compute J, J_rand, ΔJ, J̃, DI across K% thresholds.

    Inputs must be ranked lists (best-first).
    Uses N = min(len(list1), len(list2)).
    """
    n = min(len(ranked_by_variance), len(ranked_by_importance))
    if n <= 0:
        return []

    out: List[OverlapRow] = []
    for k_pct in k_pcts:
        k_pct_f = float(k_pct)
        q = k_pct_f / 100.0
        k = int(round(q * n))
        k = max(1, min(k, n))

        A = _topk(ranked_by_variance, k)
        B = _topk(ranked_by_importance, k)

        J = jaccard(A, B)
        Jr = j_rand(q)
        dJ = J - Jr
        Jt = j_tilde(J, q)
        DIv = 1.0 - Jt  # Decoupling Index used in the manuscript

        out.append(
            OverlapRow(
                k_pct=k_pct_f,
                k=k,
                q=q,
                J=float(J),
                J_rand=float(Jr),
                dJ=float(dJ),
                J_tilde=float(Jt),
                DI=float(DIv),
            )
        )
    return out


def di_auc(
    curve: Sequence[OverlapRow],
    k_min_pct: Optional[float] = None,
    k_max_pct: Optional[float] = None,
) -> float:
    """
    Area under DI(K) curve over k_pct (trapezoidal), optionally over a restricted range.
    Returns NaN if fewer than 2 points remain.
    """
    if not curve:
        return float("nan")
    xs = np.array([r.k_pct for r in curve], dtype=float)
    ys = np.array([r.DI for r in curve], dtype=float)

    if k_min_pct is not None:
        m = xs >= float(k_min_pct)
        xs, ys = xs[m], ys[m]
    if k_max_pct is not None:
        m = xs <= float(k_max_pct)
        xs, ys = xs[m], ys[m]

    if xs.size < 2:
        return float("nan")
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    return float(np.trapz(ys, xs))


# -----------------------------------
# Utility: ranks / sorting convenience
# -----------------------------------

def rank_features_desc(scores: Mapping[str, float]) -> List[str]:
    """Return feature names sorted by score descending, stable tie-break by name."""
    return sorted(scores.keys(), key=lambda k: (-float(scores[k]), str(k)))


# -----------------------------------
# Stats helpers used by Phase 8 (GMT)
# -----------------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR correction.
    Returns q-values aligned to input order.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(k_minus_1: int, N: int, M: int, n: int) -> float:
    """
    Survival function P[X >= k] for Hypergeometric(N, M, n),
    where k = k_minus_1 + 1.
    Implemented via log-sum-exp to avoid SciPy dependency.
    """
    k = int(k_minus_1) + 1
    N, M, n = int(N), int(M), int(n)

    lo = max(0, n - (N - M))
    hi = min(n, M)

    if k <= lo:
        return 1.0
    if k > hi:
        return 0.0
    if N <= 0 or n <= 0 or M <= 0:
        return 0.0

    log_den = _log_choose(N, n)
    terms: List[float] = []
    for x in range(k, hi + 1):
        lp = _log_choose(M, x) + _log_choose(N - M, n - x) - log_den
        terms.append(lp)

    m = max(terms)
    return float(math.exp(m) * sum(math.exp(t - m) for t in terms))


def enrich_hypergeom(
    gene_list: Sequence[str],
    background: Sequence[str],
    gene_sets: Mapping[str, Sequence[str]],
    min_set_size: int = 10,
    max_set_size: int = 500,
) -> np.ndarray:
    """
    Compute hypergeometric enrichment p-values for each gene set.

    Returns a structured numpy array with fields:
      pathway, pval, fdr, overlap_k, set_size_M, list_size_n, bg_size_N
    """
    bg = set(map(str, background))
    gl = set(map(str, gene_list)) & bg
    N = len(bg)
    n = len(gl)

    rows = []
    if N == 0 or n == 0:
        dtype = [
            ("pathway", "U256"),
            ("pval", "f8"),
            ("fdr", "f8"),
            ("overlap_k", "i4"),
            ("set_size_M", "i4"),
            ("list_size_n", "i4"),
            ("bg_size_N", "i4"),
        ]
        return np.array([], dtype=dtype)

    for pw, genes in gene_sets.items():
        s = set(map(str, genes)) & bg
        M = len(s)
        if M < int(min_set_size) or M > int(max_set_size):
            continue
        k = len(gl & s)
        if k <= 0:
            continue
        p = hypergeom_sf(k - 1, N=N, M=M, n=n)
        rows.append((str(pw), float(p), int(k), int(M), int(n), int(N)))

    dtype = [
        ("pathway", "U256"),
        ("pval", "f8"),
        ("fdr", "f8"),
        ("overlap_k", "i4"),
        ("set_size_M", "i4"),
        ("list_size_n", "i4"),
        ("bg_size_N", "i4"),
    ]
    if not rows:
        return np.array([], dtype=dtype)

    arr = np.array(
        rows,
        dtype=[
            ("pathway", "U256"),
            ("pval", "f8"),
            ("overlap_k", "i4"),
            ("set_size_M", "i4"),
            ("list_size_n", "i4"),
            ("bg_size_N", "i4"),
        ],
    )
    fdr = bh_fdr(arr["pval"])
    out = np.zeros(arr.shape[0], dtype=dtype)
    out["pathway"] = arr["pathway"]
    out["pval"] = arr["pval"]
    out["fdr"] = fdr
    out["overlap_k"] = arr["overlap_k"]
    out["set_size_M"] = arr["set_size_M"]
    out["list_size_n"] = arr["list_size_n"]
    out["bg_size_N"] = arr["bg_size_N"]

    # sort by fdr then pval
    order = np.lexsort((out["pval"], out["fdr"]))
    return out[order]
