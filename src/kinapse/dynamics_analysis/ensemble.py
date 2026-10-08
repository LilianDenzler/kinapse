"""Point-in-ensemble membership for low-dimensional conformational descriptors.

Given a reference ensemble of descriptor vectors (e.g. the per-frame TCR α/β
docking-geometry vectors sampled by an apo MD trajectory) and a single query
vector (e.g. the same geometry measured on a *bound* structure), quantify how
well the query falls *within* the ensemble.

Two complementary notions — the distinction the analysis cares about:

* **Joint / "exact geometry"** — :func:`mahalanobis_membership` uses the
  Mahalanobis distance to the ensemble mean/covariance, so it respects the joint
  distribution and inter-variable correlations. A point can sit inside every
  1-D range yet be far from the joint density (a geometry the ensemble never
  actually visits); Mahalanobis catches that, a per-axis range check does not.
* **Per-marginal / "within the ranges"** — whether each coordinate independently
  lies inside the ensemble's central percentile range. Weaker; reported for
  contrast.

kinapse had no such primitive (no Mahalanobis / convex-hull / KDE point-density),
so this fills that gap. Pure numpy + scipy; no MD dependency.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def mahalanobis_membership(
    ensemble: np.ndarray,
    point: Sequence[float],
    *,
    alpha: float = 0.05,
    marginal_lo: float = 1.0,
    marginal_hi: float = 99.0,
    ridge: float = 1e-9,
    kde: bool = False,
) -> dict:
    """Score how far a single ``point`` lies inside a reference ``ensemble``.

    Args:
        ensemble: ``(N, K)`` array — N samples (e.g. MD frames) of a K-dim descriptor.
        point: length-``K`` query vector.
        alpha: significance for the joint containment ellipsoid; ``within_joint`` is
            True when the point lies inside the ``1 - alpha`` (default 95%) region,
            using the chi-square distribution with K degrees of freedom.
        marginal_lo, marginal_hi: percentile bounds for the per-axis range test.
        ridge: added to the covariance diagonal before inversion (numerical safety).
        kde: also fit a Gaussian KDE on the ensemble and report the point's
            log-density and its percentile among the ensemble's own densities
            (robust to non-Gaussian shape; slower). Falls back silently if the KDE
            is singular.

    Returns a dict:
        n, k, mahalanobis, mahalanobis_sq,
        chi2_percentile   — fraction of a Gaussian ensemble closer to the mean than
                            the point (0 = at the mean, →1 = far outlier),
        within_joint      — chi2_percentile <= 1 - alpha,
        per_marginal_within (bool, all axes in range),
        per_marginal_frac (fraction of axes in range),
        per_axis_within   — list[bool] per coordinate,
        kde_logdensity, kde_percentile (only if kde=True; percentile is the fraction
                            of ensemble frames with LOWER density than the point, so
                            small = the point sits where the ensemble rarely is).
    """
    from scipy.stats import chi2

    X = np.asarray(ensemble, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"ensemble must be (N, K); got shape {X.shape}")
    X = X[~np.isnan(X).any(axis=1)]
    x = np.asarray(point, dtype=float).ravel()
    n, k = X.shape
    if x.shape[0] != k:
        raise ValueError(f"point has length {x.shape[0]}, ensemble has {k} columns")

    out = {"n": int(n), "k": int(k)}
    if n < k + 1 or np.isnan(x).any():
        out.update(mahalanobis=np.nan, mahalanobis_sq=np.nan, chi2_percentile=np.nan,
                   within_joint=None, per_marginal_within=None, per_marginal_frac=np.nan,
                   per_axis_within=[], kde_logdensity=np.nan, kde_percentile=np.nan)
        return out

    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov) + ridge * np.eye(k)
    vi = np.linalg.pinv(cov)
    d = x - mu
    d2 = float(d @ vi @ d)
    d2 = max(d2, 0.0)
    out["mahalanobis_sq"] = d2
    out["mahalanobis"] = float(np.sqrt(d2))
    out["chi2_percentile"] = float(chi2.cdf(d2, df=k))
    out["within_joint"] = bool(out["chi2_percentile"] <= 1.0 - alpha)

    lo = np.percentile(X, marginal_lo, axis=0)
    hi = np.percentile(X, marginal_hi, axis=0)
    axis_within = (x >= lo) & (x <= hi)
    out["per_axis_within"] = [bool(v) for v in axis_within]
    out["per_marginal_frac"] = float(axis_within.mean())
    out["per_marginal_within"] = bool(axis_within.all())

    out["kde_logdensity"] = np.nan
    out["kde_percentile"] = np.nan
    if kde:
        try:
            from scipy.stats import gaussian_kde
            g = gaussian_kde(X.T)
            lp_point = float(g.logpdf(x)[0])
            lp_ens = g.logpdf(X.T)
            out["kde_logdensity"] = lp_point
            out["kde_percentile"] = float((lp_ens <= lp_point).mean())
        except Exception:  # noqa: BLE001 — singular KDE, keep NaNs
            pass
    return out


def ensemble_stats(ensemble: np.ndarray, columns: Optional[Sequence[str]] = None) -> dict:
    """Per-axis mean/std/min/max/1-99pct summary of an ensemble (for reporting)."""
    X = np.asarray(ensemble, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    keys = list(columns) if columns is not None else [f"x{i}" for i in range(X.shape[1])]
    return {
        "n": int(X.shape[0]),
        "mean": dict(zip(keys, X.mean(0))),
        "std": dict(zip(keys, X.std(0))),
        "p1": dict(zip(keys, np.percentile(X, 1, axis=0))),
        "p99": dict(zip(keys, np.percentile(X, 99, axis=0))),
    }


__all__ = ["mahalanobis_membership", "ensemble_stats"]
