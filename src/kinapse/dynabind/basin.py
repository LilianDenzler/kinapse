"""Binding-compatible basin definition and conformational-access free energy.

This is the first concrete piece of the ``dynabind`` method: turning the overlap
between a bound and an unbound conformational ensemble into a free energy. It
implements the ``ΔG_access`` term of the LC13 thermodynamic cycle
(see ``experiments/LC13_calculations/PLAN.md``):

    B          a region of conformational space that covers a target fraction of
               the *bound* ensemble density ("binding-compatible" conformations).
    P_U(B)     the fraction of an *unbound* ensemble that falls inside B.
    ΔG_access  = -R T ln P_U(B).

The basin is estimated **nonparametrically** — no Gaussian assumption:

* ``method="knn"`` (default) — k-nearest-neighbour density. B is the highest
  k-NN-density region of the bound ensemble enclosing ``coverage`` of it; a query
  point is inside B if its distance to the k-th nearest bound frame is within the
  bound's own coverage-quantile of that distance. Works in any dimension and
  degrades gracefully (unlike a Gaussian ellipsoid, which misrepresents the
  banana-shaped loop distributions, or a KDE, which fails above ~3-D).
* ``method="kde"`` — KDE highest-density region (best in ≤2-3 D; exact-ish shape).

Design contract (mirrors the plan's scientific invariants):

* **B is defined from the bound *ensemble*, never from RMSD to one crystal.**
* **P_U(B) = 0 in a finite sample is UNRESOLVED, not ΔG = +∞** — escalate to
  enhanced sampling; ``access_free_energy`` returns ``inf`` and the bootstrap
  helpers flag ``unresolved`` / ``regime="unresolved"``.
* **Uncertainty is estimated by resampling whole MD replicas, never frames.**
* **ΔG_access already contains the conformational-entropy cost of the
  restriction** — do not add a separate ``-TΔS_conf`` term.

Everything takes low-dimensional *projections* (the CV embeddings from
:mod:`kinapse.dynamics_analysis`), so it is trivially unit-testable and reusable
for any TCR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Gas constant in the two unit systems used across kinapse.
R_KCAL_PER_MOL_K = 0.0019872041          # kcal / (mol K)
R_KJ_PER_MOL_K = 0.00831446261815324     # kJ  / (mol K)
_R = {"kcal": R_KCAL_PER_MOL_K, "kj": R_KJ_PER_MOL_K}

__all__ = [
    "Basin",
    "define_basin",
    "basin_population",
    "access_free_energy",
    "basin_population_bootstrap",
    "access_free_energy_bootstrap",
    "access_over_coverages",
]


def _2d(x) -> np.ndarray:
    """Coerce to a float ``(n_frames, n_dims)`` array."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D (n_frames, n_dims) array, got shape {a.shape}")
    return a


def _weighted_quantile(values, weights, q):
    """Weighted q-quantile of ``values`` (weights need not be normalized)."""
    v = np.asarray(values, float); w = np.asarray(weights, float)
    idx = np.argsort(v)
    v, w = v[idx], w[idx]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.quantile(v, q))
    return float(np.interp(q, cw / cw[-1], v))


@dataclass
class Basin:
    """A binding-compatible region ``B`` in a low-dimensional CV space.

    Construct via :func:`define_basin`. ``contains`` gives per-frame membership
    and ``population`` gives the fraction of a query ensemble inside ``B``.
    """

    method: str          # "knn" | "kde"
    coverage: float      # target fraction of the bound density enclosed
    dim: int
    # kde parameters
    _kde: Optional[object] = None
    _threshold: Optional[float] = None
    # knn parameters
    _tree: Optional[object] = None
    _k: Optional[int] = None
    _r_thr: Optional[float] = None

    def contains(self, Z) -> np.ndarray:
        """Boolean membership mask for each row of ``Z`` (shape ``(n_frames,)``)."""
        Z = _2d(Z)
        if Z.shape[1] != self.dim:
            raise ValueError(f"query has {Z.shape[1]} dims, basin was built on {self.dim}")
        if self.method == "kde":
            dens = np.asarray(self._kde(Z.T))          # gaussian_kde wants (dim, n)
            return dens >= self._threshold
        if self.method == "knn":
            d, _ = self._tree.query(Z, k=self._k)
            rq = d[:, -1] if self._k > 1 else np.asarray(d).ravel()
            return rq <= self._r_thr
        raise ValueError(f"unknown basin method {self.method!r}")

    def population(self, Z, weights=None) -> float:
        """Fraction of ``Z`` inside ``B``. With per-frame ``weights`` (e.g. MSM
        stationary weights) this is the equilibrium-reweighted ``P(B)``."""
        m = self.contains(Z)
        if not m.size:
            return 0.0
        if weights is None:
            return float(np.mean(m))
        w = np.asarray(weights, float)
        return float((m.astype(float) * w).sum() / w.sum()) if w.sum() > 0 else 0.0


def define_basin(
    Z_bound,
    coverage: float = 0.90,
    method: str = "knn",
    *,
    weights=None,
    k: Optional[int] = None,
    max_ref: Optional[int] = 8000,
    bw_method="scott",
    max_kde_samples: Optional[int] = 4000,
    seed: int = 0,
) -> Basin:
    """Define the binding-compatible basin ``B`` from a *bound* projection.

    Parameters
    ----------
    Z_bound : array (n_frames, n_dims)
        Low-dimensional projection of the bound ensemble (the CV embedding).
    coverage : float in (0, 1)
        Fraction of the bound density that ``B`` should enclose (e.g. 0.90).
    method : {"knn", "kde"}
        ``knn`` — k-NN highest-density region (nonparametric, any dimension).
        ``kde`` — Gaussian-KDE highest-density region (best in ≤2-3 D).
    k : int, optional
        Neighbour count for ``knn`` (default ``round(sqrt(n_ref))``). Larger k
        smooths the density estimate.
    max_ref : int, optional
        Cap on bound frames used as the k-NN reference set (default 8000; keeps
        the tree query tractable on MD-scale ensembles). ``None`` uses all.
    max_kde_samples : int, optional
        Cap on bound frames used to fit the KDE (``method="kde"`` only).

    Returns
    -------
    Basin
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0, 1)")
    Z = _2d(Z_bound)
    n, dim = Z.shape

    if method == "knn":
        from scipy.spatial import cKDTree

        ref = Z
        ref_w = None if weights is None else np.asarray(weights, float)
        if max_ref is not None and n > max_ref:
            rng = np.random.default_rng(seed)
            sel = rng.choice(n, size=max_ref, replace=False)
            ref = Z[sel]
            if ref_w is not None:
                ref_w = ref_w[sel]
        m = ref.shape[0]
        if m <= 2:
            raise ValueError(f"k-NN basin needs >2 reference frames, got {m}")
        kk = int(k) if k is not None else max(5, int(round(np.sqrt(m))))
        kk = min(kk, m - 1)
        tree = cKDTree(ref)
        # bound self-distances: query k+1 and drop the self match (distance 0)
        dref, _ = tree.query(ref, k=kk + 1)
        r_self = dref[:, -1]
        # highest-density region: keep the `coverage` fraction with SMALLEST k-NN
        # distance (= highest density). With weights, coverage = fraction of the
        # equilibrium MASS rather than of the frame count.
        r_thr = (float(np.quantile(r_self, coverage)) if ref_w is None
                 else _weighted_quantile(r_self, ref_w, coverage))
        return Basin(method="knn", coverage=coverage, dim=dim, _tree=tree, _k=kk, _r_thr=r_thr)

    if method == "kde":
        from scipy.stats import gaussian_kde

        if n <= dim:
            raise ValueError(f"KDE basin needs n_frames ({n}) > n_dims ({dim})")
        Zfit = Z
        if max_kde_samples is not None and n > max_kde_samples:
            rng = np.random.default_rng(seed)
            Zfit = Z[rng.choice(n, size=max_kde_samples, replace=False)]
        kde = gaussian_kde(Zfit.T, bw_method=bw_method)
        dens = np.asarray(kde(Zfit.T))
        thr = float(np.quantile(dens, 1.0 - coverage))
        return Basin(method="kde", coverage=coverage, dim=dim, _kde=kde, _threshold=thr)

    raise ValueError(f"unknown basin method {method!r} (use 'knn' or 'kde')")


def basin_population(basin: Basin, Z_query) -> float:
    """``P(B)`` — fraction of the query ensemble ``Z_query`` inside ``basin``."""
    return basin.population(Z_query)


def access_free_energy(p: float, temperature: float = 310.0, units: str = "kcal") -> float:
    """``ΔG_access = -R T ln p``.

    Returns ``inf`` when ``p <= 0`` (UNRESOLVED — the unbound MD did not sample
    the basin; a signal to use enhanced sampling, *not* a real +∞).
    """
    R = _R[units.lower()]
    p = float(p)
    if p <= 0.0:
        return float("inf")
    return -R * temperature * float(np.log(p))


def _replica_groups(replica_ids: Sequence) -> Tuple[np.ndarray, list]:
    ids = np.asarray(replica_ids)
    uniq = np.unique(ids)
    groups = [np.where(ids == r)[0] for r in uniq]
    return uniq, groups


def basin_population_bootstrap(
    basin: Basin,
    Z_query,
    replica_ids: Sequence,
    *,
    weights=None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, object]:
    """Replica-level bootstrap of ``P(B)``.

    Frames within a replica are autocorrelated, so we resample whole replicas
    with replacement (never individual frames). Optional per-frame ``weights``
    (e.g. MSM stationary weights) make this an equilibrium-reweighted population:
    each replica contributes its inside-mass and total-mass, and the pooled
    estimate is Σ(inside mass) / Σ(total mass) over the resampled replicas.
    """
    Z = _2d(Z_query)
    if len(replica_ids) != Z.shape[0]:
        raise ValueError("replica_ids length must equal n_frames of Z_query")
    inside = basin.contains(Z).astype(float)
    w = np.ones(Z.shape[0]) if weights is None else np.asarray(weights, float)
    uniq, groups = _replica_groups(replica_ids)
    win = np.array([float((inside[g] * w[g]).sum()) for g in groups])   # inside mass per replica
    wtot = np.array([float(w[g].sum()) for g in groups])                # total mass per replica
    point = float(win.sum() / wtot.sum()) if wtot.sum() > 0 else 0.0

    rng = np.random.default_rng(seed)
    R = len(uniq)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, R, size=R)
        denom = wtot[pick].sum()
        boots[b] = float(win[pick].sum() / denom) if denom > 0 else 0.0

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "p": point,
        "p_mean_boot": float(boots.mean()),
        "p_se": float(boots.std(ddof=1)) if n_boot > 1 else 0.0,
        "p_ci95": (float(lo), float(hi)),
        "n_replicas": int(R),
        "n_frames": int(Z.shape[0]),
        "boots": boots,
    }


def _regime(p: float) -> str:
    """Qualitative preorganization verdict (thresholds from the writeup)."""
    if p <= 0.0:
        return "unresolved"       # not sampled -> enhanced sampling
    if p >= 0.1:
        return "preorganized"     # unbound already largely binding-ready
    if p >= 1e-3:
        return "accessible"       # binding-compatible state is reachable
    return "rare_access"          # binding needs a rare conformational excursion


def access_free_energy_bootstrap(
    basin: Basin,
    Z_query,
    replica_ids: Sequence,
    *,
    weights=None,
    temperature: float = 310.0,
    units: str = "kcal",
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, object]:
    """``ΔG_access ± CI`` with a replica-level bootstrap and a regime flag.

    Pass per-frame ``weights`` (e.g. MSM stationary weights) to compute the
    equilibrium-reweighted access free energy instead of the raw frame-count one.
    """
    R = _R[units.lower()]
    bp = basin_population_bootstrap(basin, Z_query, replica_ids, weights=weights, n_boot=n_boot, seed=seed)
    p = bp["p"]
    unresolved = p <= 0.0
    dg = access_free_energy(p, temperature=temperature, units=units)

    boots = bp["boots"]
    with np.errstate(divide="ignore"):
        dg_boots = np.where(boots > 0.0, -R * temperature * np.log(boots), np.inf)
    finite = dg_boots[np.isfinite(dg_boots)]

    result: Dict[str, object] = {
        "p": p,
        "p_ci95": bp["p_ci95"],
        "p_se": bp["p_se"],
        "dG_access": dg,
        "dG_units": units,
        "temperature": temperature,
        "coverage": basin.coverage,
        "method": basin.method,
        "n_replicas": bp["n_replicas"],
        "n_frames": bp["n_frames"],
        "unresolved": bool(unresolved),
        "frac_boot_unresolved": float(np.mean(~np.isfinite(dg_boots))),
        "regime": _regime(p),
    }
    if finite.size:
        lo, hi = np.quantile(finite, [0.025, 0.975])
        result["dG_ci95"] = (float(lo), float(hi))
        result["dG_se"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    else:
        result["dG_ci95"] = (float("inf"), float("inf"))
        result["dG_se"] = float("nan")
    return result


def access_over_coverages(
    Z_bound,
    Z_query,
    replica_ids: Sequence,
    *,
    coverages: Sequence[float] = (0.80, 0.90, 0.95),
    method: str = "knn",
    temperature: float = 310.0,
    units: str = "kcal",
    n_boot: int = 1000,
    seed: int = 0,
    **basin_kwargs,
) -> list:
    """Run the access calculation across several basin coverages (Phase 4 → 5).

    Because ``U*`` is a construct, the coverage level is a modelling choice; this
    sweeps it so the robustness of ``ΔG_access`` to the basin definition is a
    reported output rather than a hidden assumption. Returns one
    :func:`access_free_energy_bootstrap` dict per coverage.
    """
    out = []
    for c in coverages:
        basin = define_basin(Z_bound, coverage=c, method=method, seed=seed, **basin_kwargs)
        res = access_free_energy_bootstrap(
            basin, Z_query, replica_ids,
            temperature=temperature, units=units, n_boot=n_boot, seed=seed,
        )
        out.append(res)
    return out
