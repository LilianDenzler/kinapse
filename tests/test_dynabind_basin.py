"""Tests for kinapse.dynabind.basin — binding-compatible basin + ΔG_access.

Synthetic ensembles with known overlap so P_U(B) and ΔG_access are checkable
against closed-form expectations. No MD data required.
"""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from kinapse.dynabind import (  # noqa: E402
    Basin,
    access_free_energy,
    access_free_energy_bootstrap,
    access_over_coverages,
    basin_population,
    define_basin,
)


def _gauss(n, mean=(0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2)) + np.asarray(mean, float)


def test_lazy_exports_are_wired():
    # exercised via the top-level lazy import above
    assert isinstance(Basin, type)
    for f in (define_basin, basin_population, access_free_energy):
        assert callable(f)


@pytest.mark.parametrize("method", ["kde", "knn"])
def test_bound_self_population_matches_coverage(method):
    """B built at coverage c should enclose ~c of the ensemble it was built on."""
    Z = _gauss(4000, seed=1)
    for c in (0.80, 0.90, 0.95):
        b = define_basin(Z, coverage=c, method=method)
        assert b.population(Z) == pytest.approx(c, abs=0.05)


@pytest.mark.parametrize("method", ["kde", "knn"])
def test_identical_unbound_overlaps_at_coverage(method):
    """An unbound ensemble from the same distribution overlaps B at ~coverage."""
    Z_bound = _gauss(4000, seed=1)
    Z_unbound = _gauss(4000, seed=2)  # same distribution, different sample
    b = define_basin(Z_bound, coverage=0.90, method=method)
    assert basin_population(b, Z_unbound) == pytest.approx(0.90, abs=0.06)


@pytest.mark.parametrize("method", ["kde", "knn"])
def test_far_shifted_unbound_is_unresolved(method):
    """A distant unbound ensemble never enters B -> P=0, ΔG=inf, unresolved."""
    Z_bound = _gauss(3000, mean=(0, 0), seed=1)
    Z_unbound = _gauss(3000, mean=(50, 50), seed=2)
    b = define_basin(Z_bound, coverage=0.90, method=method)
    p = basin_population(b, Z_unbound)
    assert p == 0.0
    assert access_free_energy(p) == math.inf


def test_access_free_energy_closed_form():
    # -R T ln p ; R=0.0019872041 kcal/mol/K, T=310, p=0.1
    dg = access_free_energy(0.1, temperature=310.0, units="kcal")
    assert dg == pytest.approx(-0.0019872041 * 310.0 * math.log(0.1), rel=1e-9)
    # kJ path
    dg_kj = access_free_energy(0.1, temperature=310.0, units="kj")
    assert dg_kj == pytest.approx(-0.00831446261815324 * 310.0 * math.log(0.1), rel=1e-9)
    # p<=0 -> inf, never raises
    assert access_free_energy(0.0) == math.inf


def test_partial_overlap_gives_finite_positive_dg():
    """Unbound half-in/half-out of B -> intermediate P and positive ΔG."""
    Z_bound = _gauss(4000, mean=(0, 0), seed=1)
    # unbound centred at the edge of the bound basin -> partial overlap
    Z_unbound = _gauss(4000, mean=(1.5, 0.0), seed=2)
    b = define_basin(Z_bound, coverage=0.90, method="knn")
    p = basin_population(b, Z_unbound)
    assert 0.0 < p < 0.9
    dg = access_free_energy(p, temperature=310.0)
    assert dg > 0.0 and math.isfinite(dg)


def test_replica_bootstrap_shape_and_ci_contains_point():
    Z_bound = _gauss(4000, seed=1)
    # 10 unbound replicas of 400 frames each
    reps = [_gauss(400, mean=(0.3, 0.0), seed=100 + i) for i in range(10)]
    Z_unbound = np.vstack(reps)
    replica_ids = np.repeat(np.arange(10), 400)
    b = define_basin(Z_bound, coverage=0.90, method="knn")
    res = access_free_energy_bootstrap(
        b, Z_unbound, replica_ids, temperature=310.0, n_boot=500, seed=0
    )
    assert res["n_replicas"] == 10
    assert res["n_frames"] == 4000
    lo, hi = res["p_ci95"]
    assert lo <= res["p"] <= hi
    assert res["regime"] in {"preorganized", "accessible", "rare_access", "unresolved"}
    assert res["dG_units"] == "kcal"


def test_replica_bootstrap_rejects_mismatched_ids():
    Z = _gauss(100, seed=1)
    b = define_basin(_gauss(2000, seed=2), coverage=0.9, method="knn")
    with pytest.raises(ValueError):
        access_free_energy_bootstrap(b, Z, np.arange(50))  # wrong length


def test_access_over_coverages_sweep():
    Z_bound = _gauss(4000, seed=1)
    reps = [_gauss(400, mean=(0.5, 0.0), seed=200 + i) for i in range(8)]
    Z_unbound = np.vstack(reps)
    replica_ids = np.repeat(np.arange(8), 400)
    out = access_over_coverages(
        Z_bound, Z_unbound, replica_ids,
        coverages=(0.80, 0.90, 0.95), method="kde", n_boot=200, seed=0,
    )
    assert [r["coverage"] for r in out] == [0.80, 0.90, 0.95]
    # a wider basin (higher coverage) can only include more -> P non-decreasing
    ps = [r["p"] for r in out]
    assert ps[0] <= ps[1] + 1e-9 <= ps[2] + 1e-9


def test_weighted_population_reweights():
    """Per-frame weights reweight P(B): down-weighting the outside cluster -> P->1."""
    Z_bound = _gauss(4000, seed=1)
    rng = np.random.default_rng(3)
    near = rng.normal(size=(2000, 2))                    # inside the bound basin
    far = rng.normal(size=(2000, 2)) + np.array([6.0, 0.0])  # outside
    Z_unb = np.vstack([near, far])
    b = define_basin(Z_bound, coverage=0.9, method="knn")
    p_uniform = b.population(Z_unb)
    p_near = b.population(Z_unb, weights=np.concatenate([np.ones(2000), np.full(2000, 1e-6)]))
    p_far = b.population(Z_unb, weights=np.concatenate([np.full(2000, 1e-6), np.ones(2000)]))
    assert p_far < p_uniform < p_near
    assert p_near > 0.85 and p_far < 0.1


def test_weighted_bootstrap_reduces_to_unweighted_when_uniform():
    Z_bound = _gauss(4000, seed=1)
    reps = [_gauss(400, mean=(0.3, 0.0), seed=100 + i) for i in range(10)]
    Z_unb = np.vstack(reps); ids = np.repeat(np.arange(10), 400)
    b = define_basin(Z_bound, coverage=0.9, method="knn")
    r0 = access_free_energy_bootstrap(b, Z_unb, ids, n_boot=300, seed=0)
    r1 = access_free_energy_bootstrap(b, Z_unb, ids, weights=np.ones(len(Z_unb)), n_boot=300, seed=0)
    assert abs(r0["p"] - r1["p"]) < 1e-9


def test_invalid_inputs():
    with pytest.raises(ValueError):
        define_basin(_gauss(100), coverage=1.5)
    with pytest.raises(ValueError):
        define_basin(_gauss(100), method="nope")
    # KDE needs more frames than dims
    with pytest.raises(ValueError):
        define_basin(np.zeros((2, 5)), method="kde")
