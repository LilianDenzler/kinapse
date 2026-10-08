"""dynabind — ★ NOVEL: dynamics → binding (native science tier).

The project's headline hypothesis and clearest field whitespace: TCR conformational
dynamics (CDR-loop flexibility, ensemble state occupancies, bound-state overlap)
inform binding / specificity / cross-reactivity beyond a single static structure.
No existing TCR-pMHC predictor uses conformational-ensemble features — this module
is where that loop is closed.

purpose: ensemble → dynamics features → calibrated binding/cross-reactivity head.
extra:   ``[dynabind]``
status:  scaffold — interface declared; the method is the research goal.

Consumes: :mod:`kinapse.dynamics_analysis` (features/MSM/NMA/flexibility),
:mod:`kinapse.conformer_generation` (ensembles), :mod:`kinapse.datasets`
(ATLAS ΔG / DMS cross-reactivity labels). Benchmarked in :mod:`kinapse.benchmarks`
against sequence-only and static-structure baselines on unseen-epitope splits.

First concrete piece: :mod:`kinapse.dynabind.basin` — bound/unbound ensemble
overlap → binding-compatible basin ``B`` → ``P_U(B)`` → ``ΔG_access`` (replica-level
bootstrap). TCR-agnostic: it operates on generic CV projections, so it transfers
to any receptor. Lazily exported here (needs the ``[dynabind]`` extra, i.e. scipy).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazily re-exported from .basin so ``import kinapse.dynabind`` stays cheap and
# never fails when scipy is absent (PEP 562).
_BASIN_EXPORTS = frozenset({
    "Basin",
    "define_basin",
    "basin_population",
    "access_free_energy",
    "basin_population_bootstrap",
    "access_free_energy_bootstrap",
    "access_over_coverages",
})


def featurize(ensemble, **kwargs):
    """Turn a conformational ensemble into dynamics features (RMSF, entropy, MSM
    occupancies, geometry-angle distributions, bound-state overlap). (Scaffold.)"""
    raise NotImplementedError("dynabind.featurize is the research goal; not implemented yet")


def predict(features, **kwargs):
    """Predict binding / specificity / cross-reactivity from dynamics features. (Scaffold.)"""
    raise NotImplementedError("dynabind.predict is the research goal; not implemented yet")


def __getattr__(name):
    if name in _BASIN_EXPORTS:
        from . import basin as _basin
        return getattr(_basin, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _BASIN_EXPORTS)


if TYPE_CHECKING:  # help type checkers/IDEs see the lazy names
    from .basin import (  # noqa: F401
        Basin,
        access_free_energy,
        access_free_energy_bootstrap,
        access_over_coverages,
        basin_population,
        basin_population_bootstrap,
        define_basin,
    )


__all__ = ["featurize", "predict", *sorted(_BASIN_EXPORTS)]
