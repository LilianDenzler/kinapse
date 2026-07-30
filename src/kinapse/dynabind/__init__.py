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
"""
from __future__ import annotations


def featurize(ensemble, **kwargs):
    """Turn a conformational ensemble into dynamics features (RMSF, entropy, MSM
    occupancies, geometry-angle distributions, bound-state overlap). (Scaffold.)"""
    raise NotImplementedError("dynabind.featurize is the research goal; not implemented yet")


def predict(features, **kwargs):
    """Predict binding / specificity / cross-reactivity from dynamics features. (Scaffold.)"""
    raise NotImplementedError("dynabind.predict is the research goal; not implemented yet")


__all__ = ["featurize", "predict"]
