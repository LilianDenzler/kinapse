"""structure_prediction — predict TCR / pMHC / TCR-pMHC structures (runner tier).

purpose: run external structure predictors from sequence.
extra:   ``[modelling]``
status:  scaffold — models are declared in ``registry.py``; execution wiring is WIP.

This is a *runner* tier: models are pluggable via :mod:`kinapse.runners`.
Add your own — see ``docs/ADDING_A_MODEL.md``.
"""
from __future__ import annotations

from kinapse import runners

TIER = "structure_prediction"


def available():
    """Registered structure-prediction models (:class:`kinapse.runners.RunnerSpec`)."""
    return runners.specs(TIER)


def run(name, inputs=None, timeout=600.0):
    """Run a registered model by name with a dict of inputs; returns a dict."""
    return runners.run(TIER, name, inputs, timeout)


__all__ = ["available", "run", "TIER"]
