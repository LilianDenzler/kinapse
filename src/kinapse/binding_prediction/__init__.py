"""binding_prediction — TCR-pMHC binding / specificity predictors (runner tier).

purpose: run external TCR-pMHC binding/specificity/cross-reactivity predictors.
extra:   ``[binding]``
status:  scaffold — models declared in ``registry.py``; execution wiring is WIP.

Runner tier: models are pluggable via :mod:`kinapse.runners`.
Add your own — see ``docs/ADDING_A_MODEL.md``.
"""
from __future__ import annotations

from kinapse import runners

TIER = "binding_prediction"


def available():
    """Registered binding/specificity models."""
    return runners.specs(TIER)


def run(name, inputs=None, timeout=600.0):
    """Run a registered model by name with a dict of inputs; returns a dict."""
    return runners.run(TIER, name, inputs, timeout)


__all__ = ["available", "run", "TIER"]
