"""docking — TCR-pMHC docking engines (runner tier).

purpose: run external docking tools to build/refine TCR-pMHC complexes.
extra:   ``[docking]``
status:  scaffold — engines declared in ``registry.py``; execution wiring is WIP.

Runner tier: engines are pluggable via :mod:`kinapse.runners`.
Add your own — see ``docs/ADDING_A_MODEL.md``.
"""
from __future__ import annotations

from kinapse import runners

TIER = "docking"


def available():
    """Registered docking engines."""
    return runners.specs(TIER)


def run(name, inputs=None, timeout=600.0):
    """Run a registered engine by name with a dict of inputs; returns a dict."""
    return runners.run(TIER, name, inputs, timeout)


__all__ = ["available", "run", "TIER"]
