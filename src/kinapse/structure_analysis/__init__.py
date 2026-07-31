"""structure_analysis — external TCR structure annotators (runner tier).

purpose: run external tools that *annotate/analyze* a TCR or TCR-pMHC structure —
         chains/CDRs, docking geometry, interface interactions, quality metrics.
extra:   ``[structure_analysis]``
status:  stable API — STCRpy wired as an external runner.

This is a *runner* tier: tools are pluggable via :mod:`kinapse.runners` and run in
their own (external) environment, fully separate from kinapse. Add your own — see
``docs/ADDING_A_MODEL.md``.
"""
from __future__ import annotations

from kinapse import runners

TIER = "structure_analysis"


def available():
    """Registered structure-analysis tools (:class:`kinapse.runners.RunnerSpec`)."""
    return runners.specs(TIER)


def run(name, inputs=None, timeout=600.0):
    """Run a registered tool by name with a dict of inputs; returns a dict."""
    return runners.run(TIER, name, inputs, timeout)


__all__ = ["available", "run", "TIER"]
