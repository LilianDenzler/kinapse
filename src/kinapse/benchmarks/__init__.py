"""benchmarks — leakage-aware harness + ensemble-quality benchmark (research tier).

purpose: evaluate models/scorers honestly and rank conformational ensembles.
extra:   ``[bench]``
status:  scaffold — interfaces declared; not implemented yet.

Planned:
  * model/scorer benchmark harness driven by :mod:`kinapse.datasets` splits.
  * an ensemble-quality benchmark ("DockQ-for-ensembles"): JSD + Wasserstein/RMWD
    + RMSF-correlation + bound-state recovery — to validate DiG/AlphaFlow on TCRs.
"""
from __future__ import annotations


def __getattr__(name):  # lazy: scorer_benchmark pulls pandas/sklearn/scipy
    if name in ("run_scorer_benchmark", "agreement_analysis", "discrimination_analysis",
                "split_by_group"):
        from . import scorer_benchmark
        return getattr(scorer_benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_benchmark(name, **kwargs):
    """Run a named benchmark. (Scaffold — not implemented yet.)"""
    raise NotImplementedError("benchmark harness is scaffolded but not implemented yet")


def ensemble_quality(gt, pred, **kwargs):
    """Score a predicted conformational ensemble vs a reference. (Scaffold.)"""
    raise NotImplementedError("ensemble-quality benchmark is scaffolded but not implemented yet")


__all__ = ["run_benchmark", "ensemble_quality", "run_scorer_benchmark",
           "agreement_analysis", "discrimination_analysis", "split_by_group"]
