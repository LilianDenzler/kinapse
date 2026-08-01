"""benchmarks — leakage-aware harness + ensemble-quality benchmark (research tier).

purpose: evaluate models/scorers honestly and rank conformational ensembles.
extra:   ``[bench]``
status:  scorer benchmark works; ensemble-quality benchmark is scaffolded.

Working:
  * ``run_scorer_benchmark`` — GT vs modelled vs negatives, leakage-aware split,
    per-scorer agreement + discrimination (:mod:`kinapse.benchmarks.scorer_benchmark`).
  * ``structural_agreement`` / ``assign_tier`` — model-vs-GT Cα iRMSD over the 6 CDRs
    → HQ/MQ/AQ/LQ tiers (:mod:`kinapse.benchmarks.structural`).

Planned:
  * an ensemble-quality benchmark ("DockQ-for-ensembles"): JSD + Wasserstein/RMWD
    + RMSF-correlation + bound-state recovery — to validate DiG/AlphaFlow on TCRs.
"""
from __future__ import annotations


def __getattr__(name):  # lazy: pulls pandas/sklearn/scipy only when used
    import importlib
    if name in ("run_scorer_benchmark", "agreement_analysis", "discrimination_analysis",
                "split_by_group", "tiers_summary"):
        return getattr(importlib.import_module("kinapse.benchmarks.scorer_benchmark"), name)
    if name in ("structural_agreement", "assign_tier"):
        return getattr(importlib.import_module("kinapse.benchmarks.structural"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_benchmark(name, **kwargs):
    """Run a named benchmark. (Scaffold — not implemented yet.)"""
    raise NotImplementedError("benchmark harness is scaffolded but not implemented yet")


def ensemble_quality(gt, pred, **kwargs):
    """Score a predicted conformational ensemble vs a reference. (Scaffold.)"""
    raise NotImplementedError("ensemble-quality benchmark is scaffolded but not implemented yet")


__all__ = ["run_benchmark", "ensemble_quality", "run_scorer_benchmark",
           "agreement_analysis", "discrimination_analysis", "split_by_group",
           "tiers_summary", "structural_agreement", "assign_tier"]
