"""Generative conformer sampling.

Drive the external DiG diffusion sampler to generate a TCR conformational
ensemble, and post-process its output into an analysable trajectory:

* :mod:`kinapse.conformer_generation.dig_runner`  — ``runall(...)`` orchestration (prep -> inference)
* :mod:`kinapse.conformer_generation.experiment`  — a top-level experiment driver
* :mod:`kinapse.conformer_generation.postprocess` — fold a folder of per-frame PDBs into an
  ``.xtc`` trajectory (stripping the alpha/beta linker)

The DiG side shells out to external inference scripts/checkpoints (configure
their paths via :mod:`kinapse.config`). Symbols are lazily imported.
"""
from __future__ import annotations

_LAZY = {
    "runall": (".dig_runner", "runall"),
    "process_output": (".postprocess", "process_output"),
    "unlink_pdbs": (".postprocess", "unlink_pdbs"),
    "split": (".postprocess", "split"),
}

__all__ = list(_LAZY)

# --- runner-tier surface: pluggable external generators via kinapse.runners ---
from kinapse import runners  # noqa: E402

TIER = "conformer_generation"
__all__ += ["available", "run", "TIER"]


def available():
    """Registered conformer generators (DiG / AlphaFlow / BioEmu …)."""
    return runners.specs(TIER)


def run(name, inputs=None, timeout=600.0):
    """Run a registered generator by name with a dict of inputs; returns a dict."""
    return runners.run(TIER, name, inputs, timeout)


def __getattr__(name):  # PEP 562 lazy loading
    import importlib
    if name in _LAZY:
        submod, attr = _LAZY[name]
        module = importlib.import_module(submod, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
