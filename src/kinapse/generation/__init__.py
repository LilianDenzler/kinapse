"""Generative conformer sampling.

Drive the external DiG diffusion sampler to generate a TCR conformational
ensemble, and post-process its output into an analysable trajectory:

* :mod:`kinapse.generation.dig_runner`  — ``runall(...)`` orchestration (prep -> inference)
* :mod:`kinapse.generation.experiment`  — a top-level experiment driver
* :mod:`kinapse.generation.postprocess` — fold a folder of per-frame PDBs into an
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


def __getattr__(name):  # PEP 562 lazy loading
    import importlib
    if name in _LAZY:
        submod, attr = _LAZY[name]
        module = importlib.import_module(submod, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
