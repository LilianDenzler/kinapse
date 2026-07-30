"""Sequence embedders.

Turn a TCR (or its chains) into sequence-based representations:

* :mod:`kinapse.sequence_embedding.fasta`     — extract per-chain FASTA from a PDB
* :mod:`kinapse.sequence_embedding.msa`       — build MSAs with MMseqs2-GPU
* :mod:`kinapse.sequence_embedding.evoformer` — run OpenFold / Evoformer to get embeddings

The MSA and Evoformer paths need heavy/external tooling (MMseqs2 binary, torch,
an OpenFold checkout) — installed via the ``embed`` extra. Symbols are lazily
imported so ``import kinapse.sequence_embedding`` works even without them.
"""
from __future__ import annotations

_LAZY = {
    "pdb_to_fasta": (".fasta", "pdb_to_fasta"),
}

__all__ = list(_LAZY)

# --- runner-tier surface: pluggable external embedders via kinapse.runners ---
from kinapse import runners  # noqa: E402

TIER = "sequence_embedding"
__all__ += ["available", "run", "TIER"]


def available():
    """Registered sequence-embedding models (MSA / Evoformer / ESM …)."""
    return runners.specs(TIER)


def run(name, inputs=None, timeout=600.0):
    """Run a registered embedder by name with a dict of inputs; returns a dict."""
    return runners.run(TIER, name, inputs, timeout)


def __getattr__(name):  # PEP 562 lazy loading
    import importlib
    if name in _LAZY:
        submod, attr = _LAZY[name]
        module = importlib.import_module(submod, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
