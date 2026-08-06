"""Generative conformer sampling.

Drive the external DiG diffusion sampler to generate a TCR conformational
ensemble, and post-process its output into an analysable trajectory:

* :mod:`kinapse.conformer_generation.dig_runner`  — ``run_one(pdb, out, …)`` for a single
  TCR, ``runall(folder, …)`` for a batch: prep (α/β linking) → **Evoformer embedding
  (computed in-pipeline if not supplied)** → DiG inference. Five sampling modes
  (``vanilla | no_mask | binary_mask | cdr_mask | weighted_mask``, or ``all``) share one prep +
  embedding. CLI: ``python -m kinapse.conformer_generation.dig_runner --pdb tcr.pdb --out out/ --mode all``.
* :mod:`kinapse.conformer_generation.postprocess` — fold a folder of per-frame PDBs into an
  ``.xtc`` trajectory (stripping the alpha/beta linker)

Runs **your** model: the checkpoint comes from ``KINAPSE_CHECKPOINT_MAIN_MODEL`` (or
``generation.main_model`` in ``kinapse.yaml``), the same for every mode. The DiG inference +
OpenFold embedding wrapper are external scripts configured via :mod:`kinapse.config`
(``generation.run_inference`` / ``run_inference_addnoise`` / ``get_init_state`` /
``openfold_wrapper``) and need a GPU (+ AlphaFold DBs for the embedding). Symbols are lazily imported.
"""
from __future__ import annotations

_LAZY = {
    "run_one": (".dig_runner", "run_one"),           # single TCR (prep → embedding → inference)
    "runall": (".dig_runner", "runall"),             # batch over a folder
    "compute_embedding": (".dig_runner", "compute_embedding"),
    # sampling modes (see dig_runner): the 5 canonical names + helpers
    "ALL_MODES": (".dig_runner", "ALL_MODES"),
    "MODE_DIRS": (".dig_runner", "MODE_DIRS"),
    "DEFAULT_REGION_WEIGHTS": (".dig_runner", "DEFAULT_REGION_WEIGHTS"),
    "canonical_mode": (".dig_runner", "canonical_mode"),
    "resolve_modes": (".dig_runner", "resolve_modes"),
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
