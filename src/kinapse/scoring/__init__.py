"""kinapse.scoring — protein-interface scoring for TCR / TCR-pMHC complexes.

kinapse does **not** vendor scorers. It delegates to the external `ifscore`
package, whose core depends on no scorer at all: each scorer (DockQ, PRODIGY,
`geometry`, ...) runs in its own isolated environment behind a JSON/subprocess
contract, so mutually-incompatible tools (DockQ needs numpy<2, PRODIGY needs
numpy>=2, PyRosetta, torch-based models) can coexist without ever touching
kinapse's environment.

kinapse's value-add is **chain auto-derivation**: because the `structures`
loader already knows which chains are the TCR α/β domains, it can fill in
ifscore's ``receptor``/``ligand`` chain arguments automatically for a TCR-pMHC
complex (receptor = TCR, ligand = pMHC = everything else).

``ifscore`` is imported lazily, so importing this module never requires it.

Install (ifscore is a separate, optional package)::

    pip install "ifscore @ git+https://github.com/LilianDenzler/scoring_functions"
    # or from a local checkout:
    pip install -e /path/to/scoring_functions

Then provision the scorers you want (one-off; uses `uv`)::

    ifscore install all   # provision every scorer (the `geometry` scorer needs nothing)

Licensed scorers you must supply yourself
-----------------------------------------
Some scorers are academic-licensed and are **not** shipped with ifscore or
kinapse — you obtain the licence and point ifscore at the binary (no path is
hardcoded):

* **FoldX** — get a free academic licence and download the binary from
  https://foldxsuite.crg.eu (licences are **time-limited / expire yearly**), then::

      export IFSCORE_FOLDX=/your/path/to/foldx
      # or record it once:
      ifscore install foldx --path /your/path/to/foldx

* **PyRosetta** (the ``rosetta`` scorer) — free for academics, auto-downloaded by
  ``ifscore install rosetta``; commercial use needs a licence (license@uw.edu).

``ifscore doctor`` reports whether a licensed binary is missing, ready, or
licence-expired, so a dead FoldX licence never silently yields wrong numbers.

Note: ifscore's ``geometry`` scorer (interface BSA / SASA / contacts) is a
different thing from :mod:`kinapse.geometry` (α/β domain docking angles).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_INSTALL_HINT = (
    "The 'ifscore' package is required for kinapse.scoring but is not installed.\n"
    '  pip install "ifscore @ git+https://github.com/LilianDenzler/scoring_functions"\n'
    "  (or a local checkout:  pip install -e /path/to/scoring_functions)\n"
    "Then provision scorers once:  ifscore install all"
)

_Chains = Optional[Union[str, Sequence[str]]]

__all__ = [
    "score",
    "score_batch",
    "score_tcr_pmhc",
    "infer_tcr_pmhc_chains",
    "available_scorers",
]


def _ifscore():
    """Import ifscore lazily, with an actionable error if it is missing."""
    try:
        import ifscore  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - exercised only without ifscore
        raise ImportError(_INSTALL_HINT) from exc
    return ifscore


def _as_list(chains: _Chains) -> Optional[List[str]]:
    if chains is None:
        return None
    if isinstance(chains, str):
        return [c.strip() for c in chains.split(",") if c.strip()]
    return [str(c) for c in chains]


# kinapse surfaces ifscore's interface-geometry scorer as `geometry_scoring` so it
# never clashes with the `kinapse.geometry` module (α/β domain angles).
_SCORER_ALIASES = {"geometry_scoring": "geometry"}   # kinapse name -> ifscore name
_SCORER_ALIASES_REV = {v: k for k, v in _SCORER_ALIASES.items()}


def _resolve_scorer_names(scorers):
    """Translate kinapse scorer aliases (e.g. 'geometry_scoring') to ifscore names."""
    def tr(n):
        return _SCORER_ALIASES.get(n.strip(), n.strip())
    if scorers is None:
        return scorers
    if isinstance(scorers, str):
        return ",".join(tr(s) for s in scorers.split(",") if s.strip())
    return [tr(str(s)) for s in scorers]


def score(
    model,
    native=None,
    rec: _Chains = None,
    lig: _Chains = None,
    scorers: Union[str, Sequence[str]] = "default",
    timeout: float = 600.0,
    chain_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Score one complex with ifscore. Returns ``{'<scorer>__<metric>': value}``.

    Args:
        model: path to the model complex PDB.
        native: optional native reference PDB (needed by DockQ and other metrics).
        rec, lig: receptor / ligand chain ids (list, or comma-separated string).
        scorers: a scorer set — ``"fast"`` (geometry, dockq), ``"default"``
            (+ prodigy), ``"all"``, or explicit names like ``"geometry,dockq"``.
        chain_map: maps model chain ids to native chain ids (for DockQ when the
            two structures use different chain letters).
    """
    ifs = _ifscore()
    return ifs.score(
        model=str(model),
        native=str(native) if native else None,
        rec=_as_list(rec),
        lig=_as_list(lig),
        scorers=_resolve_scorer_names(scorers),
        timeout=timeout,
        chain_map=chain_map,
    )


def score_batch(
    models=None,
    native=None,
    manifest=None,
    rec: _Chains = None,
    lig: _Chains = None,
    scorers: Union[str, Sequence[str]] = "default",
    n_jobs: int = 4,
    timeout: float = 600.0,
    strip: Optional[str] = None,
    limit: Optional[int] = None,
    chain_map: Optional[Dict[str, str]] = None,
):
    """Score a batch (a directory of models, or a manifest CSV).

    Returns a pandas DataFrame, one row per structure, columns namespaced
    ``<scorer>__<metric>`` (plus ``<scorer>__status`` / ``__runtime_s``).
    """
    ifs = _ifscore()
    return ifs.score_batch(
        models=str(models) if models else None,
        native=str(native) if native else None,
        manifest=str(manifest) if manifest else None,
        rec=_as_list(rec),
        lig=_as_list(lig),
        scorers=_resolve_scorer_names(scorers),
        n_jobs=n_jobs,
        timeout=timeout,
        strip=strip,
        limit=limit,
        chain_map=chain_map,
    )


def infer_tcr_pmhc_chains(pdb, tcr=None, legacy_anarci: bool = True) -> Tuple[List[str], List[str]]:
    """Split a TCR-pMHC complex into ``(receptor_chains, ligand_chains)``.

    * receptor = the TCR α/β chains, found via kinapse's IMGT numbering + pairing
      (:class:`kinapse.structures.TCR`).
    * ligand   = every *other* polymer chain (peptide + MHC + β2m + CD8 ...).

    Returns original PDB chain ids, ready to pass as ifscore ``rec`` / ``lig``.

    Args:
        pdb: path to the complex PDB (ignored if ``tcr`` is given).
        tcr: an already-loaded :class:`kinapse.structures.TCR` (avoids re-loading).
        legacy_anarci: use legacy ANARCI (bioconda) for numbering; set ``False``
            to use the pip-installable ANARCII (works without conda).
    """
    from Bio.PDB import is_aa

    if tcr is None:
        from kinapse.structures import TCR
        tcr = TCR(input_pdb=str(pdb), legacy_anarci=legacy_anarci)

    rec: List[str] = []
    for pair in tcr.pairs:
        for cid in (getattr(pair, "alpha_chain_id", None), getattr(pair, "beta_chain_id", None)):
            if cid and cid not in rec:
                rec.append(cid)

    lig: List[str] = []
    for chain in tcr.original_structure:  # original (un-renamed) chain ids
        if chain.id in rec or chain.id in lig:
            continue
        if any(is_aa(res, standard=False) for res in chain):  # polymer chains only
            lig.append(chain.id)

    return rec, lig


def score_tcr_pmhc(
    model,
    native=None,
    scorers: Union[str, Sequence[str]] = "default",
    tcr=None,
    rec: _Chains = None,
    lig: _Chains = None,
    chain_map: Optional[Dict[str, str]] = None,
    timeout: float = 600.0,
    legacy_anarci: bool = True,
) -> Dict[str, Any]:
    """Score a TCR-pMHC complex, auto-deriving ``rec``/``lig`` when not given.

    With no explicit chains, receptor = TCR α/β and ligand = pMHC (everything
    else), inferred by :func:`infer_tcr_pmhc_chains`. Set ``legacy_anarci=False``
    to number with the pip-installable ANARCII instead of bioconda ANARCI.
    """
    if rec is None and lig is None:
        rec, lig = infer_tcr_pmhc_chains(model, tcr=tcr, legacy_anarci=legacy_anarci)
        if not lig:
            raise ValueError(
                f"No non-TCR (ligand) chains found in {model} — is this a TCR-pMHC "
                "complex? Pass rec/lig explicitly to score a different interface."
            )
    return score(
        model,
        native=native,
        rec=rec,
        lig=lig,
        scorers=_resolve_scorer_names(scorers),
        timeout=timeout,
        chain_map=chain_map,
    )


def available_scorers() -> Dict[str, Dict[str, Any]]:
    """Return ``{name: {metrics, backend, needs_native, description}}`` for every
    scorer ifscore knows about (see also the ``ifscore doctor`` CLI for what is
    actually provisioned and ready to run)."""
    ifs = _ifscore()
    return {
        _SCORER_ALIASES_REV.get(spec.name, spec.name): {
            "metrics": list(spec.metrics),
            "backend": spec.backend,
            "needs_native": spec.needs_native,
            "description": spec.description,
        }
        for spec in ifs.resolve("all")
    }
