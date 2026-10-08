"""Add a CD8 co-receptor to a TCR-pMHC (or bare pMHC) complex.

The CD8 co-receptor docks on the α3 domain of an MHC **class I** heavy chain (and
brushes β2-microglobulin); it does *not* engage class II — that is CD4's job. No
CD8 is resolved in most TCR-pMHC structures, yet it matters for steered-MD /
docking / interface work, so we graft one in from a reference complex:

1. **Pick the right template** — a solved CD8 : pMHC-class-I structure whose MHC
   best matches the target's MHC (:func:`select_cd8_template`).
2. **Superpose** the template's MHC heavy chain onto the target's, matching Cα
   atoms by *sequence alignment* (robust to differing residue numbering).
3. **Transfer** the template's CD8 chains through that rigid transform, so the CD8
   lands exactly where the template's crystallography says it docks.
4. **Rename & merge** the CD8 chains into the target (collision-free chain ids).
5. *(optional)* PDBFixer cleanup — remove heterogens, replace nonstandard
   residues, add missing atoms/hydrogens. Requires ``pdbfixer`` + ``openmm``.

Public API
----------
* :func:`add_cd8` — the entry point (also surfaced as
  :meth:`kinapse.structures.TCRpMHC.add_cd8` and ``kinapse add-cd8``).
* :func:`select_cd8_template`, :func:`load_templates` — template machinery.
* :class:`CD8Template`, :class:`CD8Result` — data records.

Heavy dependencies (Biopython, PDBFixer) are imported lazily so importing this
module never fails on a machine without them.
"""
from __future__ import annotations

import difflib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# CD8 engages class-I MHC only; a heavy chain best-matching a template below this
# sequence-identity ratio is treated as "not a class-I MHC we can place CD8 on".
_MIN_MHC_IDENTITY = 0.35
# Minimum matched Cα pairs for a trustworthy rigid fit (mirrors the original prep).
_MIN_MATCHED_CA = 20
# Chain ids we will hand out to the transferred CD8 when the defaults collide.
_CHAIN_ID_POOL = "STUVWXYZONPQRabcdefghijklmnopqrstuvwxyz0123456789"


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class CD8Template:
    """A reference CD8 : peptide-MHC-class-I complex used as a graft donor."""

    name: str
    ref_pdb: str                      # absolute path to the reference PDB
    mhc_chain: str                    # MHC heavy-chain id in the reference
    cd8_chains: List[str]             # CD8 chain ids in the reference
    species: Optional[str] = None
    mhc_class: str = "I"
    cd8_type: Optional[str] = None    # "aa" (α/α) or "ab" (α/β)
    b2m_chain: Optional[str] = None
    peptide_chain: Optional[str] = None
    default: bool = False
    citation: Optional[str] = None

    _mhc_seq: Optional[str] = field(default=None, repr=False, compare=False)

    def mhc_sequence(self) -> str:
        """One-letter sequence of the reference MHC heavy chain (cached)."""
        if self._mhc_seq is None:
            from .io import load_pdb
            model = next(load_pdb(self.ref_pdb, self.name).get_models())
            if self.mhc_chain not in model:
                raise ValueError(
                    f"template {self.name!r}: MHC chain {self.mhc_chain!r} not in {self.ref_pdb}"
                )
            self._mhc_seq = _chain_sequence(model[self.mhc_chain])
        return self._mhc_seq


@dataclass
class CD8Result:
    """Outcome of :func:`add_cd8`."""

    out_pdb: Optional[str]
    template: str
    target_mhc_chain: str
    ref_mhc_chain: str
    cd8_chains: List[str]        # new chain ids of the added CD8 in the output
    rmsd: float                  # Cα fit RMSD (Å)
    n_matched: int               # matched Cα atoms used for the fit
    mhc_identity: float          # target-vs-template MHC sequence identity [0,1]
    structure: object = field(default=None, repr=False)  # merged Bio.PDB Structure

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CD8[{self.template}] -> chains {self.cd8_chains} "
            f"(fit RMSD {self.rmsd:.3f} Å over {self.n_matched} Cα, "
            f"MHC id {self.mhc_identity:.0%})"
        )


# ---------------------------------------------------------------------------
# Sequence / chain helpers
# ---------------------------------------------------------------------------

def _chain_sequence(chain) -> str:
    """One-letter sequence of the standard (non-hetero) residues in a chain."""
    from Bio.SeqUtils import seq1
    out = []
    for res in chain:
        if res.id[0] != " ":
            continue
        try:
            out.append(seq1(res.get_resname()) or "X")
        except Exception:
            out.append("X")
    return "".join(out)


def _ca_residues(chain, res_start: Optional[int] = None, res_end: Optional[int] = None):
    """Ordered ``[(one_letter, CA_atom), ...]`` for standard residues with a Cα.

    Restricted to resseq in ``[res_start, res_end]`` when a range is given.
    """
    from Bio.SeqUtils import seq1
    out = []
    for res in chain:
        hetflag, resseq, _icode = res.id
        if hetflag != " " or "CA" not in res:
            continue
        if res_start is not None and (resseq < res_start or resseq > res_end):
            continue
        try:
            aa = seq1(res.get_resname()) or "X"
        except Exception:
            aa = "X"
        out.append((aa, res["CA"]))
    return out


def _identity(seq_a: str, seq_b: str) -> float:
    """Rough sequence identity in [0, 1] via difflib (no alignment deps)."""
    if not seq_a or not seq_b:
        return 0.0
    return difflib.SequenceMatcher(a=seq_a, b=seq_b, autojunk=False).ratio()


def _matched_ca_pairs(target_chain, ref_chain, fit_resrange=None):
    """Pair Cα atoms of two chains by sequence alignment (numbering-agnostic).

    ``fit_resrange`` (``(start, end)`` or ``None``) restricts the *reference*
    residues used — e.g. to fit on the α3 domain only. Returns
    ``(fixed_target_atoms, moving_ref_atoms)`` — equal length, aligned positions.
    """
    rs, re = (fit_resrange or (None, None))
    tgt = _ca_residues(target_chain)
    ref = _ca_residues(ref_chain, rs, re)
    tgt_seq = "".join(a for a, _ in tgt)
    ref_seq = "".join(a for a, _ in ref)

    fixed, moving = [], []
    sm = difflib.SequenceMatcher(a=tgt_seq, b=ref_seq, autojunk=False)
    for i, j, size in sm.get_matching_blocks():
        for k in range(size):
            fixed.append(tgt[i + k][1])
            moving.append(ref[j + k][1])
    return fixed, moving


def _free_chain_ids(existing: set, wanted: Sequence[str], n: int) -> List[str]:
    """Return ``n`` single-char chain ids: honour ``wanted`` where free, else pool."""
    out: List[str] = []
    taken = set(existing)
    for w in wanted:
        if len(str(w)) == 1 and w not in taken:
            out.append(w)
            taken.add(w)
        if len(out) == n:
            return out
    for c in _CHAIN_ID_POOL:
        if len(out) == n:
            break
        if c not in taken:
            out.append(c)
            taken.add(c)
    if len(out) < n:
        raise ValueError("ran out of free single-character chain ids for CD8")
    return out


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

def load_templates(templates_dir: Optional[str] = None) -> List[CD8Template]:
    """Load CD8 templates from a directory's ``templates.yaml`` manifest.

    Defaults to the bundled templates (or ``$KINAPSE_CD8_TEMPLATES_DIR``); see
    :func:`kinapse.config.cd8_templates_dir`.
    """
    import yaml
    from kinapse import config

    root = Path(templates_dir) if templates_dir else config.cd8_templates_dir()
    manifest = root / "templates.yaml"
    if not manifest.exists():
        raise FileNotFoundError(f"no CD8 template manifest at {manifest}")
    data = yaml.safe_load(manifest.read_text()) or {}

    out: List[CD8Template] = []
    for e in data.get("templates", []):
        ref = Path(e["ref_pdb"])
        if not ref.is_absolute():
            ref = root / ref
        out.append(CD8Template(
            name=e["name"],
            ref_pdb=str(ref),
            mhc_chain=e["mhc_chain"],
            cd8_chains=list(e["cd8_chains"]),
            species=e.get("species"),
            mhc_class=str(e.get("mhc_class", "I")),
            cd8_type=e.get("cd8_type"),
            b2m_chain=e.get("b2m_chain"),
            peptide_chain=e.get("peptide_chain"),
            default=bool(e.get("default", False)),
            citation=e.get("citation"),
        ))
    if not out:
        raise ValueError(f"CD8 template manifest {manifest} lists no templates")
    return out


def _target_mhc_candidates(model, min_len: int = 140) -> List[Tuple[str, str]]:
    """``[(chain_id, sequence), ...]`` for chains long enough to be an MHC heavy chain."""
    out = []
    for chain in model:
        seq = _chain_sequence(chain)
        if len(seq) >= min_len:
            out.append((chain.id, seq))
    return out


def select_cd8_template(
    target_pdb: str,
    *,
    species: Optional[str] = None,
    target_mhc_chain: Optional[str] = None,
    templates: Optional[List[CD8Template]] = None,
    templates_dir: Optional[str] = None,
) -> Tuple[CD8Template, str, float]:
    """Pick the CD8 template best matching the target's pMHC.

    Returns ``(template, target_mhc_chain_id, identity)``. The target MHC heavy
    chain is the polymer chain whose sequence is most identical to a class-I
    template's MHC; that same identity picks the template (filtered by ``species``
    when given). Raises if nothing clears :data:`_MIN_MHC_IDENTITY` — the target is
    then not a class-I complex CD8 can dock onto (e.g. class II, or MHC-less).
    """
    from .io import load_pdb

    tmpls = templates if templates is not None else load_templates(templates_dir)
    tmpls = [t for t in tmpls if t.mhc_class == "I"]
    if species:
        want = species.strip().lower()
        by_species = [t for t in tmpls if (t.species or "").lower() == want]
        tmpls = by_species or tmpls  # fall back to any class-I if species unseen
    if not tmpls:
        raise ValueError("no MHC class-I CD8 templates available to choose from")

    model = next(load_pdb(target_pdb, "target").get_models())
    if target_mhc_chain is not None:
        if target_mhc_chain not in model:
            raise ValueError(
                f"--target-mhc-chain {target_mhc_chain!r} not found; "
                f"chains present: {[c.id for c in model]}"
            )
        candidates = [(target_mhc_chain, _chain_sequence(model[target_mhc_chain]))]
    else:
        candidates = _target_mhc_candidates(model)
        if not candidates:
            raise ValueError(
                f"no chain long enough to be an MHC heavy chain in {target_pdb}"
            )

    # Rank by (identity, is_default) so higher identity wins and a template flagged
    # `default: true` breaks near-ties — keeping selection stable.
    best = None  # (sort_key, template, chain_id, identity)
    for t in tmpls:
        ref_seq = t.mhc_sequence()
        for cid, seq in candidates:
            ident = _identity(seq, ref_seq)
            key = (round(ident, 4), bool(t.default))
            if best is None or key > best[0]:
                best = (key, t, cid, ident)

    _key, template, mhc_chain, identity = best
    if identity < _MIN_MHC_IDENTITY and target_mhc_chain is None:
        raise ValueError(
            f"best MHC match is only {identity:.0%} identical to any class-I template "
            f"— this does not look like an MHC class-I complex (CD8 binds class I only; "
            f"class II uses CD4). Pass target_mhc_chain / a species / a custom template "
            f"if this is a genuine class-I target with an unusual sequence."
        )
    return template, mhc_chain, identity


# ---------------------------------------------------------------------------
# Geometric transfer
# ---------------------------------------------------------------------------

def _apply_rt(structure, R, t, chain_ids):
    """Apply rotation ``R`` / translation ``t`` to the given chains, in place."""
    allowed = set(chain_ids)
    for chain in next(structure.get_models()):
        if chain.id not in allowed:
            continue
        for atom in chain.get_atoms():
            atom.set_coord((atom.get_coord() @ R) + t)


def _graft_cd8(target_struct, ref_struct, ref_cd8_chains, new_ids):
    """Copy renamed CD8 chains from ``ref_struct`` into a copy of ``target_struct``."""
    merged = deepcopy(target_struct)
    base_model = next(merged.get_models())
    ref_model = next(ref_struct.get_models())
    rename = dict(zip(ref_cd8_chains, new_ids))
    for old, new in rename.items():
        if old not in ref_model:
            raise ValueError(f"CD8 chain {old!r} not found in reference model")
        chain_copy = deepcopy(ref_model[old])
        chain_copy.id = new
        base_model.add(chain_copy)
    return merged


def _run_pdbfixer(in_pdb: str, out_pdb: str, ph: float,
                  remove_heterogens: bool, keep_waters: bool) -> None:
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PDBFixer cleanup requested but pdbfixer/openmm are not importable. "
            "Install them (e.g. `conda install -c conda-forge pdbfixer openmm`) "
            "or call add_cd8(..., do_fixer=False)."
        ) from e

    fixer = PDBFixer(filename=in_pdb)
    if remove_heterogens:
        fixer.removeHeterogens(keepWater=keep_waters)
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # never add whole missing residues
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=ph)
    with open(out_pdb, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def add_cd8(
    target_pdb: str,
    out_pdb: Optional[str] = None,
    *,
    template: Optional[CD8Template] = None,
    templates_dir: Optional[str] = None,
    ref_pdb: Optional[str] = None,
    ref_mhc_chain: Optional[str] = None,
    ref_cd8_chains: Optional[Sequence[str]] = None,
    species: Optional[str] = None,
    target_mhc_chain: Optional[str] = None,
    cd8_new_chain_ids: Sequence[str] = ("S", "T"),
    fit_resrange: Optional[Tuple[int, int]] = None,
    do_fixer: bool = False,
    ph: float = 7.4,
    remove_heterogens: bool = True,
    keep_waters: bool = False,
) -> CD8Result:
    """Add the correct CD8 co-receptor to a TCR-pMHC (or pMHC) complex.

    Args:
        target_pdb: path to the complex to add CD8 to.
        out_pdb: where to write the merged PDB. If ``None`` nothing is written and
            the merged structure is returned in :attr:`CD8Result.structure`.
        template: an explicit :class:`CD8Template`. If ``None`` the best-matching
            one is chosen by :func:`select_cd8_template`.
        ref_pdb / ref_mhc_chain / ref_cd8_chains: supply your own reference instead
            of the registry (all three required together; bypasses auto-selection).
        species: hint to restrict template auto-selection (e.g. ``"human"``).
        target_mhc_chain: MHC heavy-chain id in the target; auto-detected by
            sequence similarity when omitted.
        cd8_new_chain_ids: preferred chain ids for the grafted CD8 (auto-bumped to
            free ids on collision).
        fit_resrange: restrict the superposition to this reference resseq range
            (e.g. the α3 domain); default fits the whole MHC heavy chain.
        do_fixer: run PDBFixer cleanup, writing ``<out>`` from the fixed topology
            (needs pdbfixer/openmm). The raw merge is written to
            ``<out stem>.with_cd8.pdb`` alongside.
        ph, remove_heterogens, keep_waters: PDBFixer options.

    Returns:
        A :class:`CD8Result` (template used, chain ids, fit RMSD, matched Cα, …).
    """
    from Bio.PDB import Superimposer
    from .io import load_pdb, write_pdb

    # 1. Resolve the reference (explicit ref_pdb, explicit template, or auto-pick).
    if ref_pdb is not None:
        if not (ref_mhc_chain and ref_cd8_chains):
            raise ValueError("ref_pdb requires ref_mhc_chain and ref_cd8_chains too")
        template = CD8Template(
            name=Path(ref_pdb).stem, ref_pdb=ref_pdb,
            mhc_chain=ref_mhc_chain, cd8_chains=list(ref_cd8_chains),
        )
        if target_mhc_chain is None:
            tmodel = next(load_pdb(target_pdb, "target").get_models())
            cands = _target_mhc_candidates(tmodel) or [(c.id, _chain_sequence(c)) for c in tmodel]
            ref_seq = template.mhc_sequence()
            target_mhc_chain = max(cands, key=lambda cs: _identity(cs[1], ref_seq))[0]
            mhc_identity = _identity(
                _chain_sequence(tmodel[target_mhc_chain]), ref_seq)
        else:
            mhc_identity = float("nan")
    elif template is not None:
        if target_mhc_chain is None:
            template, target_mhc_chain, mhc_identity = select_cd8_template(
                target_pdb, species=species, templates=[template])
        else:
            mhc_identity = float("nan")
    else:
        template, target_mhc_chain, mhc_identity = select_cd8_template(
            target_pdb, species=species, target_mhc_chain=target_mhc_chain,
            templates_dir=templates_dir)

    # 2. Superpose reference MHC onto target MHC (alignment-matched Cα).
    target = load_pdb(target_pdb, "target")
    ref = load_pdb(template.ref_pdb, template.name)
    tgt_model = next(target.get_models())
    ref_model = next(ref.get_models())
    if target_mhc_chain not in tgt_model:
        raise ValueError(f"target MHC chain {target_mhc_chain!r} not in {target_pdb}")
    if template.mhc_chain not in ref_model:
        raise ValueError(
            f"template MHC chain {template.mhc_chain!r} not in {template.ref_pdb}")

    fixed, moving = _matched_ca_pairs(
        tgt_model[target_mhc_chain], ref_model[template.mhc_chain], fit_resrange)
    if len(fixed) < _MIN_MATCHED_CA:
        raise RuntimeError(
            f"too few matched Cα for a reliable fit (matched={len(fixed)}). "
            f"Check that target chain {target_mhc_chain!r} is really the MHC heavy "
            f"chain (fit_resrange={fit_resrange})."
        )
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    R, t = sup.rotran
    rmsd = float(sup.rms)

    # 3. Move the reference CD8 into the target frame, 4. graft it in.
    _apply_rt(ref, R, t, template.cd8_chains)
    new_ids = _free_chain_ids(
        {c.id for c in tgt_model}, cd8_new_chain_ids, len(template.cd8_chains))
    merged = _graft_cd8(target, ref, template.cd8_chains, new_ids)

    # 5. Write (raw and/or PDBFixer-cleaned).
    written = None
    if out_pdb is not None:
        out_path = Path(out_pdb)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if do_fixer:
            raw = out_path.with_suffix("")
            raw = raw.with_name(raw.name + ".with_cd8.pdb")
            write_pdb(str(raw), merged)
            _run_pdbfixer(str(raw), str(out_path), ph, remove_heterogens, keep_waters)
        else:
            write_pdb(str(out_path), merged)
        written = str(out_path)

    return CD8Result(
        out_pdb=written,
        template=template.name,
        target_mhc_chain=target_mhc_chain,
        ref_mhc_chain=template.mhc_chain,
        cd8_chains=new_ids,
        rmsd=rmsd,
        n_matched=len(fixed),
        mhc_identity=mhc_identity,
        structure=merged,
    )
