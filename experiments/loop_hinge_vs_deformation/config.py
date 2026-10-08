"""Config for the loop hinge-vs-deformation experiment.

One place for data paths, the system list, per-system loader overrides, and the
(transferable, identical-for-all-systems) numeric choices. See plan.md.
"""
from __future__ import annotations
import os

# --- data ------------------------------------------------------------------
DATA = "/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD"          # <ID>/<ID>.{pdb,xtc}

# 22 unbound TCRs (4-char PDB IDs). Discovered from DATA if left empty.
SYSTEMS: list[str] = []           # empty => auto-discover 4-char dirs with pdb+xtc

# ANARCII mistypes some alpha chains as delta -> swap so pairing works.
CHAIN_OVERRIDES: dict[str, dict[str, str]] = {
    "8YJ3": {"A": "B", "B": "A"},
}

# --- analysis choices (transferable: same for every system + every CDR) ----
STRIDE = 25                       # ~57k -> ~2.3k frames/system
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
# per-chain framework used as the rigid-core CANDIDATE set (FR4/J-region excluded)
FR_BY_CHAIN = {
    "A": ["A_FR1", "A_FR2", "A_FR3"],
    "B": ["B_FR1", "B_FR2", "B_FR3"],
}
CHAIN_OF = {c: c.split("_")[0] for c in CDRS}   # A_CDR1 -> A

# rigid-core detection: keep the maximal subset of framework CA whose every
# pairwise CA-CA distance has std (over frames) <= CORE_STD_A (Angstrom).
# A pair whose separation barely fluctuates is rigidly connected -> physical,
# absolute (not cv), applied identically to all systems.
CORE_STD_A = 0.8
CORE_MIN_ATOMS = 6                # need >=4 non-coplanar; ask for a comfortable margin

SEQ_SEP = 2                       # |i-j|>=SEQ_SEP variant of D_deform

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
