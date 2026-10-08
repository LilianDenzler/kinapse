"""Config for the elasticity experiment (three formalisms). See plan.md.

Fresh, separate approach — deliberately does NOT import the parent config so the
numeric choices here are independent. Transferable: identical for all systems/CDRs.
"""
from __future__ import annotations
import os

# --- data ------------------------------------------------------------------
DATA = "/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD"          # <ID>/<ID>.{pdb,xtc}
SYSTEMS: list[str] = []            # empty => auto-discover 4-char dirs with pdb+xtc
CHAIN_OVERRIDES = {"8YJ3": {"A": "B", "B": "A"}}             # ANARCII mistypes 8YJ3 alpha

# --- selection -------------------------------------------------------------
STRIDE = 25
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]

# --- elasticity choices (transferable: same for every system + every CDR) --
KT = 0.001987 * 300.0              # kcal/mol at 300 K

# Method 1 (finite-strain field): local neighborhood on the reference config.
STRAIN_RC = 9.0                    # Angstrom spatial cutoff for the strain neighborhood
STRAIN_KMIN = 4                    # >=4 non-coplanar neighbors needed to fit F (expand Rc if short)

# Method 2 (elastic network): contact cutoffs.
GNM_RC = 8.0                       # Angstrom (GNM Kirchhoff)
ANM_RC = 13.0                      # Angstrom (ANM Hessian)

# Method 3 (data-driven stiffness): covariance regularization.
RIDGE = 1e-3                       # Angstrom^2 ridge added to Sigma before inversion

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
