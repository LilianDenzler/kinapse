#!/usr/bin/env python
"""PyMOL cartoon of the TCR variable domains (Fv) with the graph regions coloured + labelled.

Colour key (IMGT regions, both chains):
    CDR1 = orange   CDR2 = red   CDR3 = magenta      (the loop nodes)
    green                                            = flanking residues (beta 26, beta 66)
    slate blue                                       = super-rigid CONSENSUS framework (frame nodes)
    grey                                             = other/mobile framework (not in the rigid set)

Writes the kinapse IMGT-numbered Fv to a PDB, then ray-traces a labelled figure headlessly (pymol2).
CLI:  python pymol_view.py 3QH3
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np
from graph_build import load_md, consensus, CDR_RANGES, flanking_residues, HERE

CDR_COLOR = {"CDR1": "orange", "CDR2": "red", "CDR3": "magenta"}


def write_fv_pdb(sysid, md_cache=None):
    out0 = f"{HERE}/figures/{sysid}_Fv.pdb"
    if md_cache is None and os.path.exists(out0):
        return out0                                                   # reuse (skip the MD reload)
    tv, xyz, imap = md_cache if md_cache is not None else load_md(sysid)
    vidx = np.asarray(tv.domain_idx(["A_variable", "B_variable"]))     # all atoms of both variable domains
    fv = tv.mdtraj[0].atom_slice(vidx)                                 # first frame, Fv only (IMGT-numbered)
    out = f"{HERE}/figures/{sysid}_Fv.pdb"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fv.save_pdb(out)
    return out


def render(sysid, md_cache=None):
    pdb = write_fv_pdb(sysid, md_cache)
    import pymol2
    con = consensus()
    out = f"{HERE}/figures/pymol_{sysid}_regions.png"
    with pymol2.PyMOL() as P:
        cmd = P.cmd
        cmd.load(pdb, "tcr")
        cmd.hide("everything"); cmd.show("cartoon")
        cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
        cmd.set("cartoon_fancy_helices", 1); cmd.set("cartoon_transparency", 0.0)
        cmd.color("grey70", "tcr")                                     # default = other/mobile framework
        for ch in "AB":                                               # super-rigid consensus -> slate
            sel = "+".join(str(n) for n in sorted(con[ch]))
            cmd.color("slate", f"chain {ch} and resi {sel}")
        for ch in "AB":                                               # CDR loops -> distinct colours
            for cdr, (lo, hi) in CDR_RANGES.items():
                cmd.color(CDR_COLOR[cdr], f"chain {ch} and resi {lo}-{hi}")
        for ch in "AB":                                               # flanking -> green
            fl = [n for cdr in CDR_RANGES for n in flanking_residues(ch, cdr)]
            if fl:
                cmd.color("green", f"chain {ch} and resi {'+'.join(map(str, fl))}")
        # label each CDR at its central residue (float_labels keeps them on top, not occluded)
        cmd.set("label_size", 18); cmd.set("label_color", "black")
        cmd.set("float_labels", 1); cmd.set("label_outline_color", "white")
        for ch in "AB":
            for cdr, (lo, hi) in CDR_RANGES.items():
                mid = (lo + hi) // 2
                cmd.label(f"chain {ch} and resi {mid} and name CA", f'"{ch}-{cdr}"')
        cmd.orient("tcr"); cmd.turn("y", 10)
        cmd.set("ray_trace_mode", 1)
        cmd.ray(1800, 1400)
        cmd.png(out, dpi=150)
    return out


if __name__ == "__main__":
    sysid = sys.argv[1] if len(sys.argv) > 1 else "3QH3"
    print("pymol figure ->", render(sysid))
