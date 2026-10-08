#!/usr/bin/env python
"""Structural figure of the CDR rigid hinge: the loop's reference shape swung across the MD rotation range
about its clamp pivot, with the pivot point and hinge axis drawn. Pure rigid swing (no deformation clutter)."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch


def pdb_ca(path, confs, imgt):
    """confs: list of (T,N,3)-> here list of (N,3) CA sets, one chain each."""
    ch = "ABCDEFGH"
    with open(path, "w") as f:
        for ci, X in enumerate(confs):
            for k, (x, y, z) in enumerate(X):
                f.write(f"ATOM  {k+1:5d}  CA  GLY {ch[ci]}{imgt[k]:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("TER\n")
        f.write("END\n")


def main(sysid="3QH3", cdr="A_CDR3", nconf=5):
    ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))["chain_" + ch]["ultra_rigid"]
    lk = sorted(k for k in imap[ch] if lo <= k <= hi)
    loop = xyz[:, np.array([imap[ch][k] for k in lk])]
    R = xyz[:, np.array([imap[ch][r] for r in rig if r in imap[ch]])]
    # superpose every frame's framework onto frame 0 (static-framework frame, matches the Fv PDB frame 0)
    Rref = R[0] - R[0].mean(0); Rc = R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(len(R)):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + Rc
    L0 = Lf[0]; L0m = L0.mean(0); L0c = L0 - L0m
    omega = np.empty((len(Lf), 3)); A = np.zeros((3, 3)); b = np.zeros(3); Qs = []
    for t in range(len(Lf)):
        Q = kabsch(L0c, Lf[t] - Lf[t].mean(0)); Qs.append(Q)
        omega[t] = Rot.from_matrix(Q).as_rotvec()
        c = Lf[t].mean(0) - Q @ L0m; M = np.eye(3) - Q; A += M.T @ M; b += M.T @ c
    p = np.linalg.lstsq(A, b, rcond=None)[0]                           # pivot (screw-axis location)
    u = np.linalg.eigh(omega.T @ omega)[1][:, -1]; u /= np.linalg.norm(u)   # dominant rotation axis
    phi = omega @ u                                                    # signed swing about the main axis
    sel = [int(np.argmin(np.abs(phi - q))) for q in np.percentile(phi, np.linspace(2, 98, nconf))]
    confs = [(Qs[t] @ (L0 - p).T).T + p for t in sel]                  # reference loop swung about the pivot
    theta_range = np.degrees(phi.max() - phi.min())
    loop_pdb = f"{HERE}/figures/_hinge_{sysid}_{cdr}.pdb"
    pdb_ca(loop_pdb, confs, lk)

    from pymol_view import write_fv_pdb
    fv = write_fv_pdb(sysid)
    import pymol2
    out = f"{HERE}/figures/hinge_{sysid}_{cdr}.png"
    with pymol2.PyMOL() as P:
        cmd = P.cmd
        cmd.load(fv, "fv"); cmd.hide("everything"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
        cmd.show("cartoon", f"fv and chain {ch}"); cmd.color("grey80", "fv")
        cmd.set("cartoon_transparency", 0.35, f"fv and chain {ch}")
        cmd.hide("cartoon", f"fv and chain {ch} and resi {lo}-{hi}")   # remove the static loop; fan replaces it
        # clamp residues (rigid stem near the loop) as sticks
        clampres = "104+105+106" if cdr.endswith("CDR3") else "41+40+39" if cdr.endswith("CDR1") else "55+56+57"
        cmd.show("sticks", f"fv and chain {ch} and resi {clampres}")
        cmd.color("orange", f"fv and chain {ch} and resi {clampres}")
        cmd.label(f"fv and chain {ch} and resi 104 and name CA", '"Cys104 clamp"') if cdr.endswith("CDR3") else None
        # the swinging loop fan
        cmd.load(loop_pdb, "fan")
        cmd.set("connect_mode", 1)
        cols = ["marine", "cyan", "white", "salmon", "firebrick"][:nconf]
        for ci, c in enumerate(cols):
            s = f"fan and chain {'ABCDEFGH'[ci]}"
            cmd.color(c, s); cmd.show("spheres", s); cmd.set("sphere_scale", 0.45, s)
            for k in range(len(confs[ci]) - 1):                        # bond consecutive CAs -> a trace
                cmd.bond(f"{s} and rank {k}", f"{s} and rank {k+1}")
            cmd.show("sticks", s); cmd.set("stick_radius", 0.35, s)
        # pivot + hinge axis
        cmd.pseudoatom("piv", pos=[float(x) for x in p]); cmd.show("spheres", "piv")
        cmd.color("green", "piv"); cmd.set("sphere_scale", 1.4, "piv")
        from pymol.cgo import CYLINDER
        a1 = (p - 13 * u).tolist(); a2 = (p + 13 * u).tolist()
        cmd.load_cgo([CYLINDER, *a1, *a2, 0.25, .1, .5, .1, .1, .5, .1], "hingeaxis")
        cmd.set("label_size", 16); cmd.set("float_labels", 1)
        cmd.orient(f"(fan) or (fv and chain {ch} and resi {lo-8}-{hi+8})")
        cmd.turn("y", 20); cmd.ray(1700, 1300); cmd.png(out, dpi=150)
    print(f"theta swing range = {theta_range:.1f} deg over {nconf} conformations")
    print("fig ->", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) > 0 else "3QH3", a[1] if len(a) > 1 else "A_CDR3")
