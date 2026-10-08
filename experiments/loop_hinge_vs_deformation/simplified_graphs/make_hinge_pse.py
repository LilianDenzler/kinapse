#!/usr/bin/env python
"""Build a PyMOL .pse session per CDR (6 total): the loop's rigid hinge swing about its clamp,
with framework, clamp residues, pivot point, and hinge axis clearly coloured. Also a .png each."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, traceback
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
CLAMP = {"CDR1": ([38, 39, 40, 41], "Trp41 clamp (C)"),
         "CDR2": ([54, 55, 56], "FR2 clamp (N)"),
         "CDR3": ([104, 105, 106], "Cys104 clamp (N)")}
NCONF = 7
CH8 = "ABCDEFGH"


def pdb_ca(path, confs, imgt):
    with open(path, "w") as f:
        for ci, X in enumerate(confs):
            for k, (x, y, z) in enumerate(X):                          # sequential resi (loop may have insertions)
                f.write(f"ATOM  {k+1:5d}  CA  GLY {CH8[ci]}{k+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("TER\n")
        f.write("END\n")


def bwr(f):
    return [2 * f, 2 * f, 1.0] if f < 0.5 else [1.0, 2 * (1 - f), 2 * (1 - f)]   # blue -> white -> red


def compute(xyz, imap, rig_res, ch, lo, hi):
    lk = sorted(k for k in imap[ch] if lo <= k <= hi)
    loop = xyz[:, np.array([imap[ch][k] for k in lk])]
    R = xyz[:, np.array([imap[ch][r] for r in rig_res if r in imap[ch]])]
    Rref = R[0] - R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(len(R)):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + R[0].mean(0)
    L0 = Lf[0]; L0m = L0.mean(0); L0c = L0 - L0m
    omega = np.empty((len(Lf), 3)); A = np.zeros((3, 3)); b = np.zeros(3); Qs = []
    for t in range(len(Lf)):
        Q = kabsch(L0c, Lf[t] - Lf[t].mean(0)); Qs.append(Q)
        omega[t] = Rot.from_matrix(Q).as_rotvec()
        c = Lf[t].mean(0) - Q @ L0m; M = np.eye(3) - Q; A += M.T @ M; b += M.T @ c
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    u = np.linalg.eigh(omega.T @ omega)[1][:, -1]; u /= np.linalg.norm(u)
    phi = omega @ u
    sel = [int(np.argmin(np.abs(phi - q))) for q in np.percentile(phi, np.linspace(2, 98, NCONF))]
    confs = [(Qs[t] @ (L0 - p).T).T + p for t in sel]
    return confs, p, u, lk, float(np.degrees(phi.max() - phi.min()))


def build(sysid, cdr, xyz, imap, rig, fv):
    ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
    confs, p, u, lk, rng = compute(xyz, imap, rig["chain_" + ch]["ultra_rigid"], ch, lo, hi)
    loop_pdb = f"{HERE}/figures/_hinge_{sysid}_{cdr}.pdb"; pdb_ca(loop_pdb, confs, lk)
    clampres, label = CLAMP[cdr[2:]]
    import pymol2
    from pymol.cgo import CYLINDER
    out = f"{HERE}/figures/hinge_{sysid}_{cdr}"
    with pymol2.PyMOL() as P:
        cmd = P.cmd
        cmd.load(fv, "framework"); cmd.remove(f"framework and not chain {ch}")
        cmd.hide("everything"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
        cmd.show("cartoon", "framework"); cmd.color("grey80", "framework")
        cmd.set("cartoon_transparency", 0.45, "framework")
        cmd.hide("cartoon", f"framework and resi {lo}-{hi}")            # remove static loop; the fan replaces it
        cs = "+".join(map(str, clampres))
        cmd.create("clamp", f"framework and resi {cs}"); cmd.show("sticks", "clamp")
        cmd.color("orange", "clamp"); cmd.set("stick_radius", 0.25, "clamp")
        cmd.label(f"clamp and resi {clampres[len(clampres)//2]} and name CA", f'"{label}"')
        cmd.load(loop_pdb, "loop_swing")
        for ci in range(NCONF):                                        # colour each conformation + bond into a trace
            X = CH8[ci]; cname = f"conf{ci}"
            cmd.set_color(cname, bwr(ci / (NCONF - 1))); cmd.color(cname, f"loop_swing and chain {X}")
            for k in range(len(lk) - 1):                               # bond consecutive CAs into a trace
                cmd.bond(f"loop_swing and chain {X} and resi {k+1}", f"loop_swing and chain {X} and resi {k+2}")
        cmd.show("sticks", "loop_swing"); cmd.set("stick_radius", 0.28, "loop_swing")
        cmd.show("spheres", "loop_swing"); cmd.set("sphere_scale", 0.38, "loop_swing")
        cmd.pseudoatom("pivot", pos=[float(x) for x in p]); cmd.show("spheres", "pivot")
        cmd.color("green", "pivot"); cmd.set("sphere_scale", 1.3, "pivot")
        a1 = (p - 13 * u).tolist(); a2 = (p + 13 * u).tolist()
        cmd.load_cgo([CYLINDER, *a1, *a2, 0.25, .1, .55, .1, .1, .55, .1], "hinge_axis")
        cmd.set("label_size", 15); cmd.set("float_labels", 1); cmd.set("label_color", "black")
        cmd.orient(f"loop_swing or (framework and resi {lo-8}-{hi+8})"); cmd.turn("y", 20)
        cmd.save(out + ".pse")
        cmd.ray(1500, 1150); cmd.png(out + ".png", dpi=150)
    return out + ".pse", rng


def main(sysid="3QH3"):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    from pymol_view import write_fv_pdb
    fv = write_fv_pdb(sysid)
    for cdr in CDRS:
        try:
            pse, rng = build(sysid, cdr, xyz, imap, rig, fv)
            print(f"{cdr}: swing range {rng:.0f}°  -> {pse}", flush=True)
        except Exception as e:
            print(f"{cdr} ERR {type(e).__name__}: {str(e)[:120]}", flush=True); traceback.print_exc()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3QH3")
