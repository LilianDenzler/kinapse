#!/usr/bin/env python
"""Per-CDR .pse showing the loop's three rigid-motion AXES and the rigid transform about each,
ANCHORED ON THE CLAMP foot (CDR1->C-stem, CDR2/3->N-stem) so every conformation shares the clamp:
  ĉ (green)  = feet-line  -> tip-elevation swing (tip out of plane)
  n̂ (orange) = plane normal -> in-plane sway
  ĥ (purple) = up-the-loop -> twist
Each axis = a coloured rod through the clamp; each transform = a fan of the reference loop rotated about
that axis; each swing ANGLE is drawn as a coloured arc (same axis colour) so it is clear what is measured.
Angles shown at 3x the real MD amplitude for visibility. -> figures/axes_<sys>_<CDR>.pse (+ .png)"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, traceback
import numpy as np
import mdtraj as md
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch
from viz_ensemble import superpose_all

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}          # which stem foot is the anchor
NCONF = 7
RGB = {"c": (0.18, 0.49, 0.20), "n": (0.94, 0.63, 0.19), "h": (0.48, 0.31, 0.64)}
NAME = {"c": "c: feet-line (tip-elev)", "n": "n: normal (sway)", "h": "h: up-loop (twist)"}


def rot_line(P, u, piv, th):
    c, s = np.cos(th), np.sin(th); X = P - piv
    return piv + c * X + s * np.cross(np.broadcast_to(u, X.shape), X) + (1 - c) * ((X @ u)[:, None] * u)


def arc_cgo(piv, u, eref, half, R, rgb, nseg=26):
    from pymol.cgo import CYLINDER
    e1 = eref - (eref @ u) * u
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0, 0]) - u * u[0]
    e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    pts = [piv + R * (np.cos(t) * e1 + np.sin(t) * e2) for t in np.linspace(-half, half, nseg)]
    cgo = []
    for i in range(len(pts) - 1):
        cgo += [CYLINDER, *pts[i].tolist(), *pts[i + 1].tolist(), 0.16, *rgb, *rgb]
    cgo += [CYLINDER, *piv.tolist(), *pts[0].tolist(), 0.09, *rgb, *rgb]     # bounding radii
    cgo += [CYLINDER, *piv.tolist(), *pts[-1].tolist(), 0.09, *rgb, *rgb]
    return cgo, pts[len(pts) // 2]


def pdb_ca(path, confs):
    ch = "ABCDEFGHIJKLMNOPQRSTUVWX"
    with open(path, "w") as f:
        for ci, X in enumerate(confs):
            for k, (x, y, z) in enumerate(X):
                f.write(f"ATOM  {k+1:5d}  CA  GLY {ch[ci]}{k+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("TER\n")
        f.write("END\n")


def main(sysid="3SKN", only=None):
    tv, xyz, imap = load_md(sysid)
    top = tv.mdtraj.topology
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    amps = json.load(open(f"{HERE}/results_rotframe.json"))[sysid]
    supr = {ch: superpose_all(xyz, np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])) for ch in "AB"}
    for cdr in CDRS:
        if only and cdr != only:
            continue
        try:
            ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
            xyz_s = supr[ch]
            lk = sorted(k for k in imap[ch] if lo <= k <= hi)
            loop = xyz_s[:, np.array([imap[ch][k] for k in lk])]
            N = loop.shape[1]
            dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
            L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
            fN, fC = L0[0], L0[-1]
            piv = fC if CLAMP[cdr[2:]] == "C" else fN                     # clamp foot = anchor
            c = fC - fN; c /= np.linalg.norm(c)
            perp = (L0 - piv) - ((L0 - piv) @ c)[:, None] * c
            ai = int(np.argmax(np.linalg.norm(perp, axis=1))); apex = L0[ai]
            h = perp[ai]; h /= np.linalg.norm(h); n = np.cross(c, h); n /= np.linalg.norm(n)
            axv = {"c": c, "n": n, "h": h}
            amp = {k: amps[cdr][f"amp_{k}"] for k in "cnh"}
            half = {k: float(np.radians(min(max(3 * amp[k], 8), 26))) for k in "cnh"}
            for k in "cnh":                                               # fans anchored on the clamp foot
                fan = [rot_line(L0, axv[k], piv, th) for th in np.linspace(-half[k], half[k], NCONF)]
                pdb_ca(f"{HERE}/figures/_ax_{sysid}_{cdr}_{k}.pdb", fan)
            pdb_ca(f"{HERE}/figures/_ax_{sysid}_{cdr}_ref.pdb", [L0])
            dom = np.asarray(tv.domain_idx([f"{ch}_variable"]))
            traj = md.Trajectory(xyz_s / 10.0, top)
            struct_pdb = f"{HERE}/figures/_ax_{sysid}_{cdr}_struct.pdb"
            traj[0].atom_slice(dom).save_pdb(struct_pdb)
            Rarc = 0.6 * float(np.linalg.norm(apex - piv))
            import pymol2
            from pymol.cgo import CYLINDER
            out = f"{HERE}/figures/axes_{sysid}_{cdr}"
            with pymol2.PyMOL() as P:
                cmd = P.cmd
                cmd.load(struct_pdb, "domain"); cmd.hide("everything"); cmd.bg_color("white")
                cmd.set("ray_opaque_background", 0)
                cmd.show("cartoon", "domain"); cmd.color("grey80", "domain"); cmd.set("cartoon_transparency", 0.6, "domain")
                cmd.load(f"{HERE}/figures/_ax_{sysid}_{cdr}_ref.pdb", "ref")
                cmd.hide("everything", "ref")
                for k in range(N - 1):
                    cmd.bond(f"ref and resi {k+1}", f"ref and resi {k+2}")
                cmd.show("sticks", "ref"); cmd.set("stick_radius", 0.45, "ref"); cmd.color("grey40", "ref")
                for k in "cnh":                                           # transform fans
                    obj = f"fan_{k}"; cmd.load(f"{HERE}/figures/_ax_{sysid}_{cdr}_{k}.pdb", obj)
                    cmd.hide("everything", obj)
                    for ci in range(NCONF):
                        X = "ABCDEFGHIJKLMNOPQRSTUVWX"[ci]
                        for j in range(N - 1):
                            cmd.bond(f"{obj} and chain {X} and resi {j+1}", f"{obj} and chain {X} and resi {j+2}")
                    cmd.set_color(f"col_{k}", list(RGB[k])); cmd.color(f"col_{k}", obj)
                    cmd.show("sticks", obj); cmd.set("stick_radius", 0.16, obj)
                    L = 6.5                                               # axis rod through the clamp foot
                    a1 = (piv - L * axv[k]).tolist(); a2 = (piv + L * axv[k]).tolist()
                    cmd.load_cgo([CYLINDER, *a1, *a2, 0.22, *RGB[k], *RGB[k]], f"axis_{k}")
                    cmd.pseudoatom(f"lab_{k}", pos=(piv + (L + 1.0) * axv[k]).tolist(), label=NAME[k])
                    cgo, mid = arc_cgo(piv, axv[k], apex - piv, half[k], Rarc, RGB[k])   # SWING ANGLE arc
                    cmd.load_cgo(cgo, f"angle_{k}")
                    cmd.pseudoatom(f"ang_{k}", pos=mid.tolist(), label=f"{amp[k]:.0f}deg")
                cmd.pseudoatom("clamp", pos=[float(v) for v in piv]); cmd.show("spheres", "clamp")
                cmd.color("black", "clamp"); cmd.set("sphere_scale", 0.8, "clamp")
                cmd.pseudoatom("clamp_lbl", pos=(piv - 1.5 * h).tolist(), label="CLAMP (anchor)")
                cmd.set("label_size", 15); cmd.set("float_labels", 1); cmd.set("label_color", "black")
                cmd.orient("ref or fan_c or fan_n or fan_h"); cmd.zoom("ref", 10)
                cmd.save(out + ".pse")
                cmd.turn("y", 20); cmd.ray(1600, 1200); cmd.png(out + ".png", dpi=150)
            print(f"{cdr}: clamp={CLAMP[cdr[2:]]}-foot  amp c/n/h={amp['c']:.1f}/{amp['n']:.1f}/{amp['h']:.1f} -> {out}.pse", flush=True)
        except Exception as e:
            print(f"{cdr} ERR {type(e).__name__}: {str(e)[:120]}", flush=True); traceback.print_exc()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "3SKN", a[1] if len(a) > 1 else None)
