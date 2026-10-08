#!/usr/bin/env python
"""Per-CDR PyMOL ensemble session: the whole V-domain (static, framework-aligned) + the CDR loop as a
dense multi-frame trajectory (real MD motion, ~200 frames) + pivot + hinge axis. Playable and overlay-able."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, traceback
import numpy as np
import mdtraj as md
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
# per CDR: full ensemble residue range (clamp stem + loop), the clamp-stem subset, and a label
ENS = {"CDR1": (list(range(27, 44)), list(range(39, 44)), "Trp41 clamp"),        # C-clamp
       "CDR2": (list(range(49, 66)), list(range(49, 56)), "FR2 clamp"),          # N-clamp
       "CDR3": (list(range(100, 118)), list(range(100, 105)), "Cys104 clamp")}   # N-clamp
NFRAMES = 150


def res_atoms(top, ca_idx):
    out = []
    for ca in ca_idx:
        out += [a.index for a in top.atom(int(ca)).residue.atoms]
    return np.array(sorted(set(out)))


def superpose_all(xyz, fw_ca):
    fw = xyz[:, fw_ca]; ref = fw[0] - fw[0].mean(0); ref0 = fw[0].mean(0)
    out = np.empty_like(xyz)
    for t in range(len(xyz)):
        c = fw[t].mean(0); out[t] = (xyz[t] - c) @ kabsch(fw[t] - c, ref).T + ref0
    return out


def pivot_axis(xyz_s, loop_ca):
    loop = xyz_s[:, loop_ca]; L0 = loop[0]; L0m = L0.mean(0); L0c = L0 - L0m
    om = np.empty((len(loop), 3)); A = np.zeros((3, 3)); b = np.zeros(3)
    for t in range(len(loop)):
        Q = kabsch(L0c, loop[t] - loop[t].mean(0)); om[t] = Rot.from_matrix(Q).as_rotvec()
        c = loop[t].mean(0) - Q @ L0m; M = np.eye(3) - Q; A += M.T @ M; b += M.T @ c
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    u = np.linalg.eigh(om.T @ om)[1][:, -1]; u /= np.linalg.norm(u)
    return p, u


def main(sysid="3QH3"):
    tv, xyz, imap = load_md(sysid)
    top = tv.mdtraj.topology
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    supr = {}
    for ch in "AB":
        fw_ca = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
        supr[ch] = superpose_all(xyz, fw_ca)                          # all atoms superposed on this chain's framework
    for cdr in CDRS:
        try:
            ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
            xyz_s = supr[ch]
            traj = md.Trajectory(xyz_s / 10.0, top)                    # nm
            dom = np.asarray(tv.domain_idx([f"{ch}_variable"]))
            lk = sorted(k for k in imap[ch] if lo <= k <= hi)
            loop_ca = np.array([imap[ch][k] for k in lk])
            ens_res, clamp_res, label = ENS[cdr[2:]]
            ens_ca = [imap[ch][k] for k in ens_res if k in imap[ch]]
            ens_at = res_atoms(top, ens_ca)                            # clamp stem + loop -> shown as ensemble
            step = max(1, len(traj) // NFRAMES)
            struct_pdb = f"{HERE}/figures/_ens_{sysid}_{cdr}_struct.pdb"
            loop_pdb = f"{HERE}/figures/_ens_{sysid}_{cdr}_loop.pdb"
            traj[0].atom_slice(dom).save_pdb(struct_pdb)               # static full domain (frame 0)
            traj[::step].atom_slice(ens_at).save_pdb(loop_pdb)         # clamp+loop trajectory (~150 frames)
            p, u = pivot_axis(xyz_s, loop_ca)
            import pymol2
            from pymol.cgo import CYLINDER
            out = f"{HERE}/figures/hinge_ens_{sysid}_{cdr}"
            with pymol2.PyMOL() as P:
                cmd = P.cmd
                cmd.load(struct_pdb, "domain"); cmd.hide("everything"); cmd.bg_color("white")
                cmd.set("ray_opaque_background", 0)
                cmd.show("cartoon", "domain"); cmd.color("grey80", "domain")
                cmd.set("cartoon_transparency", 0.4, "domain")
                cmd.load(loop_pdb, "ens")                             # ensemble: clamp stem + loop, all frames
                cmd.hide("everything", "ens"); cmd.show("ribbon", "ens"); cmd.set("ribbon_width", 5, "ens")
                cmd.spectrum("resi", "rainbow", f"ens and resi {lo}-{hi}")     # loop: N-term blue -> tip red
                clsel = f"ens and resi {'+'.join(map(str, clamp_res))}"
                cmd.color("orange", clsel)                            # clamp stem = orange (should be a tight bundle)
                cmd.label(f"domain and resi {clamp_res[-1]} and name CA", f'"{label}"')
                cmd.pseudoatom("pivot", pos=[float(x) for x in p]); cmd.show("spheres", "pivot")
                cmd.color("green", "pivot"); cmd.set("sphere_scale", 1.3, "pivot")
                a1 = (p - 13 * u).tolist(); a2 = (p + 13 * u).tolist()
                cmd.load_cgo([CYLINDER, *a1, *a2, 0.25, .1, .55, .1, .1, .55, .1], "hinge_axis")
                cmd.set("label_size", 15); cmd.set("float_labels", 1); cmd.set("label_color", "black")
                cmd.orient(f"ens or (domain and resi {lo-8}-{hi+8})")
                cmd.set("all_states", 0)                              # .pse: playable movie (state 1 shown)
                cmd.save(out + ".pse")
                cmd.set("all_states", 1); cmd.turn("y", 15)          # png: all frames overlaid = swing envelope
                cmd.ray(1500, 1150); cmd.png(out + ".png", dpi=150)
            print(f"{cdr}: {len(traj[::step])} frames -> {out}.pse", flush=True)
        except Exception as e:
            print(f"{cdr} ERR {type(e).__name__}: {str(e)[:140]}", flush=True); traceback.print_exc()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3QH3")
