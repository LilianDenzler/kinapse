#!/usr/bin/env python
"""Verify an amplitude OUTLIER is a real physical swing, not a numbering/clamp/axis artifact.
Diagnostics (printed): loop IMGT residue keys (insertion-code check), N, feet-line length, and the RMSF of the
CLAMP foot / non-clamp foot / tip relative to the (fixed) framework — a mobile clamp = bad anchor = spurious rotation.
Visual: the REAL MD loop conformations at percentiles [2.5,25,50,75,97.5] of the mode angle, overlaid on the framework
(blue→red by angle), clamp marked. If the whole loop fans coherently → real rigid swing; if one atom jumps → artifact.
-> figures/outlier_<sys>_<CDR>_<mode>.png + printed diagnostics"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
import mdtraj as md
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all

CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}
RGB = {"hinge": (0.18, 0.49, 0.20), "twist": (0.48, 0.31, 0.64), "sway": (0.94, 0.63, 0.19)}


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1); c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


def pdb_ca(path, confs):
    ch = "ABCDEFGHIJKLMNOPQRSTUVWX"
    with open(path, "w") as f:
        for ci, X in enumerate(confs):
            for k, (x, y, z) in enumerate(X):
                f.write(f"ATOM  {k+1:5d}  CA  GLY {ch[ci]}{k+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("TER\n")
        f.write("END\n")


def main(sysid, cdr, mode="hinge"):
    tv, xyz, imap = load_md(sysid); top = tv.mdtraj.topology
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
    fwkeys = [r for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]]
    fw = np.array([imap[ch][r] for r in fwkeys])
    supr = superpose_all(xyz, fw)
    lk = sorted(k for k in imap[ch] if lo <= k <= hi)
    loop = supr[:, np.array([imap[ch][k] for k in lk])]; T, N, _ = loop.shape
    # reference + axes
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    fN, fC = L0[0], L0[-1]; m = 0.5 * (fN + fC); cc = fC - fN; feet_len = np.linalg.norm(cc); cc /= feet_len
    perp = (L0 - m) - ((L0 - m) @ cc)[:, None] * cc
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(cc, h); n /= np.linalg.norm(n)
    axv = {"hinge": cc, "twist": h, "sway": n}; u = axv[mode]
    piv = L0[-1] if CLAMP[cdr[2:]] == "C" else L0[0]
    clamp_i = N - 1 if CLAMP[cdr[2:]] == "C" else 0
    # per-frame angle about the chosen axis
    ang = np.empty(T)
    for t in range(T):
        R, _, _ = robust_kabsch(L0, loop[t]); ang[t] = np.degrees(Rot.from_matrix(R).as_rotvec() @ u)
    # ---- DIAGNOSTICS ----
    def rmsf(idx):
        P = supr[:, idx]; return float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))
    fw_rmsf = np.mean([rmsf(i) for i in fw])
    loopidx = [imap[ch][k] for k in lk]
    print(f"\n=== {sysid} {cdr}  mode={mode} ===")
    print(f"loop IMGT keys ({N}): {lk}")
    gaps = [b - a for a, b in zip(lk[:-1], lk[1:])]
    print(f"key gaps: {gaps}  (all 1 ⇒ contiguous, no missing/odd insertion handling)")
    print(f"feet-line length |fC-fN| = {feet_len:.2f} Å  (short ⇒ hinge axis ĉ less stable)")
    print(f"framework CA RMSF (mean)   = {fw_rmsf:.2f} Å   <- the anchor")
    print(f"CLAMP foot ({'C' if clamp_i==N-1 else 'N'}, IMGT {lk[clamp_i]}) RMSF = {rmsf(loopidx[clamp_i]):.2f} Å   <- should be ~framework if truly anchored")
    print(f"non-clamp foot (IMGT {lk[0 if clamp_i==N-1 else -1]}) RMSF = {rmsf(loopidx[0 if clamp_i==N-1 else -1]):.2f} Å")
    print(f"tip (IMGT {lk[ai]}) RMSF = {rmsf(loopidx[ai]):.2f} Å")
    print(f"angle about {mode}: median {np.median(ang):+.0f}°, p2.5 {np.percentile(ang,2.5):+.0f}°, p97.5 {np.percentile(ang,97.5):+.0f}°, "
          f"range {np.percentile(ang,97.5)-np.percentile(ang,2.5):.0f}°", flush=True)
    # ---- VISUAL: real frames at angle percentiles ----
    pcts = [2.5, 25, 50, 75, 97.5]
    sel = [int(np.argmin(np.abs(ang - np.percentile(ang, p)))) for p in pcts]
    pdb_ca(f"{HERE}/figures/_ol_frames.pdb", [loop[f] for f in sel])
    dom = np.asarray(tv.domain_idx([f"{ch}_variable"]))
    md.Trajectory(supr[sel[2]:sel[2]+1] / 10.0, top).atom_slice(dom).save_pdb(f"{HERE}/figures/_ol_dom.pdb")
    import pymol2
    from pymol.cgo import CYLINDER
    with pymol2.PyMOL() as P:
        cmd = P.cmd
        cmd.load(f"{HERE}/figures/_ol_dom.pdb", "domain"); cmd.hide("everything"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
        cmd.show("cartoon", "domain"); cmd.color("grey90", "domain"); cmd.set("cartoon_transparency", 0.75, "domain")
        cmd.load(f"{HERE}/figures/_ol_frames.pdb", "frames"); cmd.hide("everything", "frames")
        for ci in range(len(sel)):
            X = "ABCDEFGHIJKLMNOPQRSTUVWX"[ci]
            for j in range(N - 1):
                cmd.bond(f"frames and chain {X} and resi {j+1}", f"frames and chain {X} and resi {j+2}")
        cmd.show("sticks", "frames"); cmd.set("stick_radius", 0.3, "frames"); cmd.spectrum("chain", "blue_white_red", "frames")
        cmd.set("all_states", 1)
        a1 = (piv - 7 * u).tolist(); a2 = (piv + 7 * u).tolist()
        cmd.load_cgo([CYLINDER, *a1, *a2, 0.3, *RGB[mode], *RGB[mode]], f"axis_{mode}")
        cmd.pseudoatom("clamp", pos=[float(v) for v in piv]); cmd.show("spheres", "clamp"); cmd.color("black", "clamp"); cmd.set("sphere_scale", 1.0, "clamp")
        cmd.orient("frames"); cmd.zoom("frames", 7)
        out = f"{HERE}/figures/outlier_{sysid}_{cdr}_{mode}.png"
        cmd.ray(1400, 1100); cmd.png(out, dpi=140)
    print("fig ->", out, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2] if len(a) > 2 else "hinge")
