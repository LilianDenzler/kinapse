#!/usr/bin/env python
"""Visualise and ISOLATE each loop-motion type for one TCR/CDR in a single .pse.
Each rigid mode is reconstructed on its own by rotating the reference loop about ONE clamp axis:
  hinge = rotate about ĉ (feet-line)   twist = about ĥ (up-loop)   sway = about n̂ (plane normal)
  deform = the residual (real shape change after the rigid fit is removed)   total = the real MD loop
Deliverables per mode: a PLAYABLE multi-state trajectory (hit ▶) + a static VECTOR porcupine (displacement field).
-> figures/motions_<sys>_<CDR>.pse (+ .png)"""
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
from viz_ensemble import superpose_all

CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}
RGB = {"hinge": (0.18, 0.49, 0.20), "twist": (0.48, 0.31, 0.64), "sway": (0.94, 0.63, 0.19),
       "deform": (0.75, 0.22, 0.17), "total": (0.4, 0.4, 0.4)}
NFR = 61


def rot_about(P, u, piv, th):
    c, s = np.cos(th), np.sin(th); X = P - piv
    return piv + c * X + s * np.cross(np.broadcast_to(u, X.shape), X) + (1 - c) * ((X @ u)[:, None] * u)


def pdb_multi(path, confs):                                             # confs: (S, N, 3), one MODEL per state
    with open(path, "w") as f:
        for si, X in enumerate(confs):
            f.write(f"MODEL     {si+1:4d}\n")
            for k, (x, y, z) in enumerate(X):
                f.write(f"ATOM  {k+1:5d}  CA  GLY A{k+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("TER\nENDMDL\n")
        f.write("END\n")


def arrow(p1, p2, rgb, r=0.14):
    from pymol.cgo import CYLINDER, CONE
    p1 = [float(v) for v in p1]; p2 = [float(v) for v in p2]
    d = np.array(p2) - np.array(p1); mid = (np.array(p1) + 0.68 * d).tolist()
    return [CYLINDER, *p1, *mid, r, *rgb, *rgb,
            CONE, *mid, *p2, r * 2.4, 0.0, *rgb, *rgb, 1.0, 1.0]


def main(sysid="3SKN", cdr="B_CDR2"):
    tv, xyz, imap = load_md(sysid)
    top = tv.mdtraj.topology
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
    fw_ca = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
    xyz_s = superpose_all(xyz, fw_ca)
    lk = sorted(k for k in imap[ch] if lo <= k <= hi)
    loop = xyz_s[:, np.array([imap[ch][k] for k in lk])]
    N = loop.shape[1]
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    L0cen = L0.mean(0); L0c = L0 - L0cen
    fN, fC = L0[0], L0[-1]; piv = fC if CLAMP[cdr[2:]] == "C" else fN
    c = fC - fN; c /= np.linalg.norm(c)
    perp = (L0 - piv) - ((L0 - piv) @ c)[:, None] * c
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); apex = L0[ai]
    h = perp[ai]; h /= np.linalg.norm(h); n = np.cross(c, h); n /= np.linalg.norm(n)
    axv = {"hinge": c, "twist": h, "sway": n}
    # per-frame rotation split + deformation residual
    T = loop.shape[0]; wsplit = {"hinge": np.empty(T), "twist": np.empty(T), "sway": np.empty(T)}
    resid = np.empty((T, N, 3))
    for t in range(T):
        Q = kabsch(L0c, loop[t] - loop[t].mean(0)); w = Rot.from_matrix(Q).as_rotvec()
        wsplit["hinge"][t] = w @ c; wsplit["twist"][t] = w @ h; wsplit["sway"][t] = w @ n
        resid[t] = loop[t] - (L0c @ Q.T + loop[t].mean(0))             # deformation displacement field
    # isolated smooth sweeps for the rigid modes (across the real observed angle range)
    trajs = {}
    for m in ("hinge", "twist", "sway"):
        angs = np.linspace(np.percentile(wsplit[m], 2), np.percentile(wsplit[m], 98), NFR)
        trajs[m] = np.array([rot_about(L0, axv[m], piv, a) for a in angs])
    step = max(1, T // NFR)
    trajs["deform"] = L0[None] + resid[::step][:NFR]                    # real shape change on the reference
    trajs["total"] = loop[::step][:NFR]                                 # the real loop motion
    dom = np.asarray(tv.domain_idx([f"{ch}_variable"]))
    traj0 = md.Trajectory(xyz_s / 10.0, top)
    struct_pdb = f"{HERE}/figures/_mm_{sysid}_{cdr}_dom.pdb"; traj0[0].atom_slice(dom).save_pdb(struct_pdb)
    files = {}
    for m, X in trajs.items():
        p = f"{HERE}/figures/_mm_{sysid}_{cdr}_{m}.pdb"; pdb_multi(p, X); files[m] = p
    # vector porcupines: displacement at +2σ of each rigid mode; deform = RMS residual direction
    vecs = {}
    for m in ("hinge", "twist", "sway"):
        disp = rot_about(L0, axv[m], piv, 2 * np.std(wsplit[m])) - L0
        vecs[m] = disp * 2.5                                            # exaggerate for visibility
    dvec = np.empty((N, 3))                                             # deform: dominant fluctuation direction per residue
    for i in range(N):
        wv, V = np.linalg.eigh(np.cov(resid[:, i, :].T))               # PCA of this residue's residual cloud
        dvec[i] = V[:, -1] * np.sqrt(max(wv[-1], 0.0))                 # top eigenvector × its std (sign arbitrary)
    vecs["deform"] = dvec * 2.5

    import pymol2
    from pymol.cgo import CYLINDER
    out = f"{HERE}/figures/motions_{sysid}_{cdr}"
    with pymol2.PyMOL() as P:
        cmd = P.cmd
        cmd.load(struct_pdb, "domain"); cmd.hide("everything"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
        cmd.show("cartoon", "domain"); cmd.color("grey80", "domain"); cmd.set("cartoon_transparency", 0.6, "domain")
        for m in ("hinge", "twist", "sway", "deform", "total"):         # playable trajectories, one object per mode
            cmd.load(files[m], m)
            cmd.hide("everything", m); cmd.show("ribbon", m); cmd.set("ribbon_width", 5, m)
            cmd.set_color(f"c_{m}", list(RGB[m])); cmd.color(f"c_{m}", m)
        for m in ("hinge", "twist", "sway"):                            # axis rods through the clamp
            L = 7.0; a1 = (piv - L * axv[m]).tolist(); a2 = (piv + L * axv[m]).tolist()
            cmd.load_cgo([CYLINDER, *a1, *a2, 0.2, *RGB[m], *RGB[m]], f"axis_{m}")
        for m in ("hinge", "twist", "sway", "deform"):                  # vector porcupines (static)
            cgo = []
            for i in range(N):
                cgo += arrow(L0[i], L0[i] + vecs[m][i], RGB[m])
                if m == "deform":                                       # sign arbitrary -> double-headed fluctuation axis
                    cgo += arrow(L0[i], L0[i] - vecs[m][i], RGB[m])
            cmd.load_cgo(cgo, f"vec_{m}")
        cmd.pseudoatom("clamp", pos=[float(v) for v in piv]); cmd.show("spheres", "clamp")
        cmd.color("black", "clamp"); cmd.set("sphere_scale", 0.6, "clamp")
        cmd.set("all_states", 0)                                        # play one state at a time (▶ animates)
        cmd.disable("twist"); cmd.disable("sway"); cmd.disable("deform"); cmd.disable("total")   # start on hinge
        cmd.disable("vec_twist"); cmd.disable("vec_sway"); cmd.disable("vec_deform")
        cmd.orient("total"); cmd.zoom("total", 7)
        cmd.save(out + ".pse")
        # per-mode overlay PNGs (all states of ONE mode shown = isolated envelope), same view
        cmd.set("all_states", 1)
        allobj = ["hinge", "twist", "sway", "deform", "total"]; allvec = ["vec_hinge", "vec_twist", "vec_sway", "vec_deform"]
        for m in ("hinge", "twist", "sway", "deform"):
            for o in allobj + allvec:
                cmd.disable(o)
            cmd.enable(m); cmd.enable(f"vec_{m}")
            cmd.ray(1000, 850); cmd.png(f"{out}_{m}.png", dpi=140)
    # montage the four isolated modes into one image
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.image as mpimg
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for a, m in zip(ax.flat, ("hinge", "twist", "sway", "deform")):
        a.imshow(mpimg.imread(f"{out}_{m}.png")); a.axis("off")
        a.set_title(f"{m}  (isolated)", fontsize=13, color={"hinge": "#2E7D32", "twist": "#7B4FA3", "sway": "#B8860B", "deform": "#C0392B"}[m], fontweight="bold")
    fig.suptitle(f"{sysid} {cdr}: isolated loop-motion modes (each = that mode alone; arrows = displacement field)", fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(f"{out}_montage.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("montage ->", out + "_montage.png")
    print(f"{sysid} {cdr}: modes = hinge/twist/sway/deform/total  -> {out}.pse", flush=True)
    print("objects: trajectories (play ▶): hinge twist sway deform total | axes: axis_* | vectors: vec_* | clamp")


if __name__ == "__main__":
    a = sys.argv[1:]
    try:
        main(a[0] if a else "3SKN", a[1] if len(a) > 1 else "B_CDR2")
    except Exception as e:
        print("ERR", type(e).__name__, str(e)[:200]); traceback.print_exc()
