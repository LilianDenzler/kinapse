#!/usr/bin/env python
"""Pedagogical visual of the three rigid rotation MODES for one loop, exaggerated & labelled with everyday analogies.
Panel 0: the loop + its 3 axes (ĉ feet-line, ĥ up-loop, n̂ normal) through the clamp.
Panels 1-3: the loop swung ±25° about ONE axis (fan blue→red), so it is obvious what hinge / twist / sway look like.
Same camera in every panel. -> figures/modes_clear_<sys>_<CDR>.png"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
import mdtraj as md
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all

CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}
RGB = {"hinge": (0.18, 0.49, 0.20), "twist": (0.48, 0.31, 0.64), "sway": (0.94, 0.63, 0.19)}
NC = 7; EXAG = 25.0


def rot_about(P, u, piv, th):
    c, s = np.cos(th), np.sin(th); X = P - piv
    return piv + c * X + s * np.cross(np.broadcast_to(u, X.shape), X) + (1 - c) * ((X @ u)[:, None] * u)


def pdb_ca(path, confs):
    ch = "ABCDEFGHIJKLMNOPQRSTUVWX"
    with open(path, "w") as f:
        for ci, X in enumerate(confs):
            for k, (x, y, z) in enumerate(X):
                f.write(f"ATOM  {k+1:5d}  CA  GLY {ch[ci]}{k+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("TER\n")
        f.write("END\n")


def main(sysid="3SKN", cdr="B_CDR3"):
    tv, xyz, imap = load_md(sysid); top = tv.mdtraj.topology
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
    fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
    supr = superpose_all(xyz, fw)
    lk = sorted(k for k in imap[ch] if lo <= k <= hi)
    loop = supr[:, np.array([imap[ch][k] for k in lk])]; N = loop.shape[1]
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    fN, fC = L0[0], L0[-1]; piv = fC if CLAMP[cdr[2:]] == "C" else fN
    m = 0.5 * (fN + fC); c = fC - fN; c /= np.linalg.norm(c)
    perp = (L0 - m) - ((L0 - m) @ c)[:, None] * c
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(c, h); n /= np.linalg.norm(n)
    axv = {"hinge": c, "twist": h, "sway": n}
    for k in axv:
        pdb_ca(f"{HERE}/figures/_mc_{k}.pdb", [rot_about(L0, axv[k], piv, np.radians(a)) for a in np.linspace(-EXAG, EXAG, NC)])
    pdb_ca(f"{HERE}/figures/_mc_ref.pdb", [L0])
    dom = np.asarray(tv.domain_idx([f"{ch}_variable"]))
    md.Trajectory(supr / 10.0, top)[0].atom_slice(dom).save_pdb(f"{HERE}/figures/_mc_dom.pdb")
    import pymol2
    from pymol.cgo import CYLINDER
    panels = {}
    with pymol2.PyMOL() as P:
        cmd = P.cmd
        cmd.load(f"{HERE}/figures/_mc_dom.pdb", "domain"); cmd.hide("everything"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
        cmd.show("cartoon", "domain"); cmd.color("grey90", "domain"); cmd.set("cartoon_transparency", 0.7, "domain")
        cmd.load(f"{HERE}/figures/_mc_ref.pdb", "ref"); cmd.hide("everything", "ref")
        for j in range(N - 1):
            cmd.bond(f"ref and resi {j+1}", f"ref and resi {j+2}")
        cmd.show("sticks", "ref"); cmd.set("stick_radius", 0.5, "ref"); cmd.color("grey50", "ref")
        L = 8.0
        for k, u in axv.items():
            a1 = (piv - L * u).tolist(); a2 = (piv + L * u).tolist()
            cmd.load_cgo([CYLINDER, *a1, *a2, 0.28, *RGB[k], *RGB[k]], f"axis_{k}")
            obj = f"fan_{k}"; cmd.load(f"{HERE}/figures/_mc_{k}.pdb", obj); cmd.hide("everything", obj)
            for ci in range(NC):
                X = "ABCDEFGHIJKLMNOPQRSTUVWX"[ci]
                for j in range(N - 1):
                    cmd.bond(f"{obj} and chain {X} and resi {j+1}", f"{obj} and chain {X} and resi {j+2}")
            cmd.show("sticks", obj); cmd.set("stick_radius", 0.22, obj)
            cmd.spectrum("chain", "blue_white_red", obj)
        cmd.pseudoatom("clamp", pos=[float(v) for v in piv]); cmd.show("spheres", "clamp"); cmd.color("black", "clamp"); cmd.set("sphere_scale", 0.9, "clamp")
        cmd.set("all_states", 1)
        cmd.orient("ref"); cmd.zoom("ref", 9); cmd.turn("y", 25); cmd.turn("x", 10)
        for name, objs in [("frame", ["ref", "axis_hinge", "axis_twist", "axis_sway"]),
                           ("hinge", ["fan_hinge", "axis_hinge"]), ("twist", ["fan_twist", "axis_twist"]), ("sway", ["fan_sway", "axis_sway"])]:
            for o in ["ref", "fan_hinge", "fan_twist", "fan_sway", "axis_hinge", "axis_twist", "axis_sway"]:
                cmd.disable(o)
            for o in objs:
                cmd.enable(o)
            png = f"{HERE}/figures/_mc_panel_{name}.png"; cmd.ray(1000, 950); cmd.png(png, dpi=140); panels[name] = png
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.image as mpimg
    titles = {"frame": ("the loop + its 3 axes", "black ball = CLAMP (fixed anchor); green ĉ=feet-line, purple ĥ=up-loop, orange n̂=normal"),
              "hinge": ("HINGE — rotate about ĉ (feet-line)", "tip lifts OUT of the loop plane · like a DRAWBRIDGE / trapdoor"),
              "twist": ("TWIST — rotate about ĥ (up-loop)", "loop turns about its own long axis · like WRINGING A TOWEL"),
              "sway": ("SWAY — rotate about n̂ (normal)", "loop swings side-to-side IN its plane · like a PENDULUM / wiper")}
    col = {"frame": "black", "hinge": "#2E7D32", "twist": "#7B4FA3", "sway": "#B8860B"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 11.5))
    for a, name in zip(ax.flat, ["frame", "hinge", "twist", "sway"]):
        a.imshow(mpimg.imread(panels[name])); a.axis("off")
        t, sub = titles[name]
        a.set_title(t, fontsize=13, fontweight="bold", color=col[name])
        a.text(0.5, -0.04, sub, transform=a.transAxes, ha="center", va="top", fontsize=10, color="0.25")
    fig.suptitle(f"{sysid} {cdr}: the three RIGID rotation modes (fan = ±25° swing, blue→red; clamp fixed)", fontsize=13, y=1.0)
    out = f"{HERE}/figures/modes_clear_{sysid}_{cdr}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "3SKN", a[1] if len(a) > 1 else "B_CDR3")
