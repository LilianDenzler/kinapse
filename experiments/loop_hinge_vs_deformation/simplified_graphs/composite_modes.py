#!/usr/bin/env python
"""Per-CDR composite (one TCR): (A) per-residue displacement — average line + min–max shaded band;
(B) average vs max per-residue (Å) by mode; (C) the structural mode visualisations (hinge/twist/sway);
(D) total-Å² budget: total = rigid + deform (basis-free; nothing unexplained). No per-mode bar — hinge/twist/sway
overlap and must not be summed. Rigid = robust (deform-excluded) fit; deform = residual (≈ CA-CA d_LL).
-> figures/composite_<sys>_<CDR>.png for each CDR."""
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

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}
RGB = {"hinge": (0.18, 0.49, 0.20), "twist": (0.48, 0.31, 0.64), "sway": (0.94, 0.63, 0.19)}
NC = 7; EXAG = 25.0


def _autocrop(im, pad=3):
    """trim near-white margins from a rendered panel (RGB or RGBA float image)."""
    rgb = im[..., :3]
    mask = (rgb < 0.985).any(2)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return im
    y0, y1 = max(0, ys.min() - pad), min(im.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(im.shape[1], xs.max() + pad)
    return im[y0:y1, x0:x1]


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1); c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


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


def geom(loop):
    T, N, _ = loop.shape
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    fN, fC = L0[0], L0[-1]; m = 0.5 * (fN + fC); cc = fC - fN; cc /= np.linalg.norm(cc)
    perp = (L0 - m) - ((L0 - m) @ cc)[:, None] * cc
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(cc, h); n /= np.linalg.norm(n)
    return L0, cc, h, n


def montage(sysid, cdr, loop, L0, axv, piv, top, dom, supr):
    for k in axv:
        pdb_ca(f"{HERE}/figures/_cm_{k}.pdb", [rot_about(L0, axv[k], piv, np.radians(a)) for a in np.linspace(-EXAG, EXAG, NC)])
    pdb_ca(f"{HERE}/figures/_cm_ref.pdb", [L0])
    md.Trajectory(supr / 10.0, top)[0].atom_slice(dom).save_pdb(f"{HERE}/figures/_cm_dom.pdb")
    import pymol2
    from pymol.cgo import CYLINDER
    N = loop.shape[1]; panels = {}
    with pymol2.PyMOL() as P:
        cmd = P.cmd
        cmd.load(f"{HERE}/figures/_cm_dom.pdb", "domain"); cmd.hide("everything"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
        cmd.show("cartoon", "domain"); cmd.color("grey90", "domain"); cmd.set("cartoon_transparency", 0.75, "domain")
        cmd.load(f"{HERE}/figures/_cm_ref.pdb", "ref"); cmd.hide("everything", "ref")
        for j in range(N - 1):
            cmd.bond(f"ref and resi {j+1}", f"ref and resi {j+2}")
        cmd.show("sticks", "ref"); cmd.set("stick_radius", 0.5, "ref"); cmd.color("grey50", "ref")
        for k, u in axv.items():
            cmd.load_cgo([CYLINDER, *(piv - 8 * u).tolist(), *(piv + 8 * u).tolist(), 0.28, *RGB[k], *RGB[k]], f"axis_{k}")
            obj = f"fan_{k}"; cmd.load(f"{HERE}/figures/_cm_{k}.pdb", obj); cmd.hide("everything", obj)
            for ci in range(NC):
                X = "ABCDEFGHIJKLMNOPQRSTUVWX"[ci]
                for j in range(N - 1):
                    cmd.bond(f"{obj} and chain {X} and resi {j+1}", f"{obj} and chain {X} and resi {j+2}")
            cmd.show("sticks", obj); cmd.set("stick_radius", 0.22, obj); cmd.spectrum("chain", "blue_white_red", obj)
        cmd.pseudoatom("clamp", pos=[float(v) for v in piv]); cmd.show("spheres", "clamp"); cmd.color("black", "clamp"); cmd.set("sphere_scale", 0.9, "clamp")
        cmd.set("all_states", 1); cmd.orient("ref"); cmd.zoom("ref", 9); cmd.turn("y", 25); cmd.turn("x", 10)
        for name, objs in [("frame", ["ref", "axis_hinge", "axis_twist", "axis_sway"]), ("hinge", ["fan_hinge", "axis_hinge"]),
                           ("twist", ["fan_twist", "axis_twist"]), ("sway", ["fan_sway", "axis_sway"])]:
            for o in ["ref", "fan_hinge", "fan_twist", "fan_sway", "axis_hinge", "axis_twist", "axis_sway"]:
                cmd.disable(o)
            for o in objs:
                cmd.enable(o)
            p = f"{HERE}/figures/_cm_panel_{name}.png"; cmd.ray(760, 720); cmd.png(p, dpi=130); panels[name] = p
    return panels


def main(sysid="3SKN", only=None):
    tv, xyz, imap = load_md(sysid); top = tv.mdtraj.topology
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.image as mpimg
    from matplotlib import gridspec
    supr_cache = {}
    for cdr in (CDRS if not only else [only]):
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        if ch not in supr_cache:
            fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
            supr_cache[ch] = superpose_all(xyz, fw)
        supr = supr_cache[ch]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = supr[:, np.array([imap[ch][k] for k in lk])]; T, N, _ = loop.shape
        L0, cc, h, n = geom(loop); axv = {"hinge": cc, "twist": h, "sway": n}
        piv = L0[-1] if CLAMP[cdr[2:]] == "C" else L0[0]
        disp = {k: np.empty((T, N)) for k in ("hinge", "twist", "sway", "rigid", "deform", "total")}
        for t in range(T):
            R, Pc, Qc = robust_kabsch(L0, loop[t]); w = Rot.from_matrix(R).as_rotvec()
            for k, u in axv.items():
                disp[k][t] = np.linalg.norm(rot_about(L0, u, piv, w @ u) - L0, axis=1)
            rigid = (L0 - Pc) @ R.T + Qc
            disp["rigid"][t] = np.linalg.norm(rigid - L0, axis=1)
            disp["deform"][t] = np.linalg.norm(loop[t] - rigid, axis=1)
            disp["total"][t] = np.linalg.norm(loop[t] - L0, axis=1)
        rms = {k: np.sqrt((disp[k] ** 2).mean(0)) for k in disp}          # per-residue RMS
        V = {k: float((disp[k] ** 2).mean(0).sum()) for k in disp}        # summed mean-square (Å²)
        dom = np.asarray(tv.domain_idx([f"{ch}_variable"]))
        panels = montage(sysid, cdr, loop, L0, axv, piv, top, dom, supr)

        fig = plt.figure(figsize=(16, 9.2))
        gs = gridspec.GridSpec(2, 3, height_ratios=[0.78, 1.0], hspace=0.16, wspace=0.28)
        # C: montage across the top (autocropped + height-matched to kill whitespace)
        axc = fig.add_subplot(gs[0, :]); axc.axis("off")
        imgs = [_autocrop(mpimg.imread(panels[k])) for k in ("frame", "hinge", "twist", "sway")]
        Hm = max(im.shape[0] for im in imgs)
        def _padh(im):
            dh = Hm - im.shape[0]; top = dh // 2
            return np.pad(im, ((top, dh - top), (5, 5), (0, 0)), constant_values=1)
        comb = np.concatenate([_padh(im) for im in imgs], axis=1)
        axc.imshow(comb)
        axc.set_title("c  structural modes:  frame (3 axes at clamp) | HINGE (drawbridge) | TWIST (wring towel) | SWAY (pendulum)", fontsize=10, fontweight="bold")
        x = np.arange(N); imgt = [str(k) for k in lk]
        # A: per-residue avg + min-max band (rigid & deform)
        axA = fig.add_subplot(gs[1, 0])
        for k, col, lw in [("hinge", RGB["hinge"], 1.4), ("twist", RGB["twist"], 1.4), ("sway", RGB["sway"], 1.4),
                           ("rigid", "#3B6EA5", 2.4), ("deform", "#C0392B", 2.4)]:
            av = disp[k].mean(0); mn = disp[k].min(0); mx = disp[k].max(0)
            axA.fill_between(x, mn, mx, color=col, alpha=0.09)
            axA.plot(x, av, "-", color=col, lw=lw, label=k)
        axA.axvline(0 if CLAMP[cdr[2:]] == "N" else N - 1, color="k", ls=":", lw=1)
        axA.set_xticks(x); axA.set_xticklabels(imgt, fontsize=7, rotation=90); axA.set_xlabel("loop residue (IMGT)")
        axA.set_ylabel("displacement (Å)"); axA.legend(fontsize=7.5, ncol=2)
        axA.set_title("a  per-residue by mode: average line + min–max band", fontsize=10, fontweight="bold")
        # B: avg vs max per-residue by mode
        axB = fig.add_subplot(gs[1, 1]); labs = ["hinge", "twist", "sway", "rigid", "deform"]; wdt = 0.38
        axB.bar(np.arange(5) - wdt / 2, [rms[k].mean() for k in labs], wdt, color="#888", label="avg/residue")
        axB.bar(np.arange(5) + wdt / 2, [rms[k].max() for k in labs], wdt, color="#333", label="max residue")
        axB.set_xticks(range(5)); axB.set_xticklabels(["hinge", "twist", "sway", "TOT\nrigid", "deform"], fontsize=8)
        axB.set_ylabel("Å RMS displacement"); axB.legend(fontsize=8)
        axB.set_title("b  average vs max per residue (Å)", fontsize=10, fontweight="bold")
        # D: total-Å² budget — total = rigid + deform (basis-free; nothing unexplained). No per-mode bar (they double-count).
        axD = fig.add_subplot(gs[1, 2])
        axD.bar(0, V["rigid"], 0.6, color="#3B6EA5", label="rigid", edgecolor="white")
        axD.bar(0, V["deform"], 0.6, bottom=V["rigid"], color=(0.75, 0.22, 0.17), label="deformation", edgecolor="white")
        axD.plot(0, V["total"], "_", color="k", ms=46, mew=2.5, label="total (check)")
        fr_d = 100 * V["deform"] / V["total"]
        axD.set_xticks([0]); axD.set_xticklabels(["total = rigid + deform"], fontsize=9); axD.set_xlim(-0.7, 0.7)
        axD.set_ylabel("summed displacement variance (Å²)")
        axD.legend(fontsize=8, loc="upper right")
        axD.set_title(f"d  Å² budget: total = rigid + deform (nothing unexplained)\nrigid {V['rigid']:.1f} + deform {V['deform']:.1f} = {V['total']:.1f} Å²  (deform {fr_d:.0f}%)", fontsize=9, fontweight="bold")
        fig.suptitle(f"{sysid} {cdr}: rigid modes + deformation — per-residue, magnitudes, structure, and the Å² budget", y=0.99, fontsize=13, fontweight="bold")
        out = f"{HERE}/figures/composite_{sysid}_{cdr}.png"
        fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
        print(f"{cdr}: total={V['total']:.1f} rigid={V['rigid']:.1f} deform={V['deform']:.1f} Å² -> {out}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "3SKN", a[1] if len(a) > 1 else None)
