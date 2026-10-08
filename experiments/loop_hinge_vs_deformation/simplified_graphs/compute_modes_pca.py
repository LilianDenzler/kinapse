#!/usr/bin/env python
"""Data-derived collective motions of a CDR loop (essential dynamics), two ways:
 (1) ALIGNMENT-FREE CA-CA distance PCA -> deformation-only spectrum + effective INTERNAL DOF (participation ratio).
     (pairwise distances are rigid-invariant, so this isolates internal shape change.)
 (2) framework-referenced Cartesian PCA (loop NOT aligned -> rigid swing preserved) -> 3D-visualisable modes.
Each Cartesian mode is tagged with its RIGID fraction (projection on the 6-D rigid group at the mean) and its
hinge/twist/sway character. The top (softest, largest-amplitude) modes are drawn as ±amplitude fans (blue->red).
-> figures/modes_pca_<sys>_<CDR>.png + printed spectrum."""
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
NC = 7; KMODES = 3; VIS_A = 3.0


def geom_axes(loop):
    T, N, _ = loop.shape
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    fN, fC = L0[0], L0[-1]; m = 0.5 * (fN + fC); cc = fC - fN; cc /= np.linalg.norm(cc)
    perp = (L0 - m) - ((L0 - m) @ cc)[:, None] * cc
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(cc, h); n /= np.linalg.norm(n)
    return {"hinge": cc, "twist": h, "sway": n}


def pca_cart(loop):
    T, N, _ = loop.shape
    X = loop.reshape(T, 3 * N); mean = X.mean(0); Xc = X - mean
    C = (Xc.T @ Xc) / (T - 1)
    lam, V = np.linalg.eigh(C); idx = np.argsort(lam)[::-1]
    return mean.reshape(N, 3), lam[idx], V[:, idx]


def dist_spectrum(loop):
    T, N, _ = loop.shape; i, j = np.triu_indices(N, 1)
    d = np.linalg.norm(loop[:, i] - loop[:, j], axis=-1); dc = d - d.mean(0)
    C = (dc.T @ dc) / (T - 1)
    return np.sort(np.linalg.eigvalsh(C))[::-1]


def participation(lam):
    lam = lam[lam > 1e-9]
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def rigid_basis(mean, piv):
    N = len(mean); cols = []
    for e in np.eye(3):
        cols.append(np.tile(e, N))                                          # translations
    for e in np.eye(3):
        cols.append(np.cross(np.broadcast_to(e, (N, 3)), mean - piv).reshape(-1))  # rotations about piv
    Q, _ = np.linalg.qr(np.array(cols).T)
    return Q                                                                # 3N x 6 orthonormal


def hts_shares(v, mean, piv, axv):
    s = {}
    for k, u in axv.items():
        g = np.cross(np.broadcast_to(u, (len(mean), 3)), mean - piv).reshape(-1); g /= np.linalg.norm(g)
        s[k] = float((v @ g) ** 2)
    tot = sum(s.values()) + 1e-12
    return {k: s[k] / tot for k in s}


def pdb_ca(path, confs):
    ch = "ABCDEFGHIJKLMNOPQRSTUVWX"
    with open(path, "w") as f:
        for ci, X in enumerate(confs):
            for k, (x, y, z) in enumerate(X):
                f.write(f"ATOM  {k+1:5d}  CA  GLY {ch[ci]}{k+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("TER\n")
        f.write("END\n")


def montage_modes(sysid, cdr, mean, V, lam, top, top_char, top_frig, top_var, top_amp, supr, top_slice, dom):
    N = len(mean)
    for mi, m in enumerate(top):
        mv = V[:, m].reshape(N, 3); mx = np.linalg.norm(mv, axis=1).max() + 1e-9
        amp = VIS_A / mx
        pdb_ca(f"{HERE}/figures/_mp_mode{mi}.pdb", [mean + a * mv for a in np.linspace(-amp, amp, NC)])
    pdb_ca(f"{HERE}/figures/_mp_mean.pdb", [mean])
    md.Trajectory(supr / 10.0, top_slice).atom_slice(dom).save_pdb(f"{HERE}/figures/_mp_dom.pdb") if False else None
    import pymol2
    panels = []
    with pymol2.PyMOL() as P:
        cmd = P.cmd
        cmd.load(f"{HERE}/figures/_mp_dom.pdb", "domain"); cmd.hide("everything"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
        cmd.show("cartoon", "domain"); cmd.color("grey90", "domain"); cmd.set("cartoon_transparency", 0.78, "domain")
        cmd.load(f"{HERE}/figures/_mp_mean.pdb", "mean0"); cmd.hide("everything", "mean0")
        for j in range(N - 1):
            cmd.bond(f"mean0 and resi {j+1}", f"mean0 and resi {j+2}")
        cmd.show("sticks", "mean0"); cmd.set("stick_radius", 0.45, "mean0"); cmd.color("grey60", "mean0")
        for mi in range(len(top)):
            obj = f"mode{mi}"; cmd.load(f"{HERE}/figures/_mp_mode{mi}.pdb", obj); cmd.hide("everything", obj)
            for ci in range(NC):
                X = "ABCDEFGHIJKLMNOPQRSTUVWX"[ci]
                for j in range(N - 1):
                    cmd.bond(f"{obj} and chain {X} and resi {j+1}", f"{obj} and chain {X} and resi {j+2}")
            cmd.show("sticks", obj); cmd.set("stick_radius", 0.22, obj); cmd.spectrum("chain", "blue_white_red", obj)
        cmd.set("all_states", 1); cmd.orient("mean0"); cmd.zoom("mean0", 9); cmd.turn("y", 25); cmd.turn("x", 10)
        for mi in range(len(top)):
            for o in ["mean0"] + [f"mode{k}" for k in range(len(top))]:
                cmd.disable(o)
            cmd.enable("mean0"); cmd.enable(f"mode{mi}")
            p = f"{HERE}/figures/_mp_panel{mi}.png"; cmd.ray(820, 780); cmd.png(p, dpi=130); panels.append(p)
    return panels


def main(sysid="3SKN", cdr="B_CDR3"):
    tv, xyz, imap = load_md(sysid); top_slice = tv.mdtraj.topology
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
    fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
    supr = superpose_all(xyz, fw)
    lk = sorted(k for k in imap[ch] if lo <= k <= hi)
    loop = supr[:, np.array([imap[ch][k] for k in lk])]; T, N, _ = loop.shape
    mean, lam, V = pca_cart(loop)
    axv = geom_axes(loop)
    piv = mean[-1] if CLAMP[cdr[2:]] == "C" else mean[0]
    Q6 = rigid_basis(mean, piv)
    lam_d = dist_spectrum(loop)
    PR_c = participation(lam); PR_d = participation(lam_d)
    tot = lam[lam > 0].sum()
    top = list(range(min(KMODES, 3 * N)))
    top_frig = [float((Q6.T @ V[:, m]) @ (Q6.T @ V[:, m])) for m in top]
    top_char = [hts_shares(V[:, m], mean, piv, axv) for m in top]
    top_var = [100 * lam[m] / tot for m in top]
    top_amp = [float(np.sqrt(lam[m])) for m in top]                          # RMS amplitude of the mode (Å)
    print(f"\n{sysid} {cdr}  (N={N} CA, T={T} frames)")
    print(f"  CARTESIAN (framework-ref, rigid preserved): eff. DOF (participation ratio) = {PR_c:.2f}")
    print(f"  CA-CA DISTANCE (alignment-free, deformation-only): eff. INTERNAL DOF = {PR_d:.2f}")
    print(f"  {'mode':5}{'%var':>7}{'RMS(Å)':>8}{'rigid%':>8}   hinge/twist/sway")
    for mi, m in enumerate(top):
        c = top_char[mi]
        print(f"  {mi+1:<5}{top_var[mi]:7.1f}{top_amp[mi]:8.2f}{100*top_frig[mi]:8.0f}   {c['hinge']:.2f}/{c['twist']:.2f}/{c['sway']:.2f}")
    print(f"  cumulative %var of top {len(top)}: {sum(top_var):.0f}%", flush=True)
    dom = np.asarray(tv.domain_idx([f"{ch}_variable"]))
    # write domain pdb (need real coords)
    md.Trajectory(supr[0:1] / 10.0, top_slice).atom_slice(dom).save_pdb(f"{HERE}/figures/_mp_dom.pdb")
    panels = montage_modes(sysid, cdr, mean, V, lam, top, top_char, top_frig, top_var, top_amp, supr[0], top_slice, dom)
    plot(sysid, cdr, lam, lam_d, tot, top, top_var, top_frig, top_char, top_amp, PR_c, PR_d, panels, N)


def plot(sysid, cdr, lam, lam_d, tot, top, top_var, top_frig, top_char, top_amp, PR_c, PR_d, panels, N):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.image as mpimg
    from matplotlib import gridspec
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 1.0], hspace=0.3, wspace=0.28)
    # top: mode fans
    axm = fig.add_subplot(gs[0, :]); axm.axis("off")
    imgs = [mpimg.imread(p) for p in panels]
    comb = np.concatenate([np.pad(im, ((0, 0), (6, 6), (0, 0)), constant_values=1) for im in imgs], axis=1)
    axm.imshow(comb)
    lab = " | ".join([f"MODE {mi+1}: {top_var[mi]:.0f}% var, {100*top_frig[mi]:.0f}% rigid "
                      f"({max(top_char[mi],key=top_char[mi].get)})" for mi in range(len(top))])
    axm.set_title(f"top (softest) data-derived modes — mean loop grey, ±amplitude fan blue→red\n{lab}", fontsize=10, fontweight="bold")
    # A: Cartesian scree colored by rigid fraction
    axA = fig.add_subplot(gs[1, 0])
    K = min(10, len(lam)); pv = 100 * lam[:K] / tot
    fr = []
    for m in range(K):
        fr.append(top_frig[m] if m < len(top_frig) else np.nan)
    cols = plt.cm.RdYlBu([f if np.isfinite(f) else 0.5 for f in [top_frig[m] if m < len(top) else np.nan for m in range(K)]])
    bars = axA.bar(np.arange(1, K + 1), pv, color="#4C72B0")
    for mi in range(min(len(top), K)):
        axA.text(mi + 1, pv[mi] + 1, f"{100*top_frig[mi]:.0f}%\nrigid", ha="center", va="bottom", fontsize=7, color="#B0300F")
    axA.set_xlabel("mode (Cartesian, framework-ref)"); axA.set_ylabel("% of variance")
    axA.set_title(f"a  Cartesian spectrum (rigid preserved)\neff. DOF (PR) = {PR_c:.2f}", fontsize=9, fontweight="bold")
    axA.set_xticks(range(1, K + 1))
    # B: alignment-free distance scree (deformation-only)
    axB = fig.add_subplot(gs[1, 1])
    Kd = min(10, len(lam_d)); totd = lam_d[lam_d > 0].sum()
    axB.bar(np.arange(1, Kd + 1), 100 * lam_d[:Kd] / totd, color="#C0392B")
    axB.set_xlabel("mode (CA-CA distance, alignment-free)"); axB.set_ylabel("% of distance variance")
    axB.set_title(f"b  DEFORMATION-only spectrum (alignment-free)\neff. INTERNAL DOF = {PR_d:.2f}", fontsize=9, fontweight="bold")
    axB.set_xticks(range(1, Kd + 1))
    # C: mode character table as text
    axC = fig.add_subplot(gs[1, 2]); axC.axis("off")
    lines = [f"{'mode':5}{'%var':>6}{'RMS Å':>7}{'rigid':>7}  h / t / s", "-" * 44]
    for mi in range(len(top)):
        c = top_char[mi]
        lines.append(f"{mi+1:<5}{top_var[mi]:6.1f}{top_amp[mi]:7.2f}{100*top_frig[mi]:6.0f}%  {c['hinge']:.2f}/{c['twist']:.2f}/{c['sway']:.2f}")
    lines += ["", f"Cartesian eff. DOF (PR)   = {PR_c:.2f}", f"Deformation eff. DOF (PR) = {PR_d:.2f}",
              "", "rigid% = projection of the mode", "onto the 6-D rigid group at the mean.",
              "PR = (Σλ)²/Σλ² = effective # of modes."]
    axC.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    axC.set_title("c  mode character", fontsize=9, fontweight="bold", loc="left")
    fig.suptitle(f"{sysid} {cdr}: data-derived collective motions — is the dominant mode a rigid swing or deformation?",
                 y=1.0, fontsize=12, fontweight="bold")
    out = f"{HERE}/figures/modes_pca_{sysid}_{cdr}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "3SKN", a[1] if len(a) > 1 else "B_CDR3")
