#!/usr/bin/env python
"""Real Å displacements for the modes shown in modes_clear (default 3SKN B_CDR3).
For each mode (hinge ĉ / twist ĥ / sway n̂, rotation about the CLAMP by the REAL per-frame angle), plus the
TOTAL rigid motion (full robust rigid fit) and the DEFORMATION (post-fit residual):
  per-residue RMS displacement (Å), and the loop TOTAL (Σ over residues), AVERAGE (per residue), MAX (per residue).
NOTE: the three modes do NOT sum to total-rigid (cross-terms); total-rigid is the honest complete number.
-> figures/angstroms_<sys>_<CDR>.png + printed table"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all

CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1)
        c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


def rot_about(P, u, piv, th):
    c, s = np.cos(th), np.sin(th); X = P - piv
    return piv + c * X + s * np.cross(np.broadcast_to(u, X.shape), X) + (1 - c) * ((X @ u)[:, None] * u)


def main(sysid="3SKN", cdr="B_CDR3"):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
    fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
    supr = superpose_all(xyz, fw)
    lk = sorted(k for k in imap[ch] if lo <= k <= hi)
    loop = supr[:, np.array([imap[ch][k] for k in lk])]; T, N, _ = loop.shape
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1))); L0 = loop[med]
    fN, fC = L0[0], L0[-1]; piv = fC if CLAMP[cdr[2:]] == "C" else fN
    m = 0.5 * (fN + fC); c = fC - fN; c /= np.linalg.norm(c)
    perp = (L0 - m) - ((L0 - m) @ c)[:, None] * c
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(c, h); n /= np.linalg.norm(n)
    axv = {"hinge": c, "twist": h, "sway": n}
    acc = {k: np.zeros(N) for k in ("hinge", "twist", "sway", "rigid", "deform")}
    for t in range(T):
        R, Pc, Qc = robust_kabsch(L0, loop[t]); w = Rot.from_matrix(R).as_rotvec()
        for k, u in axv.items():
            acc[k] += ((rot_about(L0, u, piv, w @ u) - L0) ** 2).sum(1)          # mode-k displacement about clamp, real angle
        rigid = (L0 - Pc) @ R.T + Qc                                              # full rigid placement of L0
        acc["rigid"] += ((rigid - L0) ** 2).sum(1)
        acc["deform"] += ((loop[t] - rigid) ** 2).sum(1)
    prof = {k: np.sqrt(acc[k] / T) for k in acc}                                  # per-residue RMS (Å)
    order = ["hinge", "twist", "sway", "rigid", "deform"]
    print(f"\n{sysid} {cdr}  (N={N} loop residues; Å RMS displacement)")
    print(f"{'mode':10}{'TOTAL(Σ)':>10}{'avg/res':>9}{'max/res':>9}{'  max at res':>13}")
    imgt = [str(k) for k in lk]
    for k in order:
        p = prof[k]; mi = int(np.argmax(p))
        lab = {"hinge": "hinge(ĉ)", "twist": "twist(ĥ)", "sway": "sway(n̂)", "rigid": "TOTAL rigid", "deform": "deformation"}[k]
        print(f"{lab:10}{p.sum():10.2f}{p.mean():9.2f}{p.max():9.2f}{('#'+str(mi)+' ('+imgt[mi]+')'):>13}")
    print("  (note: hinge+twist+sway do NOT sum to TOTAL rigid — cross-terms; TOTAL rigid is the complete number)")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.arange(N)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    sty = {"hinge": ("#2E7D32", "-o"), "twist": ("#7B4FA3", "-o"), "sway": ("#F0A030", "-o"),
           "rigid": ("#3B6EA5", "-s"), "deform": ("#C0392B", "--D")}
    for k in order:
        ax[0].plot(x, prof[k], sty[k][1], color=sty[k][0], ms=4, lw=2 if k in ("rigid", "deform") else 1.4,
                   label={"hinge": "hinge (ĉ)", "twist": "twist (ĥ)", "sway": "sway (n̂)", "rigid": "TOTAL rigid", "deform": "deformation"}[k])
    ax[0].axvline(0 if CLAMP[cdr[2:]] == "N" else N - 1, color="k", ls=":", lw=1)
    ax[0].text(0 if CLAMP[cdr[2:]] == "N" else N - 1, ax[0].get_ylim()[1], " clamp", fontsize=8, va="top")
    ax[0].set_xticks(x); ax[0].set_xticklabels(imgt, fontsize=8); ax[0].set_xlabel("loop residue (IMGT)")
    ax[0].set_ylabel("per-residue RMS displacement (Å)"); ax[0].legend(fontsize=9)
    ax[0].set_title(f"a  {sysid} {cdr}: per-residue Å displacement by mode", fontsize=10, fontweight="bold")
    # bar of totals (avg and max per residue)
    labs = ["hinge", "twist", "sway", "rigid", "deform"]; w = 0.38
    ax[1].bar(np.arange(5) - w / 2, [prof[k].mean() for k in labs], w, color="#888", label="average / residue")
    ax[1].bar(np.arange(5) + w / 2, [prof[k].max() for k in labs], w, color="#333", label="max residue")
    ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["hinge", "twist", "sway", "TOTAL\nrigid", "deform"], fontsize=8)
    ax[1].set_ylabel("Å RMS displacement"); ax[1].legend(fontsize=9)
    ax[1].set_title("b  average vs max per-residue (Å)", fontsize=10, fontweight="bold")
    fig.suptitle(f"{sysid} {cdr}: real Å displacements of each rigid mode + total rigid + deformation "
                 "(modes don't sum to total — cross-terms)", y=1.02, fontsize=11)
    out = f"{HERE}/figures/angstroms_{sysid}_{cdr}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "3SKN", a[1] if len(a) > 1 else "B_CDR3")
