#!/usr/bin/env python
"""DECOUPLED rigid-vs-deform, per frame, for one TCR — the clean version.
For each (sub-sampled) frame vs the medoid reference:
  DEFORM  = RMS deviation of the loop's internal CA-CA distances d_LL from the reference (alignment-free, fit-free).
  RIGID   = fluctuation-ROBUST Kabsch (ref->frame): iteratively down-weights high-residual (deforming) atoms so the
            rigid transform is anchored on the rigid core and CANNOT be contaminated by deformation. Save ω=rotvec.
Then PCA the saved ω vectors -> common/collective rigid rotation modes. Rotation is well-defined; translation is
pivot-dependent so we save it but analyse ω. -> results_decoupled_<sys>.npz + figures/decoupled_<sys>.png"""
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

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
NSUB = 200


def robust_kabsch(P, Q, iters=6):
    """Rotation R mapping P->Q, robust to a deforming subset (Cauchy reweighting down-weights high-residual atoms)."""
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1)
        c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, w


def main(sysid="3SKN"):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    rng = np.random.default_rng(0)
    out = {}
    print(f"{sysid}  decoupled rigid(robust Kabsch)  vs  deform(d_LL)")
    print(f"{'CDR':8}{'defRMS(Å)':>10}{'rot|ω|(°)':>10}{'corr(def,rot)':>14}{'P1(ω-PCA)':>11}{'domAxis: c/h/n':>18}")
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
        supr = superpose_all(xyz, fw)
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = supr[:, np.array([imap[ch][k] for k in lk])]
        T, N, _ = loop.shape
        ii, jj = np.triu_indices(N, 1)
        dLL = np.linalg.norm(loop[:, ii] - loop[:, jj], axis=-1)                    # (T, pairs)
        med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1))); L0 = loop[med]; dref = dLL[med]
        # loop-frame axes (for interpreting the PCA dominant axis)
        fN, fC = L0[0], L0[-1]; m = 0.5 * (fN + fC)
        c = fC - fN; c /= np.linalg.norm(c)
        perp = (L0 - m) - ((L0 - m) @ c)[:, None] * c
        ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
        n = np.cross(c, h); n /= np.linalg.norm(n)
        idx = rng.choice(T, min(NSUB, T), replace=False)
        omega = np.empty((len(idx), 3)); transl = np.empty((len(idx), 3)); deform = np.empty(len(idx))
        for s, t in enumerate(idx):
            deform[s] = np.sqrt(((dLL[t] - dref) ** 2).mean())                      # d_LL deviation (clean deform)
            R, w = robust_kabsch(L0, loop[t])
            omega[s] = Rot.from_matrix(R).as_rotvec()
            transl[s] = loop[t].mean(0) - R @ L0.mean(0)
        rotmag = np.degrees(np.linalg.norm(omega, axis=1))
        corr = float(np.corrcoef(deform, rotmag)[0, 1])
        C = np.cov(omega.T); w_, V = np.linalg.eigh(C); P1 = float(w_[-1] / w_.sum())
        dax = V[:, -1]                                                              # dominant collective rotation axis
        frac = np.array([abs(dax @ c), abs(dax @ h), abs(dax @ n)]); frac /= frac.sum()
        out[cdr] = dict(omega=omega, transl=transl, deform=deform, rotmag=rotmag,
                        axes=np.array([c, h, n]), pca_eig=w_, pca_vec=V, med_frame=med)
        print(f"{cdr:8}{np.median(deform):9.2f}{np.median(rotmag):10.1f}{corr:14.2f}{P1:11.2f}   {frac[0]:.2f}/{frac[1]:.2f}/{frac[2]:.2f}")
    np.savez(f"{HERE}/results_decoupled_{sysid}.npz", **{f"{c}_{k}": out[c][k] for c in CDRS for k in out[c]})
    print(f"\nsaved per-frame ω/transl/deform -> results_decoupled_{sysid}.npz")
    plot(sysid, out)


def plot(sysid, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 6, figsize=(20, 7))
    for j, cdr in enumerate(CDRS):
        o = out[cdr]
        a = ax[0, j]
        a.scatter(o["deform"], o["rotmag"], s=10, alpha=0.5, color="#3B6EA5")
        r = np.corrcoef(o["deform"], o["rotmag"])[0, 1]
        a.set_title(f"{cdr}\ncorr(def,rot)={r:+.2f}", fontsize=9)
        a.set_xlabel("deform d_LL (Å)", fontsize=8); a.set_ylabel("rigid |ω| (°)", fontsize=8) if j == 0 else None
        b = ax[1, j]
        eig = o["pca_eig"][::-1]; eig = eig / eig.sum()
        b.bar(range(3), eig, color=["#2E7D32", "#7B4FA3", "#F0A030"])
        b.set_xticks(range(3)); b.set_xticklabels(["PC1", "PC2", "PC3"], fontsize=8)
        b.set_ylim(0, 1); b.set_title(f"ω-PCA (P1={eig[0]:.2f})", fontsize=9)
        if j == 0:
            b.set_ylabel("rotation variance share", fontsize=8)
    fig.suptitle(f"{sysid}: DECOUPLED per-frame — top row: is rigid rotation independent of deformation?  "
                 "bottom row: PCA of the rigid rotation vectors ω (collective modes)", y=1.02, fontsize=12)
    out_png = f"{HERE}/figures/decoupled_{sysid}.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out_png)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3SKN")
