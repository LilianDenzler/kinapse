#!/usr/bin/env python
"""Loop-ONLY elastic network model with the FEET ANCHORED — does an elastic model predict the MD deformation?
Nodes = loop CA only. Springs: distance cutoff Rc (captures backbone neighbours + hairpin cross-contacts).
The two terminal residues (feet) are FIXED -> diagonalize the interior block H_ii (anchoring enters as restoring
forces to the pinned feet; no zero modes). Predicted per-residue deformation = diag of H_ii^{-1}.
Compare to the MD deformation profile (per-residue RMS of ‖loop - rigid_fit‖). Sequence-BLIND springs, so a good match
⇒ deformation is set by loop geometry+length+anchoring; a poor match ⇒ sequence-specific chemistry on top.
-> figures/anm_loop_<sys>.png + printed correlations. (Intrinsic deformability only — no framework packing support.)"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import pearsonr
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
RC_LIST = [8.0, 10.0, 13.0]


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1); c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


def hessian(X, Rc, gamma=1.0):
    N = len(X); H = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for j in range(i + 1, N):
            d = X[j] - X[i]; r = np.linalg.norm(d)
            if r <= Rc:
                e = d / r; k = gamma * np.outer(e, e)
                H[3*i:3*i+3, 3*j:3*j+3] -= k; H[3*j:3*j+3, 3*i:3*i+3] -= k
                H[3*i:3*i+3, 3*i:3*i+3] += k; H[3*j:3*j+3, 3*j:3*j+3] += k
    return H


def anm_profile(L0, Rc):
    """feet (res 0 and N-1) fixed; return per-residue predicted DEFORMATION (0 at feet).
    Fixing 2 point-feet leaves ONE zero mode = rigid rotation about the feet-line (= the hinge); we drop it,
    so the predicted fluctuation is pure internal deformation (the nonzero modes). Mirrors the MD rigid/deform split."""
    N = len(L0); H = hessian(L0, Rc)
    interior = list(range(1, N - 1))
    idx = np.array([3 * k + a for k in interior for a in range(3)])
    Hii = H[np.ix_(idx, idx)]
    w, V = np.linalg.eigh(Hii)
    keep = w > 1e-6 * w.max()                                  # drop hinge zero-mode + numerical negatives
    Cinv = (V[:, keep] / w[keep]) @ V[:, keep].T              # pseudo-inverse covariance over deformation modes
    prof = np.zeros(N)
    for m, k in enumerate(interior):
        prof[k] = np.trace(Cinv[3*m:3*m+3, 3*m:3*m+3])        # predicted mean-square deformation
    return np.sqrt(np.maximum(prof, 0)), interior             # RMS


def md_deform_profile(loop):
    T, N, _ = loop.shape
    ii, jj = np.triu_indices(N, 1); dLL = np.linalg.norm(loop[:, ii] - loop[:, jj], axis=-1)
    L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    acc = np.zeros(N)
    for t in range(T):
        R, Pc, Qc = robust_kabsch(L0, loop[t]); rig = (L0 - Pc) @ R.T + Qc
        acc += ((loop[t] - rig) ** 2).sum(1)
    return np.sqrt(acc / T), L0                                # per-residue RMS deformation (Å)


def main(sysid="3SKN", cdrs=("B_CDR1", "B_CDR3")):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json")); cache = {}
    results = {}
    print(f"\n{sysid}: loop-only feet-anchored ENM vs MD deformation profile")
    for cdr in cdrs:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        if ch not in cache:
            fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
            cache[ch] = superpose_all(xyz, fw)
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = cache[ch][:, np.array([imap[ch][k] for k in lk])]; N = loop.shape[1]
        md, L0 = md_deform_profile(loop)
        row = {"lk": lk, "md": md, "anm": {}, "r": {}}
        for Rc in RC_LIST:
            anm, interior = anm_profile(L0, Rc)
            # correlate over interior residues (feet are 0 in ENM by construction)
            r, p = pearsonr(md[interior], anm[interior]) if len(interior) >= 3 else (np.nan, np.nan)
            row["anm"][Rc] = anm; row["r"][Rc] = (r, p)
            print(f"  {cdr:8} N={N:2d} interior={len(interior):2d}  Rc={Rc:>4}Å  Pearson(ENM,MD interior)={r:+.2f} (p={p:.2f})", flush=True)
        results[cdr] = row
    plot(sysid, results)


def plot(sysid, results):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cdrs = list(results); fig, axes = plt.subplots(1, len(cdrs), figsize=(7.5 * len(cdrs), 5.2))
    if len(cdrs) == 1:
        axes = [axes]
    for ax, cdr in zip(axes, cdrs):
        row = results[cdr]; lk = row["lk"]; x = np.arange(len(lk)); md = row["md"]
        axr = ax.twinx()
        ax.plot(x, md, "-D", color="#C0392B", lw=2.4, label="MD deformation (Å)")
        for Rc, st in zip(RC_LIST, ["-o", "-s", "-^"]):
            anm = row["anm"][Rc]; r = row["r"][Rc][0]
            an = anm / (anm.max() + 1e-9)                      # normalise shape (arbitrary ENM units)
            axr.plot(x, an, st, color={8.0: "#2E7D32", 10.0: "#3B6EA5", 13.0: "#7B4FA3"}[Rc], lw=1.6, ms=4,
                     alpha=0.85, label=f"ENM Rc={Rc:.0f} (r={r:+.2f})")
        ax.set_xticks(x); ax.set_xticklabels([str(k) for k in lk], fontsize=7, rotation=90)
        ax.set_xlabel("loop residue (IMGT)"); ax.set_ylabel("MD deformation (Å)", color="#C0392B")
        axr.set_ylabel("ENM predicted (normalised)", color="#3B6EA5")
        ax.set_title(f"{sysid} {cdr}: feet-anchored loop ENM vs MD deformation", fontsize=10, fontweight="bold")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center")
    fig.suptitle("Does a sequence-blind, feet-anchored loop ENM predict the MD deformation PROFILE? (feet fixed ⇒ 0 there)", y=1.02, fontsize=11, fontweight="bold")
    out = f"{HERE}/figures/anm_loop_{sysid}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "3SKN", tuple(a[1:]) if len(a) > 1 else ("B_CDR1", "B_CDR3"))
