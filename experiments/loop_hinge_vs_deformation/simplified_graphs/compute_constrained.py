#!/usr/bin/env python
"""Constrained clamp-rotation split: is CDR rigid motion GENUINE anchored rotation, or is deformation
leaking into the free rigid fit? For each CDR/TCR compare two models:
  FREE     = full 6-DOF Kabsch (rotation about centroid + translation)   -> deform_free (current)
  CLAMP    = rotation ONLY about the fixed clamp point m (3 DOF, no translation) -> deform_clamp (stricter)
Decompose BOTH rotations into loop-frame axes (ĉ/ĥ/n̂); report whether twist (f_h) survives the constraint.
Also: per-frame coupling corr(|twist|, deformation) and corr(|hinge|, deformation) — high twist-coupling
means the 'twist' tracks shape change (artifact-like), not independent rigid motion.
-> results_constrained.json + constrained_split.png"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]


def rot_fit(A, B):
    """rotation R minimising |R A_i - B_i| (A,B relative to a common origin); returns R, residual sum-sq."""
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    res = ((A @ R.T - B) ** 2).sum()
    return R, res


def analyze(loop, R):
    T, N, _ = loop.shape
    Rref = R[0] - R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + R[0].mean(0)
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = Lf[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    L0cen = L0.mean(0); fN, fC = L0[0], L0[-1]; m = 0.5 * (fN + fC)
    c = fC - fN; c /= np.linalg.norm(c)
    perp = (L0 - m) - ((L0 - m) @ c)[:, None] * c
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(c, h); n /= np.linalg.norm(n)
    Lbar = Lf.mean(0)
    wf = np.empty((T, 3)); wc = np.empty((T, 3)); Ef = np.empty(T); Ec = np.empty(T)
    A_f = L0 - L0cen; A_c = L0 - m
    for t in range(T):
        Rf, ef = rot_fit(A_f, Lf[t] - Lf[t].mean(0)); Ef[t] = ef
        Rc, ec = rot_fit(A_c, Lf[t] - m); Ec[t] = ec
        wf[t] = Rot.from_matrix(Rf).as_rotvec(); wc[t] = Rot.from_matrix(Rc).as_rotvec()

    def split(w):
        pc, ph, pn = w @ c, w @ h, w @ n
        vc, vh, vn = np.var(pc), np.var(ph), np.var(pn); tot = vc + vh + vn
        return float(vc / tot), float(vh / tot), float(vn / tot), (pc, ph, pn)

    fcf, fhf, fnf, (pcf, phf, pnf) = split(wf)
    fcc, fhc, fnc, _ = split(wc)
    Dtot = float(np.sqrt(((Lf - Lbar) ** 2).sum(-1).mean()))
    deff = float(np.sqrt(Ef.mean() / N)); defc = float(np.sqrt(Ec.mean() / N))
    dperframe = np.sqrt(Ef / N)
    cpl_h = float(np.corrcoef(np.abs(phf - phf.mean()), dperframe)[0, 1])   # twist vs deformation
    cpl_c = float(np.corrcoef(np.abs(pcf - pcf.mean()), dperframe)[0, 1])   # hinge vs deformation
    return dict(f_c=fcf, f_h=fhf, f_n=fnf, fc_c=fcc, fc_h=fhc, fc_n=fnc,
                deform_free=deff, deform_clamp=defc, Dtot=Dtot, cpl_twist=cpl_h, cpl_hinge=cpl_c)


def main():
    tcrs = [os.path.basename(f)[:4] for f in sorted(glob.glob(f"{HERE}/results_swing/*.npz"))]
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    res = {}
    for t in tcrs:
        try:
            tv, xyz, imap = load_md(t)
            row = {}
            for cdr in CDRS:
                ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
                lk = sorted(k for k in imap[ch] if lo <= k <= hi)
                loop = xyz[:, np.array([imap[ch][k] for k in lk])]
                Rr = xyz[:, np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])]
                row[cdr] = analyze(loop, Rr)
            res[t] = row; print(f"{t} ok", flush=True)
        except Exception as e:
            print(f"{t} ERR {type(e).__name__}: {str(e)[:90]}", flush=True)
    json.dump(res, open(f"{HERE}/results_constrained.json", "w"))
    print(f"\n{len(res)} TCRs -> results_constrained.json")
    plot(res)


def plot(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res); x = np.arange(6); w = 0.38
    M = {c: {k: np.array([res[t][c][k] for t in tcrs]) for k in res[tcrs[0]][c]} for c in CDRS}
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    # (A) twist share: free vs constrained
    ax[0].bar(x - w / 2, [np.median(M[c]["f_h"]) for c in CDRS], w, color="#7B4FA3", label="FREE fit")
    ax[0].bar(x + w / 2, [np.median(M[c]["fc_h"]) for c in CDRS], w, color="#B79BD0", label="CLAMP-constrained")
    ax[0].set_xticks(x); ax[0].set_xticklabels(CDRS, rotation=20, fontsize=9); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("twist share (ĥ) of rigid rotation"); ax[0].legend(fontsize=8.5)
    ax[0].set_title("a  Does the twist survive the constraint?", loc="left", fontsize=10, fontweight="bold")
    # (B) deformation: free vs clamp-constrained (stricter)
    ax[1].bar(x - w / 2, [np.median(M[c]["deform_free"]) for c in CDRS], w, color="#C0392B", label="deform (free rigid)")
    ax[1].bar(x + w / 2, [np.median(M[c]["deform_clamp"]) for c in CDRS], w, color="#E8998D", label="deform (clamp-only)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(CDRS, rotation=20, fontsize=9)
    ax[1].set_ylabel("deformation (Å, median)"); ax[1].legend(fontsize=8.5)
    ax[1].set_title("b  Stricter split: deformation the free fit hid", loc="left", fontsize=10, fontweight="bold")
    # (C) per-frame coupling of each mode to deformation
    ax[2].bar(x - w / 2, [np.median(M[c]["cpl_twist"]) for c in CDRS], w, color="#7B4FA3", label="twist–deform corr")
    ax[2].bar(x + w / 2, [np.median(M[c]["cpl_hinge"]) for c in CDRS], w, color="#2E7D32", label="hinge–deform corr")
    ax[2].axhline(0, color="k", lw=.6); ax[2].set_xticks(x); ax[2].set_xticklabels(CDRS, rotation=20, fontsize=9)
    ax[2].set_ylabel("per-frame corr with deformation"); ax[2].legend(fontsize=8.5)
    ax[2].set_title("c  Is the mode coupled to shape change?\n(high twist-coupling ⇒ artifact-like)", loc="left", fontsize=10, fontweight="bold")
    fig.suptitle("Constrained clamp-rotation split: is CDR3's twist genuine rigid motion or deformation leakage? (22 TCRs)", y=1.03, fontsize=12)
    out = f"{HERE}/figures/constrained_split.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)
    for c in CDRS:
        print(f"  {c}: twist FREE={np.median(M[c]['f_h']):.2f} CLAMP={np.median(M[c]['fc_h']):.2f} | "
              f"deform free/clamp={np.median(M[c]['deform_free']):.2f}/{np.median(M[c]['deform_clamp']):.2f} | "
              f"twist-cpl={np.median(M[c]['cpl_twist']):+.2f} hinge-cpl={np.median(M[c]['cpl_hinge']):+.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(json.load(open(f"{HERE}/results_constrained.json")))
    else:
        main()
