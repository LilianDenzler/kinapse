#!/usr/bin/env python
"""Complete rigid-motion characterisation of each CDR in its OWN frame, across all 22 TCRs.
Frame from 3 fixed points (two stem feet + apex):
  c = feet-line (stem-to-stem)   -> rotation about c = tip lifts OUT of plane, feet fixed = clean tether hinge
  n = loop-plane normal          -> rotation about n = in-plane sway (a foot must move)
  h = up-the-loop (feet->apex)   -> rotation about h = twist (feet counter-swing)
Reports: rotational-variance split (f_c/f_n/f_h) + absolute amplitudes (deg);
and the COMPLETENESS check -> clamp-point slide |d| (A) vs tip rigid swing (A) vs deformation (A).
Rotation-3 is the whole rigid story only if |d| is small (clamp holds).  -> results_rotframe.json + rotframe_spread.png
"""
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


def analyze(loop, R):
    T, N, _ = loop.shape
    Rref = R[0] - R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + R[0].mean(0)
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = Lf[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]           # medoid reference
    L0cen = L0.mean(0)
    fN, fC = L0[0], L0[-1]; m = 0.5 * (fN + fC)                          # feet + clamp midpoint
    c = fC - fN; c /= np.linalg.norm(c)                                  # feet-line axis
    perp = (L0 - m) - ((L0 - m) @ c)[:, None] * c
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); apex = L0[ai]     # apex = farthest from feet-line
    h = perp[ai]; h /= np.linalg.norm(h)                                 # up-the-loop axis
    n = np.cross(c, h); n /= np.linalg.norm(n)                           # plane normal
    wc = np.empty(T); wh = np.empty(T); wn = np.empty(T); Edef = 0.0
    L0c = L0 - L0cen
    for t in range(T):
        Q = kabsch(L0c, Lf[t] - Lf[t].mean(0))
        w = Rot.from_matrix(Q).as_rotvec()
        wc[t], wh[t], wn[t] = w @ c, w @ h, w @ n
        Edef += ((Lf[t] - (L0c @ Q.T + Lf[t].mean(0))) ** 2).sum(1).mean()
    vc, vh, vn = np.var(wc), np.var(wh), np.var(wn); tot = vc + vh + vn
    # completeness + clamp asymmetry: framework-relative motion of the two stem feet and the apex (A)
    footN = float(np.sqrt(((Lf[:, 0] - Lf[:, 0].mean(0)) ** 2).sum(1).mean()))
    footC = float(np.sqrt(((Lf[:, -1] - Lf[:, -1].mean(0)) ** 2).sum(1).mean()))
    tip = float(np.sqrt(((Lf[:, ai] - Lf[:, ai].mean(0)) ** 2).sum(1).mean()))
    return dict(amp_c=float(np.degrees(np.sqrt(vc))), amp_h=float(np.degrees(np.sqrt(vh))),
                amp_n=float(np.degrees(np.sqrt(vn))), f_c=float(vc / tot), f_h=float(vh / tot),
                f_n=float(vn / tot), footN=footN, footC=footC, tip=tip, deform=float(np.sqrt(Edef / T)))


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
    json.dump(res, open(f"{HERE}/results_rotframe.json", "w"))
    print(f"\n{len(res)} TCRs -> results_rotframe.json")
    plot(res)


def plot(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    G, O, P = "#2E7D32", "#F0A030", "#7B4FA3"     # c=green(tip-elev), n=orange(sway), h=purple(twist)
    x = np.arange(6)
    M = {c: {k: np.array([res[t][c][k] for t in res]) for k in res[next(iter(res))][c]} for c in CDRS}
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.15, 1.15, 1.25]})
    # (A) rotational-variance split (which axis the rotation is about)
    fc = np.array([np.median(M[c]["f_c"]) for c in CDRS]); fn = np.array([np.median(M[c]["f_n"]) for c in CDRS])
    fh = np.array([np.median(M[c]["f_h"]) for c in CDRS])
    ax[0].bar(x, fc, color=G, label="about ĉ  (tip elevation = tether hinge)")
    ax[0].bar(x, fn, bottom=fc, color=O, label="about n̂  (in-plane sway)")
    ax[0].bar(x, fh, bottom=fc + fn, color=P, label="about ĥ  (twist)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(CDRS, rotation=20, fontsize=9); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("share of rigid rotation (median)")
    ax[0].legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(.5, 1.12), framealpha=0.9)
    ax[0].set_title("Which axis is the rotation about?", y=1.30)
    # (B) absolute rotation amplitude per axis (deg)
    w = 0.26
    for k, col, off, lab in [("amp_c", G, -w, "ĉ tip-elev"), ("amp_n", O, 0, "n̂ sway"), ("amp_h", P, w, "ĥ twist")]:
        ax[1].bar(x + off, [np.median(M[c][k]) for c in CDRS], width=w, color=col, label=lab)
    ax[1].set_xticks(x); ax[1].set_xticklabels(CDRS, rotation=20, fontsize=9)
    ax[1].set_ylabel("rotation amplitude (deg, median)"); ax[1].legend(fontsize=8); ax[1].set_title("Absolute rotation amplitude per axis")
    # (C) completeness + clamp asymmetry: feet vs tip framework-relative motion (Å)
    w4 = 0.2
    for k, col, off, lab in [("footN", "#AAAAAA", -1.5 * w4, "N foot"), ("footC", "#666666", -0.5 * w4, "C foot"),
                             ("tip", "#0072B2", 0.5 * w4, "tip (apex)"), ("deform", "#C0392B", 1.5 * w4, "deformation")]:
        ax[2].bar(x + off, [np.median(M[c][k]) for c in CDRS], width=w4, color=col, label=lab)
    ax[2].set_xticks(x); ax[2].set_xticklabels(CDRS, rotation=20, fontsize=9)
    ax[2].set_ylabel("framework-relative motion, Å (median)"); ax[2].legend(fontsize=8)
    ax[2].set_title("Completeness + clamp asymmetry:\nfeet still & tip swings ⇒ anchored rotation")
    fig.suptitle("Complete rigid-motion characterisation of each CDR in its own frame (22 TCRs): rotation-3 + clamp slide", y=1.05, fontsize=13)
    out = f"{HERE}/figures/rotframe_spread.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)
    print("median share about ĉ (tether hinge):", {c: round(float(np.median(M[c]["f_c"])), 2) for c in CDRS})
    print("median foot motion N/C (A):", {c: (round(float(np.median(M[c]["footN"])), 2), round(float(np.median(M[c]["footC"])), 2)) for c in CDRS})
    print("median tip motion (A):", {c: round(float(np.median(M[c]["tip"])), 2) for c in CDRS})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(json.load(open(f"{HERE}/results_rotframe.json")))
    else:
        main()
