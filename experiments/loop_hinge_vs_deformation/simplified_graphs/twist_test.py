#!/usr/bin/env python
"""Is the 'twist' a RIGID rotation or a DEFORMATION (corkscrew)?  Decisive test:
measure each loop residue's rotation angle Δφ about the ĥ (up-loop) axis, relative to the medoid.
  RIGID twist   -> every residue rotates by the SAME Δφ (coherent; whole loop turns as one)
  CORKSCREW     -> Δφ varies across residues (incoherent; = shape change = deformation)
Split the twist into a coherent (rigid) part and an incoherent (deformation) part, per CDR.
-> figures/twist_rigid_vs_deform_<sys>.png"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}


def analyze(loop, R):
    T, N, _ = loop.shape
    Rref = R[0] - R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + R[0].mean(0)
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))
    L0 = Lf[med]; fN, fC = L0[0], L0[-1]
    piv = fC if CLAMP_now == "C" else fN
    c = fC - fN; c /= np.linalg.norm(c)
    perp = (L0 - piv) - ((L0 - piv) @ c)[:, None] * c
    ai = int(np.argmax(np.linalg.norm(perp, axis=1)))
    h = perp[ai]; h /= np.linalg.norm(h)                       # up-loop (twist) axis
    e1 = c - (c @ h) * h; e1 /= np.linalg.norm(e1); e2 = np.cross(h, e1)   # basis in the plane ⟂ ĥ
    rel0 = L0 - piv
    phi0 = np.arctan2(rel0 @ e2, rel0 @ e1)
    rad = np.sqrt((rel0 @ e1) ** 2 + (rel0 @ e2) ** 2)         # weight: reliable φ only far from axis
    w = rad / rad.sum()
    coh = np.empty(T); inc = np.empty(T)
    dphi_all = np.empty((T, N))
    for t in range(T):
        rel = Lf[t] - piv
        phi = np.arctan2(rel @ e2, rel @ e1)
        dphi = np.angle(np.exp(1j * (phi - phi0)))             # wrapped Δφ per residue
        dphi_all[t] = dphi
        mbar = np.sum(w * dphi)                                # coherent (rigid) twist this frame
        coh[t] = mbar
        inc[t] = np.sqrt(np.sum(w * (dphi - mbar) ** 2))       # spread across residues = non-uniform
    coh_amp = float(np.degrees(np.std(coh)))                   # rigid twist amplitude
    inc_amp = float(np.degrees(np.sqrt(np.mean(inc ** 2))))    # deformation (corkscrew) twist amplitude
    frac_rigid = coh_amp ** 2 / (coh_amp ** 2 + inc_amp ** 2 + 1e-9)
    grad = np.degrees(np.std(dphi_all, axis=0))               # per-residue twist spread profile
    return coh_amp, inc_amp, frac_rigid, grad


CLAMP_now = "N"


def main(sysid="3SKN"):
    global CLAMP_now
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    print(f"{sysid}  is the twist RIGID (coherent) or DEFORMATION (corkscrew)?")
    print(f"{'CDR':8}{'rigid-twist°':>13}{'deform-twist°':>14}{'% of twist RIGID':>18}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    res = {}
    for cdr in CDRS:
        CLAMP_now = CLAMP[cdr[2:]]
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        R = xyz[:, np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])]
        ca, ia, fr, grad = analyze(loop, R)
        res[cdr] = (ca, ia, fr, grad)
        print(f"{cdr:8}{ca:12.1f}{ia:14.1f}{fr:17.0%}")
    x = np.arange(6); w = 0.4
    ax[0].bar(x - w / 2, [res[c][0] for c in CDRS], w, color="#7B4FA3", label="RIGID twist (coherent)")
    ax[0].bar(x + w / 2, [res[c][1] for c in CDRS], w, color="#C0392B", label="DEFORM twist (corkscrew)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(CDRS, rotation=20, fontsize=9)
    ax[0].set_ylabel("twist amplitude (deg)"); ax[0].legend(fontsize=8.5)
    ax[0].set_title("Is the twist rigid or a corkscrew?", fontsize=10, fontweight="bold")
    for c in CDRS:
        ax[1].plot(np.linspace(0, 1, len(res[c][3])), res[c][3], "-o", ms=3, label=c)
    ax[1].set_xlabel("loop position (clamp→…→other end)"); ax[1].set_ylabel("per-residue twist spread (deg)")
    ax[1].set_title("Δφ spread along the loop\n(flat=rigid; rising=corkscrew/deform)", fontsize=10, fontweight="bold")
    ax[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{sysid}: does the loop actually twist rigidly, or is the 'twist' just deformation?", y=1.02, fontsize=12)
    out = f"{HERE}/figures/twist_rigid_vs_deform_{sysid}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3SKN")
