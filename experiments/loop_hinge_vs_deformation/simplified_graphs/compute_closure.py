#!/usr/bin/env python
"""Closure / accounting: does the mode decomposition explain the WHOLE loop motion, or is there residual?
Work in summed displacement VARIANCE (Å², additive), relative to the medoid reference L0 (centroid c0).
Each atom's displacement u_i = Lf_i - L0_i splits EXACTLY (orthogonally) into:
    rotation-about-centroid  +  centroid-translation  +  deformation(residual)
  => Vtot = Vrot + Vtrans + Vdef   (exact; the three are mutually ⟂)
Rotation further splits onto the loop-frame axes -> Vrot = Vhinge + Vtwist + Vsway + (cross-terms).
So RESIDUAL = Vtot - Vhinge - Vtwist - Vsway - Vtrans - Vdef  =  the rotation-axis cross-terms = 'unexplained'.
-> results_closure.json + figures/closure_angstrom.png"""
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
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}          # rotation pivoted at the clamp (fixed by framework fit)


def analyze(loop, R, clampside):
    T, N, _ = loop.shape
    Rref = R[0] - R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + R[0].mean(0)
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = Lf[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    c0 = L0.mean(0); L0c = L0 - c0
    fN, fC = L0[0], L0[-1]; piv = fC if clampside == "C" else fN
    c = fC - fN; c /= np.linalg.norm(c)
    perp = (L0 - piv) - ((L0 - piv) @ c)[:, None] * c
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(c, h); n /= np.linalg.norm(n)
    relp = L0 - piv
    # 6 rigid generators (displacement fields): rotations about the clamp + 3 translations
    G = np.zeros((6, N, 3))
    G[0] = np.cross(c, relp); G[1] = np.cross(h, relp); G[2] = np.cross(n, relp)
    G[3, :, 0] = 1; G[4, :, 1] = 1; G[5, :, 2] = 1
    G = G.reshape(6, 3 * N)
    wv, U = np.linalg.eigh(G @ G.T)                                    # Löwdin symmetric orthonormalisation
    B = (U @ np.diag(1.0 / np.sqrt(np.clip(wv, 1e-9, None))) @ U.T) @ G  # (6, 3N) orthonormal rows -> NO double-counting
    Vm = np.zeros(6); Vnl = 0.0; Vdef = 0.0; Vtot = 0.0
    for t in range(T):
        Q = kabsch(L0c, Lf[t] - Lf[t].mean(0)); cen = Lf[t].mean(0)
        rigid = (L0c @ Q.T + cen - L0).reshape(3 * N)                 # finite rigid displacement from L0
        deform = Lf[t] - (L0c @ Q.T + cen)                            # Kabsch residual (shape change)
        co = B @ rigid                                                # projections onto the orthonormal rigid modes
        Vm += co ** 2; Vnl += float(((rigid - B.T @ co) ** 2).sum())  # nonlinear (finite-rotation) leftover, tiny
        Vdef += (deform ** 2).sum(); Vtot += ((Lf[t] - L0) ** 2).sum()
    Vm /= T; Vnl /= T; Vdef /= T; Vtot /= T
    V = dict(hinge=float(Vm[0]), twist=float(Vm[1]), sway=float(Vm[2]),
             slide=float(Vm[3] + Vm[4] + Vm[5]), nonlin=float(Vnl), deform=float(Vdef), tot=float(Vtot), nres=int(N))
    V["residual"] = float(V["tot"] - V["hinge"] - V["twist"] - V["sway"] - V["slide"] - V["nonlin"] - V["deform"])
    return V


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
                row[cdr] = analyze(loop, Rr, CLAMP[cdr[2:]])
            res[t] = row; print(f"{t} ok", flush=True)
        except Exception as e:
            print(f"{t} ERR {type(e).__name__}: {str(e)[:90]}", flush=True)
    json.dump(res, open(f"{HERE}/results_closure.json", "w"), default=float)
    print(f"\n{len(res)} TCRs -> results_closure.json")
    plot(res)


def plot(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res); x = np.arange(6)
    def F(k):                                                          # median over TCRs of (mode / that TCR's total), in %
        return np.array([100 * np.median([res[t][c][k] / res[t][c]["tot"] for t in tcrs]) for c in CDRS])
    absmed = np.array([np.median([res[t][c]["tot"] for t in tcrs]) for c in CDRS])   # absolute total for scale
    rigidF = 100 - F("deform"); dformF = F("deform")
    SUB = [("hinge", "#2E7D32", "hinge (ĉ)"), ("twist", "#7B4FA3", "twist (ĥ)"), ("sway", "#F0A030", "sway (n̂)"),
           ("slide", "#888888", "translation"), ("nonlin", "#CFCFCF", "finite-rot"), ("deform", "#C0392B", None)]
    fig, ax = plt.subplots(figsize=(14, 6.6)); w = 0.38
    ax.bar(x - w / 2, rigidF, w, color="#3B6EA5", label="RIGID (robust)", edgecolor="white")
    ax.bar(x - w / 2, dformF, w, bottom=rigidF, color="#C0392B", label="deformation (robust)", edgecolor="white")
    bottom = np.zeros(6)
    for k, col, lab in SUB:
        kw = dict(label=lab) if lab else {}
        ax.bar(x + w / 2, F(k), w, bottom=bottom, color=col, edgecolor="white", lw=0.3,
               hatch="//" if k in ("slide", "nonlin") else None, **kw)
        bottom += F(k)
    ax.axhline(100, color="k", lw=0.8, ls=":")
    for xi in range(6):
        ax.annotate(f"deform\n{dformF[xi]:.0f}%", (xi - w / 2, 100), (0, 3), textcoords="offset points", ha="center", fontsize=7.5, color="0.25")
        ax.annotate(f"total\n{absmed[xi]:.0f} Å²", (xi, -1), (0, -22), textcoords="offset points", ha="center", fontsize=7.5, color="0.35")
    ax.set_xticks(x); ax.set_xticklabels(CDRS, fontsize=10); ax.set_ylim(0, 108)
    ax.set_ylabel("% of the loop's total motion (median over 22 TCRs)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.set_title("LEFT = robust split (rigid vs deform — the ONLY basis-free decomposition).   RIGHT = orthogonal sub-modes of the same motion\n"
                 "(each TCR's decomposition closes to ~100%; hinge/twist/sway/translation are BASIS-DEPENDENT — one screw motion carved by chosen axes)", y=1.17, fontsize=9.5)
    out = f"{HERE}/figures/closure_angstrom.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)
    print(f"\n{'CDR':8}{'deform%':>8}{'rigid%':>8}{' |':>3}{'hinge%':>7}{'twist%':>7}{'sway%':>7}{'transl%':>8}{'resid%':>7}  (median of per-TCR fractions)")
    for c in CDRS:
        f = {k: np.median([100 * res[t][c][k] / res[t][c]["tot"] for t in tcrs]) for k in ("deform", "hinge", "twist", "sway", "slide", "residual")}
        print(f"{c:8}{f['deform']:7.0f}%{100-f['deform']:7.0f}%{' |':>3}{f['hinge']:6.0f}%{f['twist']:6.0f}%{f['sway']:6.0f}%{f['slide']:7.0f}%{f['residual']:6.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(json.load(open(f"{HERE}/results_closure.json")))
    else:
        main()
