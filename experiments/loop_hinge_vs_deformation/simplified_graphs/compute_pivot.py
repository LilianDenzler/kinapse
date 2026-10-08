#!/usr/bin/env python
"""Is the 'translation' real, or an artifact of pinning the pivot at the clamp?
Rigid motion = rotation + translation, and the translation DEPENDS on the pivot. Find the pivot that
MINIMISES it (the true centre of rotation / screw axis) and compare:
   V_trans(centroid)  vs  V_trans(clamp)  vs  V_trans(optimal)   [Å², summed over the loop]
If V_trans(optimal) collapses -> the big translation was pivot choice; the loop is ~pure rotation.
V_trans(optimal) is the IRREDUCIBLE translation no rotation can absorb. -> results_pivot.json"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}


def analyze(loop, R, clampside):
    T, N, _ = loop.shape
    Rref = R[0] - R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + R[0].mean(0)
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = Lf[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    c0 = L0.mean(0); L0c = L0 - c0
    fN, fC = L0[0], L0[-1]; clamp = fC if clampside == "C" else fN
    Qs = np.empty((T, 3, 3)); cens = np.empty((T, 3))
    A = np.zeros((3, 3)); b = np.zeros(3); Vtot = 0.0; Vdef = 0.0; Vrig = 0.0
    for t in range(T):
        Q = kabsch(L0c, Lf[t] - Lf[t].mean(0)); cen = Lf[t].mean(0); Qs[t] = Q; cens[t] = cen
        M = Q - np.eye(3); k = cen - Q @ c0                            # displacement of point p = M p + k
        A += M.T @ M; b += M.T @ k
        Vrig += ((L0c @ Q.T + cen - L0) ** 2).sum()
        Vdef += ((Lf[t] - (L0c @ Q.T + cen)) ** 2).sum()
        Vtot += ((Lf[t] - L0) ** 2).sum()
    pstar = -np.linalg.solve(A, b)                                    # pivot that minimises translation

    def vtrans(p):
        d = np.einsum("tij,j->ti", Qs, (p - c0)) + cens - p           # displacement of point p each frame
        return float(N * (d ** 2).sum(1).mean())
    r = dict(tot=Vtot / T, deform=Vdef / T, rigid=Vrig / T,
             trans_centroid=vtrans(c0), trans_clamp=vtrans(clamp), trans_opt=vtrans(pstar),
             pstar_to_clamp=float(np.linalg.norm(pstar - clamp)), nres=int(N))
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in r.items()}


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
    json.dump(res, open(f"{HERE}/results_pivot.json", "w"), default=float)
    print(f"\n{len(res)} TCRs -> results_pivot.json")
    tab(res)


def tab(res):
    tcrs = list(res)
    print(f"\n{'CDR':8}{'total':>7}{'deform':>8}{'rigid':>7} | translation (Å²) at pivot=  {'centroid':>9}{'clamp':>8}{'OPTIMAL':>9}  {'p*-clamp(Å)':>12}")
    for c in CDRS:
        m = {k: np.median([res[t][c][k] for t in tcrs]) for k in ("tot", "deform", "rigid", "trans_centroid", "trans_clamp", "trans_opt", "pstar_to_clamp")}
        print(f"{c:8}{m['tot']:7.1f}{m['deform']:8.1f}{m['rigid']:7.1f} |                              {m['trans_centroid']:9.1f}{m['trans_clamp']:8.1f}{m['trans_opt']:9.2f}  {m['pstar_to_clamp']:12.1f}")


def plot(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res); x = np.arange(6)
    absmed = np.array([np.median([res[t][c]["tot"] for t in tcrs]) for c in CDRS])
    def Fabs(k):                                                      # median (mode/total) * median total -> additive Å²
        return np.array([np.median([res[t][c][k] / res[t][c]["tot"] for t in tcrs]) for c in CDRS]) * absmed
    dform = Fabs("deform")
    rot = np.array([np.median([(res[t][c]["rigid"] - res[t][c]["trans_opt"]) / res[t][c]["tot"] for t in tcrs]) for c in CDRS]) * absmed
    trans = Fabs("trans_opt")
    tclamp = Fabs("trans_clamp"); tcent = Fabs("trans_centroid")
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.bar(x, rot, 0.62, color="#3B6EA5", label="rotation (about optimal pivot)", edgecolor="white")
    ax.bar(x, trans, 0.62, bottom=rot, color="#888888", hatch="//", label="IRREDUCIBLE translation (best pivot)", edgecolor="white")
    ax.bar(x, dform, 0.62, bottom=rot + trans, color="#C0392B", label="deformation", edgecolor="white")
    ax.plot(x, absmed, "_", color="k", ms=34, mew=2.5, label="total")
    ax.plot(x, rot + tclamp, "v", color="0.45", ms=7, label="translation if pivot = clamp")
    ax.plot(x, rot + tcent, "^", color="0.7", ms=7, label="translation if pivot = centroid")
    for xi in range(6):
        pct = 100 * trans[xi] / (rot[xi] + trans[xi])
        ax.annotate(f"transl\n{pct:.0f}% of rigid", (xi, rot[xi] + trans[xi] / 2), ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CDRS, fontsize=10)
    ax.set_ylabel("summed displacement variance over the loop (Å²), median over 22 TCRs")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.set_title("Rigid motion = rotation + IRREDUCIBLE translation (+ deformation), in Å².  The translation shrinks as the pivot moves to the\n"
                 "true rotation centre (▲centroid→▼clamp→bar) but does NOT vanish: short CDR1/CDR2 keep ~30% real translation; long CDR3 ~pure rotation.", y=1.15, fontsize=9.5)
    out = f"{HERE}/figures/pivot_translation.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tab":
        tab(json.load(open(f"{HERE}/results_pivot.json")))
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(json.load(open(f"{HERE}/results_pivot.json")))
    else:
        main()
