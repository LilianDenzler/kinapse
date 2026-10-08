#!/usr/bin/env python
"""AMPLITUDE version of angle_ranges: |rotation| about each loop-frame axis relative to the reference (central) loop,
in degrees -> the swing AMPLITUDE each TCR reaches (one-sided, from 0). Answers: is the amplitude of angle change
common across TCRs? Per TCR x CDR x mode we store |ω| percentiles p50/p90/p95/p99, max, mean.
Mirrors angle_ranges_all layout (6 CDR panels x 3 modes) but each TCR is a bar 0 -> p95 amplitude + dot at p50.
-> results_angleamp.json + figures/angle_amplitude_all.png + printed outlier table."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
RGB = {"hinge": (0.18, 0.49, 0.20), "twist": (0.48, 0.31, 0.64), "sway": (0.94, 0.63, 0.19)}
MODES = ["hinge", "twist", "sway"]
NSUB = 500


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1); c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


def geom(loop):
    T, N, _ = loop.shape
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    fN, fC = L0[0], L0[-1]; m = 0.5 * (fN + fC); cc = fC - fN; cc /= np.linalg.norm(cc)
    perp = (L0 - m) - ((L0 - m) @ cc)[:, None] * cc
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(cc, h); n /= np.linalg.norm(n)
    return L0, {"hinge": cc, "twist": h, "sway": n}


def one(sysid, rng):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    out = {}; supr_cache = {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        if ch not in supr_cache:
            fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
            supr_cache[ch] = superpose_all(xyz, fw)
        supr = supr_cache[ch]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = supr[:, np.array([imap[ch][k] for k in lk])]; T = loop.shape[0]
        L0, axv = geom(loop)
        idx = rng.choice(T, min(NSUB, T), replace=False)
        amp = {k: np.empty(len(idx)) for k in MODES}
        for s, t in enumerate(idx):
            R, _, _ = robust_kabsch(L0, loop[t]); w = Rot.from_matrix(R).as_rotvec()
            for k in MODES:
                amp[k][s] = abs(np.degrees(w @ axv[k]))                       # |angle| about axis from reference
        out[cdr] = {k: {"p50": float(np.percentile(amp[k], 50)), "p90": float(np.percentile(amp[k], 90)),
                        "p95": float(np.percentile(amp[k], 95)), "p99": float(np.percentile(amp[k], 99)),
                        "max": float(amp[k].max()), "mean": float(amp[k].mean())} for k in MODES}
    return out


def main():
    tcrs = sorted(os.path.basename(f)[:4] for f in glob.glob(f"{HERE}/results_swing/*.npz"))
    rng = np.random.default_rng(0); res = {}
    for i, s in enumerate(tcrs):
        try:
            res[s] = one(s, rng)
            print(f"[{i+1}/{len(tcrs)}] {s} done", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(tcrs)}] {s}: FAILED {type(e).__name__}: {e}", flush=True)
    json.dump(res, open(f"{HERE}/results_angleamp.json", "w"), indent=1, default=float)
    plot(res)
    # outlier table on p95 amplitude
    print("\nAmplitude outliers per CDR x mode (p95 |ω| > Q3 + 1.5*IQR across TCRs):")
    for cdr in CDRS:
        for m in MODES:
            a = np.array([res[t][cdr][m]["p95"] for t in res]); ts = list(res)
            q1, q3 = np.percentile(a, 25), np.percentile(a, 75); thr = q3 + 1.5 * (q3 - q1)
            o = sorted([(ts[i], a[i]) for i in range(len(ts)) if a[i] > thr], key=lambda x: -x[1])
            if o:
                print(f"  {cdr:8} {m:6}: median {np.median(a):3.0f}°  -> " + ", ".join(f"{t} ({v:.0f}°)" for t, v in o), flush=True)


def plot(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    tcrs = list(res.keys()); nt = len(tcrs)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for ax, cdr in zip(axes.flat, CDRS):
        for j, m in enumerate(MODES):
            p95 = np.array([res[t][cdr][m]["p95"] for t in tcrs])
            p50 = np.array([res[t][cdr][m]["p50"] for t in tcrs])
            xs = j + np.linspace(-0.32, 0.32, nt)
            band = np.median(p95)                                            # common amplitude reach
            ax.add_patch(Rectangle((j - 0.42, 0), 0.84, band, facecolor=RGB[m], alpha=0.13, edgecolor="none", zorder=0))
            ax.hlines(np.median(p50), j - 0.42, j + 0.42, color=RGB[m], lw=1.2, zorder=1)   # typical amplitude
            q1, q3 = np.percentile(p95, 25), np.percentile(p95, 75); thr = q3 + 1.5 * (q3 - q1)
            for i in range(nt):
                ax.vlines(xs[i], 0, p95[i], color=RGB[m], alpha=0.35, lw=1.1, zorder=2)      # 0 -> p95 amplitude
                ax.plot(xs[i], p50[i], ".", color="k", ms=2.5, zorder=4)                     # typical
                if p95[i] > thr:
                    ax.annotate(tcrs[i], (xs[i], p95[i]), fontsize=6, ha="center", va="bottom", color=RGB[m], fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(["hinge\n(ĉ)", "twist\n(ĥ)", "sway\n(n̂)"], fontsize=9)
        ax.set_ylabel("swing amplitude |angle| from reference (deg)"); ax.set_title(cdr, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.6, 2.6); ax.set_ylim(bottom=0)
    fig.suptitle("Swing AMPLITUDE of the three rigid modes across all 22 TCRs  "
                 "(each faint bar = one TCR, 0→p95 |angle|; dot = typical(p50); shaded = common reach; labelled = outliers)",
                 y=1.0, fontsize=12, fontweight="bold")
    out = f"{HERE}/figures/angle_amplitude_all.png"
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


if __name__ == "__main__":
    main()
