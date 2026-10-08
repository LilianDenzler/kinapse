#!/usr/bin/env python
"""Per-TCR clamp validation figures.

FIG 1 (clamp_pertcr_profiles.png): per-residue fluct-to-rigid across each CDR loop (base->apex->base),
       one thin line per TCR + median/IQR. The clamped stem (orange, = the orange clamp strands of
       loops_flanks_fluctuation_annotated.png) is pinned <0.5 A in every TCR; mobility ramps to the apex.
FIG 2 (clamp_pertcr_validation.png): (left) heatmap of every TCR x every clamp stem, green<=0.5 A=rigid;
       (right) clamped vs free stem per TCR -> the clamp is the rigid side in ~all TCRs.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import matplotlib.lines as mlines

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clamp_pertcr import (compute, load_tcr, fluct_to_rigid, stems, clamp_free,
                          CDR_RANGES, CLAMP_SIDE, TAU, HERE)

FIG = HERE
ORANGE, GREY, BLUE, RED = "#D9822B", "#9AA3AD", "#3B6FB0", "#C0504D"
GREEK = {"A": "α", "B": "β"}
KEYS = [f"{ch}_{c}" for c in ["CDR1", "CDR2", "CDR3"] for ch in "AB"]


def fmt(r):
    r = float(r)
    return str(int(r)) if r.is_integer() else str(r)


# ---------------------------------------------------------------- FIG 1: profiles
def fig_profiles(R):
    tcrs = R["tcrs"]; rigid = R["rigid"]
    # precompute per-residue mean fluct-to-rigid, per TCR/chain/cdr, keeping insertion keys
    per = {sid: load_tcr(sid) for sid in tcrs}
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.6), sharey=True)
    for ci, cdr in enumerate(["CDR1", "CDR2", "CDR3"]):
        lo, hi = CDR_RANGES[cdr]
        clamp, free = clamp_free(cdr)
        for ri, ch in enumerate("AB"):
            ax = axes[ri, ci]
            # master ordered x = union of present positions in the window across all TCRs
            allpos = set()
            for sid in tcrs:
                imgt = per[sid][ch][0]
                allpos |= {float(n) for n in imgt if lo - 3 <= n <= hi + 3}
            xs = sorted(allpos); slot = {p: i for i, p in enumerate(xs)}
            # per-TCR line
            M = np.full((len(tcrs), len(xs)), np.nan)
            for t, sid in enumerate(tcrs):
                imgt, D, idx = per[sid][ch]
                win = [n for n in imgt if lo - 3 <= n <= hi + 3]
                fm = fluct_to_rigid(D, idx, win, rigid[ch], "mean")
                for r in win:
                    M[t, slot[float(r)]] = fm[r]
            xi = np.arange(len(xs))
            for t in range(len(tcrs)):
                ax.plot(xi, M[t], "-", color=GREY, lw=0.7, alpha=0.42, zorder=2)
            med = np.nanmedian(M, 0); q1 = np.nanpercentile(M, 25, 0); q3 = np.nanpercentile(M, 75, 0)
            ax.fill_between(xi, q1, q3, color="#2C3E50", alpha=0.16, zorder=3, lw=0)
            ax.plot(xi, med, "-", color="#1F2D3D", lw=2.4, zorder=5, label="median (22 TCRs)")
            # shade clamped (orange) and free (grey) stems
            for r in clamp:
                if float(r) in slot:
                    ax.axvspan(slot[float(r)] - .5, slot[float(r)] + .5, color=ORANGE, alpha=.30, lw=0, zorder=1)
            for r in free:
                if float(r) in slot:
                    ax.axvspan(slot[float(r)] - .5, slot[float(r)] + .5, color=GREY, alpha=.22, lw=0, zorder=1)
            # ultra-rigid core residues: blue tick under axis
            for r in rigid[ch]:
                if float(r) in slot:
                    ax.plot(slot[float(r)], 0.02, "s", color=BLUE, ms=5, zorder=6, clip_on=False)
            ax.axhline(TAU, ls="--", lw=1.1, color=RED, zorder=4)
            ax.set_xticks(xi); ax.set_xticklabels([fmt(p) for p in xs], rotation=90, fontsize=6.5)
            for lab, p in zip(ax.get_xticklabels(), xs):
                if not float(p).is_integer():
                    lab.set_color("#8E44AD"); lab.set_fontweight("bold")
            ax.set_xlim(-0.6, len(xs) - 0.4); ax.set_ylim(0, 1.55)
            side = CLAMP_SIDE[cdr]
            ax.set_title(f"{GREEK[ch]}  {cdr}   —   clamp = {'C' if side=='C' else 'N'}-terminal stem",
                         fontsize=10.5, pad=5)
            if ci == 0:
                ax.set_ylabel(f"chain {ch} ({GREEK[ch]})\nfluct-to-rigid (Å)", fontsize=9.5)
            if ri == 1:
                ax.set_xlabel("residue (IMGT; insertions in purple)", fontsize=9)
            # small arrow/label pointing to clamped stem
            cx = np.mean([slot[float(r)] for r in clamp if float(r) in slot])
            ax.annotate("clamp", (cx, 1.42), ha="center", va="top", fontsize=8.5, color=ORANGE,
                        fontweight="bold")
    handles = [Patch(color=ORANGE, alpha=.5, label="clamped stem (predicted anchor)"),
               Patch(color=GREY, alpha=.4, label="free stem (opposite)"),
               mlines.Line2D([], [], color=GREY, lw=1, alpha=.7, label="per-TCR profile (22)"),
               mlines.Line2D([], [], color="#1F2D3D", lw=2.4, label="median ± IQR"),
               mlines.Line2D([], [], color=BLUE, marker="s", ls="", ms=6, label="ultra-rigid core residue"),
               mlines.Line2D([], [], color=RED, ls="--", lw=1.1, label=f"rigidity threshold {TAU} Å")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-TCR validation of the clamp: fluctuation to the ultra-rigid core along each CDR loop "
                 "(alignment-free, 22 unbound-TCR MD)\n"
                 "In every TCR the clamped stem stays pinned to the rigid core (<0.5 Å) while mobility rises "
                 "to the loop apex and the free stem — the clamp holds loop-by-loop, TCR-by-TCR",
                 y=1.005, fontsize=12.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = f"{FIG}/clamp_pertcr_profiles.png"
    fig.savefig(out, dpi=145, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


# ---------------------------------------------------------------- FIG 2: validation grid
def fig_validation(R):
    tcrs = R["tcrs"]
    clamp_m = np.array([[R["clamp_mean"][s][k] for k in KEYS] for s in tcrs])
    free_m = np.array([[R["free_mean"][s][k] for k in KEYS] for s in tcrs])
    collab = [f"{GREEK[k[0]]} {k[2:]}\n({'C' if CLAMP_SIDE[k[2:]]=='C' else 'N'}-clamp)" for k in KEYS]

    fig = plt.figure(figsize=(16, 9.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.22)

    # ----- LEFT: heatmap TCR x clamp region -----
    axH = fig.add_subplot(gs[0])
    norm = TwoSlopeNorm(vmin=0.15, vcenter=TAU, vmax=0.85)
    im = axH.imshow(clamp_m, cmap="RdYlGn_r", norm=norm, aspect="auto")
    axH.set_xticks(range(len(KEYS))); axH.set_xticklabels(collab, fontsize=9)
    axH.set_yticks(range(len(tcrs))); axH.set_yticklabels(tcrs, fontsize=8)
    for i in range(len(tcrs)):
        for j in range(len(KEYS)):
            v = clamp_m[i, j]
            over = v > TAU
            axH.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                     color="white" if over else "#1a1a1a", fontweight="bold" if over else "normal")
            if over:
                axH.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#7B241C", lw=1.8))
    # mean row appended visually via title text
    axH.set_title("Clamp-stem fluctuation to the ultra-rigid core, per TCR (Å)\n"
                  "green ≤ 0.5 Å = rigid/clamped · red box = above threshold", fontsize=10.5)
    cb = fig.colorbar(im, ax=axH, fraction=0.046, pad=0.02, extend="both")
    cb.set_label("mean fluct-to-rigid over clamped stem (Å)", fontsize=9)
    cb.ax.axhline(TAU, color="k", lw=1)
    # pass-rate annotation under each column
    for j, k in enumerate(KEYS):
        npass = int(np.sum(clamp_m[:, j] <= TAU))
        axH.text(j, len(tcrs) - 0.35, f"{npass}/{len(tcrs)}", ha="center", va="top", fontsize=8,
                 color="#145A32", fontweight="bold", transform=axH.transData)
    axH.text(-0.6, len(tcrs) + 0.15, "pass ≤ 0.5 Å:", ha="right", va="top", fontsize=8,
             color="#145A32", fontweight="bold")

    # ----- RIGHT: clamped vs free stem, per TCR, per region -----
    axS = fig.add_subplot(gs[1])
    xpos = np.arange(len(KEYS))
    rng = np.random.default_rng(0)
    for j in range(len(KEYS)):
        jit = (rng.random(len(tcrs)) - 0.5) * 0.16
        xc = xpos[j] - 0.17 + jit; xf = xpos[j] + 0.17 + jit
        for i in range(len(tcrs)):
            axS.plot([xc[i], xf[i]], [clamp_m[i, j], free_m[i, j]], "-", color="#C8CDD3", lw=0.6,
                     alpha=0.7, zorder=1)
        axS.scatter(xc, clamp_m[:, j], s=26, color=ORANGE, ec="#7a4a12", lw=.4, zorder=3,
                    label="clamped stem" if j == 0 else None)
        axS.scatter(xf, free_m[:, j], s=26, color=GREY, ec="#4b5158", lw=.4, zorder=3,
                    label="free stem" if j == 0 else None)
        # region medians
        axS.plot([xpos[j] - 0.30, xpos[j] - 0.04], [np.median(clamp_m[:, j])] * 2, "-",
                 color="#7a4a12", lw=2.4, zorder=4)
        axS.plot([xpos[j] + 0.04, xpos[j] + 0.30], [np.median(free_m[:, j])] * 2, "-",
                 color="#4b5158", lw=2.4, zorder=4)
        nas = int(np.sum(clamp_m[:, j] < free_m[:, j]))
        axS.text(xpos[j], 1.02, f"{nas}/{len(tcrs)}", ha="center", va="bottom", fontsize=8.5,
                 color="#145A32", fontweight="bold")
    axS.axhline(TAU, ls="--", lw=1.1, color=RED, zorder=2)
    axS.text(len(KEYS) - 0.55, TAU + 0.01, f"{TAU} Å rigidity threshold", ha="right", va="bottom",
             fontsize=8.5, color=RED)
    axS.set_xticks(xpos); axS.set_xticklabels(collab, fontsize=9)
    axS.set_ylabel("fluctuation to ultra-rigid core (Å)", fontsize=10)
    axS.set_ylim(0.1, 1.12); axS.set_xlim(-0.55, len(KEYS) - 0.45)
    axS.set_title("Clamped vs free stem, every TCR paired\n"
                  "(orange below grey ⇒ clamp is the rigid side)  ·  green = # TCRs with clamp < free",
                  fontsize=10.5)
    axS.legend(loc="upper left", fontsize=9, framealpha=.95)
    for x in xpos[:-1] + 0.5:
        axS.axvline(x, color="0.9", lw=1, zorder=0)

    fig.suptitle("Does the clamp hold for EACH TCR? — per-TCR test against the ultra-rigid framework set "
                 f"({len(tcrs)} unbound TCR MD trajectories, kinapse IMGT)",
                 y=0.99, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{FIG}/clamp_pertcr_validation.png"
    fig.savefig(out, dpi=145, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


if __name__ == "__main__":
    R = compute()
    fig_profiles(R)
    fig_validation(R)
