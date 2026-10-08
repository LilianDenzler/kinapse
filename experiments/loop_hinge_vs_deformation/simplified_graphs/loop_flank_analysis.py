#!/usr/bin/env python
"""Finalize the fluctuation-derived rigid framework and analyse each CDR loop + its flanking region.

RIGID DEFINITION (chosen): from the 22-TCR averaged CA-CA distance fluctuation,
    kept  = largest set with EVERY average pairwise D_std <= 0.5 Å   (mutually rigid on average)
    RIGID = ultra-rigid subset of kept whose WORST-CASE (max over TCRs) pairwise D_std <= 0.5 Å
This is written to RIGID_FRAMEWORK.md + rigid_framework.json.

FLANKING REGION of a CDR: walk outward from each end, up to 10 residues, stopping at the first residue
that is another CDR or a RIGID residue (i.e. the soft transition zone between the loop and the rigid core).

Then, for every loop + flank residue, plot two alignment-free, TCR-averaged quantities:
    fluctuation          = mean edge fluctuation (mean over all j of std_t|CA_i-CA_j|)
    max deviation to rigid = max over rigid residues j of the average pairwise D_std(i,j)
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob, json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from graph_build import consensus, CDR_RANGES, HERE

RES = f"{HERE}/results_fluct"; FIG = f"{HERE}/figures"
TAU, MAXCAP, FLANK_MAX = 0.5, 0.5, 10


def iscdr(r):
    return any(lo <= r <= hi for lo, hi in CDR_RANGES.values())


def load_all():
    data = {}
    for f in sorted(glob.glob(f"{RES}/*_fluct.npz")):
        z = np.load(f); sid = os.path.basename(f)[:4]
        data[sid] = {ch: (z[f"{ch}_imgt"], z[f"{ch}_Dstd"]) for ch in "AB"}
    return data


def rigid_set(S, tau):
    alive = np.ones(len(S), bool); V = (S > tau).copy(); np.fill_diagonal(V, False)
    while True:
        vc = (V & alive[None, :]).sum(1); vc[~alive] = -1
        if vc.max() <= 0:
            break
        alive[int(vc.argmax())] = False
    return np.where(alive)[0]


def averaged(data, ch, reduce="mean"):
    """Averaged (or max) pairwise D_std over TCRs on the intersection of IMGT positions."""
    common = None
    for sid in data:
        s = set(data[sid][ch][0].tolist()); common = s if common is None else common & s
    common = np.array(sorted(common)); acc = np.zeros((len(common), len(common)))
    for sid in data:
        imgt, S = data[sid][ch]; pos = np.array([np.where(imgt == n)[0][0] for n in common])
        blk = S[np.ix_(pos, pos)]
        acc = acc + blk if reduce == "mean" else np.maximum(acc, blk)
    return common, (acc / len(data) if reduce == "mean" else acc)


def profile(data, ch):
    vals = defaultdict(list)
    for sid in data:
        imgt, S = data[sid][ch]; S2 = S.copy(); np.fill_diagonal(S2, np.nan)
        for n, v in zip(imgt, np.nanmean(S2, axis=1)):
            vals[round(float(n), 1)].append(float(v))     # keep insertion keys (e.g. 112.1), don't collapse to int
    return {n: float(np.mean(vals[n])) for n in vals}


def position_counts(data, ch):
    """How many of the TCRs actually have a residue at each IMGT position (incl. insertion keys)."""
    c = defaultdict(int)
    for sid in data:
        for n in data[sid][ch][0].tolist():
            c[round(float(n), 1)] += 1
    return dict(c)


def rigid_definition(data, ch):
    common, S = averaged(data, ch, "mean"); _, Smax = averaged(data, ch, "max")
    kept = rigid_set(S, TAU)
    ultra = kept[rigid_set(Smax[np.ix_(kept, kept)], MAXCAP)]
    tolist = lambda idx: sorted(int(x) if float(x).is_integer() else float(x) for x in common[idx])
    return tolist(kept), tolist(ultra)                 # rigid positions are whole numbers (no insertions)


def flank_region(ch, lo, hi, rigidset, present):
    """Contiguous non-CDR, non-rigid residues within FLANK_MAX of each CDR end."""
    left = []
    for k in range(1, FLANK_MAX + 1):
        r = lo - k
        if r not in present or iscdr(r) or r in rigidset:
            break
        left.append(r)
    right = []
    for k in range(1, FLANK_MAX + 1):
        r = hi + k
        if r not in present or iscdr(r) or r in rigidset:
            break
        right.append(r)
    return sorted(left), right


def dev_to_rigid(data, ch, targets, rigidset):
    """For each target i: MEAN and MAX over the rigid residues j (j != i) of the average-over-TCRs
    pairwise D_std(i,j). Both are 'to the rigid set', so max >= mean always."""
    maps = {sid: {round(float(n), 1): k for k, n in enumerate(data[sid][ch][0])} for sid in data}
    rig = sorted(rigidset); meandev = {}; maxdev = {}
    for i in targets:
        perj = []
        for j in rig:
            if j == i:
                continue
            vals = [float(data[sid][ch][1][maps[sid][i], maps[sid][j]])
                    for sid in data if i in maps[sid] and j in maps[sid]]
            if vals:
                perj.append(np.mean(vals))
        meandev[i] = float(np.mean(perj)) if perj else np.nan
        maxdev[i] = float(max(perj)) if perj else np.nan
    return meandev, maxdev


def annotate_conserved(ax, targets):
    """Overlay conserved TCR V-domain landmarks + the beta-strand map on a (sequential-x) fluctuation axis."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    idx = {round(float(r), 1): i for i, r in enumerate(targets)}
    gi = lambda r: idx.get(round(float(r), 1))
    ytop = ax.get_ylim()[1]
    # --- beta-strand track at the bottom (DSSP-derived canonical Ig V-domain strands) ---
    ax.set_ylim(-0.17 * ytop, ytop); ax.set_yticks(np.arange(0, ytop, 0.5))
    yb, yh = -0.135 * ytop, 0.07 * ytop
    STRANDS = [("A", 11, 15), ("B", 19, 26), ("C", 39, 45), ("C'", 49, 56), ("C''", 64, 68),
               ("D", 76, 81), ("E", 86, 92), ("F", 98, 107), ("G", 118, 128)]
    for name, lo, hi in STRANDS:
        ii = [gi(r) for r in range(lo, hi + 1) if gi(r) is not None]
        if len(ii) < 2:
            continue
        xa, xb = min(ii), max(ii)
        hot = name in ("C", "C'", "F")                        # strands whose end clamps a CDR (C→1, C'→2, F→3)
        ax.add_patch(Rectangle((xa - .45, yb), xb - xa + .9, yh, color="#D35400" if hot else "#8090A0",
                               alpha=.9, lw=0, zorder=2))
        ax.text((xa + xb) / 2, yb + yh / 2, name, ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", zorder=3)
    ax.text(-0.5, yb + yh / 2, "β-strand", ha="right", va="center", fontsize=7.5, color="#555", fontstyle="italic")
    for imgt, label, col in [(23, "Cys23\n(1st-Cys)", "#B8860B"), (41, "Trp41\n(core keystone)", "#6A0DAD"),
                             (89, "hydrophobic\n89", "#2E8B57"), (104, "Cys104\n(2nd-Cys)", "#B8860B"),
                             (118, "Phe118\n(J-anchor)", "#C0392B")]:
        if imgt in idx:
            i = idx[imgt]
            ax.axvline(i, color=col, ls=":", lw=1, alpha=.6, zorder=1)
            ax.annotate(label, (i, ytop * 0.50), rotation=90, fontsize=6.5, color=col, va="bottom", ha="center",
                        fontweight="bold", bbox=dict(boxstyle="round,pad=.12", fc="white", ec=col, lw=.5, alpha=.9))
    if 23 in idx and 104 in idx:                                   # intradomain disulfide arc
        y = ytop * 0.9
        ax.add_patch(FancyArrowPatch((idx[23], y), (idx[104], y), arrowstyle="<|-|>", mutation_scale=9,
                                     color="#B8860B", lw=1.6, connectionstyle="arc3,rad=-0.22"))
        ax.text((idx[23] + idx[104]) / 2, ytop * 0.99, "Cys23–Cys104 disulfide  (pins the two β-sheets)",
                ha="center", va="top", fontsize=8, color="#8B6508", fontweight="bold")
    js = [r for r in (118, 119, 120, 121) if r in idx]            # J-motif F–G–X–G bracket
    if len(js) >= 2:
        xa, xb, yb = idx[js[0]], idx[js[-1]], ytop * 0.72
        ax.plot([xa - .3, xb + .3], [yb, yb], color="#C0392B", lw=2.2)
        ax.text((xa + xb) / 2, yb + ytop * 0.015, "J-motif  F–G–X–G", ha="center", va="bottom",
                fontsize=8, color="#C0392B", fontweight="bold")


def main():
    data = load_all(); con = consensus()
    print(f"{len(data)} TCRs")
    rigid = {}; note = {}
    for ch in "AB":
        kept, ultra = rigid_definition(data, ch)
        rigid[ch] = set(ultra)
        note[ch] = {"kept_tau0_5": kept, "ultra_rigid": ultra}

    # ---- write the note ----
    js = {"definition": "22-TCR averaged CA-CA distance fluctuation; kept = avg pairwise D_std<=0.5; "
                        "RIGID = ultra subset with worst-case (max over TCRs) pairwise D_std<=0.5",
          "tau_avg": TAU, "max_cap": MAXCAP,
          "chain_A": note["A"], "chain_B": note["B"]}
    with open(f"{HERE}/rigid_framework.json", "w") as f:
        json.dump(js, f, indent=2)
    with open(f"{HERE}/RIGID_FRAMEWORK.md", "w") as f:
        f.write("# Rigid framework definition (chosen)\n\n"
                "Derived from the **22-TCR averaged** CA-CA distance fluctuation (`D_std`), alignment-free.\n\n"
                "- **kept** = largest set whose *average* pairwise `D_std ≤ 0.5 Å` (mutually rigid on average)\n"
                "- **RIGID (use this)** = ultra-rigid subset of *kept* whose *worst-case* (max over TCRs) "
                "pairwise `D_std ≤ 0.5 Å`\n\n"
                f"Compare: kinapse consensus uses cross-TCR alignment RMSD < ~2.0 Å (a different metric).\n\n")
        for ch in "AB":
            f.write(f"## Chain {ch}\n\n"
                    f"- **RIGID ({len(note[ch]['ultra_rigid'])})**: {note[ch]['ultra_rigid']}\n"
                    f"- kept@0.5 ({len(note[ch]['kept_tau0_5'])}): {note[ch]['kept_tau0_5']}\n\n")
    print("wrote RIGID_FRAMEWORK.md + rigid_framework.json")

    # ---- loop + flank plot ----
    import matplotlib.patches as mpatches
    RED, ORANGE, BLUE = "#E15759", "#F28E2B", "#4E79A7"       # CDR loop / flank / RIGID
    prof = {ch: profile(data, ch) for ch in "AB"}
    fig, axes = plt.subplots(2, 1, figsize=(22, 12), gridspec_kw={"hspace": 0.95})
    ntcr = len(data); ymaxes = []; TARGETS = {}
    fmt = lambda r: str(int(r)) if float(r).is_integer() else str(r)     # 112.0 -> "112", 112.1 -> "112.1"
    for ax, ch in zip(axes, "AB"):
        present = set(prof[ch])
        flankinfo = []
        for cdr, (lo, hi) in CDR_RANGES.items():
            lft, rgt = flank_region(ch, lo, hi, rigid[ch], present)
            flankinfo.append((cdr, len(lft), len(rgt)))
        # SEQUENTIAL x-axis: every present residue (incl. insertions 112.1) gets its own evenly-spaced slot,
        # sorted in IMGT order (111 < 111.1 < 112 < 112.1 < 113). No cramming, no gaps.
        targets = sorted(present); TARGETS[ch] = targets
        xi = np.arange(len(targets))
        cnt = position_counts(data, ch)
        # per-slot shading: CDR loop red, RIGID residue blue on top
        for i, r in enumerate(targets):
            if iscdr(r):
                ax.axvspan(i - .5, i + .5, color=RED, alpha=.26, lw=0, zorder=0)
        for i, r in enumerate(targets):
            if r in rigid[ch]:
                ax.axvspan(i - .5, i + .5, color=BLUE, alpha=.36, lw=0, zorder=1)
        mdev_mean, mdev_max = dev_to_rigid(data, ch, targets, rigid[ch])
        yflu = np.array([mdev_mean[r] for r in targets]); ymax = np.array([mdev_max[r] for r in targets])
        l1, = ax.plot(xi, yflu, "-o", ms=4, color="#333333", lw=1.2, label="mean fluctuation to RIGID")
        l2, = ax.plot(xi, ymax, "-D", ms=4, color="#8E44AD", lw=1.2, label="max deviation to RIGID")
        # x tick label per residue = IMGT number (insertions like 112.1); colour insertion labels differently
        ax.set_xticks(xi)
        ax.set_xticklabels([fmt(r) for r in targets], rotation=90, fontsize=5)
        for lab, r in zip(ax.get_xticklabels(), targets):
            if not float(r).is_integer():
                lab.set_color("#8E44AD"); lab.set_fontweight("bold")     # insertion residue label
        # count of TCRs having each non-universal residue, above its point
        for i, r in enumerate(targets):
            if cnt.get(r, ntcr) < ntcr:
                ax.annotate(str(cnt[r]), (i, ymax[i]), textcoords="offset points", xytext=(0, 4),
                            ha="center", va="bottom", fontsize=6, color="#B03A2E")
        ax.set_ylabel("fluctuation to RIGID set, Å\n(avg over TCRs)")
        ax.set_xlabel("residue (IMGT, sequential — every residue incl. insertions shown)")
        ax.set_xlim(-1, len(targets)); ax.margins(x=0)
        ax.set_title(f"chain {ch}   |   flanks per CDR (left+right):  "
                     + ",  ".join(f"{c} {l}+{r}" for c, l, r in flankinfo), fontsize=10)
        ax.legend(handles=[mpatches.Patch(color=RED, alpha=.45, label="CDR loop"),
                           mpatches.Patch(color=BLUE, alpha=.45, label="RIGID residue"), l1, l2],
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, framealpha=.95)
        ymaxes.append(np.nanmax(ymax))
        print(f"chain {ch} flanks (left+right): " + ", ".join(f"{c}={l}+{r}" for c, l, r in flankinfo))
    for ax in axes:                                   # SAME y-axis scale for both chains
        ax.set_ylim(0, max(ymaxes) * 1.05)
    fig.suptitle(f"Fluctuation to the RIGID set (mean & max over the rigid residues), per residue — sequential x-axis  "
                 f"(alignment-free; averaged over the {len(data)} TCRs that HAVE each residue)\n"
                 f"red block = CDR loop · blue block = RIGID residue · purple x-label = IMGT insertion (e.g. 112.1) · "
                 f"red number above a point = # of {len(data)} TCRs with that residue (unlabeled = all {len(data)})",
                 y=1.02, fontsize=12)
    out = f"{FIG}/loops_flanks_fluctuation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("fig ->", out)
    # annotated copy: overlay conserved TCR sequence landmarks
    for ax, ch in zip(axes, "AB"):
        annotate_conserved(ax, TARGETS[ch])
    fig.suptitle("Fluctuation to the RIGID set, per residue — annotated with conserved TCR V-domain landmarks\n"
                 "Cys23–Cys104 disulfide · Trp41 core keystone · hydrophobic-89 · J-motif F–G–X–G (118–121) · "
                 "red = CDR loop, blue = RIGID residue", y=1.02, fontsize=12)
    out2 = f"{FIG}/loops_flanks_fluctuation_annotated.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("annotated ->", out2)


if __name__ == "__main__":
    main()
