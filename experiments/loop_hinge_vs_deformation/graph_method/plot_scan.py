#!/usr/bin/env python
"""Plot the alignment-free hinge-anchor position scan (rigidity selector + f_hinge descriptor), per CDR."""
from __future__ import annotations
import os, sys, glob
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__)); RES = f"{ROOT}/results_scan"; FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 130, "axes.axisbelow": True})
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/*_scan.csv"))], ignore_index=True)
print(f"{m.system.nunique()} TCRs")

RIG, HIN = "#C44E52", "#4C72B0"
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
best = {}
for ax, cdr in zip(axes.ravel(), CDRS):
    d = m[m.cdr == cdr]
    offs = sorted(d.s.unique())
    rig_med = [d[d.s == s].rigid.median() for s in offs]
    hin_med = [d[d.s == s].f_hinge.median() for s in offs]
    # shade the in-loop region (s<=0): base sits inside the loop, hinge undefined
    ax.axvspan(min(offs) - 0.5, 0.5, color="0.85", alpha=.5, zorder=0)
    ax.text(0.02, 0.96, "base inside loop\n(no hinge)", transform=ax.transAxes, va="top", ha="left", fontsize=7, color="0.4")
    # rigidity (left axis) -- lower = more rigid
    for s in offs:
        v = d[d.s == s].rigid.values
        ax.scatter(np.full(len(v), s) + np.random.default_rng(0).uniform(-.08, .08, len(v)), v, s=9, c=RIG, alpha=.35, edgecolor="none")
    ax.plot(offs, rig_med, "-o", color=RIG, lw=2, ms=4, label="anchor rigidity (↓ = rigid)")
    ax.set_ylabel("anchor↔framework dist. std (Å)", color=RIG); ax.tick_params(axis="y", labelcolor=RIG)
    ax.set_ylim(0, None)
    # choose best framework offset = min median rigidity among s>=1
    fw = [(s, r) for s, r in zip(offs, rig_med) if s >= 1]
    bs = min(fw, key=lambda t: t[1])[0]; best[cdr] = bs
    ax.axvline(bs, color="k", ls="--", lw=1.3, zorder=6)
    ax.text(bs, ax.get_ylim()[1] * .98, f" best s={bs}", fontsize=8, va="top")
    # f_hinge (right axis)
    ax2 = ax.twinx()
    ax2.plot(offs, hin_med, "-s", color=HIN, lw=2, ms=4, label="f_hinge (descriptor)")
    ax2.set_ylabel("f_hinge (rigid-reorient. frac.)", color=HIN); ax2.tick_params(axis="y", labelcolor=HIN)
    ax2.set_ylim(0, 1); ax2.grid(False)
    ax.set_title(cdr); ax.set_xlabel("anchor offset s   (lo−s / hi+s;  s≥1 = framework)")
    ax.set_xticks(offs)
fig.suptitle("Alignment-free hinge-anchor position scan per CDR — slide the base out of the loop into the framework\n"
             "red = anchor rigidity (pick the MINIMUM among framework offsets);  blue = f_hinge (climbs with s ⇒ never the selector);  grey = base inside loop",
             y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_hinge_scan.png", bbox_inches="tight"); plt.close(fig)

print("Best framework anchor offset per CDR (min median rigidity, s>=1):")
for c in CDRS:
    d = m[(m.cdr == c) & (m.s == best[c])]
    print(f"  {c}: s={best[c]}  ->  residues {int(d.L.median())}/{int(d.R.median())}  "
          f"rigid={d.rigid.median():.2f}  f_hinge={d.f_hinge.median():.2f}  consensus={bool(d.is_consensus.mode().iloc[0])}")
print(f"\nfig -> {FIG}/fig_hinge_scan.png")
