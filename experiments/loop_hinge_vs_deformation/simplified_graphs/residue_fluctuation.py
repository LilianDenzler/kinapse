#!/usr/bin/env python
"""Per-residue EDGE-FLUCTUATION profile along the TCR variable domains (alignment-free).

Edge fluctuation of a residue i:
    for every other CA atom j (same chain's variable domain), take the CA-CA distance d_ij(t) and its
    std over the MD frames, std_t(d_ij);  the residue's value = MEAN over j of std_t(d_ij).
It is an alignment-free flexibility measure: a residue locked in the rigid core keeps almost-constant
distances to the rest (low value); a mobile loop residue's distances swing a lot (high value). No
superposition is used -- only pairwise distances, which are invariant to global tumbling.

We plot it per chain vs IMGT position, shading the CDR loops and marking the super-rigid CONSENSUS
residues and the sequential flanking residues -- so you can see exactly how the fluctuation behaves at
the CDR<->framework boundary.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from graph_build import load_md, consensus, CDR_RANGES, flanking_residues, HERE

CDR_SHADE = "#F4C7C3"


def residue_fluctuation(xyz, imap, chain):
    """Return (imgt_sorted, fluct) for one chain's variable-domain CA atoms."""
    imgt = sorted(imap[chain]); ai = np.array([imap[chain][n] for n in imgt])
    P = xyz[:, ai]                                                    # (T, N, 3)
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)  # (T, N, N)
    S = D.std(0)                                                      # (N, N) per-edge fluctuation
    np.fill_diagonal(S, np.nan)
    return np.array(imgt), np.nanmean(S, axis=1)                     # mean incident-edge fluctuation


def main(sysid="3QH3"):
    tv, xyz, imap = load_md(sysid)
    con = consensus()
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    for ax, ch in zip(axes, "AB"):
        imgt, fl = residue_fluctuation(xyz, imap, ch)
        conset = con[ch]; flankset = {n for cdr in CDR_RANGES for n in flanking_residues(ch, cdr)}
        # shade CDR regions
        for cdr, (lo, hi) in CDR_RANGES.items():
            ax.axvspan(lo - .5, hi + .5, color=CDR_SHADE, alpha=.8, zorder=0)
            ax.text((lo + hi) / 2, ax.get_ylim()[1], cdr, ha="center", va="bottom", fontsize=8, color="#B03A2E")
        ax.plot(imgt, fl, "-", color="0.5", lw=1, zorder=1)
        # colour each residue by its role
        role = ["frame" if n in conset else "flank" if n in flankset else
                "cdr" if any(lo <= n <= hi for lo, hi in CDR_RANGES.values()) else "other" for n in imgt]
        cmap = {"frame": "#4C72B0", "flank": "#2CA02C", "cdr": "#D62728", "other": "#999999"}
        for r in cmap:
            m = [i for i, rr in enumerate(role) if rr == r]
            if m:
                ax.scatter(imgt[m], fl[m], s=26, c=cmap[r], zorder=3, edgecolor="k", linewidths=.3,
                           label={"frame": "consensus rigid", "flank": "flanking", "cdr": "CDR",
                                  "other": "framework (not in rigid set)"}[r])
        ax.set_ylabel("mean edge fluctuation (Å)\n= mean over j of std_t |CA_i–CA_j|")
        ax.set_title(f"chain {ch}"); ax.set_xlabel("IMGT residue"); ax.set_ylim(0, None)
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        ax.margins(x=.01)
    fig.suptitle(f"{sysid} — per-residue edge-fluctuation profile (alignment-free); shaded = CDR loops\n"
                 "how mobile each residue is relative to the rest of its domain, and where the rigid consensus set sits",
                 y=1.02, fontsize=12)
    out = f"{HERE}/figures/fluctuation_{sysid}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    # quick numeric check at each CDR boundary
    print(f"{sysid} boundary check (mean edge fluctuation, Å):")
    for ch in "AB":
        imgt, fl = residue_fluctuation(xyz, imap, ch); d = dict(zip(imgt.tolist(), fl.tolist()))
        for cdr, (lo, hi) in CDR_RANGES.items():
            inside = np.mean([d[n] for n in range(lo, hi + 1) if n in d])
            just_out = [d[n] for n in (lo - 1, hi + 1) if n in d]
            core = np.median(list(d.values()))
            print(f"  {ch}_{cdr}: loop-mean={inside:.2f}  boundary({lo-1},{hi+1})={np.round(just_out,2).tolist()}  domain-median={core:.2f}")
    print("fig ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3QH3")
