#!/usr/bin/env python
"""Loop-length normalisation + does the rigid/deform FRACTION depend on loop length?
Uses results_rigid_deform.json (rigid, deform, total Å², N residues per TCR x CDR).
 - total Å² should grow with N (trivial sum + physical reach).
 - the fraction = deform/total is already length-COUNT independent (ratio); test for residual PHYSICAL length dep.
 - separate the across-CDR identity confound (CDR3 long+floppy) from the within-CDR length effect.
-> figures/fraction_vs_length.png + printed correlations."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
CCOL = {"A_CDR1": "#1f77b4", "A_CDR2": "#2ca02c", "A_CDR3": "#d62728",
        "B_CDR1": "#17becf", "B_CDR2": "#9467bd", "B_CDR3": "#ff7f0e"}


def main():
    res = json.load(open(f"{HERE}/results_rigid_deform.json"))
    tcrs = list(res.keys())
    rows = []  # (tcr, cdr, N, rigid, deform, total, frac)
    for t in tcrs:
        for c in CDRS:
            d = res[t][c]; tot = d["total"]
            rows.append((t, c, d["N"], d["rigid"], d["deform"], tot, d["deform"] / tot))
    N = np.array([r[2] for r in rows]); Rg = np.array([r[3] for r in rows])
    Df = np.array([r[4] for r in rows]); Tt = np.array([r[5] for r in rows]); Fr = np.array([r[6] for r in rows])
    cdr_of = [r[1] for r in rows]
    perN = Tt / N

    def corr(a, b):
        return stats.pearsonr(a, b)[0], stats.spearmanr(a, b)[0], stats.spearmanr(a, b)[1]

    print("ALL 132 loops:")
    for lab, y in [("total Å² vs N", Tt), ("per-residue Å² vs N", perN), ("rigid Å² vs N", Rg),
                   ("deform Å² vs N", Df), ("deform FRACTION vs N", Fr)]:
        r, rho, p = corr(N, y)
        print(f"  {lab:26} Pearson r={r:+.2f}  Spearman ρ={rho:+.2f}  (p={p:.1e})")
    print("\nWITHIN each CDR (fraction vs N):")
    perc = {}
    for c in CDRS:
        m = np.array([x == c for x in cdr_of])
        if m.sum() >= 4 and np.ptp(N[m]) > 0:
            rho, p = stats.spearmanr(N[m], Fr[m])
            perc[c] = (rho, p, N[m].min(), N[m].max())
            print(f"  {c:8} ρ={rho:+.2f} (p={p:.2f})  N range {N[m].min()}-{N[m].max()}  <frac>={Fr[m].mean():.2f}")
        else:
            perc[c] = (np.nan, np.nan, N[m].min(), N[m].max())
            print(f"  {c:8} (constant length N={N[m].min()}-{N[m].max()}; no within-CDR length variation)")
    plot(tcrs, res, N, Rg, Df, Tt, Fr, perN, cdr_of, perc)


def plot(tcrs, res, N, Rg, Df, Tt, Fr, perN, cdr_of, perc):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    # A: per-residue rigid vs deform per TCR (length-normalised), sorted by fraction
    Rg_t = np.array([sum(res[t][c]["rigid"] for c in CDRS) for t in tcrs])
    Df_t = np.array([sum(res[t][c]["deform"] for c in CDRS) for t in tcrs])
    Nt = np.array([sum(res[t][c]["N"] for c in CDRS) for t in tcrs])
    rR, dR = Rg_t / Nt, Df_t / Nt; fr = Df_t / (Rg_t + Df_t); o = np.argsort(fr)
    x = np.arange(len(tcrs))
    ax[0, 0].bar(x, rR[o], 0.8, color="#3B6EA5", label="rigid / residue")
    ax[0, 0].bar(x, dR[o], 0.8, bottom=rR[o], color=(0.75, 0.22, 0.17), label="deformation / residue")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels([tcrs[i] for i in o], rotation=90, fontsize=8)
    ax[0, 0].set_ylabel("per-residue displacement variance (Å²/res)")
    ax[0, 0].set_title("a  length-NORMALISED rigid vs deform per TCR (Å²/residue)", fontsize=10, fontweight="bold")
    ax[0, 0].legend(fontsize=9)
    # B: total Å² vs N (raw grows) and per-residue vs N (flattened)
    for c in CDRS:
        m = [i for i, cc in enumerate(cdr_of) if cc == c]
        ax[0, 1].scatter(N[m], Tt[m], s=28, color=CCOL[c], label=c, alpha=0.8)
    rT, rhoT, _ = stats.pearsonr(N, Tt)[0], *stats.spearmanr(N, Tt)
    ax[0, 1].set_xlabel("loop length N (residues)"); ax[0, 1].set_ylabel("total Å² (summed over loop)")
    ax[0, 1].set_title(f"b  summed displacement vs length  (r={rT:+.2f}, ρ={rhoT:+.2f})\n→ grows with length (expected)", fontsize=10, fontweight="bold")
    ax[0, 1].legend(fontsize=7, ncol=2)
    # C: fraction vs N (the question)
    for c in CDRS:
        m = [i for i, cc in enumerate(cdr_of) if cc == c]
        ax[1, 0].scatter(N[m], Fr[m], s=34, color=CCOL[c], label=c, alpha=0.85)
    rF, pF = stats.pearsonr(N, Fr); rhoF, prho = stats.spearmanr(N, Fr)
    b, a = np.polyfit(N, Fr, 1); xx = np.array([N.min(), N.max()])
    ax[1, 0].plot(xx, a + b * xx, "k--", lw=1.5, label=f"fit (all): ρ={rhoF:+.2f}")
    ax[1, 0].set_xlabel("loop length N (residues)"); ax[1, 0].set_ylabel("deformation fraction = deform/total")
    ax[1, 0].set_title(f"c  does the FRACTION depend on length?  all: r={rF:+.2f}, ρ={rhoF:+.2f} (p={prho:.0e})\n"
                       "colour = CDR (watch the identity confound)", fontsize=10, fontweight="bold")
    ax[1, 0].legend(fontsize=7, ncol=2)
    # D: within-CDR Spearman(fraction, N) vs the pooled across-CDR value
    cs = [c for c in CDRS if np.isfinite(perc[c][0])]
    vals = [perc[c][0] for c in cs]
    ax[1, 1].bar(range(len(cs)), vals, color=[CCOL[c] for c in cs])
    ax[1, 1].axhline(rhoF, color="k", ls="--", lw=1.5, label=f"pooled across-CDR ρ={rhoF:+.2f}")
    ax[1, 1].axhline(0, color="0.6", lw=0.8)
    ax[1, 1].set_xticks(range(len(cs))); ax[1, 1].set_xticklabels(cs, rotation=20, fontsize=8)
    ax[1, 1].set_ylabel("Spearman ρ (fraction vs N) within CDR"); ax[1, 1].set_ylim(-1, 1)
    ax[1, 1].set_title("d  within-CDR length effect vs pooled\n(if within-CDR bars ≈ 0 but pooled > 0 → it's identity, not length)", fontsize=10, fontweight="bold")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle("Loop length vs rigid/deformation: magnitude grows with length, but is the FRACTION length-determined?",
                 y=1.0, fontsize=12, fontweight="bold")
    out = f"{HERE}/figures/fraction_vs_length.png"
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("\nfig ->", out, flush=True)


if __name__ == "__main__":
    main()
