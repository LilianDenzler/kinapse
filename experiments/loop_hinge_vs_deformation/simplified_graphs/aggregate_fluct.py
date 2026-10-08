#!/usr/bin/env python
"""Average the per-TCR fluctuation over all TCRs and DERIVE a rigid framework set by thresholding the
MUTUAL fluctuation. Per chain, produce one figure with:
    (1) averaged per-residue edge-fluctuation profile (mean +/- SD across TCRs; residues coloured by
        whether they survive at tau=0.5),
    (2) rigid-set size vs threshold tau  (largest set whose EVERY pairwise D_std <= tau; you pick tau),
    (3) a PyMOL image of one TCR (3QH3) with the tau=0.5 rigid residues coloured.

"Rigid relative to each other" = a mutually-rigid clique: a set of residues whose pairwise CA-CA
distances all barely fluctuate. Everything is averaged over TCRs (transferable) and alignment-free.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from graph_build import consensus, CDR_RANGES, HERE

RES = f"{HERE}/results_fluct"; FIG = f"{HERE}/figures"
TAU_PICK = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5      # avg-fluctuation threshold (Å); pass on CLI
MAX_CAP = 0.5                                                    # extra: worst-case (max over TCRs) pairwise cap


def load_all():
    data = {}
    for f in sorted(glob.glob(f"{RES}/*_fluct.npz")):
        z = np.load(f); sid = os.path.basename(f)[:4]
        data[sid] = {ch: (z[f"{ch}_imgt"], z[f"{ch}_Dstd"]) for ch in "AB"}
    return data


def avg_pairwise(data, ch):
    """Average D_std over TCRs on the INTERSECTION of IMGT positions (complete matrix)."""
    common = None
    for sid in data:
        s = set(data[sid][ch][0].tolist()); common = s if common is None else common & s
    common = np.array(sorted(common))
    acc = np.zeros((len(common), len(common)))
    for sid in data:
        imgt, S = data[sid][ch]; pos = np.array([np.where(imgt == n)[0][0] for n in common])
        acc += S[np.ix_(pos, pos)]
    return common, acc / len(data)


def max_pairwise(data, ch):
    """MAX D_std over TCRs per pair, on the intersection of IMGT positions (worst-case fluctuation)."""
    common = None
    for sid in data:
        s = set(data[sid][ch][0].tolist()); common = s if common is None else common & s
    common = np.array(sorted(common))
    acc = np.zeros((len(common), len(common)))
    for sid in data:
        imgt, S = data[sid][ch]; pos = np.array([np.where(imgt == n)[0][0] for n in common])
        acc = np.maximum(acc, S[np.ix_(pos, pos)])
    return common, acc


def per_residue_profile(data, ch):
    """Per-IMGT mean edge fluctuation, averaged over TCRs (union positions)."""
    vals = defaultdict(list)
    for sid in data:
        imgt, S = data[sid][ch]; S2 = S.copy(); np.fill_diagonal(S2, np.nan)
        for n, v in zip(imgt, np.nanmean(S2, axis=1)):
            vals[int(n)].append(float(v))
    imgt = np.array(sorted(vals))
    return imgt, np.array([np.mean(vals[n]) for n in imgt]), np.array([np.std(vals[n]) for n in imgt])


def rigid_set(S, tau):
    """Greedy largest mutually-rigid set: all pairwise D_std <= tau. Returns indices into S."""
    alive = np.ones(len(S), bool); V = (S > tau).copy(); np.fill_diagonal(V, False)
    while True:
        vc = (V & alive[None, :]).sum(1); vc[~alive] = -1
        if vc.max() <= 0:
            break
        alive[int(vc.argmax())] = False
    return np.where(alive)[0]


def render_rigid(chain, rigid_imgt, color="marine", tag="", sysid="3QH3"):
    from pymol_view import write_fv_pdb
    pdb = write_fv_pdb(sysid)
    import pymol2
    out = f"{FIG}/_rigid_{sysid}_{chain}_tau{TAU_PICK}{tag}.png"
    with pymol2.PyMOL() as P:
        cmd = P.cmd; cmd.load(pdb, "tcr")
        cmd.hide("everything"); cmd.show("cartoon", f"chain {chain}")
        cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
        cmd.color("grey80", f"chain {chain}")
        if len(rigid_imgt):
            cmd.color(color, f"chain {chain} and resi {'+'.join(map(str, map(int, rigid_imgt)))}")
        cmd.orient(f"chain {chain}"); cmd.ray(1200, 1200); cmd.png(out, dpi=150)
    return out


def make_chain_figure(data, ch, taus):
    imgt_p, mean, sd = per_residue_profile(data, ch)
    common, S = avg_pairwise(data, ch)                       # selection uses AVERAGE pairwise D_std
    _, Smax = max_pairwise(data, ch)                         # panel 3 shows the MAX across TCRs
    curve = np.array([len(rigid_set(S, t)) for t in taus])
    keptidx = rigid_set(S, TAU_PICK)                          # kept: mutually rigid on AVERAGE <= tau
    rigid = set(common[keptidx].tolist())
    # even-more-rigid subset: within kept, largest group whose WORST-CASE (max over TCRs) pairwise <= MAX_CAP
    subidx = keptidx[rigid_set(Smax[np.ix_(keptidx, keptidx)], MAX_CAP)]
    ultra = set(common[subidx].tolist())
    conset = consensus()[ch]
    png = render_rigid(ch, sorted(rigid), color="marine", tag="_kept")
    png_u = render_rigid(ch, sorted(ultra), color="firebrick", tag="_ultra")
    n05 = len(rigid); nu = len(ultra)

    fig = plt.figure(figsize=(24, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=.3, wspace=.28)
    # (1) profile
    axP = fig.add_subplot(gs[0, :]); top = float((mean + sd).max())
    for cdr, (lo, hi) in CDR_RANGES.items():
        axP.axvspan(lo - .5, hi + .5, color="#F4C7C3", alpha=.7, zorder=0)
        axP.text((lo + hi) / 2, top, cdr, ha="center", va="bottom", fontsize=8, color="#B03A2E")
    axP.fill_between(imgt_p, mean - sd, mean + sd, color="0.8", alpha=.6, zorder=1, label="±1 SD across TCRs")
    axP.plot(imgt_p, mean, "-", color="0.45", lw=1, zorder=2)
    inr = np.array([n in rigid for n in imgt_p])
    axP.scatter(imgt_p[inr], mean[inr], s=26, c="#1f77b4", ec="k", lw=.3, zorder=4, label=f"kept @τ={TAU_PICK} ({n05})")
    axP.scatter(imgt_p[~inr], mean[~inr], s=26, c="#d62728", ec="k", lw=.3, zorder=4, label="excluded")
    axP.axhline(TAU_PICK, color="green", ls="--", lw=1)
    axP.set_ylabel("mean edge fluctuation (Å)\naveraged over TCRs"); axP.set_xlabel("IMGT residue")
    axP.set_ylim(0, None); axP.legend(fontsize=8, ncol=3, loc="upper center")
    axP.set_title(f"chain {ch} — averaged per-residue edge-fluctuation profile ({len(data)} TCRs); shaded = CDR")
    # (2) threshold curve
    axC = fig.add_subplot(gs[1, 0])
    axC.plot(taus, curve, "-o", ms=3, color="#4C72B0")
    axC.axvline(TAU_PICK, color="green", ls="--"); axC.plot(TAU_PICK, n05, "o", ms=10, mfc="green", mec="k", zorder=5)
    axC.axhline(len(conset), color="0.5", ls=":", label=f"kinapse consensus size ({len(conset)})")
    axC.annotate(f"τ={TAU_PICK} → {n05} residues", (TAU_PICK, n05), textcoords="offset points", xytext=(10, -2), fontsize=10)
    axC.set_xlabel("mutual-fluctuation threshold τ (Å)"); axC.set_ylabel("# residues kept (mutually-rigid set)")
    axC.set_title("rigid-set size vs threshold\n(largest set with EVERY pairwise D_std ≤ τ)"); axC.legend(fontsize=8)
    axC.set_ylim(0, None)
    # (3) MAX pairwise deviation (across TCRs) WITHIN the kept set
    axS = fig.add_subplot(gs[1, 1])
    ridx = np.array([np.where(common == n)[0][0] for n in sorted(rigid)])
    sub = Smax[np.ix_(ridx, ridx)]
    offmax = float(sub[~np.eye(len(sub), dtype=bool)].max())
    im = axS.imshow(sub, cmap="inferno", vmin=0, vmax=offmax)
    fig.colorbar(im, ax=axS, fraction=.046, pad=.02, label="max pairwise D_std across TCRs (Å)")
    klab = sorted(rigid); step = max(1, len(klab) // 20); tk = list(range(0, len(klab), step))
    axS.set_xticks(tk); axS.set_xticklabels([klab[t] for t in tk], rotation=90, fontsize=5)
    axS.set_yticks(tk); axS.set_yticklabels([klab[t] for t in tk], fontsize=5)
    axS.set_title(f"MAX pairwise deviation across TCRs, within the kept set\n"
                  f"(worst-case fluctuation of any pair, in any TCR; max={offmax:.2f} Å)")
    # (4) pymol: kept set
    axI = fig.add_subplot(gs[1, 2]); axI.imshow(plt.imread(png)); axI.set_axis_off()
    axI.set_title(f"3QH3 chain {ch}: kept set (avg ≤ τ={TAU_PICK})\nblue = kept ({n05}), grey = excluded")
    # (5) pymol: even-more-rigid subset (worst-case max <= MAX_CAP)
    axU = fig.add_subplot(gs[1, 3]); axU.imshow(plt.imread(png_u)); axU.set_axis_off()
    axU.set_title(f"3QH3 chain {ch}: ULTRA-rigid subset\nred = kept ∩ max-dev ≤ {MAX_CAP} Å ({nu}), grey = rest")
    fig.suptitle(f"Fluctuation-derived rigid framework — chain {ch}  (averaged over {len(data)} TCRs, alignment-free)",
                 y=1.005, fontsize=13)
    suf = "" if abs(TAU_PICK - 0.5) < 1e-9 else f"_tau{TAU_PICK}"      # tau=0.5 keeps the original filename
    out = f"{FIG}/rigidpick_avg_chain{ch}{suf}.png"; fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)

    print(f"\n=== chain {ch} : rigid@{TAU_PICK} = {n05} residues   (kinapse consensus = {len(conset)}) ===")
    print(f"  τ→size:  " + "  ".join(f"{t:.2f}:{c}" for t, c in zip(taus, curve) if abs((t*100) % 10) < 1e-6))
    print(f"  kept@{TAU_PICK} (avg): {sorted(rigid)}")
    print(f"  ULTRA-rigid (kept ∩ max-dev ≤ {MAX_CAP}) = {nu} residues: {sorted(ultra)}")
    print(f"  kept but NOT in consensus: {sorted(rigid - conset)}")
    print(f"  in consensus but dropped:  {sorted(conset - rigid)}")
    return out


def main():
    data = load_all()
    print(f"{len(data)} TCRs loaded")
    taus = np.round(np.arange(0.20, 1.21, 0.02), 3)
    for ch in "AB":
        print("fig ->", make_chain_figure(data, ch, taus))
    print("AGG_DONE")


if __name__ == "__main__":
    main()
