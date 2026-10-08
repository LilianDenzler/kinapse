#!/usr/bin/env python
"""Per-TCR validation of the CDR-loop CLAMP model.

The 22-TCR *average* analysis (../simplified_graphs) established that each CDR loop is anchored on ONE
stem by a rigid flanking beta-strand ("clamp"), orange-shaded in loops_flanks_fluctuation_annotated.png:

    CDR1 -> C-terminal stem  (C-strand, 39-45; ends at the loop's C-stem)
    CDR2 -> N-terminal stem  (C'-strand, 49-56; ends at the loop's N-stem)
    CDR3 -> N-terminal stem  (F-strand, 98-107; Cys104 + V-germline C-A-x anchor)

Here we DROP the averaging and check the claim for EVERY TCR separately: is each clamp stem genuinely
rigid & static relative to the ultra-rigid framework set, in that individual trajectory?

Metric (alignment-free, identical to the averaged analysis but per-TCR):
    fluct_to_rigid(i) = mean over ultra-rigid residues j (j!=i) of  D_std(i,j)
    D_std(i,j)        = std over MD frames of the CA_i-CA_j distance   (from results_fluct/*_fluct.npz)
A residue with fluct_to_rigid <= 0.5 A does not move relative to the rigid core = it is clamped.

Everything reuses the cached per-TCR D_std matrices (kinapse IMGT numbering, insertions preserved as
112.1 keys) and the ultra-rigid set from ../simplified_graphs/rigid_framework.json. No MD re-run.
"""
from __future__ import annotations
import os, sys, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SG = os.path.abspath(f"{HERE}/../simplified_graphs")
RES = f"{SG}/results_fluct"
RIGID_JSON = f"{SG}/rigid_framework.json"

CDR_RANGES = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}   # IMGT, both chains
CLAMP_SIDE = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}                    # clamped stem per CDR (from FINDINGS)
TAU = 0.5                                                               # rigidity threshold used to build the set
STEM_N = 3                                                             # residues per stem


def stems(cdr):
    """(N-stem, C-stem) IMGT residue lists = the STEM_N terminal loop residues on each side."""
    lo, hi = CDR_RANGES[cdr]
    return [lo + k for k in range(STEM_N)], [hi - k for k in range(STEM_N)][::-1]


def clamp_free(cdr):
    """(clamped_stem, free_stem) IMGT residue lists for a CDR."""
    ns, cs = stems(cdr)
    return (cs, ns) if CLAMP_SIDE[cdr] == "C" else (ns, cs)


def load_rigid():
    j = json.load(open(RIGID_JSON))
    return {ch: list(map(int, j[f"chain_{ch}"]["ultra_rigid"])) for ch in "AB"}


def load_tcr(sid):
    """Return {ch: (imgt_float_array, Dstd, {imgt_key: index})} for one TCR."""
    z = np.load(f"{RES}/{sid}_fluct.npz")
    out = {}
    for ch in "AB":
        imgt = np.array([round(float(x), 1) for x in z[f"{ch}_imgt"]])
        out[ch] = (imgt, z[f"{ch}_Dstd"], {n: i for i, n in enumerate(imgt)})
    return out


def fluct_to_rigid(D, idx, targets, rigid, reduce="mean"):
    """Per target residue i: reduce over ultra-rigid j (present, j!=i) of D_std(i,j). NaN if i absent."""
    out = {}
    rig_present = [j for j in rigid if float(j) in idx]
    for i in targets:
        ki = idx.get(float(i))
        if ki is None:
            out[i] = np.nan; continue
        vals = [D[ki, idx[float(j)]] for j in rig_present if float(j) != float(i)]
        out[i] = float(np.mean(vals) if reduce == "mean" else np.max(vals)) if vals else np.nan
    return out


def all_tcrs():
    return sorted(os.path.basename(p)[:4] for p in glob.glob(f"{RES}/*_fluct.npz"))


def compute():
    """Return a nested dict of per-TCR fluct-to-rigid for every loop+flank residue, and clamp/free means."""
    rigid = load_rigid()
    tcrs = all_tcrs()
    # profile window per CDR = a couple residues of flank on each side, union over all TCRs handled at plot time
    windows = {cdr: list(range(lo - 3, hi + 4)) for cdr, (lo, hi) in CDR_RANGES.items()}
    R = {"tcrs": tcrs, "rigid": rigid, "windows": windows,
         "clamp_side": CLAMP_SIDE, "cdr_ranges": CDR_RANGES, "tau": TAU,
         "per_res_mean": {}, "per_res_max": {}, "clamp_mean": {}, "free_mean": {},
         "clamp_max": {}, "free_max": {}, "apex_mean": {}}
    for sid in tcrs:
        d = load_tcr(sid)
        R["per_res_mean"][sid] = {}; R["per_res_max"][sid] = {}
        R["clamp_mean"][sid] = {}; R["free_mean"][sid] = {}
        R["clamp_max"][sid] = {}; R["free_max"][sid] = {}; R["apex_mean"][sid] = {}
        for ch in "AB":
            imgt, D, idx = d[ch]
            for cdr, (lo, hi) in CDR_RANGES.items():
                key = f"{ch}_{cdr}"
                # every present residue in the window incl. insertions (112.1) -> never dropped
                win = sorted([n for n in imgt if lo - 3 <= n <= hi + 3])
                fm = fluct_to_rigid(D, idx, win, rigid[ch], "mean")
                fx = fluct_to_rigid(D, idx, win, rigid[ch], "max")
                R["per_res_mean"][sid][key] = {str(k): v for k, v in fm.items()}
                R["per_res_max"][sid][key] = {str(k): v for k, v in fx.items()}
                clamp, free = clamp_free(cdr)
                cvals = [fm[r] for r in clamp if r in fm and not np.isnan(fm[r])]
                fvals = [fm[r] for r in free if r in fm and not np.isnan(fm[r])]
                cxvals = [fx[r] for r in clamp if r in fx and not np.isnan(fx[r])]
                fxvals = [fx[r] for r in free if r in fx and not np.isnan(fx[r])]
                # apex = central loop residues (integer positions between the stems)
                apex = [n for n in win if lo + STEM_N <= n <= hi - STEM_N]
                avals = [fm[r] for r in apex if r in fm and not np.isnan(fm[r])]
                R["clamp_mean"][sid][key] = float(np.mean(cvals)) if cvals else np.nan
                R["free_mean"][sid][key] = float(np.mean(fvals)) if fvals else np.nan
                R["clamp_max"][sid][key] = float(np.max(cxvals)) if cxvals else np.nan
                R["free_max"][sid][key] = float(np.max(fxvals)) if fxvals else np.nan
                R["apex_mean"][sid][key] = float(np.mean(avals)) if avals else np.nan
    return R


def summary(R):
    keys = [f"{ch}_{c}" for c in ["CDR1", "CDR2", "CDR3"] for ch in "AB"]
    tcrs = R["tcrs"]
    print(f"\n{'='*88}\nPER-TCR CLAMP VALIDATION  (fluct-to-rigid, mean over ultra-rigid core, A)\n"
          f"clamp = predicted anchored stem; free = opposite stem; threshold = {R['tau']} A\n{'='*88}")
    print(f"{'':12}" + "".join(f"{k:>11}" for k in keys))
    print(f"{'clamp side':12}" + "".join(f"{CLAMP_SIDE[k[2:]]:>11}" for k in keys))
    print("-" * 88)
    for sid in tcrs:
        row = "".join(f"{R['clamp_mean'][sid][k]:>11.2f}" for k in keys)
        print(f"{sid:12}{row}")
    print("-" * 88)
    def col_nanmean(dd, k):
        vals = [dd[s][k] for s in tcrs if not np.isnan(dd[s][k])]
        return np.mean(vals) if vals else np.nan
    for lbl, dd in [("clamp mean", R["clamp_mean"]), ("free mean", R["free_mean"]),
                    ("apex mean", R["apex_mean"])]:
        row = "".join(f"{col_nanmean(dd, k):>11.2f}" for k in keys)
        print(f"{lbl:12}{row}")
    print(f"{'clamp MAX':12}" + "".join(f"{np.nanmax([R['clamp_mean'][s][k] for s in tcrs]):>11.2f}" for k in keys))
    print("-" * 88)
    # pass rates
    npass = {k: sum(R["clamp_mean"][s][k] <= R["tau"] for s in tcrs) for k in keys}
    print(f"{'pass<=0.5':12}" + "".join(f"{npass[k]:>8}/{len(tcrs)}" for k in keys))
    asym = {k: sum(R["clamp_mean"][s][k] < R["free_mean"][s][k] for s in tcrs) for k in keys}
    print(f"{'clamp<free':12}" + "".join(f"{asym[k]:>8}/{len(tcrs)}" for k in keys))
    # worst offenders
    print("\nWorst clamp-stem fluct-to-rigid per region (TCR : value):")
    for k in keys:
        vals = [(R["clamp_mean"][s][k], s) for s in tcrs]
        v, s = max(vals)
        print(f"  {k:10} worst = {s} {v:.2f} A   (median {np.nanmedian([x for x, _ in vals]):.2f})")


if __name__ == "__main__":
    R = compute()
    with open(f"{HERE}/results_clamp_pertcr.json", "w") as f:
        json.dump(R, f, indent=1)
    summary(R)
    print(f"\nwrote {HERE}/results_clamp_pertcr.json")
