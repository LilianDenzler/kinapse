#!/usr/bin/env python
"""Dataset-wide torsion-space feet-anchored loop ENM vs MD deformation (all 22 TCRs x 6 CDRs).
Reuses anm_loop_torsion. For each loop: per-residue torsion-ENM deformation prediction, MD deformation profile,
Pearson r over interior residues, and number of torsional deformation DOF (0 => loop is rigid by geometry).
-> results_anm_torsion.json + figures/anm_torsion_atlas.png + printed table."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
from scipy.stats import pearsonr
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all
from anm_loop_torsion import backbone_indices, torsion_modes, md_deform

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]


def main():
    tcrs = sorted(os.path.basename(f)[:4] for f in glob.glob(f"{HERE}/results_swing/*.npz"))
    rig = json.load(open(f"{HERE}/rigid_framework.json")); res = {}
    for n, s in enumerate(tcrs):
        try:
            tv, xyz, imap = load_md(s); top = tv.mdtraj.topology; cache = {}; res[s] = {}
            for cdr in CDRS:
                ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
                if ch not in cache:
                    fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
                    cache[ch] = superpose_all(xyz, fw)
                supr = cache[ch]
                lk = sorted(k for k in imap[ch] if lo <= k <= hi); Nres = len(lk)
                ca = np.array([imap[ch][k] for k in lk]); loop_ca = supr[:, ca]
                md, med = md_deform(loop_ca)
                bb = backbone_indices(top, ca)
                if bb is None:
                    res[s][cdr] = {"r": None, "ndof": None, "N": Nres}; continue
                fluct, ndof, nmode = torsion_modes(supr[med][bb], Nres)
                pred = np.sqrt(np.maximum(fluct.reshape(Nres, 3)[:, 1], 0))
                interior = list(range(1, Nres - 1))
                if ndof >= 1 and len(interior) >= 3 and pred[interior].std() > 0:
                    r = float(pearsonr(md[interior], pred[interior])[0])
                else:
                    r = None
                res[s][cdr] = {"r": r, "ndof": int(ndof), "N": int(Nres)}
            done = [c for c in CDRS if res[s][c].get("r") is not None]
            print(f"[{n+1}/{len(tcrs)}] {s}: " + " ".join(f"{c[2:]}:{res[s][c]['r']:+.2f}" for c in done), flush=True)
        except Exception as e:
            print(f"[{n+1}/{len(tcrs)}] {s}: FAILED {type(e).__name__}: {e}", flush=True)
    json.dump(res, open(f"{HERE}/results_anm_torsion.json", "w"), indent=1, default=float)
    plot(res)
    # summary: median r per CDR (loops with DOF)
    print("\nmedian torsion-ENM vs MD r per CDR (loops with >=1 deformation DOF):")
    for c in CDRS:
        rs = [res[s][c]["r"] for s in res if res[s][c].get("r") is not None]
        if rs:
            print(f"  {c:8} median r={np.median(rs):+.2f}  n={len(rs)}  (median DOF={np.median([res[s][c]['ndof'] for s in res if res[s][c].get('ndof')]):.0f})")
        else:
            print(f"  {c:8} no loops with deformation DOF (all rigid by geometry)")


def plot(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res.keys())
    Rm = np.full((len(tcrs), 6), np.nan); Dm = np.zeros((len(tcrs), 6))
    for i, s in enumerate(tcrs):
        for j, c in enumerate(CDRS):
            if res[s][c].get("r") is not None:
                Rm[i, j] = res[s][c]["r"]
            Dm[i, j] = res[s][c].get("ndof") or 0
    fig, ax = plt.subplots(1, 2, figsize=(14, 9))
    import numpy.ma as ma
    im0 = ax[0].imshow(ma.masked_invalid(Rm), aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax[0].set_title("a  torsion-ENM vs MD deformation profile (Pearson r)\ngrey = rigid (no deformation DOF)", fontsize=10, fontweight="bold")
    im1 = ax[1].imshow(Dm, aspect="auto", cmap="viridis", vmin=0, vmax=max(1, Dm.max()))
    ax[1].set_title("b  number of torsional deformation DOF (0 = rigid by geometry)", fontsize=10, fontweight="bold")
    for a, M, im, fmt in ((ax[0], Rm, im0, "{:.2f}"), (ax[1], Dm, im1, "{:.0f}")):
        a.set_xticks(range(6)); a.set_xticklabels(CDRS, rotation=30, fontsize=8)
        a.set_yticks(range(len(tcrs))); a.set_yticklabels(tcrs, fontsize=8)
        for y in range(len(tcrs)):
            for x in range(6):
                v = M[y, x]
                a.text(x, y, "" if (np.isnan(v)) else fmt.format(v), ha="center", va="center", fontsize=6.5)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle("Torsion-space feet-anchored loop ENM across all TCRs: does an elastic model predict the MD deformation?", y=1.0, fontsize=12, fontweight="bold")
    out = f"{HERE}/figures/anm_torsion_atlas.png"
    fig.tight_layout(); fig.savefig(out, dpi=125, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


if __name__ == "__main__":
    main()
