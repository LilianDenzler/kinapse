"""Per-CDR predicted flexibility for the BSC clinical TCRs (DiG cdr_mask ensembles).

For each TCR that has a complete DiG ensemble, measure how much each of the six CDR loops
moves across the 200 generated conformers:

  1. load the ensemble (backbone frames of the linked α–linker–β variable structure),
  2. superpose every frame on that domain's FRAMEWORK Cα (α-framework for the α CDRs,
     β-framework for the β CDRs) — so rigid inter-domain motion is removed and we measure
     genuine loop flexibility,
  3. per-CDR flexibility = mean Cα RMSF (root-mean-square fluctuation about the ensemble mean),
     reported in Å.

Outputs (under results/flexibility/): a per-TCR × per-CDR CSV, a heatmap, a per-CDR distribution
boxplot, and a per-residue RMSF CSV.

Usage:  python analyse_flexibility.py [--out-root <dig_out_dir>] [--min-frames 50]
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kinapse.structures import TCR

DIG_OUT = "/mnt/larry/lilian/DATA/alex_barcelona_data/tcrs_dig_out"
RESULTS = os.path.join(os.path.dirname(__file__), "results", "flexibility")
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
A_FR = ["A_FR1", "A_FR2", "A_FR3", "A_FR4"]
B_FR = ["B_FR1", "B_FR2", "B_FR3", "B_FR4"]


def _frame_files(tcr_dir, name):
    """Sorted per-conformer PDBs (exclude the *_init_state.npz siblings)."""
    d = glob.glob(os.path.join(tcr_dir, "dig_cdr_mask*"))
    if not d:
        return None
    fs = [f for f in glob.glob(os.path.join(d[0], f"{name}_*.pdb")) if "init_state" not in f]
    return sorted(fs, key=lambda p: int(p.rsplit("_", 1)[1][:-4]))


def _region_masks(name, tcr_dir, n_res):
    """Per-residue bool masks over the linked chain (α + linker + β), one per region.
    Built from the TCR's IMGT numbering — no MODELLER rebuild needed."""
    tcr = TCR(os.path.join(tcr_dir, f"{name}.pdb"), legacy_anarci=False)
    pv = tcr.pairs[0]
    a_id, b_id = pv.chain_map["alpha"], pv.chain_map["beta"]

    def mask(region_names):
        d = pv.cdr_fr_resmask(region_names=region_names, structure_used=pv.variable_structure,
                              atom_names={"CA"}, include_het=True)
        a, b = list(d[a_id]), list(d[b_id])
        linker = n_res - len(a) - len(b)
        if linker < 0:
            raise ValueError(f"{name}: residue count mismatch (frame {n_res} < α{len(a)}+β{len(b)})")
        return np.array(a + [False] * linker + b, dtype=bool)

    masks = {r: mask([r]) for r in CDRS}
    masks["_A_FR"] = mask(A_FR)
    masks["_B_FR"] = mask(B_FR)
    return masks


def _rmsf_on(traj_ca, ref_idx):
    """Per-Cα RMSF (Å) after superposing every frame on the reference atoms (frame 0)."""
    t = traj_ca[:]                                   # copy so each alignment is independent
    t.superpose(t, frame=0, atom_indices=ref_idx)
    xyz = t.xyz                                       # (F, N, 3) in nm
    return np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(-1).mean(0)) * 10.0   # → Å per Cα


def analyse_one(name, tcr_dir, min_frames):
    frames = _frame_files(tcr_dir, name)
    if not frames or len(frames) < min_frames:
        return None, None
    traj = md.load(frames, top=frames[0])
    ca = traj.topology.select("name CA")
    traj_ca = traj.atom_slice(ca)                     # one Cα per residue, in residue order
    n_res = traj_ca.n_residues
    masks = _region_masks(name, tcr_dir, n_res)

    rmsf_a = _rmsf_on(traj_ca, np.where(masks["_A_FR"])[0])   # aligned on α framework
    rmsf_b = _rmsf_on(traj_ca, np.where(masks["_B_FR"])[0])   # aligned on β framework

    row = {"tcr": name, "n_frames": traj_ca.n_frames}
    per_res = []
    for r in CDRS:
        rmsf = rmsf_a if r.startswith("A_") else rmsf_b
        idx = np.where(masks[r])[0]
        row[r] = round(float(rmsf[idx].mean()), 3) if len(idx) else np.nan
        for k, i in enumerate(idx):
            per_res.append({"tcr": name, "cdr": r, "cdr_pos": k, "rmsf_A": round(float(rmsf[i]), 3)})
    return row, per_res


def main():
    ap = argparse.ArgumentParser(description="Per-CDR predicted flexibility (Cα RMSF) for DiG ensembles.")
    ap.add_argument("--out-root", default=DIG_OUT, help="DiG output root (one sub-dir per TCR)")
    ap.add_argument("--min-frames", type=int, default=50, help="skip ensembles with fewer frames")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    tcr_dirs = sorted(d for d in glob.glob(os.path.join(a.out_root, "*")) if os.path.isdir(d))
    rows, per_res_all, skipped = [], [], []
    for td in tcr_dirs:
        name = os.path.basename(td)
        try:
            row, per_res = analyse_one(name, td, a.min_frames)
        except Exception as e:  # noqa: BLE001 - one TCR must not kill the batch
            print(f"FAIL {name}: {type(e).__name__}: {e}"); skipped.append(name); continue
        if row is None:
            skipped.append(name); continue
        rows.append(row); per_res_all.extend(per_res)
        print(f"  {name}: " + "  ".join(f"{r}={row[r]}" for r in CDRS))

    if not rows:
        print("no complete ensembles found"); return
    df = pd.DataFrame(rows).sort_values("tcr", ignore_index=True)
    df.to_csv(os.path.join(RESULTS, "per_cdr_flexibility.csv"), index=False)
    pd.DataFrame(per_res_all).to_csv(os.path.join(RESULTS, "per_residue_rmsf.csv"), index=False)

    # ---- summary ----
    summ = df[CDRS].agg(["mean", "std", "min", "max"]).round(3)
    summ.to_csv(os.path.join(RESULTS, "per_cdr_summary.csv"))
    print(f"\n{len(df)} TCRs analysed (skipped {len(skipped)}). Mean Cα RMSF per CDR (Å):")
    print(summ.loc["mean"].to_string())

    # ---- heatmap: TCR × CDR ----
    M = df.set_index("tcr")[CDRS]
    fig, ax = plt.subplots(figsize=(6, max(4, 0.22 * len(df))))
    im = ax.imshow(M.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(CDRS))); ax.set_xticklabels(CDRS, rotation=45, ha="right")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(M.index, fontsize=6)
    ax.set_title("Predicted CDR flexibility (Cα RMSF, Å)")
    fig.colorbar(im, ax=ax, label="RMSF (Å)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "flexibility_heatmap.png"), dpi=150)
    plt.close(fig)

    # ---- boxplot: distribution per CDR across TCRs ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([df[r].dropna() for r in CDRS], labels=CDRS, showmeans=True)
    ax.set_ylabel("Cα RMSF (Å)"); ax.set_title(f"Per-CDR flexibility across {len(df)} TCRs")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "flexibility_boxplot.png"), dpi=150)
    plt.close(fig)

    print(f"\nwrote CSVs + plots to {RESULTS}")


if __name__ == "__main__":
    main()
