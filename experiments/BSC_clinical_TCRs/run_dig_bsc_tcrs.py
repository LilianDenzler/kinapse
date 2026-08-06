"""Serial DiG driver for the BSC clinical TCRs — one TCR at a time (see run_dig_parallel.py
for the multi-GPU version). Runs the chosen sampling mode(s) per TCR, all sharing the one
prep + embedding + init_state.

Usage:  ANARCI_CPU=1 python run_dig_bsc_tcrs.py                         # all modes
        python run_dig_bsc_tcrs.py --modes cdr_mask,weighted_mask      # a subset
        python run_dig_bsc_tcrs.py --modes vanilla --n 50              # one mode, fewer samples
        DIG_MODE=no_mask python run_dig_bsc_tcrs.py                    # env override of --modes default
"""
import argparse
import os
from pathlib import Path

import kinapse
from kinapse.conformer_generation import run_one, resolve_modes, MODE_DIRS
from kinapse.structures import TCR
from kinapse.regions import CDR_FR_RANGES
from kinapse.structures.io import write_pdb

DEFAULT_TCR_DIR = "/mnt/larry/lilian/DATA/alex_barcelona_data/tcrs"
DEFAULT_OUT = "/mnt/larry/lilian/DATA/alex_barcelona_data/tcrs_dig_out"


def prep_tcr_structure(tcr_file, out_tcr_path, name):
    """Extract the variable-only structure + print diagnostics; returns the written PDB path.
    Named ``<name>.pdb`` so run_one's output lands under ``<out>/<name>/`` (frames ``<name>_*.pdb``)."""
    os.makedirs(out_tcr_path, exist_ok=True)
    tcr = TCR(input_pdb=tcr_file, legacy_anarci=False)
    pair = tcr.pairs[0]
    print("kinapse", getattr(kinapse, "__version__", "?"))
    print("paired chains (α, β):", pair.alpha_chain_id, pair.beta_chain_id)
    print("# α/β pairs found:", len(tcr.pairs))
    cdr_fr = pair.cdr_fr_sequences()
    print({k: v for k, v in cdr_fr.items() if "CDR" in k})
    for rname, (lo, hi) in list(CDR_FR_RANGES.items())[:6]:
        print(f"  {rname:8s} {lo}–{hi}")
    vs = pair.variable_structure
    n_res = sum(1 for ch in vs.get_chains() for r in ch if r.id[0] == " ")
    print("\nvariable-domain chains:", [c.id for c in vs.get_chains()], "| Cα residues:", n_res)
    var_pdb = os.path.join(out_tcr_path, f"{name}.pdb")
    write_pdb(var_pdb, vs)
    return var_pdb


def _done(out_tcr, modes):
    """True only if EVERY requested mode already has frames (idempotent skip)."""
    return all(any(Path(out_tcr).glob(f"{MODE_DIRS[m]}*/*_199.pdb")) for m in modes)


def main():
    ap = argparse.ArgumentParser(description="Run DiG on the BSC clinical TCRs (serial).")
    ap.add_argument("--tcr-dir", default=DEFAULT_TCR_DIR, help="folder of input TCR PDBs")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output root (one sub-dir per TCR)")
    ap.add_argument("--modes", default=os.environ.get("DIG_MODE", "all"),
                    help="'all', a single mode, or a comma-separated subset of "
                         "{vanilla, no_mask, binary_mask, cdr_mask, weighted_mask} (default: all)")
    ap.add_argument("--n", type=int, default=200, help="samples to draw per mode (default 200)")
    a = ap.parse_args()

    modes = resolve_modes(a.modes)
    print("modes:", ", ".join(modes))
    files = sorted(f for f in os.listdir(a.tcr_dir) if f.endswith(".pdb"))
    ok = skip = fail = 0
    for tcr_file in files:
        name = tcr_file[:-4]
        out_tcr_path = os.path.join(a.out, name)
        if _done(out_tcr_path, modes):
            print(f"skip (done): {name}")
            skip += 1
            continue
        try:
            var_pdb = prep_tcr_structure(os.path.join(a.tcr_dir, tcr_file), out_tcr_path, name)
            run_one(var_pdb, a.out, n_samples=a.n, dig_mode=modes)
            ok += 1
        except Exception as e:  # noqa: BLE001 - one TCR failing must not kill the batch
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            fail += 1
    print(f"\ndone — ok={ok} skipped={skip} failed={fail} | modes={', '.join(modes)}")


if __name__ == "__main__":
    main()
