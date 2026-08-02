#!/usr/bin/env python3
"""Do the kinapse (native) and STCRpy loaders identify the SAME thing?

For every TCR-pMHC PDB in a directory, this runs BOTH loaders and compares:

  * **Chains** — the TCR (receptor) α/β chain ids and the pMHC (ligand) chain ids.
    Native: :func:`kinapse.scoring.infer_tcr_pmhc_chains`. STCRpy: ``get_VA``/``get_VB``
    + ``get_antigen``/``get_MHC``. Reports whether the receptor set and ligand set match.

  * **CDRs** — the residues each loader assigns to the 6 CDR loops, keyed by
    ``(region, chain, IMGT number)``. Native: kinapse's IMGT renumbering + region ranges.
    STCRpy: its ANARCI IMGT fragments (``get_CDRs``). Reports, per CDR: how many residues
    each loader found, how many they share (Jaccard), and — for the residues BOTH assign
    to the same CDR — the Cα distance between them (≈0 means they numbered the *same
    physical residue*; large means the two IMGT numberings disagree).

STCRpy runs as an external model; point kinapse at its env with
``KINAPSE_STCRPY_PYTHON=/path/to/stcrpy-env/bin/python`` (see the folder README).

Usage::

    export KINAPSE_STCRPY_PYTHON=/path/to/stcrpy-env/bin/python
    python compare_loading.py --pdb-dir /path/to/TCR_complexes --limit 50 -j 8
    # or explicit files:  python compare_loading.py a.pdb b.pdb
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

CDR_REGIONS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]

_DEFAULT_DIR = "/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_openmm_minimised"


# --------------------------------------------------------------------- loaders
def native_load(pdb, legacy_anarci=False):
    """(receptor, ligand, {(region, chain, imgt): Cα}) via the kinapse loader."""
    from kinapse.structures import TCR
    from kinapse.scoring import infer_tcr_pmhc_chains
    from kinapse.benchmarks.structural import _ca_by_region

    t = TCR(input_pdb=str(pdb), legacy_anarci=legacy_anarci)
    if not t.pairs:
        raise RuntimeError("native loader found no TCR pair")
    rec, lig = infer_tcr_pmhc_chains(pdb, tcr=t, legacy_anarci=legacy_anarci)
    return rec, lig, _ca_by_region(t.pairs[0])


def stcrpy_load(pdb, timeout=300.0):
    """(receptor, ligand, {(region, chain, imgt): Cα}) via external STCRpy."""
    from kinapse.structure_analysis.stcrpy import annotate

    res = annotate(pdb, timeout=timeout)
    rec = list(res.get("receptor_chains") or [])
    lig = list(res.get("ligand_chains") or [])
    cmap = {}
    for r in res.get("regions") or []:
        try:
            cmap[(r["region"], r["chain"], int(r["imgt"]))] = np.asarray(r["xyz"], float)
        except (KeyError, TypeError, ValueError):
            continue
    return rec, lig, cmap


# ----------------------------------------------------------------- comparison
def _region_keys(cmap, region):
    return {k for k in cmap if k[0] == region}


def compare_one(pdb, legacy_anarci=False):
    """Return (summary_row, [per_cdr_rows]) comparing both loaders on one PDB."""
    row = {"id": Path(pdb).stem}
    try:
        rn, ln, cn = native_load(pdb, legacy_anarci)
    except Exception as e:  # noqa: BLE001
        row["native_error"] = str(e)[:200]
    try:
        rs, ls, cs = stcrpy_load(pdb)
    except Exception as e:  # noqa: BLE001
        row["stcrpy_error"] = str(e)[:200]
    if "native_error" in row or "stcrpy_error" in row:
        return row, []

    row["rec_native"], row["rec_stcrpy"] = ",".join(rn), ",".join(rs)
    row["lig_native"], row["lig_stcrpy"] = ",".join(ln), ",".join(ls)
    row["rec_match"] = set(rn) == set(rs)
    row["lig_match"] = set(ln) == set(ls)

    per_cdr, jaccards, coord_diffs = [], [], []
    for region in CDR_REGIONS:
        kn, ks = _region_keys(cn, region), _region_keys(cs, region)
        inter, union = kn & ks, kn | ks
        jac = len(inter) / len(union) if union else 1.0
        jaccards.append(jac)
        shared_rmsd = np.nan
        if inter:
            diffs = [float(np.linalg.norm(cn[k] - cs[k])) for k in inter]
            shared_rmsd = round(float(np.mean(diffs)), 3)
            coord_diffs.extend(diffs)
        per_cdr.append(dict(id=row["id"], region=region,
                            n_native=len(kn), n_stcrpy=len(ks), n_shared=len(inter),
                            n_only_native=len(kn - ks), n_only_stcrpy=len(ks - kn),
                            jaccard=round(jac, 3), shared_ca_rmsd=shared_rmsd))

    row["cdr_all_identical"] = all(_region_keys(cn, r) == _region_keys(cs, r) for r in CDR_REGIONS)
    row["cdr_mean_jaccard"] = round(float(np.mean(jaccards)), 3)
    row["cdr_shared_ca_rmsd"] = round(float(np.mean(coord_diffs)), 3) if coord_diffs else np.nan
    return row, per_cdr


def _worker(pdb, legacy_anarci):        # module-level: picklable for the spawn pool
    return compare_one(pdb, legacy_anarci)


# ------------------------------------------------------------------- driver
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdbs", nargs="*", help="explicit PDB files (else use --pdb-dir)")
    ap.add_argument("--pdb-dir", default=_DEFAULT_DIR, help="directory of TCR-pMHC PDBs")
    ap.add_argument("--limit", type=int, default=None, help="cap number of structures")
    ap.add_argument("--legacy-anarci", action="store_true", help="native: use bioconda ANARCI")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="parallel structures (process pool)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"),
                    help="output directory")
    args = ap.parse_args(argv)

    from kinapse.structure_analysis import stcrpy
    if not stcrpy.available():
        print(stcrpy._HINT)
        print("\nSTCRpy environment not found — set KINAPSE_STCRPY_PYTHON. Aborting.")
        return 2

    pdbs = list(args.pdbs) or [str(p) for p in sorted(Path(args.pdb_dir).glob("*.pdb"))]
    if args.limit:
        pdbs = pdbs[:args.limit]
    if not pdbs:
        print(f"no PDBs found (dir={args.pdb_dir})")
        return 1
    print(f"comparing native vs STCRpy loading on {len(pdbs)} structures")

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        def tqdm(x, **k):
            return x

    rows, per_cdr = [], []
    if args.jobs and args.jobs > 1 and len(pdbs) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        try:
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:
                futs = [ex.submit(_worker, p, args.legacy_anarci) for p in pdbs]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="compare", unit="pdb"):
                    r, pc = fut.result()
                    rows.append(r)
                    per_cdr.extend(pc)
        except (BrokenProcessPool, OSError, RuntimeError) as e:
            print(f"[compare] parallel pool unavailable ({type(e).__name__}); running serially.")
            rows, per_cdr = [], []
            for p in tqdm(pdbs, desc="compare", unit="pdb"):
                r, pc = compare_one(p, args.legacy_anarci)
                rows.append(r)
                per_cdr.extend(pc)
    else:
        for p in tqdm(pdbs, desc="compare", unit="pdb"):
            r, pc = compare_one(p, args.legacy_anarci)
            rows.append(r)
            per_cdr.extend(pc)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cdr = pd.DataFrame(per_cdr)
    df.to_csv(out / "per_structure.csv", index=False)
    cdr.to_csv(out / "per_cdr.csv", index=False)

    # ---- summary ----
    ok = df[~df.get("native_error", pd.Series(index=df.index)).notna()
            & ~df.get("stcrpy_error", pd.Series(index=df.index)).notna()] \
        if len(df) else df
    n_nat_fail = int(df["native_error"].notna().sum()) if "native_error" in df else 0
    n_stc_fail = int(df["stcrpy_error"].notna().sum()) if "stcrpy_error" in df else 0
    print("\n=== SUMMARY ===")
    print(f"  structures: {len(df)}  |  both loaded: {len(ok)}  |  "
          f"native failed: {n_nat_fail}  |  stcrpy failed: {n_stc_fail}")
    if len(ok):
        print(f"  receptor (TCR) chains identical: {int(ok.rec_match.sum())}/{len(ok)} "
              f"({100*ok.rec_match.mean():.0f}%)")
        print(f"  ligand (pMHC) chains identical:  {int(ok.lig_match.sum())}/{len(ok)} "
              f"({100*ok.lig_match.mean():.0f}%)")
        print(f"  all 6 CDRs residue-identical:    {int(ok.cdr_all_identical.sum())}/{len(ok)} "
              f"({100*ok.cdr_all_identical.mean():.0f}%)")
        print(f"  mean per-CDR Jaccard:            {ok.cdr_mean_jaccard.mean():.3f}")
        cr = ok.cdr_shared_ca_rmsd.dropna()
        if len(cr):
            print(f"  shared-residue Cα agreement:     mean {cr.mean():.3f} Å, max {cr.max():.3f} Å "
                  "(≈0 ⇒ same physical residues numbered the same)")
    if len(cdr):
        print("\n  per-CDR (mean over structures):")
        g = cdr.groupby("region").agg(
            mean_jaccard=("jaccard", "mean"),
            mean_n_native=("n_native", "mean"),
            mean_n_stcrpy=("n_stcrpy", "mean"),
            mean_shared_ca_rmsd=("shared_ca_rmsd", "mean")).round(3)
        print(g.to_string())
    # structures where chains disagree — worth eyeballing
    if len(ok):
        bad = ok[~ok.rec_match | ~ok.lig_match]
        if len(bad):
            print(f"\n  chain disagreements ({len(bad)}):")
            for _, r in bad.head(15).iterrows():
                print(f"    {r['id']}: rec {r['rec_native']} vs {r['rec_stcrpy']} | "
                      f"lig {r['lig_native']} vs {r['lig_stcrpy']}")

    print(f"\nwrote per_structure.csv + per_cdr.csv to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
