#!/usr/bin/env python3
"""Find every VDJdb pMHC binder for each TCR in a structure dataset.

For each ``<ID>/<ID>.pdb`` complex in a dataset directory, the alpha/beta CDR3
loops are extracted with the kinapse loader (:class:`kinapse.structures.TCR` --
IMGT renumbering + interface-based alpha/beta pairing) and matched against
VDJdb. Every pMHC the TCR is recorded to bind is reported together with its
VDJdb confidence score.

CDR3 format
-----------
kinapse's ``A_CDR3``/``B_CDR3`` regions are IMGT **105-117** (the loop only).
VDJdb's ``cdr3`` column is the full **104-118** string (conserved Cys...Phe/Trp
flanks included), so we rebuild it as ``FR3[-1] + CDR3 + FR4[0]`` to string-match
VDJdb verbatim.

Matching (default ``paired``)
-----------------------------
A VDJdb record is called the SAME TCR only when BOTH its alpha CDR3 == our
CDR3alpha AND its beta CDR3 == our CDR3beta (a VDJdb "complex.id" grouping a
paired alpha/beta observation). This is the specific option -- an individual
CDR3 alpha or beta matches many unrelated epitopes on its own. ``--match``
relaxes it to ``beta`` / ``alpha`` / ``either``.

Non-binders
-----------
VDJdb is a positive-only database: it records binders, never non-binders. There
are therefore NO confirmed non-binders to retrieve here -- every emitted row is
``label=confirmed_binder``. Confirmed non-binders must come from an assay source
(10x negative controls, IEDB negative assays, or a tcr_pmhc_db interaction with
binding=False); see kinapse.datasets CATALOG ("tenx", "iedb").

Run in the ``kinapse`` env (needs the ``[structures]`` extra: biopython, anarcii,
mdtraj)::

    python find_vdjdb_binders.py \
        --dataset /mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD \
        --vdjdb   /mnt/larry/lilian/DATA/vdjdb/vdjdb.slim.txt
    # explicit PDBs also work:  python find_vdjdb_binders.py a.pdb b.pdb
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io as _io
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

_DEFAULT_DATASET = "/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD"
_DEFAULT_VDJDB = "/mnt/larry/lilian/DATA/vdjdb/vdjdb.slim.txt"


# --------------------------------------------------------------------------- #
# 1. CDR3a / CDR3b from a TCR pdb (via the kinapse loader)
# --------------------------------------------------------------------------- #
def _full_cdr3(seqs: dict, prefix: str) -> str:
    """Rebuild the VDJdb-format CDR3 (IMGT 104-118) from kinapse region seqs."""
    fr3 = seqs.get(f"{prefix}_FR3", "") or ""
    cdr3 = seqs.get(f"{prefix}_CDR3", "") or ""
    fr4 = seqs.get(f"{prefix}_FR4", "") or ""
    cys = fr3[-1] if fr3 else ""     # IMGT 104 (conserved Cys)
    phe = fr4[0] if fr4 else ""      # IMGT 118 (conserved Phe/Trp)
    return (cys + cdr3 + phe).upper()


def extract_cdr3s(pdb_path: str, legacy_anarci: bool = True) -> list[dict]:
    """Return one dict per TCR pair: {pair, cdr3a, cdr3b, chain_map}.

    The chatty library stdout emitted while numbering/pairing is suppressed so
    the per-dataset progress stays readable; real errors still raise.
    """
    from kinapse.structures import TCR
    with contextlib.redirect_stdout(_io.StringIO()):
        tcr = TCR(input_pdb=str(pdb_path), legacy_anarci=legacy_anarci)
    out = []
    for i, pair in enumerate(tcr.pairs):
        seqs = pair.cdr_fr_sequences()
        out.append({
            "pair": i,
            "cdr3a": _full_cdr3(seqs, "A") or None,
            "cdr3b": _full_cdr3(seqs, "B") or None,
            "chain_map": dict(pair.chain_map),
        })
    return out


# --------------------------------------------------------------------------- #
# 2. Paired-complex index of VDJdb
# --------------------------------------------------------------------------- #
def load_vdjdb_paired(vdjdb_path: str) -> list[dict]:
    """Group VDJdb rows by complex.id (>0) into paired alpha/beta observations."""
    groups: dict[str, dict] = defaultdict(lambda: {
        "alpha": set(), "beta": set(),
        "epitope": "", "antigen_gene": "", "antigen_species": "",
        "mhc_a": "", "mhc_b": "", "mhc_class": "",
        "tcr_species": "", "score": 0, "reference": "",
    })
    with open(vdjdb_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            cid = row["complex.id"].strip()
            if cid in ("", "0"):
                continue  # unpaired / pooled -> cannot form a paired TCR
            g = groups[cid]
            cdr3 = row["cdr3"].strip().upper()
            if row["gene"] == "TRA":
                g["alpha"].add(cdr3)
            elif row["gene"] == "TRB":
                g["beta"].add(cdr3)
            g["epitope"] = row["antigen.epitope"].strip()
            g["antigen_gene"] = row["antigen.gene"].strip()
            g["antigen_species"] = row["antigen.species"].strip()
            g["mhc_a"] = row["mhc.a"].strip()
            g["mhc_b"] = row["mhc.b"].strip()
            g["mhc_class"] = row["mhc.class"].strip()
            g["tcr_species"] = row["species"].strip()
            g["reference"] = row["reference.id"].strip()
            try:
                g["score"] = max(g["score"], int(row["vdjdb.score"].strip() or 0))
            except ValueError:
                pass
    for cid, g in groups.items():
        g["complex_id"] = cid
    return list(groups.values())


# --------------------------------------------------------------------------- #
# 3. Match + aggregate
# --------------------------------------------------------------------------- #
def match_tcr(cdr3a, cdr3b, complexes, mode="paired") -> list[dict]:
    hits = []
    for c in complexes:
        a_ok = cdr3a is not None and cdr3a in c["alpha"]
        b_ok = cdr3b is not None and cdr3b in c["beta"]
        if mode == "paired":
            ok = a_ok and b_ok
        elif mode == "beta":
            ok = b_ok
        elif mode == "alpha":
            ok = a_ok
        elif mode == "either":
            ok = a_ok or b_ok
        else:
            raise ValueError(f"unknown match mode: {mode}")
        if ok:
            hits.append(c)
    return hits


def aggregate_pmhcs(hits) -> list[dict]:
    """Collapse matched complexes into unique pMHCs (max score, collected refs)."""
    agg: dict[tuple, dict] = {}
    for c in hits:
        key = (c["epitope"], c["mhc_a"], c["mhc_b"], c["mhc_class"])
        rec = agg.setdefault(key, {
            "epitope": c["epitope"], "antigen_gene": c["antigen_gene"],
            "antigen_species": c["antigen_species"], "mhc_a": c["mhc_a"],
            "mhc_b": c["mhc_b"], "mhc_class": c["mhc_class"],
            "tcr_species": c["tcr_species"], "vdjdb_score": 0,
            "complex_ids": set(), "references": set(),
        })
        rec["vdjdb_score"] = max(rec["vdjdb_score"], c["score"])
        rec["complex_ids"].add(c["complex_id"])
        rec["references"].add(c["reference"])
        rec["tcr_species"] = rec["tcr_species"] or c["tcr_species"]
    return sorted(agg.values(), key=lambda r: (-r["vdjdb_score"], r["epitope"]))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _iter_pdbs(dataset, explicit):
    if explicit:
        for p in explicit:
            yield Path(p)
    else:
        yield from sorted(Path(dataset).glob("*/*.pdb"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdbs", nargs="*", help="explicit PDB files (overrides --dataset)")
    ap.add_argument("--dataset", default=_DEFAULT_DATASET,
                    help="dataset root of <ID>/<ID>.pdb subfolders")
    ap.add_argument("--vdjdb", default=_DEFAULT_VDJDB, help="path to vdjdb.slim.txt")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <this folder>/results)")
    ap.add_argument("--match", default="paired",
                    choices=["paired", "beta", "alpha", "either"],
                    help="how to call a VDJdb record the same TCR (default: paired)")
    ap.add_argument("--new-anarci", action="store_true",
                    help="use ANARCII instead of legacy ANARCI for numbering")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading VDJdb paired complexes from {args.vdjdb}")
    complexes = load_vdjdb_paired(args.vdjdb)
    print(f"    {len(complexes)} paired alpha/beta complexes indexed")

    pdbs = list(_iter_pdbs(args.dataset, args.pdbs))
    print(f"[*] {len(pdbs)} PDB complexes to process "
          f"(match mode: {args.match})\n")

    tcr_rows, binder_rows = [], []
    for pdb_path in pdbs:
        pdb_id = pdb_path.stem
        try:
            pairs = extract_cdr3s(pdb_path, legacy_anarci=not args.new_anarci)
        except Exception as exc:  # noqa: BLE001
            print(f"  {pdb_id}: FAILED to load ({exc})")
            continue
        if not pairs:
            print(f"  {pdb_id}: no TCR pair found")
            continue

        for pr in pairs:
            tag = pdb_id if len(pairs) == 1 else f"{pdb_id}#pair{pr['pair']}"
            hits = match_tcr(pr["cdr3a"], pr["cdr3b"], complexes, mode=args.match)
            pmhcs = aggregate_pmhcs(hits)
            tcr_rows.append({
                "pdb_id": pdb_id, "pair": pr["pair"],
                "cdr3a": pr["cdr3a"] or "", "cdr3b": pr["cdr3b"] or "",
                "n_binders": len(pmhcs),
            })
            for p in pmhcs:
                binder_rows.append({
                    "pdb_id": pdb_id, "pair": pr["pair"],
                    "cdr3a": pr["cdr3a"] or "", "cdr3b": pr["cdr3b"] or "",
                    "label": "confirmed_binder",
                    "epitope": p["epitope"], "antigen_gene": p["antigen_gene"],
                    "antigen_species": p["antigen_species"],
                    "mhc_a": p["mhc_a"], "mhc_b": p["mhc_b"],
                    "mhc_class": p["mhc_class"], "tcr_species": p["tcr_species"],
                    "vdjdb_score": p["vdjdb_score"], "n_complexes": len(p["complex_ids"]),
                    "complex_ids": ";".join(sorted(p["complex_ids"])),
                    "references": ";".join(sorted(r for r in p["references"] if r)),
                })
            star = "" if pmhcs else "   (no VDJdb match)"
            print(f"  {tag}: a={pr['cdr3a']}  b={pr['cdr3b']}  "
                  f"-> {len(pmhcs)} binder pMHC(s){star}")

    binders_csv = out_dir / "vdjdb_confirmed_binders.csv"
    tcrs_csv = out_dir / "tcr_cdr3_summary.csv"
    pd.DataFrame(binder_rows, columns=[
        "pdb_id", "pair", "cdr3a", "cdr3b", "label", "epitope", "antigen_gene",
        "antigen_species", "mhc_a", "mhc_b", "mhc_class", "tcr_species",
        "vdjdb_score", "n_complexes", "complex_ids", "references"]).to_csv(binders_csv, index=False)
    pd.DataFrame(tcr_rows, columns=[
        "pdb_id", "pair", "cdr3a", "cdr3b", "n_binders"]).to_csv(tcrs_csv, index=False)

    matched = sum(1 for r in tcr_rows if r["n_binders"])
    print(f"\n[*] Done. {matched}/{len(tcr_rows)} TCRs matched a paired VDJdb "
          f"record; {len(binder_rows)} confirmed-binder pMHC rows.")
    print(f"    binders : {binders_csv}")
    print(f"    tcr map : {tcrs_csv}")
    print("\n[!] VDJdb is positive-only: it contains NO confirmed non-binders. "
          "Every row above is a confirmed binder. For confirmed non-binders use "
          "an assay source (10x negatives, IEDB negative assays, or a tcr_pmhc_db "
          "interaction with binding=False).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
