#!/usr/bin/env python3
"""Annotate the VDJdb confirmed binders with experimentally solved BOUND TCR-pMHC
structures (curated from RCSB / TCR3d).

Adds, per (source TCR, peptide):
  * ``bound_pdb``        - the solved bound TCR-pMHC complex PDB id(s)
  * ``structure_status`` - native_bound | modified_bound | sequence_only
  * ``ligand_note``      - what the peptide is (WT / APL name)
  * ``peptide_modified`` - the true chemically-modified ligand, if any
  * ``modification``     - description of the chemical modification
  * ``row_source``       - vdjdb_paired | literature_added

Two curation points beyond the VDJdb query (per expert review of the CSV):
  1. The canonical A6 complex **1AO7** (A6 / LLFGYPVYV / HLA-A*02:01, WT Tax) is
     ADDED — VDJdb's paired query missed it, yet it is A6's defining structure
     (and it was wrongly appearing as a negative; regenerate negatives after this).
  2. Two 3QH3/A6 "peptides" are chemically modified and the VDJdb amino-acid string
     does NOT capture the modification — flagged ``modified_bound``:
       - LLFGKPVYV -> 2GJ6 is Tax Y5K-IBA (Lys5 + 3-indolebutyric-acid hapten)
       - LLFGPVYV  -> 3D39 / 3D3V are Tax with a fluorinated Phe at pos 5; the
         VDJdb 8-mer string has dropped the modified residue entirely.

Keyed by (pdb_id, epitope) so the same peptide on different TCRs maps correctly
(e.g. FLRGRAYGL -> 1MI5 for 1KGC/LC13 but 3SJV for 3SKN/RL42).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_CSV = str(_HERE / "results" / "vdjdb_confirmed_binders.csv")

N, M, S = "native_bound", "modified_bound", "sequence_only"

# (pdb_id, epitope) -> (bound_pdb, structure_status, ligand_note, peptide_modified, modification)
BOUND = {
    ("1KGC", "EEYLKAWTF"):  ("3KPR", N, "LC13-HLA-B*4405 (self peptide)", "", ""),
    ("1KGC", "EEYLQAFTY"):  ("3KPS", N, "LC13-HLA-B*4405 (self peptide)", "", ""),
    ("1KGC", "FLRGRAYGL"):  ("1MI5", N, "canonical LC13-EBV FLR-HLA-B8", "", ""),
    ("2BNU", "SLLMWITQC"):  ("2BNR", N, "1G4-NY-ESO-1 WT (9C)", "", ""),
    ("2BNU", "SLLMWITQV"):  ("2BNQ", N, "1G4-NY-ESO-1 C9V analogue", "", ""),
    ("3DX9", "EENLLDFVRF"): ("3DXA", N, "DM1-HLA-B*4405-EBV ternary complex", "", ""),
    ("3QH3", "LGYGFVNYI"):  ("3PWP", N, "A6-HuD-HLA-A2", "", ""),
    ("3QH3", "LLFGFPVYV"):  ("3QFJ", N, "A6-Tax Y5F (natural Phe)", "", ""),
    ("3QH3", "LLFGKPVYV"):  ("2GJ6", M, "A6-Tax Y5K-IBA (hapten)",
                            "LLFG-[K(IBA)]-PVYV",
                            "Lys5 side chain conjugated to 3-indolebutyric acid (IBA); "
                            "VDJdb string 'LLFGKPVYV' omits the hapten"),
    ("3QH3", "LLFGPVYV"):   ("3D39;3D3V", M, "A6-Tax Y5 fluoro-Phe (fluorination study)",
                            "LLFG-[F(fluoro)]-PVYV",
                            "Position-5 fluorinated Phe (4-F-Phe in 3D39; 3,4-diF-Phe in "
                            "3D3V); the VDJdb 8-mer 'LLFGPVYV' has DROPPED the modified residue"),
    ("3QH3", "LLFGYAVYV"):  ("1QRN", N, "A6-Tax P6A APL", "", ""),
    ("3QH3", "LLFGYPRYV"):  ("1QSE", N, "A6-Tax V7R APL", "", ""),
    ("3QH3", "LLFGYPVAV"):  ("1QSF", N, "A6-Tax Y8A APL", "", ""),
    ("3QH3", "MLWGYLQYV"):  ("3H9S", N, "A6-Tel1p-HLA-A2", "", ""),
    ("3SKN", "FLRGRAYGL"):  ("3SJV", N, "RL42-HLA-B8-FLRGRAYGL", "", ""),
    ("3VXQ", "RYPLTFGWCF"): ("3VXR", N, "H27-14-HLA-A24-Nef WT", "", ""),
    ("3VXQ", "RYPLTLGWCF"): ("3VXS", N, "H27-14-Nef 6L variant", "", ""),
    ("4JFH", "ELAAIGILTV"): ("4JFD", N, "a24b17-Melan-A variant", "", ""),
    ("4JFH", "ELAGIGALTV"): ("4JFE", N, "a24b17-Melan-A L7A variant", "", ""),
    ("4JFH", "ELAGIGILTV"): ("4JFF", N, "a24b17-Melan-A cognate/parent", "", ""),
}

# Confirmed binders with a solved structure that the VDJdb query missed entirely.
# (pdb_id, epitope, mhc_a) -> (bound_pdb, structure_status, ligand_note)
LITERATURE_ADD = [
    ("3QH3", "LLFGYPVYV", "HLA-A*02:01", "1AO7", N,
     "canonical A6-Tax-HLA-A2 (WT); missed by VDJdb paired query"),
]

EXTRA_COLS = ["bound_pdb", "structure_status", "ligand_note",
              "peptide_modified", "modification", "row_source"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=_DEFAULT_CSV)
    ap.add_argument("--out", default=str(_HERE / "results" / "vdjdb_confirmed_binders_annotated.csv"))
    ap.add_argument("--no-add-literature", action="store_true",
                    help="do not append literature-only confirmed binders (e.g. 1AO7)")
    ap.add_argument("--drop-modified", action="store_true",
                    help="exclude chemically-modified ligands (structure_status=modified_bound; "
                         "e.g. 2GJ6 K-IBA, 3D39/3D3V fluoro-Phe) whose VDJdb sequence string "
                         "does not represent the real ligand; they are written to a sidecar CSV")
    args = ap.parse_args(argv)

    with open(args.csv, newline="") as fh:
        reader = csv.DictReader(fh)
        base_cols = reader.fieldnames
        rows = list(reader)

    out_rows, unmatched = [], []
    for r in rows:
        key = (r["pdb_id"], r["epitope"])
        m = BOUND.get(key)
        if m:
            bp, status, note, pmod, mod = m
        else:
            bp, status, note, pmod, mod = "", S, "", "", ""
            unmatched.append(key)
        r = dict(r)
        r.update(bound_pdb=bp, structure_status=status, ligand_note=note,
                 peptide_modified=pmod, modification=mod, row_source="vdjdb_paired")
        out_rows.append(r)

    added = []
    if not args.no_add_literature:
        # template a new row from the same TCR's first row (cdr3s, species, pair)
        by_tcr = {}
        for r in rows:
            by_tcr.setdefault(r["pdb_id"], r)
        for pdb_id, epitope, mhc_a, bp, status, note in LITERATURE_ADD:
            tmpl = by_tcr.get(pdb_id)
            if not tmpl:
                continue
            new = {c: "" for c in base_cols}
            new.update({
                "pdb_id": pdb_id, "pair": tmpl.get("pair", ""),
                "cdr3a": tmpl["cdr3a"], "cdr3b": tmpl["cdr3b"],
                "label": "confirmed_binder", "epitope": epitope,
                "mhc_a": mhc_a, "mhc_b": "B2M", "mhc_class": "MHCI",
                "tcr_species": tmpl.get("tcr_species", ""),
            })
            new.update(bound_pdb=bp, structure_status=status, ligand_note=note,
                       peptide_modified="", modification="", row_source="literature_added")
            out_rows.append(new)
            added.append((pdb_id, epitope, bp))

    dropped = []
    if args.drop_modified:
        kept = [r for r in out_rows if r["structure_status"] != M]
        dropped = [r for r in out_rows if r["structure_status"] == M]
        out_rows = kept
        if dropped:
            sidecar = str(Path(args.out).with_name("excluded_modified_ligands.csv"))
            with open(sidecar, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(base_cols) + EXTRA_COLS)
                w.writeheader()
                w.writerows(dropped)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(base_cols) + EXTRA_COLS)
        w.writeheader()
        w.writerows(out_rows)

    # report
    from collections import Counter
    status_counts = Counter(r["structure_status"] for r in out_rows)
    print(f"[*] annotated {len(rows)} rows; added {len(added)} literature row(s); "
          f"wrote {len(out_rows)} -> {args.out}")
    print(f"    structure_status: {dict(status_counts)}")
    if added:
        print("    added:", ", ".join(f"{p}/{e}->{b}" for p, e, b in added))
    if unmatched:
        print(f"    [warn] {len(unmatched)} rows had no bound-PDB mapping "
              f"(status=sequence_only): {unmatched}")
    show = dropped if args.drop_modified else [r for r in out_rows if r["structure_status"] == M]
    if show:
        verb = "dropped (modified_bound, written to sidecar)" if args.drop_modified else "modified_bound"
        print(f"    {verb}:")
        for r in show:
            print(f"      {r['pdb_id']}/{r['epitope']} -> {r['bound_pdb']}: {r['modification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
