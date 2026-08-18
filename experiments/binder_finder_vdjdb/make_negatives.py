#!/usr/bin/env python3
"""Build a negative (non-binder) pMHC dataset for the VDJdb confirmed binders.

For each TCR in ``vdjdb_confirmed_binders.csv`` a negative pMHC is an **existing**
pMHC drawn from a database (VDJdb) such that:

  * its MHC is one the TCR *is* a confirmed binder for  (same MHC), and
  * its peptide is **not** any confirmed-binder peptide of that TCR  (different peptide).

These are inferred/putative non-binders (the TCR is simply not recorded to bind
them) matched on MHC so only the peptide differs — the standard hard-negative
construction. The peptide is a real epitope presented by that MHC in VDJdb, so
the pMHC itself exists.

Output columns mirror ``vdjdb_confirmed_binders.csv`` (so this file is a drop-in
input to ``run_tfold_binders.py``) with ``label=non_binder`` and a few provenance
columns appended.

Note: VDJdb (positive-only) has a thin epitope pool for rare alleles — e.g.
HLA-A*24:02 has only 3 epitopes total, so those TCRs get very few negatives. A
larger presented-peptide DB (IEDB, NetMHCpan-predicted binders) would enrich
them; pass a different --vdjdb or extend the pool source for that.

Usage:
    python make_negatives.py                       # all candidate negatives
    python make_negatives.py --length-matched      # only peptides of a positive's length
    python make_negatives.py --max-per-tcr 5       # cap per TCR (hard/length-matched first)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_CSV = str(_HERE / "results" / "vdjdb_confirmed_binders.csv")
_DEFAULT_VDJDB = "/mnt/larry/lilian/DATA/vdjdb/vdjdb.slim.txt"

POS_COLS = ["pdb_id", "pair", "cdr3a", "cdr3b", "label", "epitope", "antigen_gene",
            "antigen_species", "mhc_a", "mhc_b", "mhc_class", "tcr_species",
            "vdjdb_score", "n_complexes", "complex_ids", "references"]
NEG_EXTRA = ["neg_source", "pep_len", "length_matched", "neg_epitope_vdjdb_score"]


def two_field(allele: str) -> str:
    a = str(allele).split(",")[0].strip()
    p = a.split(":")
    return ":".join(p[:2]) if len(p) >= 2 else a


def load_vdjdb_pool(path: str):
    """Scan VDJdb once. Returns:
      pool:    {2-field MHC-I allele: {epitope: {gene, species, score, mhc_class}}}
      by_cdr3: {cdr3 (any gene): set(epitopes)} — every epitope a CDR3 is recorded with.
    """
    pool: dict = defaultdict(dict)
    by_cdr3: dict = defaultdict(set)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ep = r["antigen.epitope"].strip()
            if not ep:
                continue
            cdr3 = r["cdr3"].strip().upper()
            if cdr3:
                by_cdr3[cdr3].add(ep)
            if r["mhc.class"] != "MHCI":
                continue
            al = two_field(r["mhc.a"])
            try:
                score = int(r["vdjdb.score"].strip() or 0)
            except ValueError:
                score = 0
            cur = pool[al].get(ep)
            if cur is None:
                pool[al][ep] = {"gene": r["antigen.gene"].strip(),
                                "species": r["antigen.species"].strip(),
                                "score": score, "mhc_class": "MHCI"}
            else:
                cur["score"] = max(cur["score"], score)
    return pool, by_cdr3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=_DEFAULT_CSV, help="vdjdb_confirmed_binders.csv")
    ap.add_argument("--vdjdb", default=_DEFAULT_VDJDB, help="database of existing pMHCs")
    ap.add_argument("--out", default=str(_HERE / "results" / "vdjdb_negatives.csv"))
    ap.add_argument("--combined-out", default=str(_HERE / "results" / "vdjdb_labeled_dataset.csv"),
                    help="positives + negatives in one labeled file")
    ap.add_argument("--length-matched", action="store_true",
                    help="only keep negatives whose length matches a positive peptide "
                         "of that TCR on the same MHC (hardest negatives)")
    ap.add_argument("--max-per-tcr", type=int, default=None,
                    help="cap negatives per TCR (length-matched then higher-confidence first)")
    ap.add_argument("--exclude-looser-binders", action="store_true",
                    help="also drop peptides the TCR binds under looser CDR3b (or CDR3a) "
                         "matching in VDJdb, not just the paired-confirmed peptides "
                         "(stricter negatives; e.g. removes WT TAX from the A6 TCR)")
    ap.add_argument("--exclude-peptides-csv", default=None,
                    help="CSV with an 'epitope' column of peptides to drop from the candidate "
                         "pool globally (e.g. excluded_modified_ligands.csv — their sequence "
                         "strings don't represent the real ligand, so never use them as negatives)")
    args = ap.parse_args(argv)

    # 1) read positives, group by TCR
    with open(args.csv, newline="") as fh:
        positives = list(csv.DictReader(fh))
    tcrs: dict = {}
    for r in positives:
        t = tcrs.setdefault(r["pdb_id"], {
            "pdb_id": r["pdb_id"], "pair": r.get("pair", ""),
            "cdr3a": r["cdr3a"], "cdr3b": r["cdr3b"],
            "tcr_species": r.get("tcr_species", ""),
            "mhcs": set(), "peps": set(), "pep_len_by_mhc": defaultdict(set)})
        mhc = two_field(r["mhc_a"])
        t["mhcs"].add(mhc)
        t["peps"].add(r["epitope"])
        t["pep_len_by_mhc"][mhc].add(len(r["epitope"]))
    print(f"[*] {len(positives)} positive rows across {len(tcrs)} TCRs")

    # 2) database pool of existing pMHCs (+ CDR3->epitope index for optional exclusion)
    pool, by_cdr3 = load_vdjdb_pool(args.vdjdb)
    print(f"[*] VDJdb MHC-I pool: {len(pool)} alleles, "
          f"{sum(len(v) for v in pool.values())} allele-epitope pMHCs")

    # optional: drop artefactual peptide strings (e.g. modified ligands) from the pool
    if args.exclude_peptides_csv:
        drop = {r["epitope"].strip() for r in csv.DictReader(open(args.exclude_peptides_csv))
                if r.get("epitope", "").strip()}
        removed = 0
        for al in pool:
            for ep in list(pool[al]):
                if ep in drop:
                    del pool[al][ep]; removed += 1
        print(f"[*] excluded {len(drop)} peptide string(s) from the pool "
              f"({removed} allele-epitope entries removed): {sorted(drop)}")

    # 3) per-TCR negatives: same MHC, peptide not among the TCR's confirmed binders
    neg_rows = []
    for pdb_id, t in tcrs.items():
        n_before = len(neg_rows)
        # peptides to treat as "bound" by this TCR (always the paired-confirmed set;
        # optionally also anything its CDR3b/CDR3a is recorded with in VDJdb)
        exclude = set(t["peps"])
        if args.exclude_looser_binders:
            exclude |= by_cdr3.get((t["cdr3b"] or "").upper(), set())
            exclude |= by_cdr3.get((t["cdr3a"] or "").upper(), set())
        cands = []
        for mhc in sorted(t["mhcs"]):
            pos_lens = t["pep_len_by_mhc"][mhc]
            for ep, info in pool.get(mhc, {}).items():
                if ep in exclude:
                    continue  # different peptide than ANY (confirmed / looser) binder of the TCR
                lm = len(ep) in pos_lens
                if args.length_matched and not lm:
                    continue
                cands.append((mhc, ep, info, lm))
        # deterministic order: length-matched first, then higher epitope confidence, then name
        cands.sort(key=lambda c: (not c[3], -c[2]["score"], c[0], c[1]))
        if args.max_per_tcr is not None:
            cands = cands[:args.max_per_tcr]
        for mhc, ep, info, lm in cands:
            neg_rows.append({
                "pdb_id": pdb_id, "pair": t["pair"],
                "cdr3a": t["cdr3a"], "cdr3b": t["cdr3b"],
                "label": "non_binder", "epitope": ep,
                "antigen_gene": info["gene"], "antigen_species": info["species"],
                "mhc_a": mhc, "mhc_b": "B2M", "mhc_class": "MHCI",
                "tcr_species": t["tcr_species"],
                "vdjdb_score": "", "n_complexes": "", "complex_ids": "", "references": "",
                "neg_source": "VDJdb:same-MHC,diff-peptide",
                "pep_len": len(ep), "length_matched": lm,
                "neg_epitope_vdjdb_score": info["score"],
            })
        print(f"  {pdb_id}: MHC={sorted(t['mhcs'])} pos_pep={len(t['peps'])} "
              f"-> {len(neg_rows) - n_before} negative pMHC(s)")

    # 4) write negatives
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=POS_COLS + NEG_EXTRA)
        w.writeheader()
        w.writerows(neg_rows)

    # 5) combined labeled dataset (positives + negatives)
    with open(args.combined_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=POS_COLS + NEG_EXTRA)
        w.writeheader()
        for r in positives:
            row = {c: r.get(c, "") for c in POS_COLS}
            row.update({"neg_source": "", "pep_len": len(r["epitope"]),
                        "length_matched": "", "neg_epitope_vdjdb_score": ""})
            w.writerow(row)
        w.writerows(neg_rows)

    n_lm = sum(1 for r in neg_rows if r["length_matched"])
    print(f"\n[*] {len(neg_rows)} negatives ({n_lm} length-matched) "
          f"for {len(positives)} positives.")
    print(f"    negatives : {args.out}")
    print(f"    combined  : {args.combined_out}  "
          f"({len(positives)} confirmed_binder + {len(neg_rows)} non_binder)")
    print("\n[!] Negatives are inferred non-binders (TCR not recorded to bind them), "
          "matched on MHC so only the peptide differs. Rare alleles have few VDJdb "
          "peptides; use --length-matched / --max-per-tcr to shape the set, or a "
          "larger presented-peptide DB (IEDB) for more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
