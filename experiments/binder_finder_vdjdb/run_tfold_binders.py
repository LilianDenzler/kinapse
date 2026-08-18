#!/usr/bin/env python3
"""Generate tFold-TCR structures for the VDJdb binders found by find_vdjdb_binders.py.

Pipeline: read ``vdjdb_confirmed_binders.csv`` (one row per TCR + confirmed-binder
pMHC), assemble the tFold-TCR input for each, and predict the TCR-pMHC complex via
kinapse's ``tfold_tcr`` runner (``kinapse.structure_prediction.tfold`` -> external
tFold env). One PDB per (TCR, pMHC) is written.

Where each chain's sequence comes from
--------------------------------------
  B  TCR beta   variable domain  -- from the source structure (kinapse loader)
  A  TCR alpha  variable domain  -- from the source structure (kinapse loader)
  M  MHC heavy  mature ectodomain -- IPD-IMGT/HLA fetch, truncated per MHC_TRUNCATION.csv
  N  beta-2-microglobulin        -- constant (tFold's own example B2M)
  P  peptide                     -- the VDJdb epitope

The TCR of each CSV row is the source structure it was extracted from, so its
alpha/beta come straight back from ``<dataset>/<pdb_id>/<pdb_id>.pdb``. Only the
pMHC varies across a TCR's rows.

Run in the ``kinapse`` env; point it at the tFold env (default below):
    KINAPSE_TFOLD_PYTHON=/home/lilian/miniconda3/envs/tfold-tcr/bin/python \
    python run_tfold_binders.py                     # full run
    python run_tfold_binders.py --dry-run           # assemble inputs only (no GPU)
    python run_tfold_binders.py --limit 2           # smoke-test two complexes
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent
_DEFAULT_CSV = "/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD/vdjdb_binders/vdjdb_confirmed_binders.csv"
_DEFAULT_DATASET = "/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD"
_DEFAULT_TRUNCATION = "/mnt/bob/lilian/projects/TCR_DATABASE_MANAGER/TCR_datasets/ATLAS/MHC_TRUNCATION.csv"
_DEFAULT_TFOLD_PY = "/home/lilian/.conda/envs/tfold-tcr/bin/python"
_DEFAULT_TFOLD_REPO = "/home/lilian/TCR_interface/tfold-tencent"

# beta-2-microglobulin (MHC-I light chain), taken verbatim from tFold-TCR's own
# examples/tcr_pmhc_example.json so it matches what the model was shown.
B2M = ("MIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEF"
       "TPTEKDEYACRVNHVTLSQPKIVKWDRDM")


# --------------------------------------------------------------------------- #
# TCR alpha/beta variable domains from a source structure (kinapse loader)
# --------------------------------------------------------------------------- #
def tcr_variable_seqs(pdb_path: str, legacy_anarci: bool = True) -> dict:
    """{'A': alpha_variable_seq, 'B': beta_variable_seq} for the first TCR pair."""
    import contextlib
    import io as _io
    from kinapse.structures import TCR, ops
    with contextlib.redirect_stdout(_io.StringIO()):
        tcr = TCR(input_pdb=str(pdb_path), legacy_anarci=legacy_anarci)
    if not tcr.pairs:
        raise RuntimeError(f"no TCR pair found in {pdb_path}")
    pair = tcr.pairs[0]
    seqd = ops.get_sequence_dict(pair.variable_structure)  # {chain_id: seq}
    return {"A": seqd[pair.chain_map["alpha"]].upper(),
            "B": seqd[pair.chain_map["beta"]].upper()}


# --------------------------------------------------------------------------- #
# MHC heavy chain: allele -> mature ectodomain (IPD-IMGT/HLA + truncation table)
# --------------------------------------------------------------------------- #
def _two_field(allele: str) -> str:
    a = str(allele).split(",")[0].strip()
    parts = a.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else a


def _ipd_accession(allele: str) -> tuple:
    name = allele.replace("HLA-", "")
    base = "https://www.ebi.ac.uk/cgi-bin/ipd/api/allele"
    q = (f'or(startsWith(name,"{name}"),contains(previous_nomenclature,"{name}"),'
         f'eq(accession,"{name}"))')
    url = f"{base}?{urllib.parse.urlencode({'limit': 1, 'project': 'HLA', 'fields': 'name,accession', 'query': q}, safe='(),*')}"
    r = requests.get(url, headers={"accept": "application/json"}, timeout=30)
    r.raise_for_status()
    data = r.json().get("data") or []
    if not data:
        raise ValueError(f"no IPD/HLA record for {allele}")
    return data[0]["accession"], data[0]["name"]


def _ipd_sequence(accession: str) -> str:
    url = (f"https://www.ebi.ac.uk/Tools/dbfetch/dbfetch?db=imgthlapro;id={accession}"
           f"&format=fasta&style=raw")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return "".join(r.text.split("\n")[1:]).strip()


def _load_truncation(path: str) -> dict:
    trunc = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            trunc[row["MHC_name"]] = (int(row["start"]), int(row["end"]))
    return trunc


def resolve_mhc_sequences(alleles, truncation_path: str, cache_path: str) -> dict:
    """Map each 2-field allele -> mature ectodomain seq, with a CSV cache.

    Rare sub-alleles that IPD can't resolve fall back to their locus 2-field
    parent's :01 (e.g. HLA-A*02:256 -> HLA-A*02:01) with a warning.
    """
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, newline="") as fh:
            for row in csv.DictReader(fh):
                cache[row["allele"]] = row["mhc_sequence"]
    trunc = _load_truncation(truncation_path)

    def locus(al):  # 'HLA-A*02:01' -> 'HLA-A'
        return al.split("*")[0]

    def fetch_truncated(al):
        acc, name = _ipd_accession(al)
        full = _ipd_sequence(acc)
        lo, hi = trunc.get(locus(al), (1, len(full)))
        return full[lo - 1:hi], name

    for allele in sorted(set(alleles)):
        if allele in cache:
            continue
        try:
            seq, name = fetch_truncated(allele)
            print(f"  MHC {allele}: IPD {name} -> mature ectodomain len={len(seq)}")
        except Exception as e:  # noqa: BLE001
            fb = f"{locus(allele)}*{allele.split('*')[1].split(':')[0]}:01"
            print(f"  MHC {allele}: fetch failed ({e}); falling back to {fb}")
            try:
                seq, name = fetch_truncated(fb)
            except Exception as e2:  # noqa: BLE001
                print(f"  MHC {allele}: fallback {fb} also failed ({e2}); SKIPPING")
                continue
        cache[allele] = seq

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["allele", "mhc_sequence"])
        for al in sorted(cache):
            w.writerow([al, cache[al]])
    return cache


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=_DEFAULT_CSV, help="vdjdb_confirmed_binders.csv")
    ap.add_argument("--dataset", default=_DEFAULT_DATASET,
                    help="root of <pdb_id>/<pdb_id>.pdb source TCR structures")
    ap.add_argument("--out", default=str(_HERE / "results" / "tfold_complexes"),
                    help="output dir for predicted PDBs")
    ap.add_argument("--mhc-cache", default=str(_HERE / "results" / "mhc_sequences.csv"))
    ap.add_argument("--truncation", default=_DEFAULT_TRUNCATION)
    ap.add_argument("--tfold-python", default=os.environ.get("KINAPSE_TFOLD_PYTHON", _DEFAULT_TFOLD_PY))
    ap.add_argument("--tfold-repo", default=os.environ.get("KINAPSE_TFOLD_REPO", _DEFAULT_TFOLD_REPO))
    ap.add_argument("--device", default=None, help="e.g. cuda:0 (default: auto)")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N complexes")
    ap.add_argument("--legacy-anarci", action="store_true", default=True)
    ap.add_argument("--new-anarci", dest="legacy_anarci", action="store_false")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble inputs + write jobs JSON, but do not run tFold")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) read binder rows
    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[*] {len(rows)} binder rows from {args.csv}")

    # 2) MHC sequences per (2-field) allele
    alleles = [_two_field(r["mhc_a"]) for r in rows]
    print("[*] Resolving MHC ectodomain sequences ...")
    mhc_seqs = resolve_mhc_sequences(alleles, args.truncation, args.mhc_cache)

    # 3) TCR variable domains per source structure (cache by pdb_id)
    print("[*] Extracting TCR variable domains from source structures ...")
    tcr_cache: dict = {}
    for pdb_id in sorted({r["pdb_id"] for r in rows}):
        pdb_path = Path(args.dataset) / pdb_id / f"{pdb_id}.pdb"
        try:
            tcr_cache[pdb_id] = tcr_variable_seqs(pdb_path, legacy_anarci=args.legacy_anarci)
            print(f"  {pdb_id}: A={len(tcr_cache[pdb_id]['A'])}aa  B={len(tcr_cache[pdb_id]['B'])}aa")
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb_id}: TCR extraction FAILED ({e})")
            tcr_cache[pdb_id] = None

    # 4) assemble tFold jobs
    jobs, meta = [], []
    for r in rows:
        pdb_id, epitope, allele = r["pdb_id"], r["epitope"], _two_field(r["mhc_a"])
        tcr = tcr_cache.get(pdb_id)
        mhc = mhc_seqs.get(allele)
        if not tcr or not mhc:
            print(f"  [skip] {pdb_id} / {epitope} / {allele}: missing "
                  f"{'TCR' if not tcr else 'MHC'} sequence")
            continue
        name = f"{pdb_id}__{_sanitize(epitope)}__{_sanitize(allele)}"
        out_pdb = str(out_dir / f"{name}.pdb")
        jobs.append({
            "name": name,
            "chains": [
                {"id": "B", "sequence": tcr["B"]},   # TCR beta
                {"id": "A", "sequence": tcr["A"]},   # TCR alpha
                {"id": "M", "sequence": mhc},        # MHC heavy (mature ectodomain)
                {"id": "N", "sequence": B2M},        # beta-2-microglobulin
                {"id": "P", "sequence": epitope},    # peptide
            ],
            "out_pdb": out_pdb,
        })
        meta.append({"name": name, "pdb_id": pdb_id, "epitope": epitope,
                     "mhc_allele": allele, "out_pdb": out_pdb})

    jobs_json = out_dir / "tfold_jobs.json"
    jobs_json.write_text(json.dumps(jobs, indent=2))
    print(f"\n[*] Assembled {len(jobs)} tFold jobs -> {jobs_json}")

    if args.dry_run:
        print("[*] --dry-run: not running tFold. Inputs are ready.")
        return 0

    # 5) run tFold-TCR via the kinapse runner (external env)
    os.environ["KINAPSE_TFOLD_PYTHON"] = args.tfold_python
    if args.tfold_repo:
        os.environ["KINAPSE_TFOLD_REPO"] = args.tfold_repo
    from kinapse.structure_prediction.tfold import predict_many, available
    if not available():
        print(f"[!] tFold env not found at {args.tfold_python}. "
              "Set --tfold-python / KINAPSE_TFOLD_PYTHON. Inputs are ready in "
              f"{jobs_json}; rerun without --dry-run once the env exists.")
        return 2

    print(f"[*] Running tFold-TCR on {len(jobs)} complexes "
          f"(python={args.tfold_python}) ...")
    res = predict_many(jobs, model_version="Complex", device=args.device,
                       chunk_size=args.chunk_size)
    results = res.get("results", [])
    by_name = {r["name"]: r for r in results}
    if res.get("status") != "ok":
        print(f"[!] runner status={res.get('status')}: {res.get('error')}")

    # 6) summary
    summary_csv = out_dir / "tfold_summary.csv"
    with open(summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "pdb_id", "epitope", "mhc_allele",
                                           "status", "iptm", "ptm", "out_pdb"])
        w.writeheader()
        ok = 0
        for m in meta:
            r = by_name.get(m["name"], {})
            if r.get("status") in ("ok", "skipped"):
                ok += 1
            w.writerow({**{k: m[k] for k in ("name", "pdb_id", "epitope", "mhc_allele", "out_pdb")},
                        "status": r.get("status", "missing"),
                        "iptm": r.get("iptm"), "ptm": r.get("ptm")})
    print(f"\n[*] Done. {ok}/{len(meta)} complexes produced. Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
