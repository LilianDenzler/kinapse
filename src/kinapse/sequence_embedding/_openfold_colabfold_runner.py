#!/usr/bin/env python3
"""DB-free OpenFold embedding: ColabFold (remote) MSA -> run_pretrained_openfold_shortened.

This is the user's *proven* representation script (`run_pretrained_openfold_shortened.py`,
the one from the working training-data log) but fed an MSA from the **ColabFold remote
server** instead of the local AlphaFold genetic databases (which aren't on this machine).
No jackhmmer/bfd/uniref90 DBs required — just internet + a GPU + the bundled AF2 params.

Steps (run inside openfold_env, with the OpenFold + evoformer_representation dirs on path):
  1. write a one-sequence FASTA for the linked structure (tag = pdb stem),
  2. build the ColabFold MSA (`make_MSA`/`MSA_query` -> a3m),
  3. re-lay it into OpenFold's precomputed-alignments layout: <align>/<tag>/colabfold.a3m,
  4. run run_pretrained_openfold_shortened.py --use_precomputed_alignments --save_outputs,
     which writes <outdir>/predictions/<tag>_<preset>_output_dict.pkl.

Imports nothing from kinapse.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _linked_sequence(pdb_file):
    """Single-chain sequence of the linked structure (standard residues), like make_split_csv."""
    from Bio.PDB import PDBParser, is_aa
    from Bio.SeqUtils import seq1
    st = PDBParser(QUIET=True).get_structure("s", pdb_file)
    for model in st:
        for chain in model:
            res = [r for r in chain if is_aa(r, standard=True)]
            if res:
                return "".join(seq1(r.get_resname()) for r in res)
    raise RuntimeError(f"no standard residues in {pdb_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_file", required=True, help="linked (single-chain) structure")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mmcif_dir", required=True, help="template CIFs (featurizer needs the dir to exist)")
    ap.add_argument("--openfold_dir", required=True, help="Graphormer/openfold (has run_pretrained_openfold_shortened.py)")
    ap.add_argument("--evoformer_dir", required=True, help="dir with make_MSA.py / MSA_query.py (ColabFold query)")
    ap.add_argument("--config_preset", default="model_1_ptm")
    ap.add_argument("--model_device", default="cuda:0")
    a = ap.parse_args()

    sys.path.insert(0, a.openfold_dir)
    sys.path.insert(0, a.evoformer_dir)
    outdir = os.path.abspath(a.outdir)
    os.makedirs(outdir, exist_ok=True)
    os.chdir(a.openfold_dir)                       # so openfold/resources/params/<preset>.npz resolves

    stem = Path(a.pdb_file).stem                    # openfold tag = FASTA header = stem
    # 1) one-sequence FASTA (header = stem)
    fasta_dir = os.path.join(outdir, "of_fasta")
    os.makedirs(fasta_dir, exist_ok=True)
    seq = _linked_sequence(a.pdb_file)
    with open(os.path.join(fasta_dir, f"{stem}.fasta"), "w") as f:
        f.write(f">{stem}\n{seq}\n")

    # 2) ColabFold MSA (remote) -> <msa_dir>/<stem>/a3m/<stem>.a3m
    from make_MSA import make_MSA
    msa_dir = os.path.join(outdir, "colabfold_msa")
    os.makedirs(msa_dir, exist_ok=True)
    make_MSA(a.pdb_file, msa_dir)

    # 3) re-lay into OpenFold layout: <align>/<tag>/*.a3m
    align_dir = os.path.join(outdir, "alignments")
    tag_dir = os.path.join(align_dir, stem)
    os.makedirs(tag_dir, exist_ok=True)
    src_a3m = None
    for cand in (os.path.join(msa_dir, stem, "a3m", f"{stem}.a3m"),):
        if os.path.exists(cand):
            src_a3m = cand
            break
    if src_a3m is None:                             # fall back to any a3m ColabFold produced
        hits = list(Path(msa_dir).rglob("*.a3m"))
        if not hits:
            raise FileNotFoundError(f"ColabFold produced no .a3m under {msa_dir}")
        src_a3m = str(hits[0])
    shutil.copyfile(src_a3m, os.path.join(tag_dir, "colabfold.a3m"))
    print(f"[colabfold] MSA -> {tag_dir}/colabfold.a3m")

    # 4) OpenFold representation with precomputed alignments (no local DBs)
    cmd = [sys.executable, "-u",
           os.path.join(a.openfold_dir, "run_pretrained_openfold_shortened.py"),
           fasta_dir, a.mmcif_dir,
           "--use_precomputed_alignments", align_dir,
           "--output_dir", outdir,
           "--config_preset", a.config_preset,
           "--model_device", a.model_device,
           "--save_outputs"]
    print("[colabfold] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=a.openfold_dir)
    # run_pretrained writes <outdir>/predictions/<tag>_<preset>_output_dict.pkl


if __name__ == "__main__":
    main()
