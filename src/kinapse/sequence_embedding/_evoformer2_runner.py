#!/usr/bin/env python3
"""Isolated runner for the DB-free Evoformer-representation path (evoformer2 backend).

Runs OPIG's ``predict_evoformer2.py`` flow inside an env that has alphaflow + openfold +
torch (e.g. ``openfold_env``): build the MSA via the **ColabFold remote server** (no local
AlphaFold databases), then extract AlphaFold's single/pair Evoformer representations with the
AF2 weights and save ``<outdir>/<stem>.pkl`` — exactly the conditioning the DiG sampler wants.

kinapse shells this out (via ``kinapse.sequence_embedding.evoformer``); it imports nothing
from kinapse. The alphaflow + evoformer_representation source dirs are injected on sys.path so
we don't depend on the hardcoded ``/workspaces`` paths inside the upstream scripts.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_file", required=True, help="linked (single-chain) structure")
    ap.add_argument("--outdir", required=True, help="where <stem>.pkl is written")
    ap.add_argument("--weights_path", required=True, help="AF2 params .npz (e.g. params_model_1.npz)")
    ap.add_argument("--msa_dir", required=True, help="dir for the ColabFold MSA")
    ap.add_argument("--alphaflow_dir", required=True, help="alphaflow source dir")
    ap.add_argument("--evoformer_dir", required=True,
                    help="dir holding predict_evoformer2.py / make_MSA.py / MSA_query.py / import_weights.py")
    ap.add_argument("--openfold_dir", required=True, help="OpenFold checkout (alphaflow imports openfold)")
    a = ap.parse_args()

    # Inject the real source dirs BEFORE importing the upstream scripts (they hardcode
    # /workspaces paths; our inserts take precedence). alphaflow imports openfold, so its
    # checkout must be importable too.
    sys.path.insert(0, a.openfold_dir)
    sys.path.insert(0, a.alphaflow_dir)
    sys.path.insert(0, a.evoformer_dir)
    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(a.msa_dir, exist_ok=True)
    os.chdir(a.outdir)              # writable cwd — alphaflow creates ./workdir at import

    from make_MSA import make_MSA
    import predict_evoformer2       # top-level sets up the AlphaFold config (use_templates=False, ...)

    input_csv, msa_dir = make_MSA(a.pdb_file, a.msa_dir)   # ColabFold MSA (remote)
    predict_evoformer2.main(input_csv, msa_dir, a.weights_path, a.outdir, mode="alphafold")


if __name__ == "__main__":
    main()
