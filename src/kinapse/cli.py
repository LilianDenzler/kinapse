"""``kinapse`` command-line interface.

A thin, dependency-light entry point over the library. Subcommands:

    kinapse info                       show version, config & packaged-data paths
    kinapse prep     PDB [opts]        load/renumber/pair a TCR, write prepped structures
    kinapse add-cd8  PDB [opts]        graft the correct CD8 co-receptor onto a class-I TCR-pMHC
    kinapse geometry PDB [opts]        compute TCR alpha/beta docking geometry
    kinapse pipelines                  list the migrated end-to-end workflows

The heavier benchmark/best-method pipelines still carry their original example
paths; run them as modules (see ``kinapse pipelines``) or drive them via
:mod:`kinapse.config`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_info(args) -> int:
    import kinapse
    from kinapse import config
    paths = config.get_paths()
    print(f"kinapse {kinapse.__version__}")
    print(f"  project_root : {paths.project_root}")
    print(f"  output_dir   : {paths.get_output_dir()}")
    print(f"  dataset_dir  : {paths.get_dataset_dir()}")
    try:
        print(f"  package_data : {config.package_data_dir()}")
        print(f"  so3 tables   : {config.so3_dir()}")
        print(f"  consensus    : {config.consensus_output_dir()}")
    except Exception as e:  # pragma: no cover
        print(f"  (packaged data lookup failed: {e})")
    return 0


def cmd_prep(args) -> int:
    from kinapse.structures import TCR, write_pdb
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tcr = TCR(input_pdb=args.pdb, traj_path=args.traj, legacy_anarci=not args.new_anarci)
    print(f"Loaded {args.pdb}: {len(tcr.pairs)} TCR pair(s)")
    for i, pair in enumerate(tcr.pairs):
        base = outdir / f"pair{i}"
        try:
            write_pdb(f"{base}_variable.pdb", pair.variable_structure)
            print(f"  wrote {base}_variable.pdb")
        except Exception as e:
            print(f"  [warn] variable_structure failed: {e}")
        try:
            full = getattr(pair, "full_structure", None)
            if full is not None:
                write_pdb(f"{base}_full.pdb", full)
                print(f"  wrote {base}_full.pdb")
        except Exception as e:
            print(f"  [warn] full_structure failed: {e}")
        try:
            seqs = pair.cdr_fr_sequences()
            print(f"  pair{i} CDR/FR sequences:")
            for name, seq in seqs.items():
                print(f"    {name:8s} {seq}")
        except Exception as e:
            print(f"  [warn] cdr_fr_sequences failed: {e}")
    return 0


def cmd_add_cd8(args) -> int:
    from kinapse.structures.cd8 import add_cd8
    ref_cd8 = args.cd8_chains.split(",") if args.cd8_chains else None
    res = add_cd8(
        args.pdb,
        out_pdb=args.out,
        ref_pdb=args.ref_pdb,
        ref_mhc_chain=args.ref_mhc_chain,
        ref_cd8_chains=ref_cd8,
        species=args.species,
        target_mhc_chain=args.target_mhc_chain,
        cd8_new_chain_ids=tuple(args.new_chain_ids.split(",")),
        fit_resrange=tuple(args.fit_range) if args.fit_range else None,
        do_fixer=args.fixer,
        ph=args.ph,
    )
    ident = "n/a" if res.mhc_identity != res.mhc_identity else f"{res.mhc_identity:.0%}"
    print(f"template     : {res.template} (MHC chain {res.ref_mhc_chain})")
    print(f"target MHC   : chain {res.target_mhc_chain}  (identity {ident})")
    print(f"CD8 added    : chains {res.cd8_chains}")
    print(f"fit RMSD     : {res.rmsd:.3f} Å over {res.n_matched} Cα")
    if res.out_pdb:
        print(f"wrote        : {res.out_pdb}")
    return 0


def cmd_geometry(args) -> int:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.traj:
        from kinapse.geometry import calc_tcr_geometry_MD
        df = calc_tcr_geometry_MD(args.traj, args.pdb)
        out_csv = outdir / "geometry_md.csv"
        try:
            df.to_csv(out_csv, index=False)
            print(f"wrote {out_csv} ({len(df)} frames)")
        except Exception:
            print(df)
    else:
        from kinapse.geometry import calc_tcr_geometry
        calc_tcr_geometry(args.pdb, str(outdir), vis=args.vis)
        print(f"geometry written under {outdir}")
    return 0


def cmd_pipelines(args) -> int:
    entries = [
        ("kinapse.pipelines.benchmark.benchmarks_calc_scaled",
         "Two-phase model-vs-MD benchmark (collect -> PMF/JSD -> tables)."),
        ("kinapse.pipelines.benchmark.benchmarks_calc_scaled_parallel",
         "Same, parallelised across (config, model) with a process pool."),
        ("kinapse.pipelines.best_method.run_test",
         "Screen all reducers per region; compute best (feature x reducer)."),
        ("kinapse.pipelines.best_method.global_analysis_best_metric",
         "Cross-TCR study selecting the best reducer per region."),
        ("kinapse.pipelines.analyse_md.gt_flexibility_analysis",
         "Ground-truth-only MD flexibility analysis + ridgeline overlays."),
    ]
    print("Migrated end-to-end pipelines (run with `python -m <module>`):\n")
    for mod, desc in entries:
        print(f"  python -m {mod}")
        print(f"      {desc}")
    print("\nNote: their __main__ blocks still carry the original example data")
    print("paths — edit those or drive the functions via kinapse.config.")
    return 0


def cmd_score(args) -> int:
    from kinapse import scoring
    if args.list_scorers:
        try:
            for name, info in scoring.available_scorers().items():
                metrics = ", ".join(info["metrics"][:6]) + ("…" if len(info["metrics"]) > 6 else "")
                print(f"  {name:10s} [{info['backend']}]  {metrics}")
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0
    if not args.model:
        print("error: a model PDB is required (or use --list-scorers)", file=sys.stderr)
        return 2
    rec, lig = args.rec, args.lig
    if args.auto_chains or (rec is None and lig is None):
        try:
            rec, lig = scoring.infer_tcr_pmhc_chains(args.model, legacy_anarci=not args.new_anarci)
        except Exception as e:
            print(f"error: could not auto-derive chains: {e}", file=sys.stderr)
            return 1
        print(f"chains  receptor(TCR)={rec}  ligand(pMHC)={lig}")
    try:
        res = scoring.score(args.model, native=args.native, rec=rec, lig=lig,
                            scorers=args.scorers, timeout=args.timeout)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for k, v in res.items():
        print(f"  {k}: {v}")
    if args.out:
        from pathlib import Path
        p = Path(args.out)
        if p.suffix.lower() == ".csv":
            import pandas as pd
            pd.DataFrame([res]).to_csv(p, index=False)
        else:
            import json
            p.write_text(json.dumps(res, indent=2, default=str))
        print(f"wrote {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kinapse", description="TCR/TCR-pMHC dynamics toolkit")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("info", help="show version, config and data paths")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("prep", help="load, IMGT-renumber and pair a TCR; write prepped structures")
    sp.add_argument("pdb", help="input TCR PDB")
    sp.add_argument("--traj", default=None, help="optional trajectory (.xtc) to attach")
    sp.add_argument("--out", default="kinapse_prep", help="output directory")
    sp.add_argument("--new-anarci", action="store_true", help="use ANARCII (GPU) instead of legacy ANARCI")
    sp.set_defaults(func=cmd_prep)

    sp = sub.add_parser("add-cd8", help="graft the correct CD8 co-receptor onto a TCR-pMHC (class I) complex")
    sp.add_argument("pdb", help="input TCR-pMHC (or pMHC) complex PDB")
    sp.add_argument("--out", default="with_cd8.pdb", help="output PDB path")
    sp.add_argument("--species", default=None, help="restrict template selection (e.g. human, mouse)")
    sp.add_argument("--target-mhc-chain", default=None, help="MHC heavy-chain id in the target (auto-detected if omitted)")
    sp.add_argument("--new-chain-ids", default="S,T", help="chain ids for the grafted CD8 (comma-separated; auto-bumped on collision)")
    sp.add_argument("--fit-range", nargs=2, type=int, default=None, metavar=("START", "END"),
                    help="restrict superposition to this reference resseq range (e.g. the alpha3 domain)")
    sp.add_argument("--ref-pdb", default=None, help="use your own reference PDB instead of the registry (needs --ref-mhc-chain and --cd8-chains)")
    sp.add_argument("--ref-mhc-chain", default=None, help="MHC heavy-chain id in --ref-pdb")
    sp.add_argument("--cd8-chains", default=None, help="CD8 chain ids in --ref-pdb (comma-separated, e.g. D,E)")
    sp.add_argument("--fixer", action="store_true", help="also run PDBFixer cleanup (needs pdbfixer/openmm)")
    sp.add_argument("--ph", type=float, default=7.4, help="pH for PDBFixer hydrogen placement")
    sp.set_defaults(func=cmd_add_cd8)

    sp = sub.add_parser("geometry", help="compute TCR alpha/beta docking geometry")
    sp.add_argument("pdb", help="input TCR PDB (or topology when --traj is given)")
    sp.add_argument("--traj", default=None, help="trajectory (.xtc); compute geometry per frame")
    sp.add_argument("--out", default="kinapse_geometry", help="output directory")
    sp.add_argument("--vis", action="store_true", help="also write PyMOL visualisation (needs pymol)")
    sp.set_defaults(func=cmd_geometry)

    sp = sub.add_parser("pipelines", help="list migrated end-to-end workflows")
    sp.set_defaults(func=cmd_pipelines)

    sp = sub.add_parser("score", help="score a TCR-pMHC complex via ifscore (optional dependency)")
    sp.add_argument("model", nargs="?", default=None, help="model complex PDB")
    sp.add_argument("--native", default=None, help="native reference PDB (for DockQ etc.)")
    sp.add_argument("--rec", default=None, help="receptor chains, comma-separated (e.g. D,E)")
    sp.add_argument("--lig", default=None, help="ligand chains, comma-separated (e.g. A,B,C)")
    sp.add_argument("--auto-chains", action="store_true", help="derive rec/lig from the TCR (rec=TCR, lig=pMHC)")
    sp.add_argument("--new-anarci", action="store_true", help="use ANARCII (pip) instead of legacy ANARCI when auto-deriving chains")
    sp.add_argument("--scorers", default="default", help="scorer set: fast | default | all | name,name")
    sp.add_argument("--timeout", type=float, default=600.0, help="per-scorer timeout (seconds)")
    sp.add_argument("--out", default=None, help="write scores to a .json or .csv file")
    sp.add_argument("--list-scorers", action="store_true", help="list scorers ifscore knows about and exit")
    sp.set_defaults(func=cmd_score)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
