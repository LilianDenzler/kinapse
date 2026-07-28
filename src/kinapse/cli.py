"""``kinapse`` command-line interface.

A thin, dependency-light entry point over the library. Subcommands:

    kinapse info                       show version, config & packaged-data paths
    kinapse prep     PDB [opts]        load/renumber/pair a TCR, write prepped structures
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

    sp = sub.add_parser("geometry", help="compute TCR alpha/beta docking geometry")
    sp.add_argument("pdb", help="input TCR PDB (or topology when --traj is given)")
    sp.add_argument("--traj", default=None, help="trajectory (.xtc); compute geometry per frame")
    sp.add_argument("--out", default="kinapse_geometry", help="output directory")
    sp.add_argument("--vis", action="store_true", help="also write PyMOL visualisation (needs pymol)")
    sp.set_defaults(func=cmd_geometry)

    sp = sub.add_parser("pipelines", help="list migrated end-to-end workflows")
    sp.set_defaults(func=cmd_pipelines)
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
