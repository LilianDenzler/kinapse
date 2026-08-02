#!/usr/bin/env python3
"""Scorer-benchmark experiment on the TCR3d tfold / OpenMM-minimised dataset.

Thin driver over ``kinapse.benchmarks.run_scorer_benchmark`` with this project's
real dataset paths. Edit the DEFAULTS below or override on the command line.

    python run_benchmark.py --limit 12 --scorers geometry_scoring --plots   # quick
    ANARCI_CPU=1 python run_benchmark.py --scorers all -j 16 --plots         # full

See ./README.md and ../../docs/benchmarks.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from kinapse.benchmarks import run_scorer_benchmark

_BASE = "/mnt/larry/lilian/DATA/TCR3d_datasets"
DEFAULTS = {
    "gt": f"{_BASE}/TCR_complexes_openmm_minimised",                       # ground truth (real, minimised)
    "model": f"{_BASE}/TCR_complexes_tfold_openmm_minimised",             # modelled positives (tfold)
    "neg": f"{_BASE}/negative_TCR_complexes_tfold/openmm_minimised",      # negatives (tfold)
}
HERE = Path(__file__).resolve().parent


def make_plots(res, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disc = res.get("discrimination")
    if disc is not None and len(disc):
        d = disc.sort_values("auroc_abs")
        plt.figure(figsize=(7, max(2, 0.35 * len(d))))
        plt.barh(d["metric"], d["auroc_abs"], color="#4f46e5")
        plt.axvline(0.5, ls="--", c="grey")
        plt.xlabel("AUROC (direction-agnostic)")
        plt.title("Discrimination: modelled positive vs negative (test split)")
        plt.tight_layout()
        plt.savefig(out_dir / "discrimination_auroc.png", dpi=130)
        plt.close()

    agree = res.get("agreement")
    if agree is not None and len(agree):
        a = agree.sort_values("pearson")
        plt.figure(figsize=(7, max(2, 0.35 * len(a))))
        plt.barh(a["metric"], a["pearson"], color="#0d9488")
        plt.xlabel("Pearson r (GT vs modelled positive)")
        plt.title("Agreement: modelled positive tracks ground truth")
        plt.tight_layout()
        plt.savefig(out_dir / "agreement_pearson.png", dpi=130)
        plt.close()

    tiers = res.get("tiers")
    if tiers is not None and len(tiers):
        order = {"HQ": 0, "MQ": 1, "AQ": 2, "LQ": 3, "untiered": 4}
        t = tiers.sort_values("tier", key=lambda s: s.map(lambda x: order.get(x, 9)))
        colors = {"HQ": "#16a34a", "MQ": "#65a30d", "AQ": "#d97706", "LQ": "#dc2626",
                  "untiered": "#9ca3af"}
        plt.figure(figsize=(5, 3.2))
        plt.bar(t["tier"], t["n"], color=[colors.get(x, "#6b7280") for x in t["tier"]])
        plt.ylabel("# modelled positives")
        plt.title("Structural agreement: model-vs-GT quality tiers\n(Cα iRMSD over 6 CDRs + DockQ)")
        plt.tight_layout()
        plt.savefig(out_dir / "structural_tiers.png", dpi=130)
        plt.close()
    print(f"wrote plots to {out_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", default=DEFAULTS["gt"], help="ground-truth complex dir")
    ap.add_argument("--model", default=DEFAULTS["model"], help="modelled-positive complex dir")
    ap.add_argument("--neg", default=DEFAULTS["neg"], help="negative complex dir")
    ap.add_argument("--out", default=str(HERE / "results"), help="output dir")
    ap.add_argument("--scorers", default="all", help="ifscore scorer set (all | default | geometry_scoring | a,b)")
    ap.add_argument("--limit", type=int, default=None, help="cap positives/negatives (quick run)")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-j", "--jobs", type=int, default=16)
    ap.add_argument("--legacy-anarci", action="store_true", help="use bioconda ANARCI (default: ANARCII/pip)")
    ap.add_argument("--loader", choices=("native", "stcrpy"), default="native",
                    help="backend for chain ID + CDR annotation: native (kinapse loader) "
                         "or stcrpy (external OPIG env; set KINAPSE_STCRPY_PYTHON)")
    ap.add_argument("--no-structural", action="store_true",
                    help="skip model-vs-GT Cα-iRMSD / HQ-MQ-AQ-LQ tiers")
    ap.add_argument("--plots", action="store_true", help="also write summary figures")
    args = ap.parse_args(argv)

    res = run_scorer_benchmark(
        args.gt, args.model, args.neg, out_dir=args.out, scorers=args.scorers,
        test_frac=args.test_frac, seed=args.seed, legacy_anarci=args.legacy_anarci,
        limit=args.limit, n_jobs=args.jobs, structural=not args.no_structural,
        loader=args.loader,
    )
    if args.plots:
        make_plots(res, Path(args.out) / "plots")
    print(f"\ndone — results in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
