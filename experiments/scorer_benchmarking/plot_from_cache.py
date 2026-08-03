#!/usr/bin/env python3
"""Analyse/plot the scores CACHED so far — read-only, safe while a run is still going.

Reads ``<out>/scores_cache/part_*.csv`` (written atomically by a running or finished
benchmark), rebuilds each row's role/id/label from the dataset directories, runs the
agreement + discrimination (+ structural tiers, if `structural_cache.csv` exists) analyses
on whatever is scored so far, and writes partial CSVs + plots to ``<out>/partial/``.

It never writes into the run's own files, so you can call it any time to peek at progress::

    python plot_from_cache.py --out results_native
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kinapse.benchmarks.scorer_benchmark import (
    agreement_analysis, discrimination_analysis, discover, split_by_group, tiers_summary,
)
from run_benchmark import DEFAULTS, make_plots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="results dir that contains scores_cache/")
    ap.add_argument("--gt", default=DEFAULTS["gt"])
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--neg", default=DEFAULTS["neg"])
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots-out", default=None, help="default <out>/partial")
    args = ap.parse_args(argv)

    out = Path(args.out)
    parts = sorted((out / "scores_cache").glob("part_*.csv"))
    if not parts:
        raise SystemExit(f"no scores_cache/part_*.csv under {out} — nothing cached yet")
    scored_raw = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True, sort=False)
    scored_raw = scored_raw.drop_duplicates(subset="model", keep="first")

    # rebuild role/id/label from the dataset dirs (no scoring, just globbing)
    positives, negatives = discover(args.gt, args.model, args.neg)
    meta = []
    for cid, g, m in positives:
        meta.append(dict(model=g, role="gt", id=cid, group=cid, label=float("nan")))
        meta.append(dict(model=m, role="model_pos", id=cid, group=cid, label=1))
    for cid, n in negatives:
        meta.append(dict(model=n, role="model_neg", id=cid, group=cid, label=0))
    meta = pd.DataFrame(meta)

    metric_cols = [c for c in scored_raw.columns if "__" in c]
    scored = meta.merge(scored_raw[["model"] + metric_cols], on="model", how="inner")  # scored-so-far
    n = dict(scored.role.value_counts())
    print(f"cached so far: {len(scored)} rows  (gt={n.get('gt',0)}, "
          f"model_pos={n.get('model_pos',0)}, model_neg={n.get('model_neg',0)})")
    if not len(scored):
        raise SystemExit("no cached structures matched the dataset dirs (wrong --gt/--model/--neg?)")

    scored = split_by_group(scored, test_frac=args.test_frac, seed=args.seed)

    # structural tiers only if that cache exists (it is written as structural runs)
    tiers = pd.DataFrame()
    sc = out / "structural_cache.csv"
    if sc.exists():
        sdf = pd.read_csv(sc)
        if "tier" not in sdf.columns and "struct__cdr_irmsd" in sdf.columns:
            from kinapse.benchmarks import assign_tier
            sdf["tier"] = sdf["struct__cdr_irmsd"].map(assign_tier)   # iRMSD-only (no DockQ)
        sdf["id"] = sdf["id"].astype(str)
        scored["id"] = scored["id"].astype(str)
        scored = scored.merge(sdf[["id", "tier"]], on="id", how="left")
        tiers = tiers_summary(scored)

    agree = agreement_analysis(scored)
    disc = discrimination_analysis(scored)

    pout = Path(args.plots_out) if args.plots_out else out / "partial"
    pout.mkdir(parents=True, exist_ok=True)
    agree.to_csv(pout / "agreement.csv", index=False)
    disc.to_csv(pout / "discrimination.csv", index=False)
    if len(tiers):
        tiers.to_csv(pout / "tiers.csv", index=False)

    print("\n=== AGREEMENT (GT vs modelled positive) ===")
    print(agree.to_string(index=False) if len(agree) else "  (need ≥3 complete GT/model pairs)")
    print("\n=== DISCRIMINATION (positive vs negative, test split) ===")
    print(disc.to_string(index=False) if len(disc)
          else "  (none yet — negatives are scored after positives, so wait for neg rows)")
    if len(tiers):
        print("\n=== STRUCTURAL TIERS ===")
        print(tiers.to_string(index=False))

    make_plots({"agreement": agree, "discrimination": disc, "tiers": tiers}, pout / "plots")
    print(f"\nwrote partial CSVs + plots to {pout}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
