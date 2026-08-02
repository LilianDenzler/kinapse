#!/usr/bin/env python3
"""Compare two scorer-benchmark runs — e.g. the native loader vs the STCRpy loader.

Point it at two result directories (each produced by ``run_benchmark.py --out <dir>``)
and it diffs them across every axis the benchmark reports:

  * coverage   — how many structures each loader could resolve chains for (and which
                 ones only one loader handled);
  * agreement  — Pearson/Spearman (GT vs model) per metric, side by side + Δ;
  * discrimination — direction-agnostic AUROC per metric, side by side + Δ;
  * tiers      — HQ/MQ/AQ/LQ counts;
  * structural — per-complex Cα-iRMSD from each loader: correlation, mean |Δ|, and a
                 tier-agreement crosstab (how often the two loaders assign the same tier).

Usage::

    python compare_loaders.py results_native results_stcrpy --labels native stcrpy
    # writes comparison_*.csv into ./loader_comparison (or --out)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read(dir_: Path, name: str):
    p = dir_ / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:      # e.g. no discriminatory metrics on a tiny run
        return None


def _resolved_ids(dir_: Path):
    """{stem -> True} for structures whose chains resolved (non-error) in a run."""
    cache = dir_ / "chains_cache.json"
    if not cache.exists():
        return set()
    d = json.loads(cache.read_text())
    return {Path(k).stem for k, v in d.items()
            if isinstance(v, list) and v[:1] != ["__error__"]}


def _merge(a, b, key, cols, la, lb):
    """Outer-merge two frames on `key`, keeping `cols` with _la/_lb suffixes."""
    if a is None or b is None:
        return None
    a = a[[key] + [c for c in cols if c in a.columns]]
    b = b[[key] + [c for c in cols if c in b.columns]]
    return a.merge(b, on=key, how="outer", suffixes=(f"_{la}", f"_{lb}"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir_a", help="first results dir (e.g. results_native)")
    ap.add_argument("dir_b", help="second results dir (e.g. results_stcrpy)")
    ap.add_argument("--labels", nargs=2, default=("a", "b"),
                    help="short labels for the two runs (default: a b)")
    ap.add_argument("--out", default="loader_comparison", help="dir for comparison CSVs")
    args = ap.parse_args(argv)

    A, B = Path(args.dir_a), Path(args.dir_b)
    la, lb = args.labels
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # -- coverage ------------------------------------------------------------
    ia, ib = _resolved_ids(A), _resolved_ids(B)
    print(f"=== COVERAGE (chains resolved) ===")
    print(f"  {la}: {len(ia)}   {lb}: {len(ib)}   both: {len(ia & ib)}")
    only_a, only_b = sorted(ia - ib), sorted(ib - ia)
    if only_a:
        print(f"  only {la} ({len(only_a)}): {', '.join(only_a[:12])}{' ...' if len(only_a) > 12 else ''}")
    if only_b:
        print(f"  only {lb} ({len(only_b)}): {', '.join(only_b[:12])}{' ...' if len(only_b) > 12 else ''}")

    # -- agreement -----------------------------------------------------------
    ag = _merge(_read(A, "agreement.csv"), _read(B, "agreement.csv"),
                "metric", ["pearson", "spearman", "n"], la, lb)
    if ag is not None and len(ag):
        if f"pearson_{la}" in ag and f"pearson_{lb}" in ag:
            ag["dpearson"] = (ag[f"pearson_{lb}"] - ag[f"pearson_{la}"]).round(3)
        ag = ag.sort_values(f"pearson_{la}", ascending=False, na_position="last")
        ag.to_csv(out / "comparison_agreement.csv", index=False)
        print(f"\n=== AGREEMENT (GT vs model) — Pearson: {la} vs {lb} ===")
        print(ag.to_string(index=False))

    # -- discrimination ------------------------------------------------------
    dc = _merge(_read(A, "discrimination.csv"), _read(B, "discrimination.csv"),
                "metric", ["auroc_abs", "auprc", "cohens_d"], la, lb)
    if dc is not None and len(dc):
        if f"auroc_abs_{la}" in dc and f"auroc_abs_{lb}" in dc:
            dc["dauroc"] = (dc[f"auroc_abs_{lb}"] - dc[f"auroc_abs_{la}"]).round(3)
            dc = dc.sort_values(f"auroc_abs_{la}", ascending=False, na_position="last")
        dc.to_csv(out / "comparison_discrimination.csv", index=False)
        print(f"\n=== DISCRIMINATION (pos vs neg) — AUROC_abs: {la} vs {lb} ===")
        print(dc.to_string(index=False))

    # -- tiers ---------------------------------------------------------------
    ta, tb = _read(A, "tiers.csv"), _read(B, "tiers.csv")
    if ta is not None and tb is not None:
        t = (ta[["tier", "n"]].rename(columns={"n": f"n_{la}"})
             .merge(tb[["tier", "n"]].rename(columns={"n": f"n_{lb}"}), on="tier", how="outer"))
        t.to_csv(out / "comparison_tiers.csv", index=False)
        print(f"\n=== TIERS (modelled positives) ===")
        print(t.to_string(index=False))

    # -- structural (per-complex Cα-iRMSD) -----------------------------------
    sa, sb = _read(A, "structural.csv"), _read(B, "structural.csv")
    if sa is not None and sb is not None and "struct__cdr_irmsd" in sa.columns:
        keep = ["id", "struct__cdr_irmsd", "tier"]
        m = (sa[keep].merge(sb[keep], on="id", how="inner", suffixes=(f"_{la}", f"_{lb}")))
        if len(m):
            x, y = m[f"struct__cdr_irmsd_{la}"], m[f"struct__cdr_irmsd_{lb}"]
            m["abs_diff"] = (y - x).abs().round(3)
            m.to_csv(out / "comparison_structural.csv", index=False)
            corr = x.corr(y)
            print(f"\n=== STRUCTURAL (Cα-iRMSD per complex, {len(m)} shared) ===")
            print(f"  Pearson r({la}, {lb}) = {corr:.3f} | mean |Δ iRMSD| = {m['abs_diff'].mean():.3f} Å "
                  f"| max |Δ| = {m['abs_diff'].max():.3f} Å")
            ct = pd.crosstab(m[f"tier_{la}"], m[f"tier_{lb}"])
            agree = int((m[f"tier_{la}"] == m[f"tier_{lb}"]).sum())
            print(f"  tier agreement: {agree}/{len(m)} identical")
            print("  tier crosstab (rows=%s, cols=%s):" % (la, lb))
            print(ct.to_string())

    print(f"\nwrote comparison CSVs to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
