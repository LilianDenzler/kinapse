#!/usr/bin/env python
"""Does a Glycine in a loop's FLANK correlate with the measured rotation angles (hinge/twist/sway)?
Pools all 132 (TCR,CDR) loops. gly_flank = Gly present in the +-4 IMGT flank (either side); jmotif_gly = Gly in the
C-flank (the 868 J-motif side). Compares angle amplitude (p95 |omega|, deg) WITH vs WITHOUT a flank Gly:
Mann-Whitney U, group medians, Cliff's delta effect size; pooled AND within each CDR (to strip the CDR/length confound)."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy.stats import mannwhitneyu
HERE = os.path.dirname(os.path.abspath(__file__))
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
MODES = ["hinge", "twist", "sway"]


def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b); gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def test(have, lack):
    if len(have) < 3 or len(lack) < 3:
        return None
    U, p = mannwhitneyu(have, lack, alternative="two-sided")
    return dict(n1=len(have), n0=len(lack), med1=float(np.median(have)), med0=float(np.median(lack)),
                delta=float(cliffs(have, lack)), p=float(p))


def main():
    F = json.load(open(f"{HERE}/results_features.json")); AA = json.load(open(f"{HERE}/results_angleamp.json"))
    rows = []
    for t in F:
        for c in CDRS:
            if c not in F[t] or c not in AA[t]:
                continue
            rows.append((c, F[t][c].get("gly_flank", 0), F[t][c].get("jmotif_gly", 0),
                         {m: AA[t][c][m]["p95"] for m in MODES}))
    # base rates
    gf = np.array([r[1] for r in rows]); jg = np.array([r[2] for r in rows])
    print(f"n loops={len(rows)}; gly in any flank: {gf.sum()}/{len(rows)}; gly in C-flank(J-motif): {jg.sum()}/{len(rows)}\n")

    def report(flagidx, label):
        print(f"==== {label} ====")
        print(f"{'scope':14}{'mode':7}{'n(Gly)':>7}{'n(no)':>7}{'med Gly':>9}{'med no':>8}{'Cliff d':>9}{'p':>9}")
        for scope, cdrs in [("ALL loops", CDRS)] + [(c, [c]) for c in CDRS]:
            ss = [r for r in rows if r[0] in cdrs]
            for m in MODES:
                have = [r[3][m] for r in ss if r[1 + flagidx] == 1]; lack = [r[3][m] for r in ss if r[1 + flagidx] == 0]
                res = test(have, lack)
                if res:
                    sig = " *" if res["p"] < 0.05 else ""
                    print(f"{scope:14}{m:7}{res['n1']:7d}{res['n0']:7d}{res['med1']:9.0f}{res['med0']:8.0f}{res['delta']:+9.2f}{res['p']:9.3f}{sig}")
                elif scope == "ALL loops":
                    print(f"{scope:14}{m:7}  (too few in one group: {len(have)} vs {len(lack)})")
            print()

    report(0, "Gly in ANY flank  vs  angle amplitude (deg)")
    report(1, "Gly in C-flank (J-motif)  vs  angle amplitude (deg)")


if __name__ == "__main__":
    main()
