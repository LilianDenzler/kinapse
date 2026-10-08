#!/usr/bin/env python
"""Build the requested distance-graph figures for one TCR (loads the MD once).

  fig1 : B_CDR3  vs  BETA framework            (5 panels)
  fig2 : B_CDR3  vs  ALPHA+BETA framework      (5 panels)
  fig3 : B_CDR2  vs  ALPHA+BETA framework      (5 panels; B_CDR2 has a flank residue -> panels 3&4 populated)
  framework_chainA / framework_chainB          (rigid-scaffold graphs, 3 panels each)
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
from graph_build import load_md, build_cdr_graph, build_framework_graph, save_graph, HERE
from graph_viz import draw_cdr5, draw_framework

SYS = sys.argv[1] if len(sys.argv) > 1 else "3QH3"
FIG = f"{HERE}/figures"
cache = load_md(SYS)
print(f"[{SYS}] MD loaded")

g1 = build_cdr_graph(SYS, "B", "CDR3", fw_chains=["B"], md_cache=cache)
save_graph(g1, f"{HERE}/graphs/{SYS}_BCDR3_betaFW.npz")
print("fig1:", draw_cdr5(g1, f"{FIG}/fig1_{SYS}_BCDR3_vs_betaFW.png"),
      f"({g1['n_loop']}L+{g1['n_flank']}flk+{g1['n_frame']}F)")

g2 = build_cdr_graph(SYS, "B", "CDR3", fw_chains=["A", "B"], md_cache=cache)
save_graph(g2, f"{HERE}/graphs/{SYS}_BCDR3_alphabetaFW.npz")
print("fig2:", draw_cdr5(g2, f"{FIG}/fig2_{SYS}_BCDR3_vs_alphabetaFW.png"),
      f"({g2['n_loop']}L+{g2['n_flank']}flk+{g2['n_frame']}F)")

g3 = build_cdr_graph(SYS, "B", "CDR2", fw_chains=["A", "B"], md_cache=cache)
print("fig3:", draw_cdr5(g3, f"{FIG}/fig3_{SYS}_BCDR2_vs_alphabetaFW_flankdemo.png"),
      f"({g3['n_loop']}L+{g3['n_flank']}flk+{g3['n_frame']}F)")

for ch in "AB":
    gf = build_framework_graph(SYS, ch, md_cache=cache)
    print(f"framework {ch}:", draw_framework(gf, f"{FIG}/framework_{SYS}_chain{ch}.png"),
          f"({gf['n_frame']} nodes)")
print("ALL_DONE")
