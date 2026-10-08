#!/usr/bin/env python
"""Render an overlay PNG (all frames superimposed = swing envelope) from an existing ensemble .pse.
Cheap: no MD reload. Shows the clamp stem (orange) as its own bundle -> visual check it aligns tightly."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob
from graph_build import HERE

pses = sorted(glob.glob(f"{HERE}/figures/hinge_ens_*_*.pse"))
if len(sys.argv) > 1:
    pses = [p for p in pses if sys.argv[1] in p]
import pymol2
for pse in pses:
    png = pse[:-4] + ".png"
    with pymol2.PyMOL() as P:
        cmd = P.cmd
        cmd.load(pse)
        cmd.set("all_states", 1)                 # overlay every frame = swing envelope
        cmd.set("ray_opaque_background", 0); cmd.bg_color("white")
        cmd.ray(1500, 1150)
        cmd.png(png, dpi=150)
    print(f"{os.path.basename(png)}  exists={os.path.exists(png)}", flush=True)
