# graph_method

Decompose each CDR loop's framework-relative motion into **hinge + internal deformation + residual**.

Rotation extraction is **ensemble-referenced** (generalized-Procrustes template + Karcher-mean rotation → per-frame
`ω_t = θ_t û_t`, no arbitrary frame). The flexibility decomposition is **alignment-free** in Cα-distance space: a
linear hinge mode learned by regressing loop→framework distances on `θ`, projected orthogonally →
`F_hinge`, `F_deform` (from loop→loop distances), `F_residual`, `f_hinge`. α and β are done **separately**.

Full method + the two measured caveats (linear-mode curvature; deformation that mimics rotation): **[`Plan.md`](Plan.md)**.

```bash
mamba activate kinapse
python graph_tests.py          # 5-check validation gate (all pass)
```

- [`graph_fit.py`](graph_fit.py) — distance primitives + pure-distance rigid fit (drop-in alternative to GPA).
- [`rotations.py`](rotations.py) — framework frame → loop GPA → Karcher-referenced rotation vectors.
- [`graph_decomp.py`](graph_decomp.py) — hinge mode, projection, `F_*`, `f_hinge`.
- [`graph_tests.py`](graph_tests.py) — validation gate.

Core is built & validated. MD pipeline (`compute_graph_decomp.py`), cross-TCR axis conservation, residual PCA, and
figures are next. Outputs will land in `results/` (git-ignored).
