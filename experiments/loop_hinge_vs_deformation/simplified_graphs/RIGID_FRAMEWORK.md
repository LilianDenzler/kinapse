# Rigid framework definition (chosen)

Derived from the **22-TCR averaged** CA-CA distance fluctuation (`D_std`), alignment-free.

- **kept** = largest set whose *average* pairwise `D_std ≤ 0.5 Å` (mutually rigid on average)
- **RIGID (use this)** = ultra-rigid subset of *kept* whose *worst-case* (max over TCRs) pairwise `D_std ≤ 0.5 Å`

Compare: kinapse consensus uses cross-TCR alignment RMSD < ~2.0 Å (a different metric).

## Chain A

- **RIGID (13)**: [40, 41, 89, 90, 91, 101, 102, 103, 104, 105, 106, 124, 125]
- kept@0.5 (46): [15, 18, 20, 21, 22, 23, 24, 38, 39, 40, 41, 42, 43, 44, 56, 77, 78, 80, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 117, 118, 121, 122, 123, 124, 125, 126]

## Chain B

- **RIGID (20)**: [6, 10, 21, 38, 39, 40, 41, 42, 43, 52, 53, 101, 102, 103, 104, 105, 106, 121, 122, 123]
- kept@0.5 (46): [5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 21, 22, 23, 24, 25, 38, 39, 40, 41, 42, 43, 44, 52, 53, 87, 89, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127]

