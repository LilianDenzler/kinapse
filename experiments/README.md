# experiments

Concrete studies that *use* the `kinapse` library (they are not part of the
installable package). Each subfolder is one experiment: a runnable script + a
README, with real dataset paths and configuration. Generated outputs
(`results/`) are git-ignored.

| experiment | what |
|---|---|
| [`scorer_benchmarking/`](scorer_benchmarking/) | benchmark all interface scorers: ground-truth vs modelled vs negative TCR-pMHC complexes |
| [`binder_finder_vdjdb/`](binder_finder_vdjdb/) | for each TCR structure, find every pMHC it is a confirmed VDJdb binder for (paired CDR3α+β match) |
