"""Composable, config-driven end-to-end workflows.

These stitch the five core sub-packages into full runs. They were migrated from
the original ``TCR_Metrics/pipelines`` and preserve every capability:

* :mod:`kinapse.pipelines.benchmark`   — the two-phase model-vs-MD benchmark
  (collect embeddings -> shared-bin PMF/JSD -> summary tables), plus PyMOL
  visualisation and MD baseline samplers.
* :mod:`kinapse.pipelines.best_method` — screen reducers per region and select
  the best (feature x reducer) mode.
* :mod:`kinapse.pipelines.analyse_md`  — ground-truth-only flexibility analysis.

Note: the migrated ``__main__`` blocks still contain the original absolute data
paths as *examples*. Prefer driving runs through :mod:`kinapse.config` (env vars
/ ``config.yaml``) or the ``kinapse`` CLI. See ``docs/pipelines.md``.
"""
