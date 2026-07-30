"""Structure & MD analysis + dynamics metrics.

Everything that turns an ensemble (ground-truth MD or a predicted ensemble)
into numbers and figures:

* alignment (Kabsch / TMalign / ProFit)          — :mod:`kinapse.dynamics_analysis.aligning`
* per-frame RMSD & TM-score                       — :mod:`kinapse.dynamics_analysis.rmsd_tm`
* feature extraction (Ca-dist / coords / dihed)   — :mod:`kinapse.dynamics_analysis.embeddings.features`
* dimensionality reduction (PCA/wPCA/kPCA/TICA/diffmap)
                                                  — :mod:`kinapse.dynamics_analysis.embeddings.dim_reduction`
* embedding-quality metrics (trustworthiness, Mantel)
                                                  — :mod:`kinapse.dynamics_analysis.embeddings.metrics`
* free-energy surfaces (PMF) + Jensen-Shannon divergence
                                                  — :mod:`kinapse.dynamics_analysis.pmf_kde`
* end-to-end per-region runners                   — :mod:`kinapse.dynamics_analysis.embeddings.run_*`
* plotting                                        — :mod:`kinapse.dynamics_analysis.plotters`

Symbols are lazily imported (reducers pull in sklearn/deeptime only when used),
so ``import kinapse.dynamics_analysis`` stays cheap.
"""
from __future__ import annotations

_LAZY = {
    # rmsd / tm
    "rmsd_tm": (".rmsd_tm", "rmsd_tm"),
    "run_rmsd_tm": (".rmsd_tm", "run"),
    # pmf / free energy / divergence
    "oriol_analysis": (".pmf_kde", "oriol_analysis"),
    "js_from_fe": (".pmf_kde", "js_from_fe"),
    "compute_pmf_hist": (".pmf_kde", "compute_pmf_hist"),
    "compute_pmf_kde": (".pmf_kde", "compute_pmf_kde"),
    "identify_high_density_points": (".pmf_kde", "identify_high_density_points"),
    # alignment
    "align_traj_same_tcr_fast": (".aligning", "align_traj_same_tcr_fast"),
    "align_traj_to_refstructure": (".aligning", "align_traj_to_refstructure"),
    "align_traj_different_tcr_tmalign": (".aligning", "align_traj_different_tcr_tmalign"),
    "align_MD_same_TCR_profit": (".aligning", "align_MD_same_TCR_profit"),
    # projections / plotting
    "pca_project_two": ("._legacy.PCA_methods", "pca_project_two"),
    "plot_pca": (".plotters", "plot_pca"),
    # per-region end-to-end runners
    "run_coords": (".embeddings.run_coords", "run"),
    "run_ca_dist": (".embeddings.run_ca_dist", "run"),
    "run_dihedrals": (".embeddings.run_dihedrals", "run"),
}

__all__ = list(_LAZY)


def __getattr__(name):  # PEP 562 lazy loading
    import importlib
    if name in _LAZY:
        submod, attr = _LAZY[name]
        module = importlib.import_module(submod, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
