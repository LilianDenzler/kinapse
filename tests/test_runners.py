"""Tests for the pluggable-model engine (kinapse.runners) — the extension path.

Dependency-light: exercises the registry + native backend + builtin registries +
back-compat aliases without any heavy scientific dependency.
"""
import importlib

import pytest


def test_register_run_native_extension():
    from kinapse import runners
    runners.register(
        runners.RunnerSpec(name="_t_dummy", tier="binding_prediction", backend="native",
                           status="stable", outputs=("score",),
                           func=lambda i: {"score": i.get("x", 0) * 2}),
        replace=True,
    )
    import kinapse.binding_prediction as bp
    assert "_t_dummy" in [s.name for s in bp.available()]
    out = bp.run("_t_dummy", {"x": 21})
    assert out["status"] == "ok" and out["score"] == 42


def test_runner_error_is_captured_not_raised():
    from kinapse import runners

    def _boom(_):
        raise RuntimeError("boom")

    runners.register(runners.RunnerSpec(name="_t_boom", tier="docking",
                                        backend="native", func=_boom), replace=True)
    out = runners.run("docking", "_t_boom")
    assert out["status"] == "error" and "boom" in out["error"]


@pytest.mark.parametrize("tier,expected", [
    ("structure_prediction", "tcrdock"),
    ("binding_prediction", "tulip"),
    ("docking", "haddock"),
    ("conformer_generation", "dig"),
    ("sequence_embedding", "esm2"),
    ("scoring", "geometry_scoring"),
    ("structure_analysis", "stcrpy"),
])
def test_builtin_registries_load(tier, expected):
    from kinapse import runners
    assert expected in [s.name for s in runners.specs(tier)]


def test_stcrpy_runner_without_env_returns_error():
    import importlib.util
    import os
    if importlib.util.find_spec("stcrpy") is not None or os.environ.get("KINAPSE_STCRPY_PYTHON"):
        pytest.skip("a STCRpy env is available; the missing-env path is not exercised")
    from kinapse import runners
    out = runners.run("structure_analysis", "stcrpy", {"pdb": "x.pdb"})
    # native-backend errors are captured (never raised) with the install hint
    assert out["status"] == "error"
    assert "stcrpy" in out["error"].lower() and "kinapse_stcrpy_python" in out["error"].lower()


def test_unknown_runner_raises_keyerror():
    from kinapse import runners
    with pytest.raises(KeyError):
        runners.get("binding_prediction", "does_not_exist")


@pytest.mark.parametrize("old", ["kinapse.analysis", "kinapse.embedding", "kinapse.generation"])
def test_backcompat_aliases_import(old):
    importlib.import_module(old)


def test_scaffold_modules_import_light():
    for m in ["kinapse.datasets", "kinapse.benchmarks", "kinapse.dynabind"]:
        importlib.import_module(m)
