# Adding your own model (any runner tier)

Every runner tier — `sequence_embedding`, `structure_prediction`,
`conformer_generation`, `binding_prediction`, `docking`, `scoring`,
`structure_analysis` — shares one mechanism (`kinapse.runners`). Adding a model is the same everywhere and needs no
kinapse fork.

A model is a **`RunnerSpec`** plus a **runner** (a function or an isolated script).

## 1. The quickest path — a native runner (in-process)
Good for light, dependency-free models and for testing.

```python
from kinapse.runners import RunnerSpec, register

def _run(inputs: dict) -> dict:
    # inputs is a plain dict; return a plain dict (a 'status' key is added for you)
    return {"score": my_predict(inputs["cdr3b"], inputs["peptide"])}

register(RunnerSpec(
    name="my_predictor", tier="binding_prediction", backend="native",
    status="stable", outputs=("score",), func=_run,
))
```

```python
import kinapse.binding_prediction as bp
bp.available()                      # your model shows up
bp.run("my_predictor", {"cdr3b": "CASS...", "peptide": "GILGFVFTL"})
```

## 2. The isolated path — a `uvenv` runner (own environment)
Use this when the model has heavy or conflicting dependencies (torch pins, numpy
versions, PyRosetta, …). The runner is a **script that reads JSON from stdin and
writes JSON to stdout**; kinapse runs it in a throwaway env via `uv`, so its deps
never touch kinapse. A crash yields `{"status": "error", ...}` and never breaks a batch.

`my_runner.py`:
```python
import sys, json
inp = json.load(sys.stdin)
# ... do the work in this isolated interpreter ...
json.dump({"score": 0.87}, sys.stdout)
```

```python
register(RunnerSpec(
    name="my_gpu_model", tier="binding_prediction", backend="uvenv",
    status="stable", requirements=("torch==2.3", "my-model-pkg"),
    python="3.11", runner="/abs/path/to/my_runner.py", outputs=("score",),
))
```
(Needs [`uv`](https://github.com/astral-sh/uv) on PATH.)

## 3. Ship it in your own package (no fork)
Register from your package via the `kinapse.runners` **entry point** — kinapse
auto-discovers it once your package is `pip install`ed.

`pyproject.toml` of *your* package:
```toml
[project.entry-points."kinapse.runners"]
my_models = "my_pkg.kinapse_plugin:register_all"
```
`my_pkg/kinapse_plugin.py`:
```python
from kinapse.runners import RunnerSpec, register
def register_all():
    register(RunnerSpec(name="my_predictor", tier="binding_prediction",
                        backend="native", func=..., status="stable"))
```
Now `pip install my-pkg` makes `my_predictor` appear in
`kinapse.binding_prediction.available()` automatically.

## RunnerSpec fields
| field | meaning |
|---|---|
| `name`, `tier` | identity (tier ∈ the six runner tiers) |
| `backend` | `native` \| `uvenv` \| `container` (reserved) |
| `status` | `stable` \| `scaffold` \| `planned` |
| `func` | native backend: `(inputs: dict) -> dict` |
| `runner` | uvenv backend: path to a JSON-in/JSON-out script |
| `requirements`, `python` | uvenv env spec |
| `outputs` | field names the runner returns |
| `needs_native_ref` | requires a reference structure (e.g. DockQ) |
| `description`, `homepage`, `license`, `tags` | metadata for the map |

## Conventions
- One model = one `RunnerSpec`. Register bundled ones in the tier's `registry.py`.
- Keep heavy deps out of the tier's pip extra — provision them in the runner's own env.
- Set an honest `status`. Add it to [`MODULES.md`](MODULES.md) if it's a bundled model.
