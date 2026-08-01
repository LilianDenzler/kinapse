#!/usr/bin/env bash
#
# One-command bootstrap for kinapse.
#
#   git clone <repo> && cd kinapse && ./setup.sh
#
# Creates (or updates) the `kinapse` conda environment from environment.yml —
# which also installs the package in editable mode with its numbering + reduce
# extras — then runs a quick import + the smoke tests.
#
# Optional feature stacks are NOT installed by default (they are heavy or
# licensed); see the notes at the bottom of environment.yml and the README.
set -euo pipefail

ENV_NAME="kinapse"

# Prefer mamba/micromamba if present (much faster solver), else conda.
if command -v mamba >/dev/null 2>&1; then
    SOLVER="mamba"
elif command -v micromamba >/dev/null 2>&1; then
    SOLVER="micromamba"
elif command -v conda >/dev/null 2>&1; then
    SOLVER="conda"
else
    echo "ERROR: need conda, mamba or micromamba on PATH." >&2
    echo "Install Miniforge: https://github.com/conda-forge/miniforge" >&2
    exit 1
fi
echo ">> Using '$SOLVER' to build env '$ENV_NAME'"

if "$SOLVER" env list 2>/dev/null | grep -qE "^\s*${ENV_NAME}\s"; then
    echo ">> Env exists — updating"
    "$SOLVER" env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo ">> Creating env"
    "$SOLVER" env create -f environment.yml
fi

# Run verification inside the env.
RUN=( "$SOLVER" run -n "$ENV_NAME" )
echo ">> Verifying import"
"${RUN[@]}" python -c "import kinapse; print('kinapse', kinapse.__version__, 'ready')"
echo ">> Running smoke tests (PYTHONNOUSERSITE=1 to ignore stray ~/.local plugins)"
"${RUN[@]}" env PYTHONNOUSERSITE=1 python -m pytest -q || echo "(smoke tests reported issues — see output above)"

cat <<EOF

Done. Activate the environment with:

    ${SOLVER} activate ${ENV_NAME}
    kinapse info

Optional stacks (install only if you need them):
  * embedding (Evoformer):  pip install -e ".[embed]"   # pulls torch; also needs OpenFold + MMseqs2
  * legacy ANARCI numbering: ${SOLVER} install -n ${ENV_NAME} -c bioconda anarci
  * PyMOL visualisation:     already in environment.yml (pymol-open-source)
  * MODELLER (linkers):      ${SOLVER} install -n ${ENV_NAME} -c salilab modeller   # licensed
EOF
