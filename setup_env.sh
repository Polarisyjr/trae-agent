#!/bin/bash
# Set up the trae-agent environment this box's launchers actually expect.
#
# scripts/trae/start.sh (and run_pipeline.py) hard-code
# `frameworks/trae-agent/.venv/bin/python` — i.e. a uv-managed venv IN this
# directory, not a conda env. (An earlier attempt created a conda env `trae`
# and `uv pip install -e .` into it; that env is NOT what the launcher uses
# and can be ignored/removed.)
#
# What it does (all idempotent — safe to re-run):
#   1. ensure `uv` is on PATH (official installer if missing)
#   2. `uv sync --all-extras` — creates ./.venv with base deps + the `test`
#      and `evaluation` extras (datasets, docker, pexpect, unidiff, swebench —
#      required for scripts/trae/run_pipeline.py's generate->prune->select
#      pipeline; without `evaluation`, import fails with
#      ModuleNotFoundError: pexpect / docker / swebench)
#   3. verify: .venv/bin/python can `import trae_agent` and the evaluation
#      extras, and `trae-cli --help` runs
#
# Usage (from anywhere):
#   bash frameworks/trae-agent/setup_env.sh             # full setup
#   bash frameworks/trae-agent/setup_env.sh --verify    # skip install, just check
#
# Known gotcha when actually RUNNING the pipeline (not a setup_env.sh concern,
# noted here so it isn't rediscovered): `~/.bash_profile` sets
# HF_HOME=/mnt/azureuser/huggingface, which is not writable by this user and
# shadows ~/.bashrc's HF_HOME=/mnt/raid0/jirong/hf (where the models actually
# are). Export HF_HOME=/mnt/raid0/jirong/hf explicitly when invoking
# scripts/trae/start.sh / run_pipeline.py.

set -euo pipefail

TRAE_DIR="$(cd "$(dirname "$0")" && pwd)"     # this script lives in frameworks/trae-agent
VENV_PY="$TRAE_DIR/.venv/bin/python"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m %s\n' "$*"; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

if [ "$VERIFY_ONLY" = "0" ]; then
    say "1. uv"
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || die "uv install did not put uv on PATH"
    ok "uv: $(uv --version)"

    say "2. uv sync --all-extras (creates .venv)"
    ( cd "$TRAE_DIR" && uv sync --all-extras )
    ok ".venv ready"
fi

say "verify"
[ -x "$VENV_PY" ] || die ".venv python not found at $VENV_PY (run without --verify first)"
"$VENV_PY" -c "
import trae_agent, docker, swebench, datasets, pexpect, unidiff
print('  ok imports: trae_agent, docker, swebench, datasets, pexpect, unidiff')
"
"$TRAE_DIR/.venv/bin/trae-cli" --help >/dev/null && ok "trae-cli --help runs"

printf '\n\033[1;32mtrae-agent environment ready.\033[0m  Run a task with: bash %s/../../scripts/trae/start.sh\n' "$TRAE_DIR"
