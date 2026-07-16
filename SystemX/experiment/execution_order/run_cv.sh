#!/usr/bin/env bash
_REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-${_REPO_ROOT}/.venv/bin/python3}"
if [ ! -f "$PYTHON" ]; then PYTHON="python3"; fi

set -euo pipefail
cd "$_REPO_ROOT"

PYTHONPATH="$_REPO_ROOT" exec "$PYTHON" -m SystemX.experiment.execution_order.evaluate "$@"
