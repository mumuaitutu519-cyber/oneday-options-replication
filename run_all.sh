#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PACKAGE_ROOT"
MODE=quick
OFFLINE=0
for arg in "$@"; do
 case "$arg" in
  --quick) MODE=quick ;;
  --full) MODE=full ;;
  --offline) OFFLINE=1 ;;
  *) echo "Usage: bash run_all.sh [--quick|--full] [--offline]" >&2; exit 2 ;;
 esac
done
export REPLICATION_MODE="$MODE" REPLICATION_WORKERS="${REPLICATION_WORKERS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20261107
export MPLCONFIGDIR="${TMPDIR:-/tmp}/OneDay270-matplotlib"
mkdir -p logs output verify "$MPLCONFIGDIR"
exec > >(tee "logs/${MODE}_run.log") 2>&1
START=$(date +%s)
echo "START mode=$MODE offline=$OFFLINE workers=$REPLICATION_WORKERS"
if [[ ! -x .venv/bin/python ]]; then
 if command -v python3.12 >/dev/null 2>&1; then PYTHON_COMMAND=python3.12; else PYTHON_COMMAND=python3; fi
 "$PYTHON_COMMAND" -c 'import sys; assert sys.version_info[:2]==(3,12), "Python 3.12 is required"'
 "$PYTHON_COMMAND" -m venv .venv
 if ! .venv/bin/python -m pip install --no-index --find-links wheels -r requirements.txt; then
  if [[ "$OFFLINE" == 1 ]]; then exit 1; fi
  .venv/bin/python -m pip install -r requirements.txt
 fi
fi
.venv/bin/python code/run.py "$MODE" "$OFFLINE"
END=$(date +%s)
echo "FINISH mode=$MODE wall_seconds=$((END-START))"
