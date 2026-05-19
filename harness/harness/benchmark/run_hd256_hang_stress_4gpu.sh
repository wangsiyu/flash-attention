#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HARNESS_ROOT="$(cd "$HARNESS_CODE_ROOT/.." && pwd)"
REPO="$(cd "$HARNESS_ROOT/.." && pwd)"
LOG_ROOT="$HARNESS_CODE_ROOT/logs/hang_stress"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$RUN_ID"
PYTHON="${PYTHON:-python3}"
NPROC="${NPROC:-4}"
VERIFY_ENV="${VERIFY_ENV:-1}"
STRESS_TIMEOUT_SECONDS="${STRESS_TIMEOUT_SECONDS:-0}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/state"

GIT_HEAD="$(git -C "$REPO" rev-parse --short HEAD)"
echo "[hang-stress] repo=$REPO"
echo "[hang-stress] git_head=$GIT_HEAD"
echo "[hang-stress] logs=$LOG_DIR"

if [[ "$VERIFY_ENV" == "1" ]]; then
    bash "$REPO/harness/harness/environment/setup_fa4_editable.sh" --verify-only \
        2>&1 | tee "$LOG_DIR/env_verify.log"
fi

CMD=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC"
    "$REPO/harness/harness/benchmark/hd256_hang_stress_4gpu.py"
    "$@"
)

echo "[hang-stress] command=${CMD[*]}" | tee "$LOG_DIR/command.log"

(
    cd /tmp
    if [[ "$STRESS_TIMEOUT_SECONDS" != "0" ]]; then
        REPO="$REPO" GIT_HEAD="$GIT_HEAD" HANG_STRESS_STATE_DIR="$LOG_DIR/state" timeout --signal=INT --kill-after=30s \
            "$STRESS_TIMEOUT_SECONDS" "${CMD[@]}"
    else
        REPO="$REPO" GIT_HEAD="$GIT_HEAD" HANG_STRESS_STATE_DIR="$LOG_DIR/state" "${CMD[@]}"
    fi
) 2>&1 | tee "$LOG_DIR/stress.log"
