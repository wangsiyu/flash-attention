#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash harness/harness/hang_detect_fix/cuda_gdb_snapshot.sh <pid> [log]" >&2
    exit 2
fi

PID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG="${2:-$HARNESS_ROOT/logs/hang_detect_fix/gdb_hang.log}"

mkdir -p "$(dirname "$LOG")"

cuda-gdb --pid "$PID" \
    -batch \
    -ex "set confirm off" \
    -ex "set pagination off" \
    -ex "set debuginfod enabled off" \
    -ex "set logging file $LOG" \
    -ex "set logging on" \
    -ex "interrupt" \
    -ex "echo ===HOST_THREADS===\\n" \
    -ex "info threads" \
    -ex "echo ===HOST_BACKTRACES===\\n" \
    -ex "thread apply all bt" \
    -ex "echo ===CUDA_KERNELS===\\n" \
    -ex "info cuda kernels" \
    -ex "echo ===CUDA_WARPS===\\n" \
    -ex "info cuda warps" \
    -ex "echo ===CUDA_THREADS===\\n" \
    -ex "info cuda threads sm 0" \
    -ex "echo ===CUDA_PC===\\n" \
    -ex "x/8i \$pc" \
    -ex "bt" \
    -ex "echo ===SNAPSHOT_END===\\n" \
    -ex "detach" \
    -ex "set logging off"

echo "[hang_detect_fix] wrote $LOG"
