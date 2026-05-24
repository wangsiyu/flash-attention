#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG="${KEEPALIVE_LOG:-$HARNESS_ROOT/logs/keepalive.log}"
INTERVAL_SECONDS="${KEEPALIVE_INTERVAL_SECONDS:-3600}"
DURATION_SECONDS="${KEEPALIVE_DURATION_SECONDS:-300}"
LOGICAL_CPUS="$(nproc)"
TARGET_CPU_UTIL_PERCENT="${KEEPALIVE_TARGET_CPU_UTIL_PERCENT:-35}"
MIN_CPU_WORKERS="${KEEPALIVE_MIN_CPU_WORKERS:-32}"
DEFAULT_CPU_WORKERS="$(( (LOGICAL_CPUS * TARGET_CPU_UTIL_PERCENT + 99) / 100 ))"
if (( DEFAULT_CPU_WORKERS < MIN_CPU_WORKERS )); then
    DEFAULT_CPU_WORKERS="$MIN_CPU_WORKERS"
fi
CPU_WORKERS="${KEEPALIVE_CPU_WORKERS:-$DEFAULT_CPU_WORKERS}"
NICE_LEVEL="${KEEPALIVE_NICE_LEVEL:-15}"
RUN_IMMEDIATELY="${KEEPALIVE_RUN_IMMEDIATELY:-1}"

mkdir -p "$(dirname "$LOG")"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

protected_workload_running() {
    ps -eo pid=,args= | awk '
        /keepalive_gate_idle_load|keepalive_idle_load|awk |ps -eo|grep |rg / { next }
        /harness\/harness\/test\/monitor_ut\.py|harness\/harness\/test\/run_hd256_ut\.sh|python3 -m pytest|tests\/cute|cuda-gdb/ { found=1 }
        /harness\/harness\/benchmark|run_.*benchmark|benchmark.*\.py|hd256_hang_stress|run_hd256_hang_stress|torchrun|nsys|ncu/ { found=1 }
        END { exit found ? 0 : 1 }
    '
}

log "keepalive loop started interval=${INTERVAL_SECONDS}s duration=${DURATION_SECONDS}s cpu_workers=${CPU_WORKERS} logical_cpus=${LOGICAL_CPUS} target_cpu_util_percent=${TARGET_CPU_UTIL_PERCENT} reclaim_breaker=cpu_util_gt_10 mode=cpu-only"

first_iteration=1
while true; do
    if [[ "$first_iteration" == 1 && "$RUN_IMMEDIATELY" == 1 ]]; then
        first_iteration=0
    else
        first_iteration=0
        sleep "$INTERVAL_SECONDS"
    fi

    if protected_workload_running; then
        log "protected workload detected; skip keepalive load"
        continue
    fi

    log "starting CPU-only keepalive load"

    setsid nice -n "$NICE_LEVEL" python3 "$SCRIPT_DIR/keepalive_idle_load.py" \
        --duration "$DURATION_SECONDS" \
        --cpu-workers "$CPU_WORKERS" >> "$LOG" 2>&1 &
    load_pid="$!"

    while kill -0 "$load_pid" 2>/dev/null; do
        if protected_workload_running; then
            log "protected workload started during keepalive load; stopping load pid=$load_pid"
            kill -TERM -- "-$load_pid" 2>/dev/null || kill -TERM "$load_pid" 2>/dev/null || true
            sleep 2
            kill -KILL -- "-$load_pid" 2>/dev/null || true
            break
        fi
        sleep 5
    done
    wait "$load_pid" 2>/dev/null || true
    log "keepalive load finished"
done
