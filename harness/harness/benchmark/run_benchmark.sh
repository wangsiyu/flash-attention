#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HARNESS_ROOT="$(cd "$HARNESS_CODE_ROOT/.." && pwd)"
REPO="$(cd "$HARNESS_ROOT/.." && pwd)"
LOG_ROOT="$HARNESS_CODE_ROOT/logs/benchmark"
CURRENT_DIR="$LOG_ROOT/current"
PREVIOUS_DIR="$LOG_ROOT/previous"
SOURCE_STAMP="$CURRENT_DIR/source_stamp.log"
RUNS="${BENCHMARK_RUNS:-1}"
REP="${BENCHMARK_REP:-1000}"
WARMUP="${BENCHMARK_WARMUP:-100}"
SDPA_REP="${BENCHMARK_SDPA_REP:-$REP}"
SDPA_WARMUP="${BENCHMARK_SDPA_WARMUP:-$WARMUP}"
FLASH_MOE_ENABLED="${BENCHMARK_FLASH_MOE:-0}"
FLASH_MOE_REPO="${FLASH_MOE_REPO:-$(cd "$REPO/.." && pwd)/flash-moe}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

CMD=(
    python3
    "$REPO/harness/harness/benchmark/bench_sm100_hd256.py"
    --suite gate
    --rep "$REP"
    --warmup "$WARMUP"
)
SDPA_CMD=(
    python3
    "$REPO/harness/harness/benchmark/bench_sm100_hd256.py"
    --suite gate
    --sdpa-only
    --rep "$SDPA_REP"
    --warmup "$SDPA_WARMUP"
)
FLASH_MOE_CMD=(
    python3
    "$REPO/harness/harness/benchmark/bench_sm100_hd256.py"
    --suite gate
    --impl flash_moe
    --flash-moe-src "$FLASH_MOE_REPO/src"
    --rep "$REP"
    --warmup "$WARMUP"
)

mkdir -p "$LOG_ROOT"

echo "[benchmark] repo=$REPO"
echo "[benchmark] logs=$LOG_ROOT"
echo "[benchmark] runs=$RUNS"
echo "[benchmark] rep=$REP"
echo "[benchmark] warmup=$WARMUP"
echo "[benchmark] sdpa_rep=$SDPA_REP"
echo "[benchmark] sdpa_warmup=$SDPA_WARMUP"
echo "[benchmark] flash_moe_enabled=$FLASH_MOE_ENABLED"
echo "[benchmark] flash_moe_repo=$FLASH_MOE_REPO"
echo "[benchmark] command=${CMD[*]}"
echo "[benchmark] sdpa_command=${SDPA_CMD[*]}"
echo "[benchmark] flash_moe_command=${FLASH_MOE_CMD[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

keepalive_load_running() {
    ps -eo args= | awk '
        /keepalive_idle_load\.py/ && !/awk / { found=1 }
        END { exit found ? 0 : 1 }
    '
}

while keepalive_load_running; do
    echo "[benchmark] waiting for keepalive CPU load to finish"
    sleep 5
done

if [[ -d "$CURRENT_DIR" ]]; then
    if [[ -f "$CURRENT_DIR/benchmark_report.md" ]]; then
        rm -rf "$PREVIOUS_DIR"
        mv "$CURRENT_DIR" "$PREVIOUS_DIR"
    else
        echo "[benchmark] discarding incomplete current benchmark directory: $CURRENT_DIR"
        rm -rf "$CURRENT_DIR"
    fi
fi
mkdir -p "$CURRENT_DIR"

(
    cd /tmp
    echo "[benchmark] source_stamp=$SOURCE_STAMP"
    REPO="$REPO" FLASH_MOE_REPO="$FLASH_MOE_REPO" FLASH_MOE_ENABLED="$FLASH_MOE_ENABLED" python3 - <<'PY' | tee "$SOURCE_STAMP"
import hashlib
import importlib.metadata as md
import os
import subprocess
import sys
from pathlib import Path
import flash_attn.cute.interface as interface
import flash_attn.cute.pipeline as pipeline
import flash_attn.cute.flash_fwd as flash_fwd
import flash_attn.cute.block_info as block_info
import flash_attn.cute.block_sparse_utils as block_sparse_utils
import flash_attn.cute.mask as mask
import flash_attn.cute.sm100_hd256_2cta_fmha_forward as fwd
import flash_attn.cute.sm100_hd256_2cta_fmha_backward as bwd
import flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel as dkdv
import flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel as dq

repo = Path(os.environ["REPO"]).resolve()
flash_moe_repo = Path(os.environ.get("FLASH_MOE_REPO", "")).resolve()
flash_moe_enabled = os.environ.get("FLASH_MOE_ENABLED") == "1"
interface_path = Path(interface.__file__).resolve()
expected = repo / "flash_attn" / "cute" / "interface.py"
print(f"[benchmark] flash_attn.cute.interface={interface_path}")
if interface_path != expected:
    raise SystemExit(f"benchmark must use repo-local CuteDSL interface: expected {expected}")

print(f"[benchmark] repo={repo}")
print(f"[benchmark] cwd={Path.cwd()}")
print(f"[benchmark] sys.path[0]={sys.path[0]!r}")
for pkg in [
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "quack-kernels",
]:
    try:
        print(f"[benchmark] package {pkg}={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"[benchmark] package {pkg}=NOT_INSTALLED")

try:
    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
except Exception as exc:
    git_head = f"UNKNOWN:{exc}"
print(f"[benchmark] git_head={git_head}")

def stamp(label: str, path: Path) -> None:
    path = path.resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    stat = path.stat()
    under_repo = repo in path.parents or path == repo
    print(
        f"[source] {label} path={path} under_repo={under_repo} "
        f"sha256={digest} mtime_ns={stat.st_mtime_ns} size={stat.st_size}"
    )

for label, module in [
    ("interface", interface),
    ("pipeline", pipeline),
    ("flash_fwd", flash_fwd),
    ("block_info", block_info),
    ("block_sparse_utils", block_sparse_utils),
    ("mask", mask),
    ("sm100_hd256_fwd", fwd),
    ("sm100_hd256_bwd", bwd),
    ("sm100_hd256_dkdv", dkdv),
    ("sm100_hd256_dq", dq),
]:
    stamp(label, Path(module.__file__))

if flash_moe_enabled:
    print(f"[benchmark] flash_moe_repo={flash_moe_repo}")
    for label, rel in [
        ("flash_moe_interface", "src/flash_moe/nn/functional/flash_attn/interface.py"),
        ("flash_moe_fwd", "src/flash_moe/nn/functional/flash_attn/fmha_forward_sm100.py"),
        ("flash_moe_bwd", "src/flash_moe/nn/functional/flash_attn/fmha_backward_sm100.py"),
    ]:
        path = flash_moe_repo / rel
        if path.exists():
            stamp(label, path)
        else:
            print(f"[source] {label} path={path} MISSING")
PY
    for i in $(seq 1 "$RUNS"); do
        log="$CURRENT_DIR/run_${i}.log"
        echo "[benchmark] START run $i -> $log"
        "${CMD[@]}" 2>&1 | tee "$log"
        echo "[benchmark] DONE run $i"
    done
    if [[ "$FLASH_MOE_ENABLED" == "1" ]]; then
        if [[ ! -d "$FLASH_MOE_REPO/src/flash_moe/nn/functional/flash_attn" ]]; then
            echo "[benchmark] missing flash-moe flash_attn directory: $FLASH_MOE_REPO/src/flash_moe/nn/functional/flash_attn" >&2
            exit 2
        fi
        for i in $(seq 1 "$RUNS"); do
            log="$CURRENT_DIR/flash_moe_run_${i}.log"
            echo "[benchmark] START flash-moe run $i -> $log"
            "${FLASH_MOE_CMD[@]}" 2>&1 | tee "$log"
            echo "[benchmark] DONE flash-moe run $i"
        done
    fi
    sdpa_log="$CURRENT_DIR/sdpa_baseline.log"
    echo "[benchmark] START sdpa baseline -> $sdpa_log"
    "${SDPA_CMD[@]}" 2>&1 | tee "$sdpa_log"
    echo "[benchmark] DONE sdpa baseline"
)

COMPARE_CMD=(
    python3
    "$SCRIPT_DIR/compare_benchmark.py"
    --current "$CURRENT_DIR"
    --previous "$PREVIOUS_DIR"
    --sdpa-baseline "$CURRENT_DIR/sdpa_baseline.log"
    --source-stamp "$SOURCE_STAMP"
    --report "$CURRENT_DIR/benchmark_report.md"
)
if [[ "$FLASH_MOE_ENABLED" == "1" ]]; then
    COMPARE_CMD+=(--flash-moe "$CURRENT_DIR")
fi
"${COMPARE_CMD[@]}"
