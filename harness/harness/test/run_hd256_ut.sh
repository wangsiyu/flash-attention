#!/usr/bin/env bash
# Run the standard CuteDSL head_dim=256 UT groups sequentially.
# Usage:
#   bash harness/harness/test/run_hd256_ut.sh
#   bash harness/harness/test/run_hd256_ut.sh --preflight-only
#   bash harness/harness/test/run_hd256_ut.sh --start-at hd256_varlen_output
#
# Logs:
#   harness/logs/test/preflight.log
#   harness/logs/test/ut_hd256_output.log
#   harness/logs/test/ut_hd256_varlen_output.log
#   harness/logs/test/ut_hd256_layout_buffers.log
#   harness/logs/test/ut_hd256_score_mod.log
#   harness/logs/test/ut_hd256_mask_mod_block_sparse.log
#   harness/logs/test/ut_hd256_varlen_mask_mod.log
#   harness/logs/test/ut_hd256_paged_kv.log
#   harness/logs/test/ut_hd256_dlse.log
#   harness/logs/test/ut_hd256_edge_cases.log
#   harness/logs/test/ut_varlen.log

set -u

export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$HARNESS_ROOT/.." && pwd)"
LOGDIR="${HD256_UT_LOGDIR:-$HARNESS_ROOT/logs/test}"
TEST_FILE="$REPO/tests/cute/test_flash_attn.py"
SCORE_MOD_TEST_FILE="$REPO/tests/cute/test_score_mod.py"
MASK_MOD_TEST_FILE="$REPO/tests/cute/test_mask_mod.py"
MASK_MOD_VARLEN_TEST_FILE="$REPO/tests/cute/test_mask_mod_varlen.py"
VARLEN_TEST_FILE="$REPO/tests/cute/test_flash_attn_varlen.py"
PREFLIGHT_ONLY=0
START_AT="${HD256_UT_START_AT:-hd256_output}"
PYTEST_IMPORT_ARGS=(--import-mode=importlib --rootdir="$REPO")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight-only)
            PREFLIGHT_ONLY=1
            shift
            ;;
        --start-at)
            if [[ $# -lt 2 ]]; then
                echo "--start-at requires a group name" >&2
                exit 2
            fi
            START_AT="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

normalize_group() {
    case "$1" in
        hd256_output|output)
            echo "hd256_output"
            ;;
        hd256_varlen_output|varlen_output)
            echo "hd256_varlen_output"
            ;;
        hd256_layout_buffers|layout_buffers)
            echo "hd256_layout_buffers"
            ;;
        hd256_score_mod|score_mod)
            echo "hd256_score_mod"
            ;;
        hd256_mask_mod_block_sparse|mask_mod_block_sparse|mask_mod)
            echo "hd256_mask_mod_block_sparse"
            ;;
        hd256_varlen_mask_mod|varlen_mask_mod)
            echo "hd256_varlen_mask_mod"
            ;;
        hd256_paged_kv|paged_kv)
            echo "hd256_paged_kv"
            ;;
        hd256_dlse|dlse)
            echo "hd256_dlse"
            ;;
        hd256_edge_cases|edge_cases)
            echo "hd256_edge_cases"
            ;;
        varlen)
            echo "varlen"
            ;;
        *)
            echo ""
            ;;
    esac
}

START_AT="$(normalize_group "$START_AT")"
if [[ -z "$START_AT" ]]; then
    echo "unknown --start-at group. Expected one of: hd256_output, hd256_varlen_output, hd256_layout_buffers, hd256_score_mod, hd256_mask_mod_block_sparse, hd256_varlen_mask_mod, hd256_paged_kv, hd256_dlse, hd256_edge_cases, varlen" >&2
    exit 2
fi

mkdir -p "$LOGDIR"

run_preflight() {
    (cd /tmp && REPO="$REPO" python3 - <<'PY'
import hashlib
import importlib.metadata as md
import os
import re
import subprocess
import sys
from pathlib import Path

import flash_attn
import flash_attn.cute.interface as interface
import flash_attn.cute.pipeline as pipeline
import flash_attn.cute.flash_fwd as flash_fwd
import flash_attn.cute.sm100_hd256_2cta_fmha_backward as bwd
import flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel as dkdv
import flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel as dq

print(f"[preflight] flash_attn={flash_attn.__file__}")
print(f"[preflight] flash_attn.__path__={list(getattr(flash_attn, '__path__', []))}")
print(f"[preflight] flash_attn.cute.flash_fwd={flash_fwd.__file__}")

repo = Path(os.environ["REPO"])
print(f"[preflight] repo={repo}")
print(f"[preflight] cwd={Path.cwd()}")
print(f"[preflight] sys.path[0]={sys.path[0]!r}")
for pkg in [
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "quack-kernels",
]:
    try:
        print(f"[preflight] package {pkg}={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"[preflight] package {pkg}=NOT_INSTALLED")

try:
    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
except Exception as exc:
    git_head = f"UNKNOWN:{exc}"
print(f"[preflight] git_head={git_head}")

try:
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"],
        text=True,
    ).splitlines()
except Exception as exc:
    dirty = [f"UNKNOWN:{exc}"]
print(f"[preflight] git_dirty_count={len(dirty)}")
for line in dirty[:40]:
    print(f"[preflight] git_dirty {line}")

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
    ("sm100_hd256_bwd", bwd),
    ("sm100_hd256_dkdv", dkdv),
    ("sm100_hd256_dq", dq),
]:
    stamp(label, Path(module.__file__))

targets = [
    (repo / "tests/cute/test_flash_attn.py", "test_flash_attn_output", "d"),
    (repo / "tests/cute/test_flash_attn.py", "test_flash_attn_varlen_output", "d"),
    (repo / "tests/cute/test_flash_attn.py", "test_flash_attn_lse_grad", "d"),
    (repo / "tests/cute/test_flash_attn.py", "test_flash_attn_lse_grad_unused", "d"),
    (repo / "tests/cute/test_flash_attn_varlen.py", "test_varlen", "D"),
]
for label, path in [
    ("test_flash_attn", targets[0][0]),
    ("test_score_mod", repo / "tests/cute/test_score_mod.py"),
    ("test_mask_mod", repo / "tests/cute/test_mask_mod.py"),
    ("test_mask_mod_varlen", repo / "tests/cute/test_mask_mod_varlen.py"),
    ("test_flash_attn_varlen", targets[-1][0]),
]:
    stamp(label, path)

def ensure_hd256_enabled(path: Path, func_name: str, param_name: str) -> None:
    text = path.read_text()
    marker = f"def {func_name}("
    def_index = text.find(marker)
    if def_index < 0:
        raise RuntimeError(f"{path}:{func_name} not found")

    pattern = re.compile(
        rf'(?m)^@pytest\.mark\.parametrize\(\s*(["\']){re.escape(param_name)}\1\s*,\s*\[[^\]]*\]\s*\)'
    )
    matches = [m for m in pattern.finditer(text) if m.end() < def_index]
    if not matches:
        raise RuntimeError(f"{path}:{func_name} has no active parametrize for {param_name!r}")

    match = matches[-1]
    current = match.group(0)
    values_match = re.search(r"\[([^\]]*)\]", current)
    if values_match is None:
        raise RuntimeError(f"{path}:{func_name}:{param_name} has no literal value list")
    values = {value.strip() for value in values_match.group(1).split(",")}
    if "256" not in values:
        raise RuntimeError(
            f"{path}:{func_name}:{param_name} does not enable 256: {current}"
        )
    print(f"[preflight] {path}:{func_name}:{param_name} includes 256")

for item in targets:
    ensure_hd256_enabled(*item)
PY
    )
}

if ! run_preflight > "$LOGDIR/preflight.log" 2>&1; then
    cat "$LOGDIR/preflight.log"
    exit 1
fi
cat "$LOGDIR/preflight.log"

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    echo "[preflight] preflight-only mode; UT execution skipped"
    exit 0
fi

PYTEST_XDIST_ARGS=()
if cd /tmp && python3 -m pytest --help 2>/dev/null | grep -q -- "--numprocesses"; then
    PYTEST_XDIST_ARGS=(-n 0)
fi
PYTEST_DURATION_ARGS=(--durations=0 --durations-min=45.0)

pass=0
fail=0
results=()
GROUP_ORDER=(
    hd256_output
    hd256_varlen_output
    hd256_layout_buffers
    hd256_score_mod
    hd256_mask_mod_block_sparse
    hd256_varlen_mask_mod
    hd256_paged_kv
    hd256_dlse
    hd256_edge_cases
    varlen
)

group_index() {
    local key="$1"
    local i
    for i in "${!GROUP_ORDER[@]}"; do
        if [[ "${GROUP_ORDER[$i]}" == "$key" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

START_INDEX="$(group_index "$START_AT")"

run_test_group() {
    local key="$1"
    local name="$2"
    local logfile="$3"
    shift 3
    local index
    index="$(group_index "$key")"
    if (( index < START_INDEX )); then
        echo "[$(date '+%H:%M:%S')] SKIP   $name  (--start-at $START_AT)"
        results+=("SKIP  $name")
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] START  $name -> $logfile"
    if cd /tmp && PYTHONPATH="$REPO/tests/cute:${PYTHONPATH:-}" python3 -m pytest -v -s "${PYTEST_IMPORT_ARGS[@]}" "${PYTEST_XDIST_ARGS[@]}" "${PYTEST_DURATION_ARGS[@]}" --tb=long "$@" > "$logfile" 2>&1; then
        results+=("PASS  $name")
        ((pass++))
    else
        results+=("FAIL  $name")
        ((fail++))
    fi
    echo "[$(date '+%H:%M:%S')] DONE   $name  ($(tail -1 "$logfile"))"
}

run_test_group "hd256_output" "hd256 output" "$LOGDIR/ut_hd256_output.log" \
    "$TEST_FILE::test_flash_attn_output"

run_test_group "hd256_varlen_output" "hd256 varlen output" "$LOGDIR/ut_hd256_varlen_output.log" \
    "$TEST_FILE::test_flash_attn_varlen_output"

run_test_group "hd256_layout_buffers" "hd256 layout and preallocated buffers" "$LOGDIR/ut_hd256_layout_buffers.log" \
    "$TEST_FILE::test_flash_attn_hd256_sm100_noncontiguous_transpose" \
    "$TEST_FILE::test_flash_attn_bwd_preallocated_outputs"

run_test_group "hd256_score_mod" "hd256 score_mod" "$LOGDIR/ut_hd256_score_mod.log" \
    "$SCORE_MOD_TEST_FILE::test_cute_vs_flex_attention" \
    "$SCORE_MOD_TEST_FILE::test_cute_score_mod_vectorized" \
    "$SCORE_MOD_TEST_FILE::test_cute_vs_flex_attention_with_aux_tensors" \
    "$SCORE_MOD_TEST_FILE::test_cute_score_mod_with_aux_tensors_vectorized" \
    "$SCORE_MOD_TEST_FILE::test_cute_vs_flex_attention_backward" \
    "$SCORE_MOD_TEST_FILE::test_cute_vs_flex_attention_backward_with_aux" \
    "$SCORE_MOD_TEST_FILE::test_cute_vs_flex_attention_backward_pack_gqa" \
    "$SCORE_MOD_TEST_FILE::test_cute_score_mod_bwd_aux_scalars_matches_flex"

run_test_group "hd256_mask_mod_block_sparse" "hd256 mask_mod block_sparse" "$LOGDIR/ut_hd256_mask_mod_block_sparse.log" \
    "$MASK_MOD_TEST_FILE::test_cute_mask_mod_vectorized" \
    "$MASK_MOD_TEST_FILE::test_q_boundary_masking_block_sparse_bwd" \
    "$MASK_MOD_TEST_FILE::test_static_masks" \
    "$MASK_MOD_TEST_FILE::test_parameterized_masks" \
    "$MASK_MOD_TEST_FILE::test_sm100_block_sparse_sink_all_masked" \
    "$MASK_MOD_TEST_FILE::test_sm100_block_sparse_coarse_blocks" \
    "$MASK_MOD_TEST_FILE::test_sm100_block_sparse_coarse_blocks_mismatch" \
    "$MASK_MOD_TEST_FILE::test_block_sparse_bwd_deterministic"

run_test_group "hd256_varlen_mask_mod" "hd256 varlen mask_mod" "$LOGDIR/ut_hd256_varlen_mask_mod.log" \
    "$MASK_MOD_VARLEN_TEST_FILE::test_varlen_static_masks" \
    "$MASK_MOD_VARLEN_TEST_FILE::test_varlen_parameterized_masks" \
    "$MASK_MOD_VARLEN_TEST_FILE::test_varlen_global_masks" \
    "$MASK_MOD_VARLEN_TEST_FILE::test_varlen_mask_mod_vectorized" \
    "$MASK_MOD_VARLEN_TEST_FILE::test_varlen_block_sparse"

run_test_group "hd256_paged_kv" "hd256 paged KV" "$LOGDIR/ut_hd256_paged_kv.log" \
    "$TEST_FILE::test_flash_attn_paged_hd256_sm100_tma" \
    "$TEST_FILE::test_flash_attn_paged_hd256_sm100_tma_gqa" \
    "$TEST_FILE::test_flash_attn_paged_hd256_sm100_tma_shuffled"

run_test_group "hd256_dlse" "hd256 dlse" "$LOGDIR/ut_hd256_dlse.log" \
    "$TEST_FILE::test_flash_attn_lse_grad" \
    "$TEST_FILE::test_flash_attn_lse_grad_unused"

run_test_group "hd256_edge_cases" "hd256 empty-work edge cases" "$LOGDIR/ut_hd256_edge_cases.log" \
    "$TEST_FILE::test_flash_attn_seqlen_k_zero" \
    "$TEST_FILE::test_flash_attn_empty_q_dense" \
    "$TEST_FILE::test_flash_attn_empty_q_varlen"

run_test_group "varlen" "varlen" "$LOGDIR/ut_varlen.log" \
    "$VARLEN_TEST_FILE::test_varlen"

echo ""
echo "=============================="
echo "  SUMMARY: $pass passed, $fail failed"
echo "=============================="
for r in "${results[@]}"; do
    echo "  $r"
done

exit "$fail"
