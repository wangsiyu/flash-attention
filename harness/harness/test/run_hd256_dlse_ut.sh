#!/usr/bin/env bash
# Run the HD256-only dLSE / differentiable-LSE UT group.
# Usage:
#   bash harness/harness/test/run_hd256_dlse_ut.sh
#   bash harness/harness/test/run_hd256_dlse_ut.sh --preflight-only
#
# Log:
#   harness/logs/test/ut_hd256_dlse.log

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$HARNESS_ROOT/.." && pwd)"
LOGDIR="$HARNESS_ROOT/logs/test"
TEST_FILE="$REPO/tests/cute/test_flash_attn.py"
PREFLIGHT_ONLY=0
PYTEST_IMPORT_ARGS=(--import-mode=importlib --rootdir="$REPO")

if [[ "${1:-}" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=1
fi

mkdir -p "$LOGDIR"

run_preflight() {
    (cd /tmp && REPO="$REPO" python3 - <<'PY'
import os
import re
from pathlib import Path

repo = Path(os.environ["REPO"])
test_file = repo / "tests/cute/test_flash_attn.py"

def ensure_hd256_only(path: Path, func_name: str, param_name: str) -> bool:
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
    desired = f'@pytest.mark.parametrize("{param_name}", [256])'
    if re.sub(r"\s+", "", current) == re.sub(r"\s+", "", desired):
        print(f"[preflight] {path}:{func_name}:{param_name} already [256]")
        return False

    path.write_text(text[:match.start()] + desired + text[match.end():])
    print(f"[preflight] patched {path}:{func_name}:{param_name} to [256]")
    return True

changed = False
for func in ("test_flash_attn_lse_grad", "test_flash_attn_lse_grad_unused"):
    changed = ensure_hd256_only(test_file, func, "d") or changed
print("[preflight] hd256 dLSE parametrization ready")
PY
    )
}

if ! run_preflight > "$LOGDIR/ut_hd256_dlse_preflight.log" 2>&1; then
    cat "$LOGDIR/ut_hd256_dlse_preflight.log"
    exit 1
fi
cat "$LOGDIR/ut_hd256_dlse_preflight.log"

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    echo "[preflight] preflight-only mode; dLSE UT execution skipped"
    exit 0
fi

PYTEST_XDIST_ARGS=()
if cd /tmp && python3 -m pytest --help 2>/dev/null | grep -q -- "--numprocesses"; then
    PYTEST_XDIST_ARGS=(-n 0)
fi

echo "[$(date '+%H:%M:%S')] START  hd256 dlse -> $LOGDIR/ut_hd256_dlse.log"
if cd /tmp && python3 -m pytest -v -s "${PYTEST_IMPORT_ARGS[@]}" "${PYTEST_XDIST_ARGS[@]}" --tb=long \
    "$TEST_FILE::test_flash_attn_lse_grad" \
    "$TEST_FILE::test_flash_attn_lse_grad_unused" \
    > "$LOGDIR/ut_hd256_dlse.log" 2>&1; then
    echo "[$(date '+%H:%M:%S')] DONE   hd256 dlse  ($(tail -1 "$LOGDIR/ut_hd256_dlse.log"))"
    exit 0
else
    echo "[$(date '+%H:%M:%S')] FAIL   hd256 dlse  ($(tail -1 "$LOGDIR/ut_hd256_dlse.log"))"
    exit 1
fi
