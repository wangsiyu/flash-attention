---
id: test
kind: command
command: /test
script: ../harness/test/monitor_ut.py
logs: ../logs/test/
---

# Test

Monitored UT command for HD256 validation.

## Invocation

| Mode | Command |
| ---- | ------- |
| Full monitored UT | `python3 harness/harness/test/monitor_ut.py` |
| Preflight through UT script | `python3 harness/harness/test/monitor_ut.py --preflight-only` |
| Resume after passed groups | `python3 harness/harness/test/monitor_ut.py --start-at hd256_varlen_output` |

## Behavior

| Event | Script Action |
| ----- | ------------- |
| Run starts | Calls internal runner `bash harness/harness/test/run_hd256_ut.sh`. |
| Failure pattern appears in UT logs | Kills the UT process group and writes failure summary. |
| No new command or UT-log output while the UT process group's GPU utilization remains 100% | Captures live cuda-gdb diagnostics and leaves the UT process group alive for diagnosis. |
| No failure or hang | Waits until UT process exits. |

Hang detection is based on idle time, not total runtime. Long HD256 groups can
keep the GPU busy for several minutes; do not treat normal log progress as a
hang. Suspected hangs are diagnosed in place; do not launch a smaller
reproduction before capturing the live state.

The HD256 score_mod group keeps the `torch.compile` FlexAttention reference.
Because that group has only a small number of cases and compile latency can be
long, the monitor does not classify score_mod as a hang; its slow-case interval
is 450 seconds.

## Internal Runner

| Runner | Scope |
| ------ | ----- |
| `harness/harness/test/run_hd256_ut.sh` | Runs the full CuteDSL `head_dim=256` UT gate: output, varlen output, score_mod, mask_mod/block_sparse, dLSE, and varlen. Internal only; use `/test` as the command entry. |
| `harness/harness/test/run_hd256_dlse_ut.sh` | Runs only the HD256 dLSE / differentiable-LSE UT group for narrow reproduction. |

If no kernel/source code changed and an earlier group already passed, the
runner may continue from the next group with `--start-at <group>`. If any
kernel/source code changes, rerun the full monitored UT from the beginning.
After a false hang or tool-induced stop with no code change, rerun only the
affected group. Before every monitored run, stale hang/debug artifacts are
cleared so `hang_report.txt` and cuda-gdb logs cannot be mistaken for current
evidence.

The runner must execute import preflight and pytest from `/tmp`, not from the
repo root. It passes absolute test paths plus `--rootdir=$REPO` and
`--import-mode=importlib` so editable FA4 resolves as a namespace package and
root `flash_attn/__init__.py` does not import FA2 `flash_attn_2_cuda`.

## Preflight

Preflight logs a source stamp before any pytest group runs. The stamp includes
package versions, git HEAD/dirty files, and path/SHA/mtime for key modules:
`interface`, `pipeline`, `flash_fwd`, `sm100_hd256_bwd`, `sm100_hd256_dkdv`,
and `sm100_hd256_dq`. Use this section of `preflight.log` to prove each run is
using the latest working-tree kernel files. Full UT coverage means complete
HD256 parametrization; preflight must still keep `d` / `D` restricted to `[256]`.

| Test | Required Param |
| ---- | -------------- |
| `tests/cute/test_flash_attn.py::test_flash_attn_output` | `d == [256]` |
| `tests/cute/test_flash_attn.py::test_flash_attn_varlen_output` | `d == [256]` |
| `tests/cute/test_flash_attn.py::test_flash_attn_lse_grad` | `d == [256]` |
| `tests/cute/test_flash_attn.py::test_flash_attn_lse_grad_unused` | `d == [256]` |
| `tests/cute/test_flash_attn_varlen.py::test_varlen` | `D == [256]` |

If any active parametrization is not `[256]`, the internal runner temporarily
patches it before running.

## Test Groups

| Group | Pytest Node | Log |
| ----- | ----------- | --- |
| HD256 output | `tests/cute/test_flash_attn.py::test_flash_attn_output` | `harness/logs/test/ut_hd256_output.log` |
| HD256 varlen output | `tests/cute/test_flash_attn.py::test_flash_attn_varlen_output` | `harness/logs/test/ut_hd256_varlen_output.log` |
| HD256 score_mod | All HD256 score_mod gate functions in `tests/cute/test_score_mod.py` | `harness/logs/test/ut_hd256_score_mod.log` |
| HD256 mask_mod/block_sparse | All HD256 mask_mod/block-sparse gate functions in `tests/cute/test_mask_mod.py` | `harness/logs/test/ut_hd256_mask_mod_block_sparse.log` |
| HD256 dLSE | `tests/cute/test_flash_attn.py::test_flash_attn_lse_grad` and `tests/cute/test_flash_attn.py::test_flash_attn_lse_grad_unused` | `harness/logs/test/ut_hd256_dlse.log` |
| Varlen | `tests/cute/test_flash_attn_varlen.py::test_varlen` | `harness/logs/test/ut_varlen.log` |

## Outputs

The monitored UT gate runs complete HD256 test groups with head dim restricted
to 256. Do not replace any group with representative nodeids or sampled cases.

| File | Purpose |
| ---- | ------- |
| `harness/logs/test/monitor.log` | Monitor events |
| `harness/logs/test/preflight.log` | Import and head_dim preflight |
| `harness/logs/test/failure_cases.txt` | Failed case or group hints |
| `harness/logs/test/hang_report.txt` | Hang detection summary |

## Follow-Up

| Result | Next Step |
| ------ | --------- |
| UT pass | Proceed to `../skills/benchmark.md` / `../commands/benchmark.md`; after benchmark passes, run the `../skills/hd256_hang_stress.md` 5000-iteration stress gate. |
| Failed | Reproduce failed case, fix, rerun `/test`. |
| Hang | Load `../skills/hang_detect_fix.md` and `hang_detect_fix.md`. |
