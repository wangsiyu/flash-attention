---
id: test
kind: skill
triggers:
  - test
  - UT
  - unit test
  - correctness
  - precision
  - failed
  - failure
  - rerun
---

# Test Skill

UT policy for CuteDSL HD256 validation.

## When to Use

| Signal | Action |
| ------ | ------ |
| UT, test, correctness, precision | Load `../commands/test.md`; use monitored test execution. |
| failed, failure | Stop all UT, reproduce failed case, fix, rerun monitored UT. |
| hang, stuck, GPU 100% | Jump to `hang_detect_fix.md`. |

## Test Loop

| Event | Required Action |
| ----- | --------------- |
| Start validation | Invoke `/test`; do not call pytest directly from memory. |
| UT failed detected in logs | Immediately stop the full UT run. |
| After failure stop | Reproduce the failing case, fix the bug, rerun `/test`. |
| Repeated failure | Repeat reproduce-fix-rerun until all UT pass. |
| Potential hang | Confirm there has been no command or UT-log progress before following `hang_detect_fix.md`. |
| No code change after passed group | Continue from the next incomplete group instead of restarting earlier passed groups. |
| False hang / no code change | If the stale process was stopped after diagnostics but no source/kernel code changed, rerun only the affected group. |
| Kernel/source code changed | Rerun the full monitored UT from the beginning. |
| HD256 score_mod | Keep `torch.compile` in the FlexAttention reference; do not classify this group as a hang, and use a 450s slow-case interval. |

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| No dead waiting | Do not wait indefinitely after failed logs or hang signal. |
| Fail fast | Kill the running UT process group when failure is detected. |
| Hang means idle | GPU 100% with continuing pytest log progress is not a hang. |
| Fix before rerun | Rerun only after investigating and attempting a fix. |
| No blind rerun | Do not restart from earlier passed groups unless kernel/source code changed or the prior result is invalid. |
| Fresh hang logs | Every monitored run must clear stale hang/debug logs before starting. |
| Command handoff | Monitoring must be triggered through `../commands/test.md`. |

## Permanent HD256 Gate Coverage

The monitored `/test` gate must keep all standard HD256 groups enabled while
keeping head dim restricted to 256: the complete `test_flash_attn_output`, the
complete `test_flash_attn_varlen_output`, all HD256 score_mod gate functions,
all HD256 mask_mod/block-sparse gate functions, the HD256-only dLSE group
(`test_flash_attn_lse_grad` and `test_flash_attn_lse_grad_unused`), and
`test_varlen`.

Do not replace any monitored UT group with representative nodeids, sampled
cases, or a shortened parametrization. If a complete group becomes too slow or
too large, treat that as a gate infrastructure problem to solve explicitly
rather than silently narrowing coverage.
