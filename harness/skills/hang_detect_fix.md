---
id: hang_detect_fix
kind: skill
triggers:
  - hang
  - hang detect
  - hang fix
  - hang detection fix
  - stuck
  - gpu 100
  - cuda-gdb
---

# Hang Detect Fix Skill

Hang detection and repair flow for HD256 UT.

## When to Use

| Signal | Action |
| ------ | ------ |
| UT runtime > 45s and the UT process group's GPU util stays 100% without log progress | Treat as suspected hang and preserve the live process. |
| hang detected by `test.md` monitor | Attach diagnostic tools to the live broad UT process before making a hang decision. |
| cuda-gdb needed | Load `../commands/hang_detect_fix.md`. |

## Hang Rule

| Condition | Decision |
| --------- | -------- |
| No log progress, GPU utilization stays at 100%, and cuda-gdb shows active kernel/warp state that is not making progress | Kernel hang |
| No log progress, GPU utilization is low or cuda-gdb shows no active CUDA kernel | Likely compile/setup slowness; keep waiting and re-sample. |
| New log output appears or the process exits | Not a hang. |

## Fix Flow

| Step | Required Action |
| ---- | --------------- |
| H1 Preserve | Do not kill or restart the broad UT process on a suspected hang. |
| H2 Attach | Use cuda-gdb or equivalent live diagnostics on the existing pytest process. |
| H3 Classify | Decide whether the pause is compile/setup slowness or a kernel hang from live diagnostics. |
| H4 Wait or analyze | If compile/setup slowness, keep waiting. If kernel hang, inspect the captured kernel/warp/disassembly state and analyze source directly. |
| H5 Fix | Patch the source only after identifying a code-level hang cause. If kernel/source changes are made, rerun the full gate from the beginning. |
| H6 False positive | If diagnostics show it was not a kernel/source hang and no code changed, keep waiting when possible. If the stale process must be stopped, rerun only the affected group with `/test --start-at <group>`. |

Do not reproduce by launching a smaller standalone case as the first hang
response. That loses the live state and can turn a broad-run interaction into a
false negative. Smaller reproductions are allowed only after live diagnostics
have already identified the failing site and a focused experiment is needed.

Before every new monitored run, clear old hang/debug logs so decisions are not
made from stale `hang_report.txt`, `hang_processes.txt`, `slow_case_report.txt`,
or `gdb_hang_pid*.log` files.

## cuda-gdb Commands

```gdb
set logging file gdb_hang.log
set logging on
info cuda kernels
info cuda warps
info cuda threads sm 0
x/8i $pc
bt
set logging off
```

Use `../commands/hang_detect_fix.md` for the executable helper.
