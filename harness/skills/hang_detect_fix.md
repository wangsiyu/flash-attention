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
| H5 Stress the same case | After a UT case is confirmed hung, keep the original hung process alive and build a focused repeat stress for that exact case. Record probabilistic metrics such as hang count, no-hang count, iteration-to-hang, time-to-hang, min/mean/max, source hash, and cuda-gdb log path. |
| H6 Debug reproduced hang | When the focused stress reproduces the hang, capture cuda-gdb on that live stress process too, then debug against source and SASS/disassembly before patching. |
| H7 Fix | Patch the source only after identifying a code-level hang cause. Consider performance impact for hot kernels and run targeted performance checks when the fix touches a performance path. |
| H8 Verify fix under pressure | Run the same focused stress on the fixed code until a long enough observation window shows no hang. Only after this stress passes, run the relevant benchmark comparison against FA4 2CTA for hot-kernel fixes. Attribute benchmark deltas to the modified path before deciding whether they block the fix. If the modified path is not negatively affected, the originally preserved hung process may be stopped and the full gate restarted from the beginning. |
| H9 Handle regression | If the fix creates a negative performance effect on the modified path, keep the original hang evidence, do not restart the full gate yet, and design a better fix. The next candidate must pass the same focused pressure and benchmark comparison before it can replace the previous candidate. |
| H10 False positive | If diagnostics show it was not a kernel/source hang and no code changed, keep waiting when possible. If the stale process must be stopped, rerun only the affected group with `/test --start-at <group>`. |

Do not reproduce by launching a smaller standalone case as the first hang
response. That loses the live state and can turn a broad-run interaction into a
false negative. Smaller reproductions are allowed only after live diagnostics
have already identified the failing site and a focused experiment is needed.

Before every new monitored run, clear old hang/debug logs so decisions are not
made from stale `hang_report.txt`, `hang_processes.txt`, `slow_case_report.txt`,
or `gdb_hang_pid*.log` files.

## Probabilistic Hang Protocol

Low-probability hangs are not fixed by a single rerun. Once a UT case has
produced a confirmed kernel hang, preserve that live UT process until a fix has
survived post-fix stress. Use the exact failing case for a focused pressure
loop, preferably in one Python process after the first compile so repeated
kernel launches exercise runtime state instead of repeatedly recompiling. For
each run, record whether it hung, the first unfinished iteration, elapsed time
from the case start, elapsed time after compilation when measurable, the source
hash, and the cuda-gdb snapshot path.

After the focused stress reproduces the hang, attach cuda-gdb to the stress
process before stopping it. Use the captured active kernel, warp state,
program counters, and disassembly together with the source to identify the
likely code-level cause. Patch only after this analysis. After the fix, run the
same stress for a longer no-hang window. If the fix touches a hot kernel or
scheduling path, rerun the relevant benchmark comparison against FA4 2CTA after
stress passes and before releasing the originally preserved UT process.

Benchmark failures must be attributed to the code path changed by the fix
before they block the release. For example, a backward-only dkdv fix is blocked
by a dkdv/backward regression, not by an unrelated forward row; similarly, a
forward-only fix is not blocked by an unrelated backward row. Path-unrelated
negative rows should be recorded with their cases and treated as noise or an
existing issue unless there is evidence that the change can affect that path.
If the benchmark shows a negative performance effect on the modified path, do
not release the preserved state and do not restart the full gate. Instead,
analyze a better fix candidate and repeat the same stress plus benchmark loop
until both no-hang and relevant performance requirements are satisfied. Do not
accept a hang fix that silently introduces relevant performance regressions
unless the tradeoff is explicitly approved.

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
