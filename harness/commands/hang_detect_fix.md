---
id: hang_detect_fix
kind: command
command: /hang_detect_fix
script: ../harness/hang_detect_fix/cuda_gdb_snapshot.sh
logs: ../logs/hang_detect_fix/
---

# Hang Detect Fix

Helper command for cuda-gdb diagnostics on a live suspected hang.

## Invocation

| Mode | Command |
| ---- | ------- |
| Capture live PID | `bash harness/harness/hang_detect_fix/cuda_gdb_snapshot.sh <pid>` |
| Custom log | `bash harness/harness/hang_detect_fix/cuda_gdb_snapshot.sh <pid> harness/logs/hang_detect_fix/gdb_hang.log` |

## Required Flow

| Step | Action |
| ---- | ------ |
| 1 | Keep the broad UT process alive when a hang is suspected. |
| 2 | Record the stalled node ID, heartbeat age, and neighboring-process progress. Treat the node ID as a seed, not a mandatory final reproducer. |
| 3 | Without changing source, run repeated in-process pressure over the affected changed-kernel path. Start with the triggering case, then include neighboring shapes, branch toggles, or the original pre-trigger test sequence when needed to preserve JIT-cache, allocator, or asynchronous execution history. |
| 4 | Run every accepted focused pressure process for at least 10 minutes after first-iteration compilation. Iteration count is diagnostic only: finishing a nominal iteration count before 10 minutes is inconclusive, not a pass. Record hang count, no-hang count, iteration-to-hang, time-to-hang, source hash, and the exact pressure configuration. |
| 5 | Only after focused pressure has reproduced a stalled iteration, keep that stress process alive and attach cuda-gdb to it. |
| 6 | Use the snapshot to classify compile/setup slowness vs an active kernel hang. If it is compile/setup slowness, adjust the pressure threshold and repeat without patching. |
| 7 | Analyze the reproduced kernel/warp/disassembly state against the changed source before patching. No kernel edit is allowed before both pressure reproduction and cuda-gdb evidence exist. |
| 8 | If a source/kernel fix is made, rerun the same pressure matrix until a long no-hang window passes. For hot-kernel fixes, run the relevant benchmark comparison against FA4 2CTA after stress passes, then attribute any benchmark deltas to the modified path. |
| 9 | If stress passes and the modified path is not negatively affected, stop the originally preserved hang process and rerun the full gate from the beginning. |
| 10 | If benchmark regresses on the modified path, keep the original evidence, do not rerun the full gate yet, and design a better fix. Repeat focused pressure and benchmark comparison for the next candidate. Path-unrelated negative rows, such as a forward row after a backward-only fix or a backward row after a forward-only fix, should be recorded but should not block the current fix unless there is evidence that the change can affect that path. |
| 11 | If no code changed and the process had to be stopped, rerun only the affected group. |

Do not use cuda-gdb as a substitute for focused pressure reproduction.
Do not require the final reproducer to be the exact broad-gate node ID when a
neighboring case exercises the same changed kernel path more reliably.
Do not kill the broad UT process merely because a slow case was observed.
Clear stale hang/debug logs before each monitored run.

## Focused Pressure Script

Use this script after the broad gate identifies the affected kernel path:

```bash
python3 harness/harness/hang_detect_fix/stress_hang_case.py \
  --gpus 1,2,3 \
  --runs-per-gpu 5 \
  --max-iters 100000000 \
  --max-seconds 900
```

Edit the `DEFAULT_CASE` block, pass `--case-file path/to/case.py`, or launch
multiple focused cases for a shape/branch matrix. The exact UT case is the
first seed, but the selected pressure case only needs to exercise the same
changed kernel path. The script runs each case repeatedly in one Python
process, records iteration/time-to-hang statistics, and captures cuda-gdb only
after its monitor has identified a reproduced stalled iteration. It kills only
the focused stress process unless `--keep-hung` is requested.

The configured iteration ceiling must be high enough that wall-clock pressure
reaches at least 600 seconds after first compilation. A short run such as 5000
iterations in under one minute is useful for smoke testing only and must never
be reported as a no-hang result. When the broad gate hangs only after many
earlier cases, replay the original manifest prefix in the same order alongside
the isolated long-duration pressure.

For asynchronous launch or pipeline hangs, do not put
`torch.cuda.synchronize()` at every pressure iteration. Use
`stress_hd256_varlen_forward_async.py` for the HD256 varlen forward path. It
keeps several event-delimited batches in flight and checks progress with
non-blocking `CUDAEvent.query()`. A bounded event queue prevents unbounded
allocation while preserving asynchronous launches. Synchronization-heavy
correctness tests remain the final gate, not the primary hang reproducer.
