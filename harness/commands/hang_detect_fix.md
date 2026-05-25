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
| 2 | Find the live pytest PID from the existing UT process group. |
| 3 | Run this command to collect cuda-gdb diagnostics on that live PID. |
| 4 | Use the snapshot to classify compile/setup slowness vs kernel hang. |
| 5 | If it is compile/setup slowness, keep waiting and re-sample. If it is a kernel hang, fill the exact case into `harness/harness/hang_detect_fix/stress_hang_case.py` or pass a case file to that script, then run repeated in-process pressure to collect probability metrics. |
| 6 | When focused pressure reproduces the hang, capture cuda-gdb on that live stress process too, then analyze the captured kernel/warp/disassembly state against the source before patching. |
| 7 | If a source/kernel fix is made, rerun the same pressure script until a long no-hang window passes. For hot-kernel fixes, run the relevant benchmark comparison against FA4 2CTA after stress passes, then attribute any benchmark deltas to the modified path. |
| 8 | If stress passes and the modified path is not negatively affected, stop the originally preserved hang process and rerun the full gate from the beginning. |
| 9 | If benchmark regresses on the modified path, keep the original evidence, do not rerun the full gate yet, and design a better fix. Repeat focused pressure and benchmark comparison for the next candidate. Path-unrelated negative rows, such as a forward row after a backward-only fix or a backward row after a forward-only fix, should be recorded but should not block the current fix unless there is evidence that the change can affect that path. |
| 10 | If no code changed and the process had to be stopped, rerun only the affected group. |

Do not restart with a smaller reproduction before capturing live diagnostics.
Do not kill the broad UT process merely because a slow case was observed.
Clear stale hang/debug logs before each monitored run.

## Focused Pressure Script

Use this script after live gate diagnostics have confirmed the hanging case:

```bash
python3 harness/harness/hang_detect_fix/stress_hang_case.py \
  --gpus 1,2,3 \
  --runs-per-gpu 5 \
  --max-iters 5000 \
  --max-seconds 900
```

Edit the `DEFAULT_CASE` block in the script for the exact UT case, or pass
`--case-file path/to/case.py` with `MODULE`, `FUNCTION`, and `CASE` variables.
The script runs each case repeatedly in one Python process, records
iteration/time-to-hang statistics, captures cuda-gdb for reproduced hangs, and
kills only the focused stress process unless `--keep-hung` is requested.
