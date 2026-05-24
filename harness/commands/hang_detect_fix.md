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
| 5 | If it is compile/setup slowness, keep waiting and re-sample. If it is a kernel hang, analyze the captured kernel/warp/disassembly state against the source. |
| 6 | If a source/kernel fix is made, rerun the full gate from the beginning. If no code changed and the process had to be stopped, rerun only the affected group. |

Do not restart with a smaller reproduction before capturing live diagnostics.
Do not kill the broad UT process merely because a slow case was observed.
Clear stale hang/debug logs before each monitored run.
