---
id: hd256_hang_stress
kind: command
command: /hd256_hang_stress
script: ../harness/benchmark/run_hd256_hang_stress_4gpu.sh
logs: ../harness/logs/hang_stress/
---

# HD256 Hang Stress

Fixed 4-GPU all-to-all HD256 FA4 hang pressure command.

## Launch

```bash
env NCCL_DEBUG=WARN STRESS_TIMEOUT_SECONDS=0 VERIFY_ENV=0 NPROC=4 \
  bash harness/harness/benchmark/run_hd256_hang_stress_4gpu.sh \
  --case heavy_h16_q8192_k8192_b2 \
  --direction fwd-bwd \
  --log-interval 10 \
  --sync-every 1 \
  --async-a2a \
  --calls-per-iter 24 \
  --state-every-iters 1
```

For the cuda-gdb dK/dV trap case whose kernel grid was `(168,1,1)`, switch to
the targeted shape and raise calls per iteration:

```bash
env NCCL_DEBUG=WARN STRESS_TIMEOUT_SECONDS=0 VERIFY_ENV=0 NPROC=4 \
  bash harness/harness/benchmark/run_hd256_hang_stress_4gpu.sh \
  --case gdb_grid168_q1024_k2560_b2 \
  --direction fwd-bwd \
  --log-interval 10 \
  --sync-every 1 \
  --async-a2a \
  --calls-per-iter 96 \
  --state-every-iters 1
```

For detached infinite launch:

```bash
mkdir -p harness/harness/logs/hang_stress/launchers
launcher_log="harness/harness/logs/hang_stress/launchers/launcher_$(date +%Y%m%d_%H%M%S)_a2a_h16_8192.log"
setsid bash -lc 'cd /mnt/workspace/siyu.wsy/qoder_workspace/flash-attention && env NCCL_DEBUG=WARN STRESS_TIMEOUT_SECONDS=0 VERIFY_ENV=0 NPROC=4 bash harness/harness/benchmark/run_hd256_hang_stress_4gpu.sh --case heavy_h16_q8192_k8192_b2 --direction fwd-bwd --log-interval 10 --sync-every 1 --async-a2a --calls-per-iter 24 --state-every-iters 1' > "$launcher_log" 2>&1 < /dev/null &
```

## Expected Runtime Contract

| Item | Requirement |
| ---- | ----------- |
| Ranks | `NPROC=4`, one process per GPU. |
| Communication | all-to-all enabled for Q/K/V dispatch, O combine, dO dispatch, and dQ/dK/dV combine. |
| FA calls | direct `_flash_attn_fwd` and `_flash_attn_bwd`, not the high-level interface. |
| Iteration | `24` forward+backward calls followed by one `cuda.synchronize()`. |
| Trace | state files written once per iteration; no stage trace by default. |

## Logs

| Path | Meaning |
| ---- | ------- |
| `harness/harness/logs/hang_stress/<run>/stress.log` | Main torchrun log. |
| `harness/harness/logs/hang_stress/<run>/state/rank*.txt` | Last per-rank iteration state. |
| `harness/harness/logs/hang_stress/launchers/*.log` | Detached launcher stdout/stderr. |
