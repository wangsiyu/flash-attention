---
id: hd256_hang_stress
kind: skill
triggers:
  - hd256-hang-stress
  - hang stress
  - 4gpu stress
  - all-to-all stress
  - online hang repro
---

# HD256 Hang Stress Skill

4-GPU HD256 FA4 hang stress that preserves the online communication pattern.

## When to Use

| Signal | Action |
| ------ | ------ |
| Random online hang, low-probability hang, long pressure test | Run the fixed 4-GPU all-to-all stress command. |
| Need GPU pressure while preserving online topology | Use `heavy_h16_q8192_k8192_b2`; do not disable all-to-all. |
| Need to target the cuda-gdb `GridDim.x=168` dK/dV trap | Use `gdb_grid168_q1024_k2560_b2` first, then `gdb_grid168_q1024_k2432_b3`. |
| Need debug state for a stuck run | Read `harness/harness/logs/hang_stress/<run>/state/rank*.txt`. |

## Fixed Case

| Field | Value |
| ----- | ----- |
| Case | `heavy_h16_q8192_k8192_b2` |
| Heads | `16` |
| Head dim | `256` |
| Local Q shape per rank | `[16384, 16, 256]` |
| Local K/V shape per rank | `[16384, 16, 256]` |
| Varlen batch | `2` sequences, each `8192` |
| `cu_seqlens_q` / `cu_seqlens_k` | `[0, 8192, 16384]` |
| `max_seqlen_q` / `max_seqlen_k` | `8192` |
| Causal | `True` |
| Iteration shape | `24` forward+backward calls, then one sync |

## Required Command

Use the command documented in `../commands/hd256_hang_stress.md`.

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

For the `tcgen05_guardrail_trap_unallocated_columns_being_dealloced` dK/dV
site with `GridDim.x=168`, use the same command shape with:

```bash
--case gdb_grid168_q1024_k2560_b2 --calls-per-iter 96
```

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| Four ranks | Keep `NPROC=4`; one rank per GPU. |
| Four-GPU communication | All-to-all must remain enabled. Never pass `--disable-a2a` for this skill. |
| Direct FA4 calls | Use `_flash_attn_fwd` and `_flash_attn_bwd` through `harness/harness/benchmark/hd256_hang_stress_4gpu.py`. |
| Low trace overhead | Do not enable `--stage-trace` during pressure runs. Keep state writes at iteration granularity. |
| Sync granularity | Do not sync around each all-to-all or FA stage. Sync only after the full outer iteration. |
| Runtime source | Run from the repo-local editable `flash_attn/cute`; never patch `flash_attn/__init__.py`. |

## Hang Handling

If progress stops while any GPU remains busy, use `hang_detect_fix.md`:

1. Capture the latest state files under `harness/harness/logs/hang_stress/<run>/state/`.
2. Kill the whole stress process group, not one rank.
3. Reproduce with the same fixed command.
4. Attach cuda-gdb according to `../commands/hang_detect_fix.md`.
