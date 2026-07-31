# HD256 Rebase Stage 10

## Tested Source

- Main checkpoint: `c75d019dea9d910312974417bc28f190dfdda6d9`
- Source HEAD before lock commit: `50f16403f01acb5ceab298683a33efdd8d59efa5`
- Worktree diff SHA256: `f25c1027f47b8ca0930bf404f3c16c4b23d92575f9d53e88633a0e5a7f4af683`
- Benchmark SHA256: `4481ed1b901cdab1435eb082c87b3c58b2a39ef932ab33486576bd6999030bf7`
- Performance control HEAD: `41924c218b5d26681b4906035384a9caff8b2944`
- Performance control diff SHA256: `19c4f18fb9f583c73c9b7ead2f5530c7800edf0f6703d3d998054b94205dc93c`
- Control toolchain: `stage=10 cutlass=4.6.0.dev0 cuda_libs=4.6.0 quack=0.5.3 tvm=apache-tvm-ffi>=0.1.12,<0.2`
- Candidate toolchain: `stage=10 cutlass=4.6.0.dev0 cuda_libs=4.6.0 quack=0.5.3 tvm=apache-tvm-ffi>=0.1.12,<0.2`
- Container image: `dsw-registry-vpc.ap-southeast-1.cr.aliyuncs.com/pai/training:26.07.20-pytorch2.11-python3.12-cuda13.0-ubuntu24.04_accelerated`
- UT DLC job: `dlc13swd8q5qa3yf`
- Performance DLC job: `dlc17yq9ze3iusl0`
- UT result: `/cpfs01/shared/Group_m6/siyu.wsy/cc_fusion/flash-attention-rebase/agent_space/rebase_harness/results/fa4-hd256-final-strict-full-gate-20260731-0258`
- Performance result: `/cpfs01/shared/Group_m6/siyu.wsy/cc_fusion/flash-attention-rebase/agent_space/rebase_harness/results/fa4-hd256-final-source-perf-cutedsl46-20260731-0300`

The worktree diff is part of the tested source identity. It contains no temporary HD256 collection filter or source-side skip.

## Precision

- Verdict: pass
- Shards: 128
- Selected nodeids: 204429
- Executed: 204429
- Skipped by permanent test constraints: 76040
- Failures: 0
- Errors: 0

## Performance

- Verdict: pass
- Allowed stable jitter: 2.0%
- Maximum accepted coefficient of variation: 5.0%

| Direction | Causal | Seqlen | Control mean (ms) | Candidate mean (ms) | Delta | Candidate CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| backward | causal | 1024 | 0.744333 | 0.746000 | +0.224% | 0.109% |
| backward | causal | 2048 | 1.123333 | 1.123000 | -0.030% | 0.073% |
| backward | causal | 4096 | 1.893667 | 1.884000 | -0.510% | 0.043% |
| backward | causal | 8192 | 3.406000 | 3.389000 | -0.499% | 0.169% |
| backward | causal | 16384 | 6.430333 | 6.403333 | -0.420% | 0.116% |
| backward | causal | 32768 | 12.450667 | 12.358333 | -0.742% | 0.071% |
| backward | non-causal | 1024 | 1.057667 | 1.057333 | -0.032% | 0.045% |
| backward | non-causal | 2048 | 1.806667 | 1.812667 | +0.332% | 0.094% |
| backward | non-causal | 4096 | 3.282333 | 3.295333 | +0.396% | 0.076% |
| backward | non-causal | 8192 | 6.220667 | 6.260000 | +0.632% | 0.081% |
| backward | non-causal | 16384 | 12.093333 | 12.162667 | +0.573% | 0.089% |
| backward | non-causal | 32768 | 23.793000 | 23.983333 | +0.800% | 0.102% |
| forward | causal | 1024 | 0.182333 | 0.181000 | -0.731% | 0.781% |
| forward | causal | 2048 | 0.286667 | 0.284667 | -0.698% | 1.580% |
| forward | causal | 4096 | 0.488000 | 0.493333 | +1.093% | 0.191% |
| forward | causal | 8192 | 0.888333 | 0.890667 | +0.263% | 0.265% |
| forward | causal | 16384 | 1.671333 | 1.673667 | +0.140% | 0.373% |
| forward | causal | 32768 | 3.229000 | 3.234333 | +0.165% | 0.186% |
| forward | non-causal | 1024 | 0.264000 | 0.262000 | -0.758% | 0.312% |
| forward | non-causal | 2048 | 0.477667 | 0.475667 | -0.419% | 0.864% |
| forward | non-causal | 4096 | 0.857333 | 0.859000 | +0.194% | 0.285% |
| forward | non-causal | 8192 | 1.623000 | 1.618333 | -0.288% | 0.343% |
| forward | non-causal | 16384 | 3.139000 | 3.150000 | +0.350% | 0.162% |
| forward | non-causal | 32768 | 6.176667 | 6.177667 | +0.016% | 0.015% |

## Rebase Decisions

- Port current-main HD256 training features through generic HD128/main contracts while preserving the dedicated 2CTA HD256 QK loop/order; keep shared pipeline code unchanged via a local producer-tail PipelineState adapter; add the exact K=1 softmax-Jacobian zero guard without tolerance relaxation; exercise existing main tests through HD256 parametrization; use the unchanged FlexAttention oracle formula with only a 32x32 oracle compile tile to avoid a Triton compiler crash.

## Stress

- Verdict: pass
- DLC job: `dlcsq4paivynx6y4`
- Result: `/cpfs01/shared/Group_m6/siyu.wsy/cc_fusion/flash-attention-rebase/agent_space/rebase_harness/results/fa4-hd256-final-stress-5000-20260731-041940`
- Source HEAD before lock commit: `50f16403f01acb5ceab298683a33efdd8d59efa5`
- Worktree diff SHA256: `41485f02fd280ece5601d5a3207ef3142e731aab5455ab70d9c50dfdb9b43bd2`
- Topology: 1 worker, 4 GPUs, NCCL all-to-all enabled
- Case: `heavy_h16_q8192_k8192_b2`
- Direction: forward and backward
- Load: 5000 outer iterations, 24 calls per iteration, 120000 calls per rank
- Synchronization: every outer iteration
- Rank 0 final state: `iter=5000`, `total_calls=120000`, `stage=iter_done`
- Rank 1 final state: `iter=5000`, `total_calls=120000`, `stage=iter_done`
- Rank 2 final state: `iter=5000`, `total_calls=120000`, `stage=iter_done`
- Rank 3 final state: `iter=5000`, `total_calls=120000`, `stage=iter_done`
- Watchdog stalls: 0
- cuda-gdb captures: 0

The first stress submission stopped before kernel execution because the harness
still unpacked the old two-value low-level forward API. The final stress source
differs from the correctness and performance source only by changing that
harness unpack to accept the current four-value API; kernel and test logic are
unchanged.
