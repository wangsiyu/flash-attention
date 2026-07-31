# HD256 Rebase Stage 02

## Tested Source

- Main checkpoint: `2d5d5a1c95a34884b3708ef66e93db482c3a5902`
- Source HEAD before lock commit: `efe95297f67909d8f96609340527d2a38bc3cf25`
- Worktree diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Benchmark SHA256: `4481ed1b901cdab1435eb082c87b3c58b2a39ef932ab33486576bd6999030bf7`
- Performance control HEAD: `f2662a214472a929bd763b66f1931c4819806d3f`
- Performance control diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Control toolchain: `stage=01 cutlass=4.4.2 quack=0.4.1 tvm=apache-tvm-ffi>=0.1.5,<0.2`
- Candidate toolchain: `stage=02 cutlass=4.4.2 quack=0.4.1 tvm=apache-tvm-ffi>=0.1.5,<0.2`
- Container image: `dsw-registry-vpc.ap-southeast-1.cr.aliyuncs.com/pai/training:26.07.20-pytorch2.11-python3.12-cuda13.0-ubuntu24.04_accelerated`
- UT DLC job: `dlcgrpukqdxc0zh0`
- Performance DLC job: `dlcx4gtmy7mpmwau`
- UT result: `/cpfs01/shared/Group_m6/siyu.wsy/cc_fusion/flash-attention-rebase/agent_space/rebase_harness/results/fa4-hd256-ut-s02-20260730-082821`
- Performance result: `/cpfs01/shared/Group_m6/siyu.wsy/cc_fusion/flash-attention-rebase/agent_space/rebase_harness/results/fa4-hd256-perf-ab-s02-20260730-072536`

The worktree diff is part of the tested source identity. It contains no temporary HD256 collection filter or source-side skip.

## Precision

- Verdict: pass
- Shards: 256
- Selected nodeids: 200629
- Executed: 200629
- Skipped by permanent test constraints: 75096
- Failures: 0
- Errors: 0

## Performance

- Verdict: pass
- Allowed stable jitter: 2.0%
- Maximum accepted coefficient of variation: 5.0%

| Direction | Causal | Seqlen | Control mean (ms) | Candidate mean (ms) | Delta | Candidate CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| backward | causal | 1024 | 0.642545 | 0.644000 | +0.226% | 0.459% |
| backward | causal | 2048 | 0.996455 | 0.997545 | +0.109% | 0.506% |
| backward | causal | 4096 | 1.693000 | 1.692455 | -0.032% | 0.941% |
| backward | causal | 8192 | 3.062273 | 3.063273 | +0.033% | 0.223% |
| backward | causal | 16384 | 5.791364 | 5.792182 | +0.014% | 0.160% |
| backward | causal | 32768 | 11.189000 | 11.187545 | -0.013% | 0.157% |
| backward | non-causal | 1024 | 0.941818 | 0.942636 | +0.087% | 0.227% |
| backward | non-causal | 2048 | 1.679364 | 1.676909 | -0.146% | 0.551% |
| backward | non-causal | 4096 | 3.026182 | 3.024818 | -0.045% | 0.214% |
| backward | non-causal | 8192 | 5.688364 | 5.692727 | +0.077% | 0.170% |
| backward | non-causal | 16384 | 11.101455 | 11.106909 | +0.049% | 0.082% |
| backward | non-causal | 32768 | 21.673545 | 21.676273 | +0.013% | 0.041% |
| forward | causal | 1024 | 0.175545 | 0.175545 | +0.000% | 3.957% |
| forward | causal | 2048 | 0.258273 | 0.255455 | -1.091% | 2.704% |
| forward | causal | 4096 | 0.458091 | 0.457636 | -0.099% | 2.871% |
| forward | causal | 8192 | 0.834364 | 0.843636 | +1.111% | 1.827% |
| forward | causal | 16384 | 1.574182 | 1.569727 | -0.283% | 0.560% |
| forward | causal | 32768 | 3.101364 | 3.103909 | +0.082% | 0.255% |
| forward | non-causal | 1024 | 0.230727 | 0.228636 | -0.906% | 1.994% |
| forward | non-causal | 2048 | 0.426000 | 0.427000 | +0.235% | 1.926% |
| forward | non-causal | 4096 | 0.811727 | 0.822000 | +1.266% | 1.852% |
| forward | non-causal | 8192 | 1.538273 | 1.537545 | -0.047% | 0.839% |
| forward | non-causal | 16384 | 3.029273 | 3.031273 | +0.066% | 0.387% |
| forward | non-causal | 32768 | 5.955182 | 5.954727 | -0.008% | 0.133% |

## Rebase Decisions

- Replayed the next seven main commits while preserving the HD256 training implementation and excluding inference-only feature work.
- Retained upstream zero-length safety semantics; the historical HD256 branch supplies its independent zero-length implementation in a later stage.
- The first distributed UT attempt had one shard stop producing progress on one worker. A 61-nodeid one-GPU replay covering the stall location passed in 495 seconds, and a clean full 256-shard rerun passed all 200629 selected nodeids.
- The first seven-repeat performance sample had one 1K forward point at 5.295% CV; an eleven-repeat same-toolchain fixed-path rerun passed all 24 points.
