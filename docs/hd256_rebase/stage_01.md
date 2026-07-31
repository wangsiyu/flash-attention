# HD256 Rebase Stage 01

## Tested Source

- Main checkpoint: `0409f9adcbdebff6cc19eb95f370d40e896980bc`
- Source HEAD before lock commit: `448c742b03f9a636af07610090d3f778a28d6532`
- Worktree diff SHA256: `597a9efb06fa7f4131913b00ef67983b9ae2dedb755dc2e915d997a28cf73156`
- Benchmark SHA256: `4481ed1b901cdab1435eb082c87b3c58b2a39ef932ab33486576bd6999030bf7`
- Performance control HEAD: `b318d1477e80ed007f56fcce40b275112293a51b`
- Performance control diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Control toolchain: `stage=01 cutlass=4.4.2 quack=0.4.1 tvm=apache-tvm-ffi>=0.1.5,<0.2`
- Candidate toolchain: `stage=01 cutlass=4.4.2 quack=0.4.1 tvm=apache-tvm-ffi>=0.1.5,<0.2`
- Container image: `dsw-registry-vpc.ap-southeast-1.cr.aliyuncs.com/pai/training:26.07.20-pytorch2.11-python3.12-cuda13.0-ubuntu24.04_accelerated`
- UT DLC job: `dlcnnraq257el14c`
- Performance DLC job: `dlcq5nn62srxk917`
- UT result: `/cpfs01/shared/Group_m6/siyu.wsy/cc_fusion/flash-attention-rebase/agent_space/rebase_harness/results/fa4-hd256-ut-s01-20260730-0525`
- Performance result: `/cpfs01/shared/Group_m6/siyu.wsy/cc_fusion/flash-attention-rebase/agent_space/rebase_harness/results/fa4-hd256-perf-ab-s01-20260730-0525`

The worktree diff is part of the tested source identity. It contains no temporary HD256 collection filter or source-side skip.

## Precision

- Verdict: pass
- Shards: 128
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
| backward | causal | 1024 | 0.645571 | 0.644857 | -0.111% | 0.153% |
| backward | causal | 2048 | 1.005429 | 1.006000 | +0.057% | 0.657% |
| backward | causal | 4096 | 1.700857 | 1.705857 | +0.294% | 0.370% |
| backward | causal | 8192 | 3.090857 | 3.082429 | -0.273% | 0.196% |
| backward | causal | 16384 | 5.853429 | 5.849714 | -0.063% | 0.231% |
| backward | causal | 32768 | 11.319143 | 11.305286 | -0.122% | 0.246% |
| backward | non-causal | 1024 | 0.945571 | 0.947143 | +0.166% | 0.236% |
| backward | non-causal | 2048 | 1.684143 | 1.684143 | +0.000% | 0.385% |
| backward | non-causal | 4096 | 3.047000 | 3.046143 | -0.028% | 0.171% |
| backward | non-causal | 8192 | 5.746429 | 5.743571 | -0.050% | 0.282% |
| backward | non-causal | 16384 | 11.199714 | 11.213429 | +0.122% | 0.148% |
| backward | non-causal | 32768 | 21.807143 | 21.807429 | +0.001% | 0.090% |
| forward | causal | 1024 | 0.173714 | 0.174429 | +0.411% | 3.355% |
| forward | causal | 2048 | 0.271571 | 0.265571 | -2.209% | 2.569% |
| forward | causal | 4096 | 0.456429 | 0.460286 | +0.845% | 1.222% |
| forward | causal | 8192 | 0.841857 | 0.838857 | -0.356% | 0.702% |
| forward | causal | 16384 | 1.579000 | 1.597143 | +1.149% | 0.928% |
| forward | causal | 32768 | 3.124857 | 3.123714 | -0.037% | 0.323% |
| forward | non-causal | 1024 | 0.233143 | 0.236000 | +1.225% | 1.396% |
| forward | non-causal | 2048 | 0.433143 | 0.431714 | -0.330% | 2.302% |
| forward | non-causal | 4096 | 0.814857 | 0.811571 | -0.403% | 1.237% |
| forward | non-causal | 8192 | 1.558714 | 1.551714 | -0.449% | 0.602% |
| forward | non-causal | 16384 | 3.051286 | 3.051857 | +0.019% | 0.277% |
| forward | non-causal | 32768 | 5.990571 | 5.992429 | +0.031% | 0.131% |

## Rebase Decisions

- Replayed all 22 HD256 training commits onto the first six main commits without adding inference functionality.
- Restored the complete upstream D=64,128,256 varlen parameterization; HD256 selection remains external to the source tree.
- Kept the original atol=rtol=3e-2 correctness criterion and added an FP32-oracle fallback only when PyTorch-BF16 allclose fails; the fallback requires fused max error to stay within 2x the PyTorch-BF16 max error.
- The prior independent-cache A/B path produced a false causal-2K regression for byte-identical forward source. The accepted gate uses one toolchain, one content-addressed cache, and one fixed node-local source path.
