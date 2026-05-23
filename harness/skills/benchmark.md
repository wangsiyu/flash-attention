---
id: benchmark
kind: skill
triggers:
  - benchmark
  - performance
  - perf
  - regression
  - SASS
  - compare-baseline
---

# Benchmark Skill

Performance gate for CuteDSL HD256 work.

## When to Use

| Signal | Action |
| ------ | ------ |
| benchmark, performance, perf | Load `../commands/benchmark.md`. |
| after UT passes | Run benchmark gate before stress, version lock, or commit. |
| regression | Compare high-repetition benchmark results, export SASS or run equivalent experiment analysis. |
| SASS | Use benchmark command helper for before/after SASS export. |

## Benchmark Gate

| Step | Requirement |
| ---- | ----------- |
| B1 Run | Use `../commands/benchmark.md`; do not invent benchmark commands. |
| B2 Record | Store benchmark results under `harness/harness/logs/benchmark/`. |
| B3 Rotate | Keep only current run and previous run for comparison. |
| B4 Compare | Compare current high-repetition benchmark rows against previous benchmark rows. |
| B5 Decide | This round passes only if no systemic performance regression is detected. |
| B6 Stress handoff | Return to `workflow.md` W4 and run `hd256_hang_stress.md` before W5/W6. |

## Required Coverage

| Area | Requirement |
| ---- | ----------- |
| Dense MHA | Include `headq:headk = 16:16` from seqlen `1k` through `32k`. |
| Dense GQA | Include `headq:headk = 16:1` from seqlen `1k` through `32k`. |
| Varlen GQA | Include only `headq:headk = 16:1`; for every total len from `1k` through `32k`, run replicated fixed-doc cases for doc len `128,256,512,...,total_len`. |
| Varlen construction | Do not randomize document lengths; build `cu_seqlens` by repeating the requested fixed document length until the requested total length is reached. Generate doc lengths per total length by doubling from `128` through that total length. |
| Report keys | Compare by suite, direction, mask, total/seqlen, and varlen doc length so GQA/MHA and dense/varlen cases cannot be merged. |
| SDPA baseline | Run PyTorch SDPA once separately for the same logical workload and merge it into the report. Reports must include `SDPA TFLOPS`, `FA/SDPA`, and `Gap vs SDPA`; positive gap means FA is faster. |
| Source stamp | Write git head, package versions, and key CuteDSL module path/SHA/mtime to `source_stamp.log`, then embed it in `benchmark_report.md` so run-to-run noise can be distinguished from code changes. |

## Regression Policy

| Condition | Required Action |
| --------- | --------------- |
| Normal run-to-run noise | Do not treat as regression; rely on the default high-repetition inner timing instead of repeated full-matrix reruns. |
| Systemic regression | Export before/after SASS or perform equivalent experiment analysis. |
| Regression root cause found | Modify/optimize until performance recovers. |
| Regression unresolved | Do not lock version, do not commit, and do not revert code as a substitute for analysis and optimization. |

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| No skipped benchmark | Benchmark is mandatory after UT passes for performance-sensitive work. |
| Latest UT only | Benchmark must run after the latest successful UT, not before it. |
| No benchmark regression | Code cannot pass this round with systemic performance regression. |
| No rollback-to-pass | Do not skip benchmark or revert code just to pass this round; diagnose and optimize until performance recovers. |
| Command handoff | Use `../commands/benchmark.md`; benchmark scripts live under `../harness/benchmark/`. |
