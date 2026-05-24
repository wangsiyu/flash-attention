---
id: benchmark
kind: command
command: /benchmark
script: ../harness/benchmark/run_benchmark.sh
logs: ../harness/logs/benchmark/
---

# Benchmark

HD256 benchmark command with high-repetition timing and previous-run comparison.

## Invocation

| Mode | Command |
| ---- | ------- |
| Full benchmark | `bash harness/harness/benchmark/run_benchmark.sh` |
| Dry run | `bash harness/harness/benchmark/run_benchmark.sh --dry-run` |
| Override full-matrix runs | `BENCHMARK_RUNS=2 bash harness/harness/benchmark/run_benchmark.sh` |
| Override inner timing | `BENCHMARK_REP=500 BENCHMARK_WARMUP=50 bash harness/harness/benchmark/run_benchmark.sh` |
| Override SDPA timing | `BENCHMARK_SDPA_REP=500 BENCHMARK_SDPA_WARMUP=50 bash harness/harness/benchmark/run_benchmark.sh` |

## Fixed Benchmark Command

```bash
python3 harness/harness/benchmark/bench_sm100_hd256.py --suite gate --rep 1000 --warmup 100
```

## Gate Matrix

| Suite | Heads | Lengths |
| ----- | ----- | ------- |
| Dense MHA | `headq:headk = 16:16` | `seqlen = 1024,2048,4096,8192,16384,32768` |
| Dense GQA | `headq:headk = 16:1` | `seqlen = 1024,2048,4096,8192,16384,32768` |
| Varlen GQA | `headq:headk = 16:1` | each total len from `1024` to `32768`, with replicated document lengths `128,256,512,...,total_len` |

Each FA row uses `BENCHMARK_WARMUP=100` warmup calls followed by
`BENCHMARK_REP=1000` measured calls by default. `BENCHMARK_RUNS` defaults to
`1`; do not rerun the full matrix multiple times unless explicitly needed for a
separate investigation. PyTorch SDPA baseline is run once separately over the
same logical workload and written to `sdpa_baseline.log`. Reports include FA
TFLOPS, SDPA TFLOPS, `FA/SDPA`, and `Gap vs SDPA`.
`Gap vs SDPA = FA / SDPA - 1`; positive means FA is faster, negative means FA
trails SDPA.

Varlen cases are deterministic: for a given total length and document length,
`cu_seqlens` is built by repeating that exact document length until the total
length is reached. Do not randomize varlen benchmark lengths. The default
varlen doc list is generated per total length by doubling from `128` through
that total length.

## Environment

| Requirement | Command Or Setting |
| ----------- | ------------------ |
| Editable CuteDSL package | `python3 -m pip install --no-deps -e flash_attn/cute` |
| FA4 import routing | Wrapper runs from `/tmp` and verifies editable `flash_attn.cute.interface`. |
| Benchmark helper imports | Benchmark script must not import FA2 benchmark helpers that require root `flash_attn`. |
| Do not modify | Do not edit `flash_attn/__init__.py` for benchmark import routing. |

## Files

| File | Purpose |
| ---- | ------- |
| `harness/harness/benchmark/bench_sm100_hd256.py` | Benchmark implementation copied into harness. |
| `harness/harness/benchmark/run_benchmark.sh` | Fixed command wrapper, log rotation, default single high-repetition run. |
| `harness/harness/benchmark/compare_benchmark.py` | Parses suite-aware dense/varlen logs, compares current vs previous benchmark rows. |
| `harness/harness/benchmark/export_sass.sh` | Helper to export before/after SASS from provided cubin/shared-object paths. |

## Logs

| Path | Meaning |
| ---- | ------- |
| `harness/harness/logs/benchmark/current/` | Current benchmark run logs and report. |
| `harness/harness/logs/benchmark/previous/` | Previous benchmark run logs used for comparison. |
| `harness/harness/logs/benchmark/current/benchmark_report.md` | Current-vs-previous comparison and regression decision. |
| `harness/harness/logs/benchmark/current/sdpa_baseline.log` | One-pass PyTorch SDPA baseline merged into the report. |
| `harness/harness/logs/benchmark/current/source_stamp.log` | Git head, package versions, and key CuteDSL module path/SHA/mtime embedded into the report. |

Each benchmark generation writes the same source stamp used by UT: package
versions plus path/SHA/mtime for key CuteDSL modules. Check
`source_stamp.log` or the `Source Stamp` section in `benchmark_report.md` to
prove the benchmark used the latest working-tree kernel files.

Only current and previous benchmark generations are retained.

## Result Gate

| Result | Meaning |
| ------ | ------- |
| Exit `0` | Benchmark gate passed or no previous baseline exists yet. |
| Nonzero exit | Benchmark failed or systemic regression detected. Do not lock or commit. |

After benchmark exits `0`, workflow W4 is still incomplete until
`/hd256_hang_stress` finishes `--max-iters 5000` successfully.
