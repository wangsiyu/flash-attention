#!/usr/bin/env python3
"""Focused in-process stress runner for a confirmed hanging UT case.

Edit DEFAULT_MODULE, DEFAULT_FUNCTION, and DEFAULT_CASE below, or pass a
case file with MODULE/FUNCTION/CASE definitions via --case-file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Fill this block with the exact UT case after a monitored gate hang.
DEFAULT_MODULE = "test_flash_attn"
DEFAULT_FUNCTION = "test_flash_attn_varlen_output"
DEFAULT_CASE: dict[str, Any] = {
    "seqlen_q": 113,
    "seqlen_k": 211,
    "d": 256,
    "add_unused_qkv": False,
    "causal": False,
    "local_enum": 0,
    "softcap": 15.0,
    "deterministic": True,
    "has_qv": False,
    "has_learnable_sink": False,
    "mha_type": "gqa",
    "dtype": "torch.bfloat16",
    "varlen_mode": "random",
    "zero_lengths_q": True,
    "zero_lengths_k": False,
    "unpad_q": True,
    "unpad_kv": True,
}


CHILD_CODE = r"""
import contextlib
import importlib
import json
import os
import sys
import time
from datetime import datetime

import torch


def resolve_value(value):
    if isinstance(value, str) and value.startswith("torch."):
        return getattr(torch, value.split(".", 1)[1])
    if isinstance(value, list):
        return [resolve_value(v) for v in value]
    if isinstance(value, dict):
        return {k: resolve_value(v) for k, v in value.items()}
    return value


repo = os.environ["HANG_STRESS_REPO"]
module_name = os.environ["HANG_STRESS_MODULE"]
function_name = os.environ["HANG_STRESS_FUNCTION"]
case = resolve_value(json.loads(os.environ["HANG_STRESS_CASE_JSON"]))
label = os.environ["HANG_STRESS_LABEL"]
max_iters = int(os.environ["HANG_STRESS_MAX_ITERS"])
max_seconds = float(os.environ["HANG_STRESS_MAX_SECONDS"])
suppress_case_output = os.environ["HANG_STRESS_SUPPRESS_CASE_OUTPUT"] == "1"

sys.path.insert(0, os.path.join(repo, "tests", "cute"))
module = importlib.import_module(module_name)
func = getattr(module, function_name)

print(f"[{label}] imported module={module_name} function={function_name}", flush=True)
print(f"[{label}] first iteration may compile; later iterations stay in this process", flush=True)
print(f"[{label}] max_iters={max_iters} max_seconds={max_seconds}", flush=True)
print(f"[{label}] case={case}", flush=True)

start_all = time.time()
devnull = open(os.devnull, "w")
for i in range(1, max_iters + 1):
    now = time.time()
    if now - start_all >= max_seconds:
        print(f"[{label}] TIME_LIMIT before_iter={i} elapsed={now - start_all:.6f}", flush=True)
        break
    wall = datetime.now().isoformat(timespec="seconds")
    print(f"[{label}] START iter={i} epoch={now:.6f} wall={wall}", flush=True)
    iter_start = time.time()
    if suppress_case_output:
        with contextlib.redirect_stdout(devnull):
            func(**case)
    else:
        func(**case)
    torch.cuda.synchronize()
    end = time.time()
    wall = datetime.now().isoformat(timespec="seconds")
    print(f"[{label}] END iter={i} seconds={end - iter_start:.6f} epoch={end:.6f} wall={wall}", flush=True)
print(f"[{label}] COMPLETE elapsed={time.time() - start_all:.6f}", flush=True)
"""


@dataclass
class Job:
    gpu: str
    run: int
    label: str
    proc: subprocess.Popen[str]
    log: Path
    start_epoch: int
    status: str = "running"
    gdb_log: Path | None = None
    last_start_iter: int | None = None
    last_end_iter: int | None = None
    hang_iter: int | None = None
    hang_start_epoch: int | None = None
    end_epoch: int | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def jsonable(value: Any) -> Any:
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_case(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    module = args.module or DEFAULT_MODULE
    function = args.function or DEFAULT_FUNCTION
    case: dict[str, Any] = dict(DEFAULT_CASE)

    if args.case_file:
        data = runpy.run_path(args.case_file)
        module = args.module or data.get("MODULE", module)
        function = args.function or data.get("FUNCTION", function)
        case = dict(data["CASE"])

    if args.case_json:
        case = json.loads(args.case_json)

    return module, function, case


def source_hash(repo: Path, relpath: str) -> str:
    path = repo / relpath
    if not path.exists():
        return f"missing {relpath}"
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() + f"  {relpath}"


def start_job(
    *,
    repo: Path,
    gpu: str,
    run: int,
    module: str,
    function: str,
    case: dict[str, Any],
    log_dir: Path,
    max_iters: int,
    max_seconds: float,
    suppress_case_output: bool,
) -> Job:
    label = f"hang-stress-gpu{gpu}-run{run}"
    log = log_dir / f"gpu{gpu}_run{run}.stress.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["HANG_STRESS_REPO"] = str(repo)
    env["HANG_STRESS_MODULE"] = module
    env["HANG_STRESS_FUNCTION"] = function
    env["HANG_STRESS_CASE_JSON"] = json.dumps(jsonable(case), sort_keys=True)
    env["HANG_STRESS_LABEL"] = label
    env["HANG_STRESS_MAX_ITERS"] = str(max_iters)
    env["HANG_STRESS_MAX_SECONDS"] = str(max_seconds)
    env["HANG_STRESS_SUPPRESS_CASE_OUTPUT"] = "1" if suppress_case_output else "0"
    env["PYTHONPATH"] = f"{repo / 'tests' / 'cute'}:{env.get('PYTHONPATH', '')}"
    with log.open("w") as out:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD_CODE],
            cwd=repo,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return Job(gpu=gpu, run=run, label=label, proc=proc, log=log, start_epoch=int(time.time()))


def last_progress(job: Job) -> tuple[int | None, int | None, int | None]:
    try:
        text = job.log.read_text(errors="replace")
    except FileNotFoundError:
        return None, None, None
    starts = list(re.finditer(r"START iter=(\d+) epoch=(\d+)\.", text))
    ends = list(re.finditer(r"END iter=(\d+) seconds=[0-9.]+ epoch=(\d+)\.", text))
    start_iter = int(starts[-1].group(1)) if starts else None
    start_epoch = int(starts[-1].group(2)) if starts else None
    end_iter = int(ends[-1].group(1)) if ends else None
    return start_iter, end_iter, start_epoch


def capture_gdb(repo: Path, pid: int, log: Path) -> None:
    helper = repo / "harness" / "harness" / "hang_detect_fix" / "cuda_gdb_snapshot.sh"
    subprocess.run(
        ["bash", str(helper), str(pid), str(log)],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=180,
        check=False,
    )


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def monitor_jobs(
    *,
    repo: Path,
    jobs: list[Job],
    log_dir: Path,
    hang_seconds: int,
    compile_hang_seconds: int,
    keep_hung: bool,
    no_gdb: bool,
) -> None:
    active = set(range(len(jobs)))
    while active:
        time.sleep(2)
        for idx in list(active):
            job = jobs[idx]
            start_iter, end_iter, start_epoch = last_progress(job)
            job.last_start_iter = start_iter
            job.last_end_iter = end_iter
            ret = job.proc.poll()
            if ret is not None:
                job.end_epoch = int(time.time())
                job.status = "pass" if ret == 0 else f"exit_{ret}"
                active.remove(idx)
                continue
            if start_iter is None:
                continue
            finished = end_iter or 0
            if start_iter <= finished:
                continue
            threshold = compile_hang_seconds if start_iter == 1 else hang_seconds
            iter_start_epoch = start_epoch or job.start_epoch
            if int(time.time()) - iter_start_epoch < threshold:
                continue
            job.status = "hang"
            job.hang_iter = start_iter
            job.hang_start_epoch = iter_start_epoch
            job.gdb_log = log_dir / f"gpu{job.gpu}_run{job.run}.gdb_hang_pid{job.proc.pid}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            if not no_gdb:
                capture_gdb(repo, job.proc.pid, job.gdb_log)
            if not keep_hung:
                stop_process(job.proc)
            job.end_epoch = int(time.time())
            active.remove(idx)


def write_summary(summary: Path, jobs: list[Job]) -> None:
    lines = [
        "gpu\trun\tpid\tstatus\tlast_start_iter\tlast_end_iter\thang_iter\t"
        "start_epoch\tend_epoch\telapsed_s\tgdb_log\tstress_log"
    ]
    for job in jobs:
        end_epoch = job.end_epoch or int(time.time())
        lines.append(
            "\t".join(
                [
                    job.gpu,
                    str(job.run),
                    str(job.proc.pid),
                    job.status,
                    str(job.last_start_iter or ""),
                    str(job.last_end_iter or ""),
                    str(job.hang_iter or ""),
                    str(job.start_epoch),
                    str(end_epoch),
                    str(end_epoch - job.start_epoch),
                    str(job.gdb_log or ""),
                    str(job.log),
                ]
            )
        )
    summary.write_text("\n".join(lines) + "\n")


def print_stats(jobs: list[Job]) -> None:
    hangs = [job for job in jobs if job.status == "hang"]
    no_hangs = [job for job in jobs if job.status != "hang"]
    print(f"[hang-stress] total={len(jobs)} hangs={len(hangs)} no_hang={len(no_hangs)}")
    if hangs:
        iters = [job.hang_iter for job in hangs if job.hang_iter is not None]
        elapsed = [(job.end_epoch or int(time.time())) - job.start_epoch for job in hangs]
        print(
            "[hang-stress] hang_iter "
            f"min={min(iters)} mean={statistics.mean(iters):.2f} "
            f"median={statistics.median(iters):.2f} max={max(iters)}"
        )
        print(
            "[hang-stress] elapsed_s "
            f"min={min(elapsed)} mean={statistics.mean(elapsed):.2f} "
            f"median={statistics.median(elapsed):.2f} max={max(elapsed)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo_root())
    parser.add_argument("--module", default=None)
    parser.add_argument("--function", default=None)
    parser.add_argument("--case-json", default=None)
    parser.add_argument("--case-file", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--runs-per-gpu", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--hang-seconds", type=int, default=30)
    parser.add_argument("--compile-hang-seconds", type=int, default=180)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--keep-hung", action="store_true")
    parser.add_argument("--no-gdb", action="store_true")
    parser.add_argument("--show-case-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    module, function, case = load_case(args)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_root = args.log_root or (repo / "harness" / "logs" / "hang_detect_fix")
    log_dir = log_root / f"stress_case_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "case.json").write_text(json.dumps(jsonable(case), indent=2, sort_keys=True) + "\n")
    (log_dir / "source_hash.txt").write_text(
        source_hash(repo, "flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py") + "\n"
    )
    (log_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")

    all_jobs: list[Job] = []
    for run in range(1, args.runs_per_gpu + 1):
        jobs = [
            start_job(
                repo=repo,
                gpu=gpu,
                run=run,
                module=module,
                function=function,
                case=case,
                log_dir=log_dir,
                max_iters=args.max_iters,
                max_seconds=args.max_seconds,
                suppress_case_output=not args.show_case_output,
            )
            for gpu in gpus
        ]
        all_jobs.extend(jobs)
        for job in jobs:
            print(f"[hang-stress] launched gpu={job.gpu} run={job.run} pid={job.proc.pid} log={job.log}")
        monitor_jobs(
            repo=repo,
            jobs=jobs,
            log_dir=log_dir,
            hang_seconds=args.hang_seconds,
            compile_hang_seconds=args.compile_hang_seconds,
            keep_hung=args.keep_hung,
            no_gdb=args.no_gdb,
        )
        write_summary(log_dir / "summary.tsv", all_jobs)
        print_stats(all_jobs)

    write_summary(log_dir / "summary.tsv", all_jobs)
    print(f"[hang-stress] logs={log_dir}")
    print_stats(all_jobs)
    return 1 if any(job.status == "hang" for job in all_jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
