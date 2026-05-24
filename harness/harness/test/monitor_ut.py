#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


FAIL_RE = re.compile(
    r"(^|\n)(FAILED\s+|FAIL\s+)|=+\s+[^\n]*\b([1-9]\d*)\s+failed\b",
    re.MULTILINE,
)
CRASH_RE = re.compile(r"Segfault|Segmentation fault|SIGSEGV|SIGBUS|Illegal memory access", re.IGNORECASE)
CASE_RE = re.compile(r"^(FAILED|ERROR)\s+(\S+)", re.MULTILINE)
TEST_NODE_RE = re.compile(r"(?P<node>\S*tests/\S+\.py::\S+)")
RUNNER_START_RE = re.compile(r"START\s+(?P<name>.*?)\s+->\s+(?P<log>\S+)")
SCORE_MOD_GROUP = "hd256_score_mod"
LOG_GROUPS = {
    "ut_hd256_output.log": "hd256_output",
    "ut_hd256_varlen_output.log": "hd256_varlen_output",
    "ut_hd256_score_mod.log": SCORE_MOD_GROUP,
    "ut_hd256_mask_mod_block_sparse.log": "hd256_mask_mod_block_sparse",
    "ut_hd256_dlse.log": "hd256_dlse",
    "ut_varlen.log": "varlen",
}


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        bufsize=1,
        env=env,
    )


def kill_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(2)
    except ProcessLookupError:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGSTOP)
    except ProcessLookupError:
        pass


def gpu_util_for_process_group(proc: subprocess.Popen[str]) -> int | None:
    pids = set(process_group_pids(proc))
    if not pids:
        return None
    try:
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        return None

    active_uuids: set[str] = set()
    for line in apps.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid in pids:
            active_uuids.add(parts[0])
    if not active_uuids:
        return None

    try:
        util_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        return None

    vals: list[int] = []
    for line in util_out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or parts[0] not in active_uuids:
            continue
        try:
            vals.append(int(parts[1]))
        except ValueError:
            continue
    return max(vals) if vals else None


def process_group_pids(proc: subprocess.Popen[str]) -> list[int]:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,pgid=,comm=,args="],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        if pgid == proc.pid:
            pids.append(pid)
    return pids


def cuda_gdb_snapshot(proc: subprocess.Popen[str], log_dir: Path, note) -> None:
    pids = process_group_pids(proc)
    if not pids:
        note("hang debug: no process-group pids found for cuda-gdb attach")
        return
    ps_log = log_dir / "hang_processes.txt"
    try:
        ps_log.write_text(
            subprocess.check_output(
                ["ps", "-fp", ",".join(str(pid) for pid in pids)],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=3,
            )
        )
    except Exception as exc:
        ps_log.write_text(f"failed to capture ps output: {exc}\n")

    candidates: list[int] = []
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,pgid=,comm=,args="],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        for line in out.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) < 4:
                continue
            pid = int(parts[0])
            pgid = int(parts[1])
            args = parts[3]
            if pgid == proc.pid and "python" in args and "pytest" in args:
                candidates.append(pid)
    except Exception:
        candidates = []
    if not candidates:
        candidates = [pid for pid in pids if pid != proc.pid]
    if not candidates:
        note("hang debug: no attach candidate found")
        return

    pid = candidates[-1]
    gdb_log = log_dir / f"gdb_hang_pid{pid}.log"
    helper = log_dir.parents[1] / "harness" / "hang_detect_fix" / "cuda_gdb_snapshot.sh"
    note(f"hang debug: attaching cuda-gdb to live pid={pid}, log={gdb_log}")
    try:
        result = subprocess.run(
            ["bash", str(helper), str(pid), str(gdb_log)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        (log_dir / "cuda_gdb_snapshot_stdout.log").write_text(result.stdout)
        note(f"hang debug: cuda-gdb snapshot exit={result.returncode}")
    except subprocess.TimeoutExpired as exc:
        (log_dir / "cuda_gdb_snapshot_stdout.log").write_text(
            (exc.stdout or "") + "\n[cuda-gdb snapshot timed out]\n"
        )
        note("hang debug: cuda-gdb snapshot timed out")
    except Exception as exc:
        (log_dir / "cuda_gdb_snapshot_stdout.log").write_text(f"{exc}\n")
        note(f"hang debug: cuda-gdb snapshot failed: {exc}")


def capture_text(cmd: list[str], *, timeout: int = 5) -> str:
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except Exception as exc:
        return f"failed to run {' '.join(cmd)}: {exc}\n"


def capture_crash_artifacts(
    proc: subprocess.Popen[str],
    log_dir: Path,
    *,
    current_group: str | None,
    current_case: str | None,
    current_log_path: Path | None,
    reason: str,
) -> None:
    report = log_dir / "crash_report.txt"
    pids = process_group_pids(proc)
    report.write_text(
        f"reason={reason}\n"
        f"group={current_group or 'UNKNOWN'}\n"
        f"case={current_case or 'UNKNOWN'}\n"
        f"log={current_log_path or 'UNKNOWN'}\n"
        f"runner_pid={proc.pid}\n"
        f"process_group_pids={','.join(str(pid) for pid in pids) or 'NONE'}\n"
        "The process group is preserved. Do not rerun before debugging this run directory.\n"
    )
    (log_dir / "crash_ps.txt").write_text(
        capture_text(["ps", "-eo", "pid,ppid,pgid,stat,etime,pcpu,pmem,args"])
    )
    (log_dir / "crash_nvidia_smi.txt").write_text(capture_text(["nvidia-smi"], timeout=10))
    dmesg_lines = capture_text(["dmesg"], timeout=10).splitlines()[-300:]
    (log_dir / "crash_dmesg_tail.txt").write_text("\n".join(dmesg_lines) + "\n")


def read_new(path: Path, offsets: dict[Path, int]) -> str:
    if not path.exists():
        return ""
    offset = offsets.get(path, 0)
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size < offset:
        offset = 0
    with path.open("r", errors="replace") as f:
        f.seek(offset)
        data = f.read()
        offsets[path] = f.tell()
    return data


def extract_cases(text: str) -> list[str]:
    return [m.group(2) for m in CASE_RE.finditer(text)]


def extract_last_test_node(text: str) -> str | None:
    matches = list(TEST_NODE_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group("node")


def normalize_group_name(name: str) -> str:
    return name.strip().replace("/", "_").replace(" ", "_")


def extract_runner_group(line: str) -> tuple[str, Path | None] | None:
    match = RUNNER_START_RE.search(line)
    if not match:
        return None
    return normalize_group_name(match.group("name")), Path(match.group("log"))


def group_from_log_path(path: Path) -> str | None:
    return LOG_GROUPS.get(path.name)


def cleanup_hang_debug_logs(test_log_dir: Path, hang_fix_log_dir: Path) -> None:
    patterns = [
        "hang_report.txt",
        "hang_processes.txt",
        "slow_case_report.txt",
        "cuda_gdb_snapshot_stdout.log",
        "gdb_hang_pid*.log",
        "gdb_hang.log",
    ]
    test_log_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in test_log_dir.glob(pattern):
            unlink_if_possible(path)

    hang_fix_log_dir.mkdir(parents=True, exist_ok=True)


def log_files(log_dir: Path) -> list[Path]:
    visible = list(log_dir.glob("*.log"))
    hidden = list(log_dir.glob(".*.log"))
    return sorted({*visible, *hidden})


def unlink_if_possible(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def append_slow_case_report(
    path: Path,
    *,
    elapsed: float,
    idle: float,
    util: int | None,
    current_case: str | None,
    log_path: Path | None,
    proc: subprocess.Popen[str],
) -> None:
    try:
        ps = subprocess.check_output(
            ["ps", "-o", "pid,ppid,pgid,stat,etime,pcpu,pmem,args", "-g", str(proc.pid)],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=3,
        )
    except Exception as exc:
        ps = f"failed to capture ps output: {exc}\n"
    kind = "compile-like" if util is None or util < 50 else "gpu-runtime"
    with path.open("a") as f:
        f.write(
            f"kind={kind} elapsed={elapsed:.1f}s idle={idle:.1f}s gpu_util={util} "
            f"log={log_path or 'UNKNOWN'}\n"
        )
        f.write(f"current_case={current_case or 'UNKNOWN'}\n")
        f.write(ps)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--start-at",
        choices=[
            "hd256_output",
            "hd256_varlen_output",
            "hd256_score_mod",
            "hd256_mask_mod_block_sparse",
            "hd256_dlse",
            "varlen",
        ],
        help="Start the monitored gate at a later HD256 group after earlier groups already passed.",
    )
    parser.add_argument("--hang-seconds", type=int, default=180)
    parser.add_argument("--slow-case-seconds", type=int, default=45)
    parser.add_argument("--score-mod-slow-case-seconds", type=int, default=450)
    parser.add_argument("--sample-seconds", type=int, default=5)
    parser.add_argument("--gpu-100-samples", type=int, default=3)
    parser.add_argument("--run-id", help="Optional run directory name under harness/logs/test.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    harness_root = script_dir.parents[1]
    repo = harness_root.parent
    base_test_log_dir = harness_root / "logs" / "test"
    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    test_log_dir = base_test_log_dir / run_id
    test_runner_log_dir = test_log_dir
    test_log_dir.mkdir(parents=True, exist_ok=True)
    base_test_log_dir.mkdir(parents=True, exist_ok=True)
    (base_test_log_dir / "latest_run.txt").write_text(str(test_log_dir) + "\n")
    latest_link = base_test_log_dir / "latest"
    try:
        if latest_link.is_symlink() or latest_link.is_file():
            latest_link.unlink()
        latest_link.symlink_to(test_log_dir, target_is_directory=True)
    except OSError:
        pass
    monitor_log = test_log_dir / "monitor.log"
    failure_cases = test_log_dir / "failure_cases.txt"
    hang_report = test_log_dir / "hang_report.txt"
    slow_case_report = test_log_dir / "slow_case_report.txt"
    hang_fix_log_dir = harness_root / "logs" / "hang_detect_fix"

    cmd = ["bash", "harness/harness/test/run_hd256_ut.sh"]
    if args.preflight_only:
        cmd.append("--preflight-only")
    if args.start_at:
        cmd.extend(["--start-at", args.start_at])

    with monitor_log.open("w") as log:
        def note(msg: str) -> None:
            line = f"[monitor] {msg}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        note(f"repo={repo}")
        note(f"run_dir={test_log_dir}")
        note(f"cmd={' '.join(cmd)}")
        test_runner_log_dir.mkdir(parents=True, exist_ok=True)
        cleanup_hang_debug_logs(test_log_dir, hang_fix_log_dir)
        for old_log in log_files(test_runner_log_dir):
            if old_log.name != "monitor.log" and not args.start_at:
                unlink_if_possible(old_log)
        unlink_if_possible(failure_cases)
        offsets: dict[Path, int] = {
            path: path.stat().st_size
            for path in log_files(test_runner_log_dir)
            if path != monitor_log
        }
        env = os.environ.copy()
        env["HD256_UT_LOGDIR"] = str(test_runner_log_dir)
        proc = run(cmd, env=env)
        start = time.monotonic()
        last_progress = start
        gpu_100_streak = 0
        collected = ""
        current_group = args.start_at or "hd256_output"
        current_case: str | None = None
        current_log_path: Path | None = None
        slow_case_reported = False

        while True:
            if proc.stdout is not None and select.select([proc.stdout], [], [], 0)[0]:
                line = proc.stdout.readline()
                if line:
                    print(line, end="")
                    log.write(line)
                    log.flush()
                    collected += line
                    last_progress = time.monotonic()
                    slow_case_reported = False
                    group = extract_runner_group(line)
                    if group is not None:
                        current_group, current_log_path = group
                        current_case = None
                    if CRASH_RE.search(line):
                        note("crash signature detected in command output; freezing UT process group for debugging")
                        capture_crash_artifacts(
                            proc,
                            test_log_dir,
                            current_group=current_group,
                            current_case=current_case,
                            current_log_path=current_log_path,
                            reason=line.strip(),
                        )
                        stop_process_group(proc)
                        return 3
                    if FAIL_RE.search(line):
                        note("failure detected in command output; killing UT process group")
                        kill_process_group(proc)
                        return 1

            for path in log_files(test_runner_log_dir):
                if path == monitor_log:
                    continue
                chunk = read_new(path, offsets)
                if chunk:
                    collected += "\n" + chunk
                    last_progress = time.monotonic()
                    slow_case_reported = False
                    path_group = group_from_log_path(path)
                    if path_group is not None:
                        current_group = path_group
                    node = extract_last_test_node(chunk)
                    if node is not None:
                        current_case = node
                        current_log_path = path
                    if CRASH_RE.search(chunk):
                        note(f"crash signature detected in {path}; freezing UT process group for debugging")
                        capture_crash_artifacts(
                            proc,
                            test_log_dir,
                            current_group=current_group,
                            current_case=current_case,
                            current_log_path=current_log_path,
                            reason="crash signature in UT log",
                        )
                        stop_process_group(proc)
                        return 3
                    if FAIL_RE.search(chunk):
                        cases = extract_cases(collected)
                        failure_cases.write_text(
                            "\n".join(cases)
                            + ("\n" if cases else "")
                            + "\nReproduce with:\n"
                            + "\n".join(f"python3 -m pytest -v -s --tb=long {c}" for c in cases)
                            + "\n",
                        )
                        note(f"failure detected in {path}; killing UT process group")
                        kill_process_group(proc)
                        return 1

            if args.preflight_only and proc.poll() is not None:
                return proc.returncode or 0

            now = time.monotonic()
            elapsed = now - start
            idle = now - last_progress
            util = gpu_util_for_process_group(proc)
            slow_case_seconds = (
                args.score_mod_slow_case_seconds
                if current_group == SCORE_MOD_GROUP
                else args.slow_case_seconds
            )
            if util == 100:
                gpu_100_streak += 1
            else:
                gpu_100_streak = 0

            if (
                idle >= slow_case_seconds
                and not slow_case_reported
                and proc.poll() is None
            ):
                note(
                    f"slow case observed: elapsed={elapsed:.1f}s, idle={idle:.1f}s, "
                    f"gpu_util={util}, case={current_case or 'UNKNOWN'}"
                )
                append_slow_case_report(
                    slow_case_report,
                    elapsed=elapsed,
                    idle=idle,
                    util=util,
                    current_case=current_case,
                    log_path=current_log_path,
                    proc=proc,
                )
                slow_case_reported = True

            if (
                current_group != SCORE_MOD_GROUP
                and idle >= args.hang_seconds
                and gpu_100_streak >= args.gpu_100_samples
                and proc.poll() is None
            ):
                msg = (
                    f"hang detected: elapsed={elapsed:.1f}s, "
                    f"idle={idle:.1f}s, "
                    f"gpu_util=100 for {gpu_100_streak} samples"
                )
                hang_report.write_text(
                    msg
                    + f"\ngroup={current_group or 'UNKNOWN'}\n"
                    + f"case={current_case or 'UNKNOWN'}\n"
                    + f"log={current_log_path or 'UNKNOWN'}\n"
                    + "\nCaptured cuda-gdb snapshot on the live broad UT process.\n"
                    + "The process was left alive for source-level analysis; do not rerun before diagnosing this state.\n"
                    + "If source/kernel code is changed after diagnosis, rerun the full gate from the beginning.\n"
                    + "If this is a false hang and no code changed, rerun only this group after stopping the stale process if needed.\n"
                    + "See harness/logs/test/gdb_hang_pid*.log and cuda_gdb_snapshot_stdout.log.\n"
                )
                note(msg + "; capturing cuda-gdb on live UT process")
                cuda_gdb_snapshot(proc, test_log_dir, note)
                note("hang debug complete; leaving UT process group alive for diagnosis")
                return 2

            if proc.poll() is not None:
                rest = proc.stdout.read() if proc.stdout is not None else ""
                if rest:
                    print(rest, end="")
                    log.write(rest)
                    collected += rest
                if CRASH_RE.search(collected):
                    note("crash signature detected after UT process exit; preserving run directory for debugging")
                    capture_crash_artifacts(
                        proc,
                        test_log_dir,
                        current_group=current_group,
                        current_case=current_case,
                        current_log_path=current_log_path,
                        reason="crash signature after process exit",
                    )
                    return 3
                if proc.returncode != 0 and FAIL_RE.search(collected):
                    cases = extract_cases(collected)
                    failure_cases.write_text("\n".join(cases) + ("\n" if cases else ""))
                return proc.returncode or 0

            time.sleep(args.sample_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
