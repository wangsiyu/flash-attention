#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time


def cpu_worker(deadline: float) -> None:
    value = os.getpid() & 0xFFFFFFFF
    while time.time() < deadline:
        for _ in range(200_000):
            value = (value * 1664525 + 1013904223) & 0xFFFFFFFF


def gpu_worker(device: int, deadline: float) -> None:
    import torch

    torch.cuda.set_device(device)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    while time.time() < deadline:
        c = a @ b
        torch.cuda.synchronize()
        a, c = c, a


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--cpu-workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    duration = max(1, args.duration)
    deadline = time.time() + duration
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    cpu_workers = max(1, args.cpu_workers)

    print(
        f"[keepalive] load start duration={duration}s "
        f"cpu_workers={cpu_workers} gpus={gpus}",
        flush=True,
    )

    procs: list[mp.Process] = []
    for _ in range(cpu_workers):
        proc = mp.Process(target=cpu_worker, args=(deadline,))
        proc.start()
        procs.append(proc)
    for gpu in gpus:
        proc = mp.Process(target=gpu_worker, args=(gpu, deadline))
        proc.start()
        procs.append(proc)

    exit_code = 0
    try:
        for proc in procs:
            proc.join()
            if proc.exitcode not in (0, None):
                exit_code = proc.exitcode or 1
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)

    print(f"[keepalive] load done exit={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
