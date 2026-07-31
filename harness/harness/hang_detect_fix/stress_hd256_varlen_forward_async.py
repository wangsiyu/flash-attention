#!/usr/bin/env python3
"""Asynchronous pressure for the HD256 varlen forward hang path."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from collections import deque
from pathlib import Path
from typing import Any

import torch

from flash_attn.cute.interface import flash_attn_varlen_func
from flash_attn.cute.testing import generate_qkv, generate_random_padding_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--batch-launches", type=int, default=64)
    parser.add_argument("--max-pending-batches", type=int, default=8)
    parser.add_argument("--stall-seconds", type=float, default=90.0)
    parser.add_argument("--compile-stall-seconds", type=float, default=300.0)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--stall-file", type=Path, required=True)
    args = parser.parse_args()
    if args.max_seconds < 600:
        parser.error("--max-seconds must be at least 600")
    if args.batch_launches < 1 or args.max_pending_batches < 1:
        parser.error("batch and pending limits must be positive")
    return args


def write_state(path: Path, **state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")


def make_case() -> tuple[
    Callable[[], tuple[torch.Tensor, torch.Tensor | None]], dict[str, Any]
]:
    seqlen_q = 1024
    seqlen_k = 1024  # local attention rewrites the node-id value 1023 to seqlen_q
    head_dim = 256
    batch_size = 7
    num_heads = 6
    num_kv_heads = 1
    softcap = 15.0
    dtype = torch.bfloat16
    device = "cuda"

    seed = seqlen_q + seqlen_k + head_dim + 1
    random.seed(seed)
    torch.random.manual_seed(seed)

    q_ref = torch.randn(
        batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype
    )
    q_ref = (q_ref * softcap / 4).detach().requires_grad_()
    k_ref = torch.randn(
        batch_size, seqlen_k, num_kv_heads, head_dim, device=device, dtype=dtype
    ).requires_grad_()
    v_ref = torch.randn(
        batch_size, seqlen_k, num_kv_heads, head_dim, device=device, dtype=dtype
    ).requires_grad_()
    window_size = tuple(random.randrange(0, seqlen_k) for _ in range(2))
    learnable_sink = torch.randn(num_heads, dtype=torch.bfloat16, device=device)

    q, k, v = [tensor.detach().requires_grad_() for tensor in (q_ref, k_ref, v_ref)]
    query_padding_mask = generate_random_padding_mask(
        seqlen_q,
        batch_size,
        device,
        mode="random",
        zero_lengths=False,
    )
    key_padding_mask = query_padding_mask
    (
        q_unpad,
        _,
        _,
        _,
        cu_seqlens_q,
        _,
        _,
        seqused_k,
        _,
        _,
        _,
        k,
        v,
        _,
        _,
        _,
        _,
    ) = generate_qkv(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        qv=None,
        kvpacked=False,
        query_unused_mask=None,
        key_unused_mask=None,
    )
    q_unpad = q_unpad.detach().to(dtype).requires_grad_()

    def launch() -> tuple[torch.Tensor, torch.Tensor | None]:
        return flash_attn_varlen_func(
            q_unpad,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            seqused_q=None,
            seqused_k=seqused_k,
            causal=False,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=1,
            pack_gqa=False,
            deterministic=False,
        )

    case = {
        "batch_size": batch_size,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "window_size": window_size,
        "softcap": softcap,
        "unpad_q": True,
        "unpad_kv": False,
        "has_learnable_sink": True,
    }
    return launch, case


def wait_event(
    event: torch.cuda.Event,
    *,
    timeout: float,
    state_file: Path,
    stall_file: Path,
    phase: str,
    enqueued: int,
    completed: int,
) -> None:
    wait_start = time.monotonic()
    last_state = 0.0
    while not event.query():
        now = time.monotonic()
        if now - last_state >= 1.0:
            write_state(
                state_file,
                phase=phase,
                enqueued=enqueued,
                completed=completed,
                event_wait_seconds=now - wait_start,
                epoch=time.time(),
            )
            last_state = now
        if now - wait_start >= timeout:
            state = {
                "phase": "stall",
                "stalled_phase": phase,
                "enqueued": enqueued,
                "completed": completed,
                "event_wait_seconds": now - wait_start,
                "epoch": time.time(),
            }
            write_state(state_file, **state)
            write_state(stall_file, **state)
            while True:
                time.sleep(1)
        time.sleep(0.001)


def main() -> int:
    args = parse_args()
    launch, case = make_case()
    write_state(args.state_file, phase="compiled_input", case=case, epoch=time.time())

    # Separate compilation/setup latency from the post-compilation pressure window.
    output, lse = launch()
    del output, lse
    warmup_done = torch.cuda.Event()
    warmup_done.record()
    wait_event(
        warmup_done,
        timeout=args.compile_stall_seconds,
        state_file=args.state_file,
        stall_file=args.stall_file,
        phase="warmup",
        enqueued=1,
        completed=0,
    )

    pending: deque[tuple[torch.cuda.Event, int, float]] = deque()
    pressure_start = time.monotonic()
    enqueued = 1
    completed = 1
    batch = 0
    last_state = 0.0
    while time.monotonic() - pressure_start < args.max_seconds:
        for _ in range(args.batch_launches):
            output, lse = launch()
            del output, lse
            enqueued += 1
        batch += 1
        event = torch.cuda.Event()
        event.record()
        pending.append((event, enqueued, time.monotonic()))

        while pending and pending[0][0].query():
            _, completed, _ = pending.popleft()
        if len(pending) >= args.max_pending_batches:
            event, target, _ = pending[0]
            wait_event(
                event,
                timeout=args.stall_seconds,
                state_file=args.state_file,
                stall_file=args.stall_file,
                phase="pressure",
                enqueued=enqueued,
                completed=completed,
            )
            pending.popleft()
            completed = target

        now = time.monotonic()
        if now - last_state >= 1.0:
            write_state(
                args.state_file,
                phase="pressure",
                batch=batch,
                enqueued=enqueued,
                completed=completed,
                pending_batches=len(pending),
                pressure_seconds=now - pressure_start,
                epoch=time.time(),
            )
            last_state = now

    while pending:
        event, target, _ = pending.popleft()
        wait_event(
            event,
            timeout=args.stall_seconds,
            state_file=args.state_file,
            stall_file=args.stall_file,
            phase="drain",
            enqueued=enqueued,
            completed=completed,
        )
        completed = target

    write_state(
        args.state_file,
        phase="pass",
        enqueued=enqueued,
        completed=completed,
        pressure_seconds=time.monotonic() - pressure_start,
        epoch=time.time(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
