#!/usr/bin/env python3
"""4-GPU HD256 varlen hang stress for direct FA4 low-level calls.

This intentionally calls _flash_attn_fwd / _flash_attn_bwd directly.  The
provided q/k/v shapes are treated as the already-dispatched per-rank tensors;
NCCL all-to-all calls are used only to reproduce the online communication
pressure around FA.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


HEAD_DIM = 256
NHEADS = 4
DTYPE = torch.bfloat16


@dataclass
class StressCase:
    name: str
    total_q: int
    total_k: int
    cu_q_host: tuple[int, ...]
    cu_k_host: tuple[int, ...]
    max_q: int
    max_k: int
    causal: bool
    nheads: int = NHEADS
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    cu_q: torch.Tensor | None = None
    cu_k: torch.Tensor | None = None
    q_dispatch: torch.Tensor | None = None
    k_dispatch: torch.Tensor | None = None
    v_dispatch: torch.Tensor | None = None
    out: torch.Tensor | None = None
    out_combine: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    dout: torch.Tensor | None = None
    dout_dispatch: torch.Tensor | None = None
    dq: torch.Tensor | None = None
    dk: torch.Tensor | None = None
    dv: torch.Tensor | None = None
    dq_combine: torch.Tensor | None = None
    dk_combine: torch.Tensor | None = None
    dv_combine: torch.Tensor | None = None


def uniform_case(
    name: str,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    causal: bool,
    nheads: int = NHEADS,
) -> StressCase:
    cu_q = tuple(idx * seqlen_q for idx in range(batch_size + 1))
    cu_k = tuple(idx * seqlen_k for idx in range(batch_size + 1))
    return StressCase(
        name=name,
        total_q=batch_size * seqlen_q,
        total_k=batch_size * seqlen_k,
        cu_q_host=cu_q,
        cu_k_host=cu_k,
        max_q=seqlen_q,
        max_k=seqlen_k,
        causal=causal,
        nheads=nheads,
    )


def varlen_case(
    name: str,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    causal: bool,
    nheads: int = NHEADS,
) -> StressCase:
    assert len(q_lens) == len(k_lens)
    cu_q = [0]
    cu_k = [0]
    for q_len, k_len in zip(q_lens, k_lens):
        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + k_len)
    return StressCase(
        name=name,
        total_q=cu_q[-1],
        total_k=cu_k[-1],
        cu_q_host=tuple(cu_q),
        cu_k_host=tuple(cu_k),
        max_q=max(q_lens),
        max_k=max(k_lens),
        causal=causal,
        nheads=nheads,
    )


def dkdv_grid_stats(case: StressCase) -> str:
    k_lens = [
        case.cu_k_host[idx + 1] - case.cu_k_host[idx]
        for idx in range(len(case.cu_k_host) - 1)
    ]
    physical_blocks_per_head = (case.total_k + len(k_lens) * (2 * 64 - 1)) // 64
    physical_blocks_per_head = physical_blocks_per_head // 2 * 2
    physical_clusters_per_head = physical_blocks_per_head // 2
    valid_clusters_per_head = sum((k_len + 127) // 128 for k_len in k_lens)
    return (
        f"dkdv_grid_x={physical_blocks_per_head * case.nheads} "
        f"physical_clusters={physical_clusters_per_head * case.nheads} "
        f"valid_clusters={valid_clusters_per_head * case.nheads}"
    )


CASES = [
    StressCase(
        name="case1_q1024_k1024_b4",
        total_q=1024,
        total_k=1024,
        cu_q_host=(0, 75, 389, 553, 1024),
        cu_k_host=(0, 75, 389, 553, 1024),
        max_q=471,
        max_k=471,
        causal=True,
    ),
    StressCase(
        name="case2_q1024_k2083_b2",
        total_q=1024,
        total_k=2083,
        cu_q_host=(0, 442, 1024),
        cu_k_host=(0, 1501, 2083),
        max_q=582,
        max_k=1501,
        causal=True,
    ),
    StressCase(
        name="case3_q1024_k2048_b3",
        total_q=1024,
        total_k=2048,
        cu_q_host=(0, 1, 259, 1024),
        cu_k_host=(0, 1025, 1283, 2048),
        max_q=765,
        max_k=1025,
        causal=True,
    ),
    StressCase(
        name="gdb_grid168_q1024_k2560_b2",
        total_q=1024,
        total_k=2560,
        cu_q_host=(0, 442, 1024),
        cu_k_host=(0, 1501, 2560),
        max_q=582,
        max_k=1501,
        causal=True,
    ),
    StressCase(
        name="gdb_grid168_q1024_k2432_b3",
        total_q=1024,
        total_k=2432,
        cu_q_host=(0, 1, 259, 1024),
        cu_k_host=(0, 1025, 1283, 2432),
        max_q=765,
        max_k=1149,
        causal=True,
    ),
    StressCase(
        name="gdb_grid168_padding_q1024_k2304_b4",
        total_q=1024,
        total_k=2304,
        cu_q_host=(0, 75, 389, 553, 1024),
        cu_k_host=(0, 75, 389, 1414, 2304),
        max_q=471,
        max_k=1025,
        causal=True,
    ),
    StressCase(
        name="gdb_grid168_padding_q1024_k2560_b2_qshort",
        total_q=1024,
        total_k=2560,
        cu_q_host=(0, 1, 1024),
        cu_k_host=(0, 2048, 2560),
        max_q=1023,
        max_k=2048,
        causal=True,
    ),
    varlen_case(
        name="gdb_grid168_padding_mild_q1024_k2435_b2",
        q_lens=(512, 512),
        k_lens=(1280, 1155),
        causal=True,
    ),
    varlen_case(
        name="gdb_grid168_padding_mild_q1024_k2499_b2",
        q_lens=(442, 582),
        k_lens=(1408, 1091),
        causal=True,
    ),
    varlen_case(
        name="gdb_grid168_padding_mild_q1024_k2307_b3",
        q_lens=(256, 256, 512),
        k_lens=(1024, 768, 515),
        causal=True,
    ),
    varlen_case(
        name="gdb_grid168_half_cluster_tail_q1024_k2498_b2",
        q_lens=(960, 64),
        k_lens=(2434, 64),
        causal=True,
    ),
    varlen_case(
        name="gdb_grid168_half_cluster_tail_q1024_k2560_b2",
        q_lens=(960, 64),
        k_lens=(2496, 64),
        causal=True,
    ),
    varlen_case(
        name="gdb_grid168_half_cluster_tail_qlong_q1024_k2498_b2",
        q_lens=(512, 512),
        k_lens=(2434, 64),
        causal=True,
    ),
    varlen_case(
        name="gdb_grid168_half_cluster_tail_qlong_q1024_k2560_b2",
        q_lens=(64, 960),
        k_lens=(2496, 64),
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_exact_q1024_k2392_b3",
        q_lens=(103, 629, 292),
        k_lens=(1471, 629, 292),
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_tail_k1_q1024_k2307_b3",
        q_lens=(103, 629, 292),
        k_lens=(1536, 770, 1),
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_tail_k64_q1024_k2307_b3",
        q_lens=(103, 629, 292),
        k_lens=(1536, 707, 64),
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_tail_k128_q1024_k2307_b3",
        q_lens=(103, 629, 292),
        k_lens=(1471, 708, 128),
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_boundary_q1024_k2434_b3",
        q_lens=(103, 629, 292),
        k_lens=(1536, 642, 256),
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_qperm_q1024_k2392_b3",
        q_lens=(292, 629, 103),
        k_lens=(1471, 629, 292),
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_qtail_gt_k_q1024_k2392_b3",
        q_lens=(1, 731, 292),
        k_lens=(1471, 629, 292),
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_many_half_empty_b16_q1024_k2064",
        q_lens=(64,) * 16,
        k_lens=(129,) * 16,
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_many_half_empty_b16_q1024_k2384",
        q_lens=(64,) * 16,
        k_lens=(149,) * 16,
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_many_res36_b16_q1024_k2624",
        q_lens=(64,) * 16,
        k_lens=(164,) * 16,
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_many_half_empty_b8_q1024_k2056",
        q_lens=(128,) * 8,
        k_lens=(257,) * 8,
        causal=True,
    ),
    varlen_case(
        name="gdb_pid183_exact_h16_q1024_k2392_b3",
        q_lens=(103, 629, 292),
        k_lens=(1471, 629, 292),
        causal=True,
        nheads=16,
    ),
    varlen_case(
        name="gdb_pid183_many_res36_h16_b16_q1024_k2624",
        q_lens=(64,) * 16,
        k_lens=(164,) * 16,
        causal=True,
        nheads=16,
    ),
    uniform_case(
        name="heavy_q4096_k4096_b4",
        batch_size=4,
        seqlen_q=4096,
        seqlen_k=4096,
        causal=True,
    ),
    uniform_case(
        name="heavy_q4096_k8192_b2",
        batch_size=2,
        seqlen_q=4096,
        seqlen_k=8192,
        causal=True,
    ),
    uniform_case(
        name="heavy_q8192_k8192_b4",
        batch_size=4,
        seqlen_q=8192,
        seqlen_k=8192,
        causal=True,
    ),
    uniform_case(
        name="heavy_q16384_k16384_b2",
        batch_size=2,
        seqlen_q=16384,
        seqlen_k=16384,
        causal=True,
    ),
    uniform_case(
        name="heavy_h16_q8192_k8192_b2",
        batch_size=2,
        seqlen_q=8192,
        seqlen_k=8192,
        causal=True,
        nheads=16,
    ),
]

CASE_GROUPS = {
    "all": CASES,
    "gdb_grid168_suite": [
        case for case in CASES if case.name.startswith("gdb_grid168_")
    ],
    "gdb_grid168_forward_safe": [
        case for case in CASES
        if case.name in ("gdb_grid168_q1024_k2560_b2", "gdb_grid168_q1024_k2432_b3")
    ],
    "gdb_grid168_padding_suite": [
        case for case in CASES if case.name.startswith("gdb_grid168_padding_")
    ],
    "gdb_grid168_padding_mild": [
        case for case in CASES if case.name.startswith("gdb_grid168_padding_mild_")
    ],
    "gdb_grid168_half_cluster_tail": [
        case for case in CASES if case.name.startswith("gdb_grid168_half_cluster_tail_")
    ],
    "gdb_pid183_search": [
        case for case in CASES if case.name.startswith("gdb_pid183_")
    ],
    "gdb_pid183_half_empty_search": [
        case for case in CASES if case.name.startswith("gdb_pid183_many_half_empty_")
    ],
    "gdb_pid183_res36_search": [
        case for case in CASES
        if case.name.startswith("gdb_pid183_many_res36_")
    ],
    "gdb_pid183_h16_search": [
        case for case in CASES if "_h16_" in case.name
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direction",
        choices=("fwd", "bwd", "fwd-bwd", "bwd-only"),
        default="fwd-bwd",
    )
    parser.add_argument(
        "--case",
        choices=(*CASE_GROUPS.keys(), *(case.name for case in CASES)),
        default="all",
    )
    parser.add_argument("--max-iters", type=int, default=0, help="outer iterations; 0 means loop forever")
    parser.add_argument("--calls-per-iter", type=int, default=24)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--sync-every", type=int, default=1)
    parser.add_argument("--state-every-iters", type=int, default=1)
    parser.add_argument(
        "--state-every-calls",
        type=int,
        default=0,
        help="write heartbeat after every N completed calls; 0 disables per-call state",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle-cases", action="store_true")
    parser.add_argument("--disable-a2a", action="store_true")
    parser.add_argument("--async-a2a", action="store_true")
    parser.add_argument("--comm-mode", choices=("tensor", "dummy"), default="tensor")
    parser.add_argument("--rank-cases", default="")
    parser.add_argument("--dummy-a2a-elements", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--stage-trace", action="store_true", help="print every completed stage on rank 0")
    return parser.parse_args()


def log_rank0(message: str) -> None:
    if dist.get_rank() == 0:
        print(message, flush=True)


def balanced_splits(total: int, world_size: int) -> list[int]:
    base = total // world_size
    rem = total % world_size
    return [base + (1 if idx < rem else 0) for idx in range(world_size)]


def all_to_all_same_shape(
    tensor: torch.Tensor,
    output: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        output.copy_(tensor)
        return output
    splits = balanced_splits(tensor.shape[0], dist.get_world_size())
    dist.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=splits,
        input_split_sizes=splits,
    )
    return output


def all_to_all_many_same_shape(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    enabled: bool,
    async_a2a: bool,
) -> list[torch.Tensor]:
    if not enabled:
        for tensor, output in pairs:
            output.copy_(tensor)
        return [output for _, output in pairs]
    if not async_a2a:
        return [all_to_all_same_shape(tensor, output, enabled=True) for tensor, output in pairs]
    input_tensors = []
    works = []
    for tensor, output in pairs:
        splits = balanced_splits(tensor.shape[0], dist.get_world_size())
        input_tensor = tensor.contiguous()
        input_tensors.append(input_tensor)
        works.append(
            dist.all_to_all_single(
                output,
                input_tensor,
                output_split_sizes=splits,
                input_split_sizes=splits,
                async_op=True,
            )
        )
    for work in works:
        work.wait()
    return [output for _, output in pairs]


def dummy_all_to_all(
    dummy_comm: tuple[torch.Tensor, torch.Tensor] | None,
    enabled: bool,
    async_a2a: bool,
) -> None:
    if dummy_comm is None:
        return
    all_to_all_many_same_shape([dummy_comm], enabled, async_a2a)


def verify_runtime(repo: Path) -> None:
    import flash_attn.cute.interface as interface

    interface_path = Path(interface.__file__).resolve()
    if repo not in interface_path.parents:
        raise RuntimeError(
            f"flash_attn.cute.interface is not repo-local: {interface_path}, repo={repo}"
        )
    log_rank0(f"[runtime] interface={interface_path}")
    log_rank0(f"[runtime] torch={torch.__version__} cuda={torch.version.cuda}")
    log_rank0(f"[runtime] git_head={os.environ.get('GIT_HEAD', 'unknown')}")


def materialize_cases(cases: list[StressCase], device: torch.device, seed: int) -> None:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + dist.get_rank() * 1009)
    for case in cases:
        case.q = torch.randn(
            case.total_q, case.nheads, HEAD_DIM, device=device, dtype=DTYPE, generator=gen
        )
        case.k = torch.randn(
            case.total_k, case.nheads, HEAD_DIM, device=device, dtype=DTYPE, generator=gen
        )
        case.v = torch.randn(
            case.total_k, case.nheads, HEAD_DIM, device=device, dtype=DTYPE, generator=gen
        )
        case.cu_q = torch.tensor(case.cu_q_host, device=device, dtype=torch.int32)
        case.cu_k = torch.tensor(case.cu_k_host, device=device, dtype=torch.int32)
        case.q_dispatch = torch.empty_like(case.q)
        case.k_dispatch = torch.empty_like(case.k)
        case.v_dispatch = torch.empty_like(case.v)
        case.out = torch.empty_like(case.q)
        case.out_combine = torch.empty_like(case.out)
        case.lse = torch.zeros(case.nheads, case.total_q, device=device, dtype=torch.float32)
        case.dout = torch.randn(
            case.out.shape, device=device, dtype=DTYPE, generator=gen
        )
        case.dout_dispatch = torch.empty_like(case.dout)
        case.dq = torch.empty_like(case.q)
        case.dk = torch.empty_like(case.k)
        case.dv = torch.empty_like(case.v)
        case.dq_combine = torch.empty_like(case.dq)
        case.dk_combine = torch.empty_like(case.dk)
        case.dv_combine = torch.empty_like(case.dv)
        case.out.copy_(
            torch.randn(case.out.shape, device=device, dtype=DTYPE, generator=gen)
        )


def maybe_sync(iter_idx: int, sync_every: int) -> None:
    if sync_every > 0 and iter_idx % sync_every == 0:
        torch.cuda.synchronize()


def trace(
    stage_trace: bool,
    iteration: int,
    case: StressCase,
    stage: str,
    call_in_iter: int,
    calls_per_iter: int,
    total_calls: int,
) -> None:
    if not stage_trace:
        return
    if dist.get_rank() == 0:
        print(
            f"[trace] iter={iteration} call={call_in_iter}/{calls_per_iter} "
            f"total_calls={total_calls} case={case.name} stage={stage}",
            flush=True,
        )


def write_state(
    iteration: int,
    case: StressCase,
    stage: str,
    call_in_iter: int,
    calls_per_iter: int,
    total_calls: int,
) -> None:
    state_dir = os.environ.get("HANG_STRESS_STATE_DIR")
    if state_dir:
        rank = dist.get_rank()
        path = Path(state_dir) / f"rank{rank}.txt"
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            f"time={time.time():.6f}\n"
            f"rank={rank}\n"
            f"iter={iteration}\n"
            f"call_in_iter={call_in_iter}\n"
            f"calls_per_iter={calls_per_iter}\n"
            f"total_calls={total_calls}\n"
            f"case={case.name}\n"
            f"stage={stage}\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)


def run_forward(
    case: StressCase,
    iteration: int,
    use_a2a: bool,
    async_a2a: bool,
    comm_mode: str,
    dummy_comm: tuple[torch.Tensor, torch.Tensor] | None,
    stage_trace: bool,
    call_in_iter: int,
    calls_per_iter: int,
    total_calls: int,
):
    from flash_attn.cute.interface import _flash_attn_fwd

    if comm_mode == "tensor":
        q, k, v = all_to_all_many_same_shape(
            [
                (case.q, case.q_dispatch),
                (case.k, case.k_dispatch),
                (case.v, case.v_dispatch),
            ],
            use_a2a,
            async_a2a,
        )
    else:
        dummy_all_to_all(dummy_comm, use_a2a, async_a2a)
        q, k, v = case.q, case.k, case.v
    trace(
        stage_trace,
        iteration,
        case,
        "fwd_dispatch_done",
        call_in_iter,
        calls_per_iter,
        total_calls,
    )

    out, lse, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=case.cu_q,
        cu_seqlens_k=case.cu_k,
        max_seqlen_q=case.max_q,
        max_seqlen_k=case.max_k,
        causal=case.causal,
        return_lse=True,
        out=case.out,
        lse=case.lse,
    )
    trace(
        stage_trace,
        iteration,
        case,
        "fwd_done",
        call_in_iter,
        calls_per_iter,
        total_calls,
    )

    if comm_mode == "tensor":
        all_to_all_same_shape(out, case.out_combine, use_a2a)
    else:
        dummy_all_to_all(dummy_comm, use_a2a, async_a2a)
    trace(
        stage_trace,
        iteration,
        case,
        "fwd_combine_done",
        call_in_iter,
        calls_per_iter,
        total_calls,
    )
    return q, k, v, out, lse


def run_backward(
    case: StressCase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    iteration: int,
    use_a2a: bool,
    async_a2a: bool,
    comm_mode: str,
    dummy_comm: tuple[torch.Tensor, torch.Tensor] | None,
    stage_trace: bool,
    call_in_iter: int,
    calls_per_iter: int,
    total_calls: int,
) -> None:
    from flash_attn.cute.interface import _flash_attn_bwd

    if comm_mode == "tensor":
        dout = all_to_all_same_shape(case.dout, case.dout_dispatch, use_a2a)
    else:
        dummy_all_to_all(dummy_comm, use_a2a, async_a2a)
        dout = case.dout
    trace(
        stage_trace,
        iteration,
        case,
        "bwd_dispatch_done",
        call_in_iter,
        calls_per_iter,
        total_calls,
    )

    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        causal=case.causal,
        cu_seqlens_q=case.cu_q,
        cu_seqlens_k=case.cu_k,
        max_seqlen_q=case.max_q,
        max_seqlen_k=case.max_k,
        dq=case.dq,
        dk=case.dk,
        dv=case.dv,
    )
    trace(
        stage_trace,
        iteration,
        case,
        "bwd_done",
        call_in_iter,
        calls_per_iter,
        total_calls,
    )

    if comm_mode == "tensor":
        all_to_all_many_same_shape(
            [
                (dq, case.dq_combine),
                (dk, case.dk_combine),
                (dv, case.dv_combine),
            ],
            use_a2a,
            async_a2a,
        )
    else:
        dummy_all_to_all(dummy_comm, use_a2a, async_a2a)
    trace(
        stage_trace,
        iteration,
        case,
        "bwd_combine_done",
        call_in_iter,
        calls_per_iter,
        total_calls,
    )


def run_backward_only(
    case: StressCase,
    iteration: int,
    use_a2a: bool,
    async_a2a: bool,
    comm_mode: str,
    dummy_comm: tuple[torch.Tensor, torch.Tensor] | None,
    stage_trace: bool,
    call_in_iter: int,
    calls_per_iter: int,
    total_calls: int,
) -> None:
    if comm_mode == "tensor":
        q, k, v = all_to_all_many_same_shape(
            [
                (case.q, case.q_dispatch),
                (case.k, case.k_dispatch),
                (case.v, case.v_dispatch),
            ],
            use_a2a,
            async_a2a,
        )
    else:
        dummy_all_to_all(dummy_comm, use_a2a, async_a2a)
        q, k, v = case.q, case.k, case.v
    trace(
        stage_trace,
        iteration,
        case,
        "bwd_only_qkv_dispatch_done",
        call_in_iter,
        calls_per_iter,
        total_calls,
    )
    run_backward(
        case,
        q,
        k,
        v,
        case.out,
        case.lse,
        iteration,
        use_a2a,
        async_a2a,
        comm_mode,
        dummy_comm,
        stage_trace,
        call_in_iter,
        calls_per_iter,
        total_calls,
    )


def main() -> int:
    args = parse_args()
    if args.calls_per_iter <= 0:
        raise ValueError(f"calls_per_iter must be positive, got {args.calls_per_iter}")
    if args.state_every_iters <= 0:
        raise ValueError(f"state_every_iters must be positive, got {args.state_every_iters}")
    if args.state_every_calls < 0:
        raise ValueError(f"state_every_calls must be non-negative, got {args.state_every_calls}")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device_count = torch.cuda.device_count()
    device_idx = local_rank
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    use_a2a = not args.disable_a2a
    if not use_a2a:
        raise RuntimeError("this stress must keep 4-GPU all-to-all enabled")
    if world_size != 4:
        raise RuntimeError(f"all-to-all stress expects 4 ranks, got {world_size}")

    repo = Path(os.environ["REPO"]).resolve()
    verify_runtime(repo)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    case_rng = random.Random(args.seed)

    if args.rank_cases:
        rank_case_names = [name.strip() for name in args.rank_cases.split(",")]
        if len(rank_case_names) != world_size:
            raise RuntimeError(
                f"--rank-cases must provide {world_size} comma-separated cases, "
                f"got {len(rank_case_names)}"
            )
        case_name = rank_case_names[rank]
        selected_cases = [case for case in CASES if case.name == case_name]
    else:
        selected_cases = CASE_GROUPS.get(args.case)
        if selected_cases is None:
            selected_cases = [case for case in CASES if case.name == args.case]
    if not selected_cases:
        raise RuntimeError(f"no stress cases selected for rank {rank}")
    materialize_cases(selected_cases, device, args.seed)
    dummy_comm = None
    if args.comm_mode == "dummy":
        dummy_in = torch.randn(
            args.dummy_a2a_elements,
            device=device,
            dtype=DTYPE,
            generator=torch.Generator(device=device).manual_seed(args.seed + rank + 9001),
        )
        dummy_out = torch.empty_like(dummy_in)
        dummy_comm = (dummy_in, dummy_out)
    dist.barrier(device_ids=[device_idx])

    log_rank0(
        "[stress] "
        f"world={world_size} device_count={device_count} direction={args.direction} "
        f"cases={[case.name for case in selected_cases]} "
        f"max_iters={args.max_iters or 'infinite'} calls_per_iter={args.calls_per_iter} "
        f"use_a2a={use_a2a} async_a2a={args.async_a2a} comm_mode={args.comm_mode} "
        f"dummy_a2a_elements={args.dummy_a2a_elements} sync_every={args.sync_every} "
        f"state_every_iters={args.state_every_iters} state_every_calls={args.state_every_calls}"
    )
    for case in selected_cases:
        print(f"[rank {rank} case] {case.name} {dkdv_grid_stats(case)}", flush=True)

    start = time.monotonic()
    iteration = 0
    total_calls = 0
    while args.max_iters == 0 or iteration < args.max_iters:
        iteration += 1
        cases_for_iter = []
        while len(cases_for_iter) < args.calls_per_iter:
            case_order = list(selected_cases)
            if args.shuffle_cases:
                case_rng.shuffle(case_order)
            cases_for_iter.extend(case_order)
        cases_for_iter = cases_for_iter[: args.calls_per_iter]

        last_case = cases_for_iter[-1]
        for call_in_iter, case in enumerate(cases_for_iter, start=1):
            total_calls += 1

            q = k = v = out = lse = None
            if args.direction == "bwd-only":
                run_backward_only(
                    case,
                    iteration,
                    use_a2a,
                    args.async_a2a,
                    args.comm_mode,
                    dummy_comm,
                    args.stage_trace,
                    call_in_iter,
                    args.calls_per_iter,
                    total_calls,
                )
                if args.state_every_calls and total_calls % args.state_every_calls == 0:
                    write_state(
                        iteration,
                        case,
                        "call_done",
                        call_in_iter,
                        args.calls_per_iter,
                        total_calls,
                    )
                continue

            if args.direction in ("fwd", "fwd-bwd", "bwd"):
                q, k, v, out, lse = run_forward(
                    case,
                    iteration,
                    use_a2a,
                    args.async_a2a,
                    args.comm_mode,
                    dummy_comm,
                    args.stage_trace,
                    call_in_iter,
                    args.calls_per_iter,
                    total_calls,
                )
            if args.direction in ("bwd", "fwd-bwd"):
                run_backward(
                    case,
                    q,
                    k,
                    v,
                    out,
                    lse,
                    iteration,
                    use_a2a,
                    args.async_a2a,
                    args.comm_mode,
                    dummy_comm,
                    args.stage_trace,
                    call_in_iter,
                    args.calls_per_iter,
                    total_calls,
                )
            if args.state_every_calls and total_calls % args.state_every_calls == 0:
                write_state(
                    iteration,
                    case,
                    "call_done",
                    call_in_iter,
                    args.calls_per_iter,
                    total_calls,
                )

        maybe_sync(iteration, args.sync_every)
        if iteration % args.state_every_iters == 0:
            write_state(
                iteration,
                last_case,
                "iter_done",
                args.calls_per_iter,
                args.calls_per_iter,
                total_calls,
            )

        if iteration <= args.warmup_iters or iteration % args.log_interval == 0:
            elapsed = time.monotonic() - start
            it_s = iteration / elapsed if elapsed > 0 else 0.0
            call_s = total_calls / elapsed if elapsed > 0 else 0.0
            log_rank0(
                f"[progress] iter={iteration} calls={total_calls} "
                f"calls_per_iter={args.calls_per_iter} last_case={last_case.name} "
                f"elapsed_s={elapsed:.1f} iter_per_s={it_s:.3f} calls_per_s={call_s:.3f}"
            )

    torch.cuda.synchronize()
    dist.barrier(device_ids=[device_idx])
    log_rank0("[stress] completed without detected hang")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[rank {os.environ.get('RANK', '?')}] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
