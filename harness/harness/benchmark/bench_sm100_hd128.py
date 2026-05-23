#!/usr/bin/env python3
"""Focused SM100 head_dim=128 benchmark for HD256 performance comparisons.

This script is intentionally independent from bench_sm100_hd256.py so HD128
comparisons do not inherit HD256-specific assumptions or labels.
"""

import argparse
import sys

import torch

from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd


HEAD_DIM = 128


def csv_ints(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def auto_batch(seqlen, batch_arg, total_tokens=32768):
    return batch_arg if batch_arg > 0 else max(1, total_tokens // seqlen)


def fwd_flops(batch, nheads, seqlen, hdim, causal=False):
    avg_seqlen = seqlen / 2 if causal else seqlen
    return batch * nheads * 2 * seqlen * avg_seqlen * (hdim + hdim)


def bwd_flops(batch, nheads, seqlen, hdim, causal=False):
    return 2.5 * fwd_flops(batch, nheads, seqlen, hdim, causal=causal)


def varlen_flops(total_len, doc_len, nheads, hdim, causal=False):
    return fwd_flops(total_len // doc_len, nheads, doc_len, hdim, causal=causal)


def make_replicated_cu_seqlens(total_len, doc_len):
    if total_len % doc_len != 0:
        raise ValueError(f"total_len={total_len} must be divisible by doc_len={doc_len}")
    return torch.arange(0, total_len + doc_len, doc_len, dtype=torch.int32, device="cuda")


def varlen_doc_lens_for_total(total_len, doc_lens):
    if doc_lens is not None:
        return [doc_len for doc_len in doc_lens if doc_len <= total_len and total_len % doc_len == 0]
    result = []
    doc_len = 128
    while doc_len <= total_len:
        if total_len % doc_len == 0:
            result.append(doc_len)
        doc_len *= 2
    return result


def check_sm100():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found", file=sys.stderr)
        sys.exit(1)
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"GPU: {name} (SM{cap[0]}{cap[1]})")
    if cap[0] not in (10, 11):
        print("WARNING: this benchmark is intended for SM100/SM110", file=sys.stderr)


def bench_fwd(batch, seqlen, nheads, nheads_kv, causal, warmup=5, rep=30):
    q = torch.randn(batch, seqlen, nheads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, seqlen, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, seqlen, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    scale = HEAD_DIM ** -0.5

    try:
        _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal)
        torch.cuda.synchronize()
    except Exception as exc:
        return None, None, str(exc)[:160]

    for _ in range(warmup):
        _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / rep
    tflops = fwd_flops(batch, nheads, seqlen, HEAD_DIM, causal=causal) / ms / 1e9
    return ms, tflops, None


def bench_bwd(batch, seqlen, nheads, nheads_kv, causal, warmup=5, rep=30):
    q = torch.randn(batch, seqlen, nheads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, seqlen, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, seqlen, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    scale = HEAD_DIM ** -0.5

    try:
        out, lse = _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal, return_lse=True)
        torch.cuda.synchronize()
    except Exception as exc:
        return None, None, str(exc)[:160]

    dout = torch.randn_like(out)

    def fn():
        return _flash_attn_bwd(q, k, v, out, dout, lse, softmax_scale=scale, causal=causal)

    try:
        grads = fn()
        torch.cuda.synchronize()
    except Exception as exc:
        return None, None, str(exc)[:160]

    for _ in range(warmup):
        grads = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        grads = fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / rep
    tflops = bwd_flops(batch, nheads, seqlen, HEAD_DIM, causal=causal) / ms / 1e9
    return ms, tflops, None


def bench_varlen_fwd(total_len, doc_len, nheads, nheads_kv, causal, warmup=5, rep=30):
    q = torch.randn(total_len, nheads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total_len, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(total_len, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    cu_seqlens = make_replicated_cu_seqlens(total_len, doc_len)
    scale = HEAD_DIM ** -0.5

    def fn(return_lse=False):
        return _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=doc_len,
            max_seqlen_k=doc_len,
            softmax_scale=scale,
            causal=causal,
            return_lse=return_lse,
        )

    try:
        fn()
        torch.cuda.synchronize()
    except Exception as exc:
        return None, None, str(exc)[:160]

    for _ in range(warmup):
        grads = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        grads = fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / rep
    tflops = varlen_flops(total_len, doc_len, nheads, HEAD_DIM, causal=causal) / ms / 1e9
    return ms, tflops, None


def bench_varlen_bwd(total_len, doc_len, nheads, nheads_kv, causal, warmup=5, rep=30):
    q = torch.randn(total_len, nheads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total_len, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(total_len, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    cu_seqlens = make_replicated_cu_seqlens(total_len, doc_len)
    scale = HEAD_DIM ** -0.5

    try:
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=doc_len,
            max_seqlen_k=doc_len,
            softmax_scale=scale,
            causal=causal,
            return_lse=True,
        )
        torch.cuda.synchronize()
    except Exception as exc:
        return None, None, str(exc)[:160]

    dout = torch.randn_like(out)

    def fn():
        return _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=doc_len,
            max_seqlen_k=doc_len,
            softmax_scale=scale,
            causal=causal,
        )

    try:
        grads = fn()
        torch.cuda.synchronize()
    except Exception as exc:
        return None, None, str(exc)[:160]

    for _ in range(warmup):
        grads = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        grads = fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / rep
    tflops = 2.5 * varlen_flops(total_len, doc_len, nheads, HEAD_DIM, causal=causal) / ms / 1e9
    return ms, tflops, None


def print_result(kind, nheads, nheads_kv, causal, total, unit, count, ms, tflops, err):
    if ms is None:
        print(f"{kind},128,{nheads},{nheads_kv},{int(causal)},{total},{unit},{count},None,None,{err}", flush=True)
    else:
        print(f"{kind},128,{nheads},{nheads_kv},{int(causal)},{total},{unit},{count},{ms},{tflops}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Focused SM100 head_dim=128 benchmark")
    parser.add_argument("--suite", choices=["dense", "varlen", "both"], default="both")
    parser.add_argument("--direction", choices=["fwd", "bwd", "both"], default="both")
    parser.add_argument("--seqlen", type=csv_ints, default=[128, 256, 512, 1024, 2048, 8192, 32768])
    parser.add_argument("--varlen-doc-lens", type=csv_ints, default=None)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--nheads", type=int, default=16)
    parser.add_argument("--nheads-kv", type=int, default=16, dest="nheads_kv")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument("--non-causal-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    check_sm100()
    directions = ["fwd", "bwd"] if args.direction == "both" else [args.direction]
    causals = [True] if args.causal_only else ([False] if args.non_causal_only else [False, True])
    print("TYPE,head_dim,nheads,nheads_kv,causal,total,seqlen_or_doc,batch_or_docs,ms,tflops")

    if args.suite in ("dense", "both"):
        for direction in directions:
            for seqlen in args.seqlen:
                batch = auto_batch(seqlen, args.batch)
                for causal in causals:
                    bench = bench_fwd if direction == "fwd" else bench_bwd
                    ms, tflops, err = bench(
                        batch,
                        seqlen,
                        args.nheads,
                        args.nheads_kv,
                        causal,
                        warmup=args.warmup,
                        rep=args.rep,
                    )
                    print_result(f"DENSE_{direction.upper()}", args.nheads, args.nheads_kv, causal, seqlen, seqlen, batch, ms, tflops, err)

    if args.suite in ("varlen", "both"):
        for direction in directions:
            for total_len in args.seqlen:
                for doc_len in varlen_doc_lens_for_total(total_len, args.varlen_doc_lens):
                    docs = total_len // doc_len
                    for causal in causals:
                        bench = bench_varlen_fwd if direction == "fwd" else bench_varlen_bwd
                        ms, tflops, err = bench(
                            total_len,
                            doc_len,
                            args.nheads,
                            args.nheads_kv,
                            causal,
                            warmup=args.warmup,
                            rep=args.rep,
                        )
                        print_result(f"VARLEN_{direction.upper()}", args.nheads, args.nheads_kv, causal, total_len, doc_len, docs, ms, tflops, err)


if __name__ == "__main__":
    main()
