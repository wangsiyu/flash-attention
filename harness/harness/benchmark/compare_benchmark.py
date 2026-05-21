#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROW_RE = re.compile(
    r"^(?P<mask>non-causal|causal)\s+/\s+"
    r"(?:(?:seqlen=(?P<seqlen>\d+))|(?:total=(?P<total>\d+)\s+doc=(?P<doc>\d+)))\s+"
    r"(?P<batch>\d+)\s+(?P<fa_ms>\d+(?:\.\d+)?)\s+"
    r"(?P<tflops>\d+(?:\.\d+)?)(?:\(\d+(?:\.\d+)?%\))?"
    r"(?:\s+(?P<sdpa_ms>\d+(?:\.\d+)?)\s+(?P<sdpa_tflops>\d+(?:\.\d+)?)\s+"
    r"(?P<speedup>\d+(?:\.\d+)?x)\s+(?P<gap>[+-]\d+(?:\.\d+)?%))?"
)
SUITE_RE = re.compile(r"\bsuite=(?P<suite>\S+)")
HEAD_RE = re.compile(r"\bnheads=(?P<nheads>\d+)\s+nheads_kv=(?P<nheads_kv>\d+)")
DIRECTION_RE = re.compile(r"\b(?P<direction>Forward|Backward)\b")
MASK_ORDER = {"non-causal": 0, "causal": 1}
DIRECTION_ORDER = ("Forward", "Backward")


@dataclass(frozen=True)
class BenchStats:
    fa: float
    sdpa: float | None = None


def parse_log_paths(
    paths: list[Path],
    metric: str,
) -> dict[tuple[str, str, str, int, int], dict[str, list[float]]]:
    values: dict[tuple[str, str, str, int, int], dict[str, list[float]]] = defaultdict(
        lambda: {"fa": [], "sdpa": []}
    )
    for path in sorted(paths):
        direction = "unknown"
        suite = "unknown"
        for line in path.read_text(errors="replace").splitlines():
            direction_match = DIRECTION_RE.search(line)
            if direction_match:
                direction = direction_match.group("direction")
            suite_match = SUITE_RE.search(line)
            if suite_match:
                suite = suite_match.group("suite")
            else:
                head_match = HEAD_RE.search(line)
                if head_match:
                    suite = f"dense_h{head_match.group('nheads')}_kv{head_match.group('nheads_kv')}"
            match = ROW_RE.match(line)
            if not match:
                continue
            total = int(match.group("seqlen") or match.group("total"))
            doc = int(match.group("doc") or 0)
            key = (suite, direction, match.group("mask"), total, doc)
            if metric == "sdpa":
                values[key]["sdpa"].append(float(match.group("tflops")))
            else:
                values[key]["fa"].append(float(match.group("tflops")))
            if metric == "fa" and match.group("sdpa_tflops") is not None:
                values[key]["sdpa"].append(float(match.group("sdpa_tflops")))
    return values


def parse_logs(directory: Path, metric: str = "fa") -> dict[tuple[str, str, str, int, int], dict[str, list[float]]]:
    if not directory.exists():
        return defaultdict(lambda: {"fa": [], "sdpa": []})
    return parse_log_paths(list(directory.glob("run_*.log")), metric)


def parse_sdpa_baseline(path: Path | None) -> dict[tuple[str, str, str, int, int], dict[str, list[float]]]:
    if path is None or not path.exists():
        return defaultdict(lambda: {"fa": [], "sdpa": []})
    return parse_log_paths([path], "sdpa")


def merge_values(
    dst: dict[tuple[str, str, str, int, int], dict[str, list[float]]],
    src: dict[tuple[str, str, str, int, int], dict[str, list[float]]],
) -> None:
    for key, value in src.items():
        dst[key]["fa"].extend(value["fa"])
        dst[key]["sdpa"].extend(value["sdpa"])


def median_map(
    values: dict[tuple[str, str, str, int, int], dict[str, list[float]]]
) -> dict[tuple[str, str, str, int, int], BenchStats]:
    return {
        key: BenchStats(
            fa=statistics.median(v["fa"]),
            sdpa=statistics.median(v["sdpa"]) if v["sdpa"] else None,
        )
        for key, v in values.items()
        if v["fa"]
    }


def doc_text(doc: int) -> str:
    return "dense" if doc == 0 else str(doc)


def sort_row_key(key: tuple[str, str, str, int, int]) -> tuple[int, int, int, str]:
    _suite, _direction, mask, total, doc = key
    return (total, doc, MASK_ORDER.get(mask, 99), mask)


def ordered_directions(keys: set[tuple[str, str, str, int, int]], suite: str) -> list[str]:
    directions = {direction for key_suite, direction, *_ in keys if key_suite == suite}
    ordered = [direction for direction in DIRECTION_ORDER if direction in directions]
    ordered.extend(sorted(directions - set(DIRECTION_ORDER)))
    return ordered


def sdpa_columns(stats: BenchStats) -> tuple[str, str, str]:
    if stats.sdpa is None:
        return "n/a", "n/a", "n/a"
    speedup = stats.fa / stats.sdpa
    gap = (speedup - 1.0) * 100
    return f"{stats.sdpa:.1f}", f"{speedup:.2f}x", f"{gap:+.1f}%"


def append_source_stamp(lines: list[str], path: Path | None) -> None:
    if path is None or not path.exists():
        return
    lines += [
        "## Source Stamp",
        "",
        "```text",
        *path.read_text(errors="replace").splitlines(),
        "```",
        "",
    ]


def append_current_tables(
    lines: list[str],
    current: dict[tuple[str, str, str, int, int], BenchStats],
) -> None:
    keys = set(current)
    for suite in sorted({key[0] for key in keys}):
        lines += [f"## {suite}", ""]
        for direction in ordered_directions(keys, suite):
            rows = [
                key for key in keys
                if key[0] == suite and key[1] == direction
            ]
            if not rows:
                continue
            lines += [
                f"### {direction}",
                "",
                "| Mask | Total/Seqlen | Doc Len | FA Median TFLOPS | SDPA Median TFLOPS | FA/SDPA | Gap vs SDPA |",
                "| ---- | ------------ | ------- | ---------------- | ------------------ | ------- | ----------- |",
            ]
            for key in sorted(rows, key=sort_row_key):
                _suite, _direction, mask, total, doc = key
                stats = current[key]
                if stats.sdpa is None:
                    lines.append(f"| {mask} | {total} | {doc_text(doc)} | {stats.fa:.1f} | n/a | n/a | n/a |")
                else:
                    speedup = stats.fa / stats.sdpa
                    gap = (speedup - 1.0) * 100
                    lines.append(
                        f"| {mask} | {total} | {doc_text(doc)} | {stats.fa:.1f} | {stats.sdpa:.1f} | "
                        f"{speedup:.2f}x | {gap:+.1f}% |"
                    )
            lines.append("")


def append_comparison_tables(
    lines: list[str],
    current: dict[tuple[str, str, str, int, int], BenchStats],
    previous: dict[tuple[str, str, str, int, int], BenchStats],
    regression_threshold: float,
) -> list[tuple[tuple[str, str, str, int, int], float, float, float]]:
    keys = set(previous) | set(current)
    regressions: list[tuple[tuple[str, str, str, int, int], float, float, float]] = []
    for suite in sorted({key[0] for key in keys}):
        lines += [f"## {suite}", ""]
        for direction in ordered_directions(keys, suite):
            rows = [
                key for key in keys
                if key[0] == suite and key[1] == direction
            ]
            if not rows:
                continue
            lines += [
                f"### {direction}",
                "",
                "| Mask | Total/Seqlen | Doc Len | Previous FA TFLOPS | Current FA TFLOPS | FA Delta | Status | SDPA Median TFLOPS | FA/SDPA | Gap vs SDPA |",
                "| ---- | ------------ | ------- | ------------------ | ----------------- | -------- | ------ | ------------------ | ------- | ----------- |",
            ]
            for key in sorted(rows, key=sort_row_key):
                prev = previous.get(key)
                cur = current.get(key)
                _suite, _direction, mask, total, doc = key
                if prev is None:
                    sdpa_text, speedup_text, gap_text = sdpa_columns(cur)
                    lines.append(f"| {mask} | {total} | {doc_text(doc)} | n/a | {cur.fa:.1f} | n/a | new | {sdpa_text} | {speedup_text} | {gap_text} |")
                    continue
                if cur is None:
                    lines.append(f"| {mask} | {total} | {doc_text(doc)} | {prev.fa:.1f} | n/a | n/a | missing | n/a | n/a | n/a |")
                    regressions.append((key, prev.fa, 0.0, -1.0))
                    continue
                delta = (cur.fa - prev.fa) / prev.fa if prev.fa else 0.0
                status = "REGRESSION" if delta < -regression_threshold else "ok"
                if status == "REGRESSION":
                    regressions.append((key, prev.fa, cur.fa, delta))
                sdpa_text, speedup_text, gap_text = sdpa_columns(cur)
                lines.append(
                    f"| {mask} | {total} | {doc_text(doc)} | {prev.fa:.1f} | {cur.fa:.1f} | "
                    f"{delta:.1%} | {status} | {sdpa_text} | {speedup_text} | {gap_text} |"
                )
            lines.append("")
    return regressions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sdpa-baseline", type=Path)
    parser.add_argument("--source-stamp", type=Path)
    parser.add_argument("--regression-threshold", type=float, default=0.03)
    args = parser.parse_args()

    current_values = parse_logs(args.current)
    merge_values(current_values, parse_sdpa_baseline(args.sdpa_baseline))
    current = median_map(current_values)
    previous = median_map(parse_logs(args.previous))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Report",
        "",
        f"Current: `{args.current}`",
        f"Previous: `{args.previous}`",
        f"SDPA baseline: `{args.sdpa_baseline}`" if args.sdpa_baseline else "SDPA baseline: `n/a`",
        f"Source stamp: `{args.source_stamp}`" if args.source_stamp else "Source stamp: `n/a`",
        f"Regression threshold: `{args.regression_threshold:.1%}`",
        "Gap vs SDPA: `(FA / SDPA - 1)`, positive means FA is faster.",
        "",
    ]
    append_source_stamp(lines, args.source_stamp)

    if not current:
        lines += ["No current benchmark rows parsed.", ""]
        args.report.write_text("\n".join(lines))
        return 2

    if not previous:
        lines += ["No previous benchmark directory found. Current run becomes the baseline for the next comparison.", ""]
        append_current_tables(lines, current)
        args.report.write_text("\n".join(lines) + "\n")
        print(f"[benchmark] wrote {args.report}")
        return 0

    regressions = append_comparison_tables(lines, current, previous, args.regression_threshold)

    if regressions:
        lines += [
            "",
            "## Regression Action",
            "",
            "Systemic regression detected from repeated benchmark medians.",
            "Export SASS for before/after kernels or run equivalent experiment analysis before modifying further.",
            "Do not submit or lock this round until performance is recovered.",
            "",
        ]
    else:
        lines += ["", "No systemic regression detected.", ""]

    args.report.write_text("\n".join(lines) + "\n")
    print(f"[benchmark] wrote {args.report}")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
