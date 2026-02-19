"""Lightweight runtime benchmark for SKTK agent invocation.

Purpose:
- Track baseline invoke latency and throughput.
- Catch obvious performance regressions over time.

Usage:
    python benchmarks/agent_runtime_benchmark.py --iterations 200
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from sktk import SKTKAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SKTK agent invoke runtime")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of invoke calls")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument(
        "--max-p95-ms",
        type=float,
        default=None,
        help="Fail if p95 latency exceeds this threshold (milliseconds).",
    )
    parser.add_argument(
        "--min-qps",
        type=float,
        default=None,
        help="Fail if throughput drops below this threshold (queries/sec).",
    )
    return parser.parse_args()


async def run_benchmark(
    iterations: int,
    warmup: int,
    *,
    max_p95_ms: float | None = None,
    min_qps: float | None = None,
) -> int:
    responses = ["ok"] * (iterations + warmup)
    agent = SKTKAgent.with_responses("bench", responses)

    for _ in range(warmup):
        await agent.invoke("warmup")

    durations_ms: list[float] = []
    start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        await agent.invoke("ping")
        durations_ms.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - start

    p50 = statistics.median(durations_ms)
    p95 = (
        statistics.quantiles(durations_ms, n=20)[18]
        if len(durations_ms) >= 20
        else max(durations_ms)
    )
    qps = iterations / elapsed if elapsed > 0 else 0.0

    print("SKTK Agent Runtime Benchmark")
    print(f"iterations      : {iterations}")
    print(f"total_time_s    : {elapsed:.4f}")
    print(f"throughput_qps  : {qps:.1f}")
    print(f"latency_ms_p50  : {p50:.4f}")
    print(f"latency_ms_p95  : {p95:.4f}")

    failed = False
    if max_p95_ms is not None and p95 > max_p95_ms:
        print(f"FAIL: p95 latency {p95:.4f}ms exceeds threshold {max_p95_ms:.4f}ms")
        failed = True
    if min_qps is not None and qps < min_qps:
        print(f"FAIL: throughput {qps:.1f} qps below threshold {min_qps:.1f} qps")
        failed = True

    if not failed:
        print("SLO status      : PASS")
    return 1 if failed else 0


def main() -> None:
    args = parse_args()
    raise SystemExit(
        asyncio.run(
            run_benchmark(
                iterations=args.iterations,
                warmup=args.warmup,
                max_p95_ms=args.max_p95_ms,
                min_qps=args.min_qps,
            )
        )
    )


if __name__ == "__main__":
    main()
