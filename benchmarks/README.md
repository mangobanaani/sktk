# Benchmarks

These scripts provide quick, repeatable performance baselines for core runtime paths.

## Agent Runtime

`agent_runtime_benchmark.py` measures invoke latency and throughput with deterministic mock responses.

```bash
python benchmarks/agent_runtime_benchmark.py --iterations 200
```

Output includes:
- total runtime
- throughput (QPS)
- p50/p95 invoke latency

Use this in local perf checks before and after runtime changes.

Optional SLO gates:

```bash
python benchmarks/agent_runtime_benchmark.py \
  --iterations 500 \
  --max-p95-ms 2.0 \
  --min-qps 50000
```

The script exits non-zero when thresholds are violated.
