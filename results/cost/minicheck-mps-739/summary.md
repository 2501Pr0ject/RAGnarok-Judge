# Cost run: minicheck

Regime: one sentence per call, device cache released after each call.
Judge: stack torch, model minicheck, device mps, dtype float32, ragnarok_ai 1.13.0, torch 2.15.0, transformers 5.22.2.
Inputs: 739 from `ragtruth-out/test.jsonl`, order of `bench-1.13/quota/hhem-mps/attempts.jsonl`; lengths 175 to 2475 tokens, 0 truncated.
Started 2026-11-09T14:14:31+00:00, ended 2026-11-09T14:22:11+00:00, 459.4 s. Previous run of this folder ended 30.1 min before the start.

| Measure | Value |
|---|---|
| Median latency, call and cache release | 407.37 ms |
| p90 latency | 1340.13 ms |
| Mean latency | 607.22 ms |
| Median latency, call alone | 406.91 ms |
| Footprint after loading | 3.63 GB (3461 MiB) |
| Peak footprint over the inputs | 4.74 GB (4525 MiB) |
| Maximum resident memory (no GPU memory) | 0.42 GB (403 MiB) |

Latency by fifth of the run, on the inputs of at most 512 tokens only (call and release). The
inputs are not in random order and their length changes along the run, so only short inputs are
compared; the last column divides each latency by the length of its input. A rise from the first
fifth to the last points to thermal drift.

| Fifth | n | p50 ms | mean tokens | p50 ms per 100 tokens |
|---|---|---|---|---|
| 1 | 34 | 202.31 | 362 | 57.78 |
| 2 | 36 | 369.5 | 394 | 86.37 |
| 3 | 97 | 376.1 | 381 | 93.54 |
| 4 | 137 | 278.64 | 346 | 80.39 |
| 5 | 137 | 303.77 | 322 | 101.26 |

Latency by input length:

| Tokens | n | p50 ms | p90 ms |
|---|---|---|---|
| 0 to 512 | 441 | 292.0 | 446.66 |
| 513 to 1024 | 163 | 628.43 | 1008.32 |
| 1025 to 2048 | 123 | 1295.99 | 1792.62 |
| 2049 to 4096 | 12 | 1939.71 | 3277.32 |

Targets:

- median latency per call: 0.407 s against < 1.0 s, met
- peak memory footprint over the 739 inputs: 4.74 GB (4525 MiB) against < 3.0 GB, MISSED

Scores against `bench-1.13/quota/minicheck/attempts.jsonl`: maximum absolute difference 0.0e+00 over 739 inputs.

`pmset -g therm` before: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
After: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
