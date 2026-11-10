# Cost run: hhem

Regime: one sentence per call, device cache released after each call.
Judge: stack torch, model hhem, device mps, dtype float32, ragnarok_ai 1.13.0, torch 2.15.0, transformers 5.22.2.
Inputs: 739 from `ragtruth-out/test.jsonl`, order of `bench-1.13/quota/hhem-mps/attempts.jsonl`; lengths 191 to 2356 tokens, 0 truncated.
Started 2026-11-09T13:42:50+00:00, ended 2026-11-09T13:44:25+00:00, 95.4 s. Previous run of this folder ended 41.2 min before the start.

| Measure | Value |
|---|---|
| Median latency, call and cache release | 71.89 ms |
| p90 latency | 260.07 ms |
| Mean latency | 118.28 ms |
| Median latency, call alone | 71.88 ms |
| Footprint after loading | 1.44 GB (1376 MiB) |
| Peak footprint over the inputs | 2.54 GB (2418 MiB) |
| Maximum resident memory (no GPU memory) | 0.94 GB (892 MiB) |

Latency by fifth of the run, on the inputs of at most 512 tokens only (call and release). The
inputs are not in random order and their length changes along the run, so only short inputs are
compared; the last column divides each latency by the length of its input. A rise from the first
fifth to the last points to thermal drift.

| Fifth | n | p50 ms | mean tokens | p50 ms per 100 tokens |
|---|---|---|---|---|
| 1 | 33 | 52.64 | 374 | 14.89 |
| 2 | 34 | 65.29 | 403 | 14.92 |
| 3 | 88 | 54.97 | 385 | 14.86 |
| 4 | 136 | 49.88 | 361 | 14.63 |
| 5 | 132 | 47.84 | 331 | 15.1 |

Latency by input length:

| Tokens | n | p50 ms | p90 ms |
|---|---|---|---|
| 0 to 512 | 423 | 50.02 | 72.23 |
| 513 to 1024 | 183 | 94.74 | 145.85 |
| 1025 to 2048 | 122 | 263.07 | 455.59 |
| 2049 to 4096 | 11 | 649.38 | 675.73 |

Targets:

- median latency per call: 0.072 s against < 1.0 s, met
- peak memory footprint over the 739 inputs: 2.54 GB (2418 MiB) against < 3.0 GB, met

Scores against `bench-1.13/quota/hhem-mps/attempts.jsonl`: maximum absolute difference 0.0e+00 over 739 inputs.

`pmset -g therm` before: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
After: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
