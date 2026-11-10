# Cost run: mlx:Qwen/Qwen3-0.6B

Regime: one sentence per call, device cache released after each call.
Judge: stack mlx, model Qwen/Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca, device gpu, dtype mlx.core.bfloat16, published_dtype mlx.core.bfloat16, converted_to_bfloat16 False, parameters 596049920, max_length 4096, template chat, template_tokens 43, mlx 0.32.4, mlx_lm 0.32.2.
Inputs: 739 from `ragtruth-out/test.jsonl`, order of `bench-1.13/quota/hhem-mps/attempts.jsonl`; lengths 211 to 2129 tokens, 0 truncated.
Started 2026-11-09T12:46:30+00:00, ended 2026-11-09T12:51:11+00:00, 281.3 s. Previous run of this folder ended 0.8 min before the start.

| Measure | Value |
|---|---|
| Median latency, call and cache release | 293.87 ms |
| p90 latency | 707.82 ms |
| Mean latency | 371.6 ms |
| Median latency, call alone | 292.41 ms |
| Footprint after loading | 1.52 GB (1445 MiB) |
| Peak footprint over the inputs | 2.15 GB (2046 MiB) |
| Maximum-length input (4096 tokens): latency | 3416.65 ms |
| Peak footprint including that input | 2.15 GB (2046 MiB) |
| Maximum resident memory (no GPU memory) | 0.69 GB (657 MiB) |

Latency by fifth of the run, on the inputs of at most 512 tokens only (call and release). The
inputs are not in random order and their length changes along the run, so only short inputs are
compared; the last column divides each latency by the length of its input. A rise from the first
fifth to the last points to thermal drift.

| Fifth | n | p50 ms | mean tokens | p50 ms per 100 tokens |
|---|---|---|---|---|
| 1 | 32 | 169.09 | 369 | 46.07 |
| 2 | 40 | 309.71 | 410 | 81.53 |
| 3 | 88 | 249.47 | 398 | 60.14 |
| 4 | 136 | 239.92 | 366 | 64.79 |
| 5 | 143 | 218.85 | 350 | 66.68 |

Latency by input length:

| Tokens | n | p50 ms | p90 ms |
|---|---|---|---|
| 0 to 512 | 439 | 231.25 | 315.23 |
| 513 to 1024 | 174 | 361.64 | 558.56 |
| 1025 to 2048 | 123 | 727.92 | 1048.54 |
| 2049 to 4096 | 3 | 1146.93 | 1169.24 |

Targets:

- median latency per call: 0.294 s against < 1.0 s, met
- peak memory footprint over the 739 inputs: 2.15 GB (2046 MiB) against < 3.0 GB, met

Reported without threshold:

- maximum-length input of 4096 tokens, latency: 3.417 s
- peak memory footprint including that input: 2.15 GB (2046 MiB)

`pmset -g therm` before: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
After: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
