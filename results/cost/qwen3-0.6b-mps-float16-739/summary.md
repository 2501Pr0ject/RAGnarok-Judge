# Cost run: hf:Qwen/Qwen3-0.6B

Regime: one sentence per call, device cache released after each call.
Judge: stack torch, model Qwen/Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca, model_type qwen3, device mps, dtype float16, parameters 596049920, max_position_embeddings 40960, max_length 4096, template chat, template_tokens 43, torch 2.15.0, transformers 5.22.2.
Inputs: 739 from `ragtruth-out/test.jsonl`, order of `bench-1.13/quota/hhem-mps/attempts.jsonl`; lengths 211 to 2129 tokens, 0 truncated.
Started 2026-11-09T12:54:58+00:00, ended 2026-11-09T13:01:39+00:00, 401.9 s. Previous run of this folder ended 0.2 min before the start.

| Measure | Value |
|---|---|
| Median latency, call and cache release | 396.35 ms |
| p90 latency | 1122.73 ms |
| Mean latency | 526.74 ms |
| Median latency, call alone | 395.89 ms |
| Footprint after loading | 2.09 GB (1991 MiB) |
| Peak footprint over the inputs | 2.98 GB (2839 MiB) |
| Maximum-length input (4096 tokens): latency | 5972.04 ms |
| Peak footprint including that input | 2.98 GB (2839 MiB) |
| Maximum resident memory (no GPU memory) | 2.07 GB (1969 MiB) |

Latency by fifth of the run, on the inputs of at most 512 tokens only (call and release). The
inputs are not in random order and their length changes along the run, so only short inputs are
compared; the last column divides each latency by the length of its input. A rise from the first
fifth to the last points to thermal drift.

| Fifth | n | p50 ms | mean tokens | p50 ms per 100 tokens |
|---|---|---|---|---|
| 1 | 32 | 269.15 | 369 | 65.85 |
| 2 | 40 | 359.62 | 410 | 81.81 |
| 3 | 88 | 338.91 | 398 | 82.07 |
| 4 | 136 | 285.07 | 366 | 84.23 |
| 5 | 143 | 268.42 | 350 | 86.61 |

Latency by input length:

| Tokens | n | p50 ms | p90 ms |
|---|---|---|---|
| 0 to 512 | 439 | 295.09 | 411.5 |
| 513 to 1024 | 174 | 492.31 | 737.89 |
| 1025 to 2048 | 123 | 1188.69 | 1709.09 |
| 2049 to 4096 | 3 | 1849.38 | 1922.07 |

Targets:

- median latency per call: 0.396 s against < 1.0 s, met
- peak memory footprint over the 739 inputs: 2.98 GB (2839 MiB) against < 3.0 GB, met

Reported without threshold:

- maximum-length input of 4096 tokens, latency: 5.972 s
- peak memory footprint including that input: 2.98 GB (2839 MiB)

`pmset -g therm` before: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
After: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
