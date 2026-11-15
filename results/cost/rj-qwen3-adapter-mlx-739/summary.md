# Cost run: mlx:Qwen/Qwen3-0.6B

Regime: one sentence per call, device cache released after each call.
Judge: stack mlx, model Qwen/Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca, device gpu, dtype mlx.core.bfloat16, published_dtype mlx.core.bfloat16, converted_to_bfloat16 False, parameters 596049920, max_length 4096, template declared (qwen3), template_tokens 84, adapters separate, not merged, adapter_folder checkpoint-3462, adapter_update 3462, adapter_rank 8, adapter_scale 2.0, labels 3, mlx 0.32.4, mlx_lm 0.32.2.
Inputs: 739 from `../ragnarok-judge-ref/ragtruth-out/test.jsonl`, order of `../ragnarok-judge-ref/bench-1.13/quota/hhem-mps/attempts.jsonl`; lengths 252 to 2170 tokens, 0 truncated.
Started 2026-11-15T13:54:17+00:00, ended 2026-11-15T13:59:08+00:00, 291.8 s. Previous run of this folder ended 8612.1 min before the start.

| Measure | Value |
|---|---|
| Median latency, call and cache release | 302.27 ms |
| p90 latency | 775.79 ms |
| Mean latency | 388.23 ms |
| Median latency, call alone | 300.38 ms |
| Footprint after loading | 1.54 GB (1467 MiB) |
| Peak footprint over the inputs | 2.68 GB (2552 MiB) |
| Maximum-length input (4096 tokens): latency | 3010.6 ms |
| Peak footprint including that input | 2.68 GB (2552 MiB) |
| Maximum resident memory (no GPU memory) | 1.62 GB (1542 MiB) |

Latency by fifth of the run, on the inputs of at most 512 tokens only (call and release). The
inputs are not in random order and their length changes along the run, so only short inputs are
compared; the last column divides each latency by the length of its input. A rise from the first
fifth to the last points to thermal drift.

| Fifth | n | p50 ms | mean tokens | p50 ms per 100 tokens |
|---|---|---|---|---|
| 1 | 26 | 226.62 | 380 | 60.29 |
| 2 | 32 | 270.34 | 430 | 57.72 |
| 3 | 67 | 253.14 | 408 | 60.8 |
| 4 | 116 | 225.44 | 383 | 61.43 |
| 5 | 125 | 221.46 | 369 | 62.5 |

Latency by input length:

| Tokens | n | p50 ms | p90 ms |
|---|---|---|---|
| 0 to 512 | 366 | 224.91 | 286.11 |
| 513 to 1024 | 241 | 342.53 | 488.56 |
| 1025 to 2048 | 126 | 783.05 | 992.28 |
| 2049 to 4096 | 6 | 1316.54 | 1367.25 |

Targets:

- median latency per call: 0.302 s against < 1.0 s, met
- peak memory footprint over the 739 inputs: 2.68 GB (2552 MiB) against < 3.0 GB, met

Reported without threshold:

- maximum-length input of 4096 tokens, latency: 3.011 s
- peak memory footprint including that input: 2.68 GB (2552 MiB)

`pmset -g therm` before: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
After: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
