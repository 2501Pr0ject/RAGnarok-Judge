# Cost run: mlx:PleIAs/Pleias-RAG-350M

Regime: one sentence per call, device cache released after each call.
Judge: stack mlx, model PleIAs/Pleias-RAG-350M, revision db001b29a33532583d14979c3c38ef04b6d59352, device gpu, dtype mlx.core.bfloat16, published_dtype mlx.core.bfloat16, converted_to_bfloat16 False, parameters 353424384, max_length 4096, template pleias, template_tokens 21, mlx 0.32.4, mlx_lm 0.32.2.
Inputs: 739 from `ragtruth-out/test.jsonl`, order of `bench-1.13/quota/hhem-mps/attempts.jsonl`; lengths 195 to 2203 tokens, 0 truncated.
Started 2026-11-09T12:43:22+00:00, ended 2026-11-09T12:45:42+00:00, 139.9 s. No previous run in this folder.

| Measure | Value |
|---|---|
| Median latency, call and cache release | 144.16 ms |
| p90 latency | 369.03 ms |
| Mean latency | 183.95 ms |
| Median latency, call alone | 141.1 ms |
| Footprint after loading | 0.87 GB (827 MiB) |
| Peak footprint over the inputs | 1.57 GB (1496 MiB) |
| Maximum-length input (4096 tokens): latency | 1624.37 ms |
| Peak footprint including that input | 1.57 GB (1496 MiB) |
| Maximum resident memory (no GPU memory) | 0.51 GB (487 MiB) |

Latency by fifth of the run, on the inputs of at most 512 tokens only (call and release). The
inputs are not in random order and their length changes along the run, so only short inputs are
compared; the last column divides each latency by the length of its input. A rise from the first
fifth to the last points to thermal drift.

| Fifth | n | p50 ms | mean tokens | p50 ms per 100 tokens |
|---|---|---|---|---|
| 1 | 34 | 108.29 | 359 | 30.68 |
| 2 | 43 | 130.59 | 404 | 30.73 |
| 3 | 96 | 131.86 | 389 | 32.08 |
| 4 | 139 | 107.31 | 351 | 31.87 |
| 5 | 145 | 108.11 | 334 | 35.67 |

Latency by input length:

| Tokens | n | p50 ms | p90 ms |
|---|---|---|---|
| 0 to 512 | 457 | 109.85 | 149.26 |
| 513 to 1024 | 154 | 180.78 | 252.47 |
| 1025 to 2048 | 122 | 382.86 | 473.25 |
| 2049 to 4096 | 6 | 676.49 | 693.49 |

Targets:

- median latency per call: 0.144 s against < 1.0 s, met
- peak memory footprint over the 739 inputs: 1.57 GB (1496 MiB) against < 3.0 GB, met

Reported without threshold:

- maximum-length input of 4096 tokens, latency: 1.624 s
- peak memory footprint including that input: 1.57 GB (1496 MiB)

`pmset -g therm` before: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
After: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
