# Cost run: hf:PleIAs/Pleias-RAG-350M

Regime: one sentence per call, device cache released after each call.
Judge: stack torch, model PleIAs/Pleias-RAG-350M, revision db001b29a33532583d14979c3c38ef04b6d59352, model_type llama, device mps, dtype float16, parameters 353424384, max_position_embeddings 4096, max_length 4096, template pleias, template_tokens 21, torch 2.15.0, transformers 5.22.2.
Inputs: 739 from `ragtruth-out/test.jsonl`, order of `bench-1.13/quota/hhem-mps/attempts.jsonl`; lengths 195 to 2203 tokens, 0 truncated.
Started 2026-11-09T12:51:28+00:00, ended 2026-11-09T12:54:46+00:00, 198.1 s. Previous run of this folder ended 0.3 min before the start.

| Measure | Value |
|---|---|
| Median latency, call and cache release | 196.35 ms |
| p90 latency | 505.69 ms |
| Mean latency | 258.58 ms |
| Median latency, call alone | 196.21 ms |
| Footprint after loading | 1.95 GB (1859 MiB) |
| Peak footprint over the inputs | 2.15 GB (2053 MiB) |
| Maximum-length input (4096 tokens): latency | 2010.76 ms |
| Peak footprint including that input | 2.15 GB (2053 MiB) |
| Maximum resident memory (no GPU memory) | 1.29 GB (1235 MiB) |

Latency by fifth of the run, on the inputs of at most 512 tokens only (call and release). The
inputs are not in random order and their length changes along the run, so only short inputs are
compared; the last column divides each latency by the length of its input. A rise from the first
fifth to the last points to thermal drift.

| Fifth | n | p50 ms | mean tokens | p50 ms per 100 tokens |
|---|---|---|---|---|
| 1 | 34 | 119.79 | 359 | 35.16 |
| 2 | 43 | 211.18 | 404 | 55.9 |
| 3 | 96 | 162.86 | 389 | 41.09 |
| 4 | 139 | 139.46 | 351 | 41.54 |
| 5 | 145 | 138.47 | 334 | 45.49 |

Latency by input length:

| Tokens | n | p50 ms | p90 ms |
|---|---|---|---|
| 0 to 512 | 457 | 145.79 | 208.37 |
| 513 to 1024 | 154 | 247.29 | 397.22 |
| 1025 to 2048 | 122 | 507.87 | 860.71 |
| 2049 to 4096 | 6 | 1166.17 | 1257.44 |

Targets:

- median latency per call: 0.196 s against < 1.0 s, met
- peak memory footprint over the 739 inputs: 2.15 GB (2053 MiB) against < 3.0 GB, met

Reported without threshold:

- maximum-length input of 4096 tokens, latency: 2.011 s
- peak memory footprint including that input: 2.15 GB (2053 MiB)

`pmset -g therm` before: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
After: Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded.
