# Judge benchmark: hhem (faithfulness, sentence level)

Config hash `950d5ebb1845fe54ffece081c9797c5e49ffa472e202121ca680c66cc990ca0b`. Records: 10385 evaluated of 10385, 10385 scored, 0 failed.
Threshold 0.25 (source: dev (6011 scored records, F1 of not supported maximized)).
Prediction rule: PASS if score >= 0.25 (source: dev (6011 scored records, F1 of not supported maximized)); PARTIAL and FAIL counted as not supported.
`prev_sentence` present in 8726 records, not transmitted to the judge.

| Metric | Value |
|---|---|
| Precision (not supported) | 0.445 |
| Recall (not supported) | 0.488 |
| F1 (not supported) | 0.466 |
| Balanced accuracy | 0.725 |
| Accuracy | 0.934 |
| Kappa | 0.431 |
| Minority human class (n) | 608 |
| Kappa interpretable (minority >= 10) | yes |
| Majority-class baseline (accuracy) | 0.941 |
| Human PASS rate | 0.941 |

Bootstrap: 1000 draws grouped on 300 groups (seed 0), 0 single-class draws excluded.
- f1_not_supported: 95% CI 0.410 to 0.515
- balanced_accuracy: 95% CI 0.695 to 0.752
- kappa: 95% CI 0.375 to 0.480

Calibration: Brier 0.0758, ECE (10 bins) 0.1028.

## Failures and cost

- failures by kind: none
- probable truncations: 0
- inputs beyond the judge's declared training length (512 tokens): 4688 of 10385 with a token count (45.1%), longest 2397 tokens
- latency p50 172 ms, p90 645 ms
- peak RSS 2085.7 MiB (harness process only; an Ollama model runs in another process), duration 2816.63 s

## By task

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| QA | 5491 | 350 | yes | 0.535 | 0.767 | 0.502 |
| Summary | 4894 | 258 | yes | 0.367 | 0.669 | 0.331 |

## By generator

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| gpt-3.5-turbo-0613 | 1588 | 17 | yes | 0.233 | 0.640 | 0.222 |
| gpt-4-0613 | 1525 | 9 | no | 0.174 | 0.607 | 0.168 |
| llama-2-13b-chat | 1900 | 109 | yes | 0.368 | 0.701 | 0.321 |
| llama-2-70b-chat | 1783 | 114 | yes | 0.500 | 0.750 | 0.463 |
| llama-2-7b-chat | 2065 | 184 | yes | 0.480 | 0.710 | 0.431 |
| mistral-7B-instruct | 1524 | 175 | yes | 0.557 | 0.737 | 0.504 |
