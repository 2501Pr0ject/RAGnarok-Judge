# Judge benchmark: minicheck (faithfulness, sentence level)

Config hash `d5ab1a32c50afb483cf1fce482efd679c4bdc43212624fdd33a2db121dcb47dd`. Records: 10385 evaluated of 10385, 10385 scored, 0 failed.
Threshold 0.20 (source: dev (6011 scored records, F1 of not supported maximized)).
Prediction rule: PASS if score >= 0.20 (source: dev (6011 scored records, F1 of not supported maximized)); PARTIAL and FAIL counted as not supported.
`prev_sentence` present in 8726 records, not transmitted to the judge.

| Metric | Value |
|---|---|
| Precision (not supported) | 0.287 |
| Recall (not supported) | 0.637 |
| F1 (not supported) | 0.396 |
| Balanced accuracy | 0.769 |
| Accuracy | 0.886 |
| Kappa | 0.342 |
| Minority human class (n) | 608 |
| Kappa interpretable (minority >= 10) | yes |
| Majority-class baseline (accuracy) | 0.941 |
| Human PASS rate | 0.941 |

Bootstrap: 1000 draws grouped on 300 groups (seed 0), 0 single-class draws excluded.
- f1_not_supported: 95% CI 0.344 to 0.450
- balanced_accuracy: 95% CI 0.737 to 0.797
- kappa: 95% CI 0.293 to 0.394

Calibration: Brier 0.1243, ECE (10 bins) 0.1541.

## Failures and cost

- failures by kind: none
- probable truncations: 0
- inputs beyond the judge's declared official truncation length (max_model_len) (2048 tokens): 162 of 10385 with a token count (1.6%), longest 2672 tokens
- latency p50 350 ms, p90 1424 ms
- peak RSS 531.3 MiB (harness process only; an Ollama model runs in another process), duration 6117.87 s

## By task

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| QA | 5491 | 350 | yes | 0.398 | 0.801 | 0.337 |
| Summary | 4894 | 258 | yes | 0.390 | 0.722 | 0.348 |

## By generator

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| gpt-3.5-turbo-0613 | 1588 | 17 | yes | 0.133 | 0.655 | 0.118 |
| gpt-4-0613 | 1525 | 9 | no | 0.025 | 0.533 | 0.015 |
| llama-2-13b-chat | 1900 | 109 | yes | 0.353 | 0.777 | 0.293 |
| llama-2-70b-chat | 1783 | 114 | yes | 0.421 | 0.802 | 0.363 |
| llama-2-7b-chat | 2065 | 184 | yes | 0.395 | 0.734 | 0.312 |
| mistral-7B-instruct | 1524 | 175 | yes | 0.549 | 0.771 | 0.482 |
