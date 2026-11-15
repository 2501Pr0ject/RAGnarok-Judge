# Judge benchmark: python:rj_judge:RJQwen3 (faithfulness, sentence level)

Config hash `b2d8fe21e651d30a5aff2ba6d30d81605df50b3fae4da1c03256b31fd5da43a8`. Records: 10385 evaluated of 10385, 10385 scored, 0 failed.
Threshold 0.50 (source: verdict_policy).
Prediction rule: PASS if score >= 0.50 (source: verdict_policy); PARTIAL and FAIL counted as not supported.
`prev_sentence` present in 8726 records, not transmitted to the judge.

| Metric | Value |
|---|---|
| Precision (not supported) | 0.423 |
| Recall (not supported) | 0.625 |
| F1 (not supported) | 0.505 |
| Balanced accuracy | 0.786 |
| Accuracy | 0.928 |
| Kappa | 0.467 |
| Minority human class (n) | 608 |
| Kappa interpretable (minority >= 10) | yes |
| Majority-class baseline (accuracy) | 0.941 |
| Human PASS rate | 0.941 |

Bootstrap: 1000 draws grouped on 300 groups (seed 0), 0 single-class draws excluded.
- f1_not_supported: 95% CI 0.447 to 0.552
- balanced_accuracy: 95% CI 0.754 to 0.817
- kappa: 95% CI 0.410 to 0.514

## Three classes

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CONTRADICTED | 0.125 | 0.115 | 0.120 | 139 |
| SUPPORTED | 0.975 | 0.952 | 0.964 | 9777 |
| UNSUPPORTED | 0.454 | 0.691 | 0.548 | 469 |

Macro F1 0.544. Confusion (rows: truth, columns: predicted):
| | CONTRADICTED | SUPPORTED | UNSUPPORTED |
|---|---|---|---|
| CONTRADICTED | 16 | 101 | 22 |
| SUPPORTED | 101 | 9308 | 368 |
| UNSUPPORTED | 11 | 134 | 324 |

Calibration: Brier 0.0564, ECE (10 bins) 0.0559.

## Failures and cost

- failures by kind: none
- probable truncations: 0
- latency p50 369 ms, p90 856 ms
- peak RSS 1731.6 MiB (harness process only; an Ollama model runs in another process), duration 4736.3 s

## By task

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| QA | 5491 | 350 | yes | 0.547 | 0.831 | 0.508 |
| Summary | 4894 | 258 | yes | 0.437 | 0.723 | 0.402 |

## By generator

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| gpt-3.5-turbo-0613 | 1588 | 17 | yes | 0.310 | 0.755 | 0.300 |
| gpt-4-0613 | 1525 | 9 | no | 0.077 | 0.598 | 0.068 |
| llama-2-13b-chat | 1900 | 109 | yes | 0.464 | 0.787 | 0.421 |
| llama-2-70b-chat | 1783 | 114 | yes | 0.500 | 0.799 | 0.457 |
| llama-2-7b-chat | 2065 | 184 | yes | 0.509 | 0.764 | 0.453 |
| mistral-7B-instruct | 1524 | 175 | yes | 0.633 | 0.795 | 0.585 |
