# Judge benchmark: python:rj_judge:RJQwen3 (faithfulness, sentence level)

Config hash `b2d8fe21e651d30a5aff2ba6d30d81605df50b3fae4da1c03256b31fd5da43a8`. Records: 8424 evaluated of 8424, 8424 scored, 0 failed.
Threshold 0.50 (source: verdict_policy).
Prediction rule: PASS if score >= 0.50 (source: verdict_policy); PARTIAL and FAIL counted as not supported.
`prev_sentence` present in 0 records, not transmitted to the judge.

| Metric | Value |
|---|---|
| Precision (not supported) | 0.654 |
| Recall (not supported) | 0.802 |
| F1 (not supported) | 0.720 |
| Balanced accuracy | 0.728 |
| Accuracy | 0.721 |
| Kappa | 0.447 |
| Minority human class (n) | 3777 |
| Kappa interpretable (minority >= 10) | yes |
| Majority-class baseline (accuracy) | 0.552 |
| Human PASS rate | 0.552 |

Bootstrap: 1000 draws grouped on 6884 groups (seed 0), 0 single-class draws excluded.
- f1_not_supported: 95% CI 0.708 to 0.733
- balanced_accuracy: 95% CI 0.719 to 0.739
- kappa: 95% CI 0.427 to 0.468

## Three classes

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CONTRADICTED | 0.000 | 0.000 | 0.000 | 0 |
| SUPPORTED | 0.798 | 0.664 | 0.725 | 4647 |
| UNSUPPORTED | 0.659 | 0.776 | 0.713 | 3777 |

Macro F1 0.479. Confusion (rows: truth, columns: predicted):
| | CONTRADICTED | SUPPORTED | UNSUPPORTED |
|---|---|---|---|
| CONTRADICTED | 0 | 0 | 0 |
| SUPPORTED | 43 | 3087 | 1517 |
| UNSUPPORTED | 63 | 783 | 2931 |

Calibration: Brier 0.2276, ECE (10 bins) 0.1949.

## Failures and cost

- failures by kind: none
- probable truncations: 282
- latency p50 222 ms, p90 1495 ms
- peak RSS 1744.9 MiB (harness process only; an Ollama model runs in another process), duration 4718.179 s

## By dataset

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| ClaimVerify | 1088 | 299 | yes | 0.571 | 0.702 | 0.449 |
| ExpertQA | 3702 | 731 | yes | 0.359 | 0.585 | 0.111 |
| FactCheck-GPT | 1566 | 376 | yes | 0.885 | 0.755 | 0.515 |
| REVEAL | 1710 | 400 | yes | 0.925 | 0.887 | 0.714 |
| WiCE | 358 | 111 | yes | 0.850 | 0.735 | 0.486 |

## By length_band

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| words_0000_0250 | 6412 | 3160 | yes | 0.738 | 0.708 | 0.415 |
| words_0251_1000 | 947 | 313 | yes | 0.646 | 0.735 | 0.462 |
| words_1001_2500 | 710 | 199 | yes | 0.619 | 0.733 | 0.481 |
| words_2501_plus | 355 | 105 | yes | 0.550 | 0.680 | 0.352 |
