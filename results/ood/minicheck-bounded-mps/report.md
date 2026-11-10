# Judge benchmark: python:minicheck_bounded:MiniCheckBoundedMps (faithfulness, sentence level)

Config hash `cbccef85dc7ad2138c39b0adc01f883056a7aa539e6334d6bbb1c28a01949d30`. Records: 8424 evaluated of 8424, 8424 scored, 0 failed.
Threshold 0.50 (source: verdict_policy).
Prediction rule: PASS if score >= 0.50 (source: verdict_policy); PARTIAL and FAIL counted as not supported.
`prev_sentence` present in 0 records, not transmitted to the judge.

| Metric | Value |
|---|---|
| Precision (not supported) | 0.627 |
| Recall (not supported) | 0.784 |
| F1 (not supported) | 0.697 |
| Balanced accuracy | 0.702 |
| Accuracy | 0.694 |
| Kappa | 0.395 |
| Minority human class (n) | 3777 |
| Kappa interpretable (minority >= 10) | yes |
| Majority-class baseline (accuracy) | 0.552 |
| Human PASS rate | 0.552 |

Bootstrap: 1000 draws grouped on 6884 groups (seed 0), 0 single-class draws excluded.
- f1_not_supported: 95% CI 0.685 to 0.709
- balanced_accuracy: 95% CI 0.692 to 0.712
- kappa: 95% CI 0.376 to 0.415

Calibration: Brier 0.2566, ECE (10 bins) 0.2255.

## Failures and cost

- failures by kind: none
- probable truncations: 6
- inputs beyond the judge's declared official truncation length (max_model_len) (2048 tokens): 793 of 8424 with a token count (9.4%), longest 32268 tokens
- latency p50 179 ms, p90 2064 ms
- peak RSS 581.1 MiB (harness process only; an Ollama model runs in another process), duration 7348.05 s

## By dataset

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| ClaimVerify | 1088 | 299 | yes | 0.630 | 0.746 | 0.488 |
| ExpertQA | 3702 | 731 | yes | 0.364 | 0.589 | 0.110 |
| FactCheck-GPT | 1566 | 376 | yes | 0.864 | 0.747 | 0.472 |
| REVEAL | 1710 | 400 | yes | 0.882 | 0.862 | 0.611 |
| WiCE | 358 | 111 | yes | 0.855 | 0.724 | 0.478 |

## By length_band

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| words_0000_0250 | 6412 | 3160 | yes | 0.711 | 0.682 | 0.363 |
| words_0251_1000 | 947 | 313 | yes | 0.655 | 0.742 | 0.453 |
| words_1001_2500 | 710 | 199 | yes | 0.623 | 0.744 | 0.449 |
| words_2501_plus | 355 | 105 | yes | 0.550 | 0.675 | 0.314 |
