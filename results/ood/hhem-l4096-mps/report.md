# Judge benchmark: python:hhem_l4096:HHEM4096Mps (faithfulness, sentence level)

Config hash `cfd9eb00c7c54380f23057a2635e49d33a5f6359c6caff4388d5ffcb5cae9adf`. Records: 8424 evaluated of 8424, 8424 scored, 0 failed.
Threshold 0.50 (source: verdict_policy).
Prediction rule: PASS if score >= 0.50 (source: verdict_policy); PARTIAL and FAIL counted as not supported.
`prev_sentence` present in 0 records, not transmitted to the judge.

| Metric | Value |
|---|---|
| Precision (not supported) | 0.644 |
| Recall (not supported) | 0.741 |
| F1 (not supported) | 0.689 |
| Balanced accuracy | 0.704 |
| Accuracy | 0.701 |
| Kappa | 0.403 |
| Minority human class (n) | 3777 |
| Kappa interpretable (minority >= 10) | yes |
| Majority-class baseline (accuracy) | 0.552 |
| Human PASS rate | 0.552 |

Bootstrap: 1000 draws grouped on 6884 groups (seed 0), 0 single-class draws excluded.
- f1_not_supported: 95% CI 0.677 to 0.702
- balanced_accuracy: 95% CI 0.694 to 0.714
- kappa: 95% CI 0.384 to 0.423

Calibration: Brier 0.2310, ECE (10 bins) 0.1726.

## Failures and cost

- failures by kind: none
- probable truncations: 307
- inputs beyond the judge's declared training length (512 tokens): 1876 of 8424 with a token count (22.3%), longest 4096 tokens
- latency p50 57 ms, p90 833 ms
- peak RSS 1247.5 MiB (harness process only; an Ollama model runs in another process), duration 2493.934 s

## By dataset

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| ClaimVerify | 1088 | 299 | yes | 0.624 | 0.736 | 0.501 |
| ExpertQA | 3702 | 731 | yes | 0.349 | 0.576 | 0.101 |
| FactCheck-GPT | 1566 | 376 | yes | 0.847 | 0.718 | 0.414 |
| REVEAL | 1710 | 400 | yes | 0.872 | 0.866 | 0.598 |
| WiCE | 358 | 111 | yes | 0.827 | 0.757 | 0.490 |

## By length_band

| Value | n | Minority class n | Kappa interpretable (>= 10) | F1 (not supported) | Balanced accuracy | Kappa |
|---|---|---|---|---|---|---|
| words_0000_0250 | 6412 | 3160 | yes | 0.709 | 0.694 | 0.388 |
| words_0251_1000 | 947 | 313 | yes | 0.642 | 0.732 | 0.453 |
| words_1001_2500 | 710 | 199 | yes | 0.570 | 0.702 | 0.386 |
| words_2501_plus | 355 | 105 | yes | 0.489 | 0.618 | 0.204 |
