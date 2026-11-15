# Training and selection of RAGnarok Judge: aggregates

Training plan: `TRAINING.md` (tag `training-plan-v1`), protocol: `TARGETS.md` (tag
`protocol-v1`) and its amendment 1. Code: commits `53e10dc` (runs) and `73b6084` (scoring of
the retained judge). Everything here was fixed before the retained judge was scored on the
test set or out of domain. Aggregates only: no text, no label and no score of a sentence. The
declarations that go with these figures are in `declarations.md`.

| File | Content |
|---|---|
| `runs.json` | the pilot and the three runs: checkpoints, durations, memory, resumptions |
| `base-tie-break.json` | output of `training/dev_eval.py`: the tie-break of amendment 1 |
| `dev-threshold.json` | the dev threshold of the retained judge, with the hash of the scores used |
| `ood-untruncated.json` | counts of the common untruncated out-of-domain subset |
| `train-check-2026-11-14.json` | check on training examples of 2026-11-14 (`declarations.md`) |
| `plugin-equivalence-2026-11-14.json` | equivalence checks of the scoring plugin (`declarations.md`) |
| `../cost/rj-qwen3-adapter-mlx-739/` | cost of the retained judge (section 8 of `TARGETS.md`) |

## The 12 checkpoints scored on dev (section 5 of `TRAINING.md`)

3 configurations, 4 checkpoints each, whole dev set (6 011 sentences), adapters separate, each
scored once. Criterion of section 12 of `TARGETS.md`: mean of the two standardised partial
AUCs over a false alarm rate of 0 to 0.10.

| Configuration | Checkpoint | Criterion | pAUC UNSUPPORTED | pAUC CONTRADICTED |
|---|---|---|---|---|
| Pleias-RAG-350M, 1e-4 | 866 | 0.5879 | 0.6601 | 0.5156 |
| | 1 731 | 0.5987 | 0.6758 | 0.5216 |
| | 2 597 | 0.6038 | 0.6877 | 0.5199 |
| | 3 462 | 0.6050 | 0.6914 | 0.5186 |
| Pleias-RAG-350M, 3e-4 | 866 | 0.5835 | 0.6566 | 0.5104 |
| | 1 731 | 0.6030 | 0.7021 | 0.5040 |
| | 2 597 | 0.6036 | 0.7010 | 0.5062 |
| | 3 462 | 0.6111 | 0.7068 | 0.5154 |
| Qwen3-0.6B, 3e-4 | 866 | 0.6736 | 0.7861 | 0.5611 |
| | 1 731 | 0.6836 | 0.8082 | 0.5589 |
| | 2 597 | 0.7068 | 0.8149 | 0.5987 |
| | 3 462 | **0.7112** | 0.8116 | 0.6108 |

Dev reference values (section 12 of `TARGETS.md`): HHEM 0.641, MiniCheck 0.619. The dev values
of the retained checkpoint are the best of these 12.

## Winner of runs 1 and 2 (section 5 of `TRAINING.md`)

Best checkpoint of run 1 (1e-4): 0.6050; of run 2 (3e-4): 0.6111. Not a tie: **3e-4**, the
peak learning rate of run 3 (Qwen3-0.6B).

## Tie-break between the bases (amendment 1 of `TARGETS.md`)

| | Value |
|---|---|
| Best of Pleias-RAG-350M (8 checkpoints) | `pleias-lr3e-4/checkpoint-3462`, 0.6111 [0.5832, 0.6403] |
| Best of Qwen3-0.6B (4 checkpoints) | `qwen3-lr3e-4/checkpoint-3462`, 0.7112 [0.6751, 0.7470] |
| Difference, Pleias-RAG-350M minus Qwen3-0.6B | -0.1001 [-0.1301, -0.0712] |
| Standard error of the difference | 0.0151 (paired bootstrap by source, 163 sources, 10 000 resamples, seed 7) |
| Within one standard error | no (6.6 standard errors) |
| **Base retained** | **Qwen3-0.6B** |

## The retained judge

| | Value |
|---|---|
| Base | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, bfloat16, MLX |
| Adapters | LoRA rank 8, scale 2.0, every linear module of the 28 blocks, kept separate, not merged |
| Checkpoint | `qwen3-lr3e-4/checkpoint-3462`, update 3 462 of 3 462, no resumption |
| `adapters.safetensors` SHA-256 | `44cdeecac3ae75c9bb53d538a14968667f9af2fa730a9dfae9477b6f2b1bed31` (20 227 058 bytes) |

The scoring plugin `training/rj_judge.py` refuses any other adapters.

## Dev threshold (sections 4 and 12 of `TARGETS.md`)

Derived once, on 2026-11-14, from the dev scores of the run, the checkpoint not being scored
on dev again.

| | Value |
|---|---|
| Scores used | `dev_scores.json` of the checkpoint, SHA-256 `f68740b28e563a8f659aac7b93e948296242d7d86f626d15b4cd4dbb1139bd3c` |
| Dev file | SHA-256 `18a8e0ff0126ba12c8b76ec05391397b7f8d58438b0184ba0bd2d8907f89ac32` |
| Rule | n = 5 536 SUPPORTED dev sentences, k = floor(0.04 n) = 221, the (k+1)-th smallest score |
| **Dev threshold** | **0.20575958490371704**; a sentence is flagged when its score is below it |
| Dev specificity | 0.9601 (221 SUPPORTED flagged, one score at the threshold) |

The same value is returned by `Split.thr` of `analysis/ragtruth_analysis.py`.

## Common untruncated out-of-domain subset (sections 7 and 14 of `TARGETS.md`)

The inputs that neither RJ nor HHEM truncates, as `ood/ood_metrics.py` takes them (union of
the `context_truncated` marks of the runs given). RJ's marks are computed with the tokenizer of
Qwen3-0.6B and the declared template at L = 4096, without any scoring and without the labels;
HHEM's come from its run of 2026-11-09.

| Dataset | Inputs | Truncated by HHEM | Truncated by RJ | By RJ only | Untruncated, HHEM alone | Common subset |
|---|---|---|---|---|---|---|
| WiCE | 358 | 35 | 31 | 0 | 323 | 323 |
| REVEAL | 1 710 | 0 | 0 | 0 | 1 710 | 1 710 |
| ClaimVerify | 1 088 | 147 | 141 | 2 | 941 | 939 |
| FactCheck-GPT | 1 566 | 0 | 0 | 0 | 1 566 | 1 566 |
| ExpertQA | 3 702 | 125 | 110 | 1 | 3 577 | 3 576 |
| All five | 8 424 | 307 | 282 | 3 | 8 117 | **8 114** |

As announced in section 7 of `TARGETS.md` for Qwen3-0.6B: 282 inputs, 3 of them outside HHEM's
307, a common subset of 8 114.

## Cost of the retained judge (section 8 of `TARGETS.md`)

Measured on 2026-11-15 from 13:54 to 13:59 UTC, in the configuration of use: the 739
sentences, one sentence per call, cache released after each call. The machine was on mains
power and under no heavy load; the run was started right after a few version-control
commands, and the 15 minutes of rest of section 8 were not timed. No other measurement run
preceded it (the previous cost run had ended 8 612 minutes before), and no thermal warning
was recorded before or after. Full report: `results/cost/rj-qwen3-adapter-mlx-739/summary.md`.

Both targets are met with a margin: the median latency is less than a third of the target,
and the peak footprint is 0.32 GB (11%) below it.

| Measure | Value | Target |
|---|---|---|
| Median latency per call | 0.302 s | < 1 s, met |
| Peak memory footprint over the 739 inputs | 2.68 GB | < 3 GB, met |
| p90 latency | 0.776 s | reported |
| Input of 4096 tokens: latency, peak with it | 3.011 s, 2.68 GB | reported |

## Durations and memory of the runs

On the training machine (MacBook Air M2, 16 GB), on mains power, under `caffeinate -i` and
`/usr/bin/time -l`. 1 GB = 10^9 bytes.

| Run | Wall clock | Training | 4 dev passes | Per update | Peak footprint | MLX peak | Resumptions |
|---|---|---|---|---|---|---|---|
| Pleias-RAG-350M, 1e-4 | 9 h 16 min 51 s | 7.23 h | 2.03 h | 7.52 s | 2.90 GB | 2.10 GB | none |
| Pleias-RAG-350M, 3e-4 | 8 h 41 min 01 s | 6.78 h | 1.89 h | 7.05 s | 2.97 GB | 2.10 GB | none |
| Qwen3-0.6B, 3e-4 | 14 h 21 min 33 s | 11.29 h | 3.06 h | 11.74 s | 3.51 GB | 2.67 GB | none |
| Pilot, Pleias-RAG-350M, 1e-4, 300 updates | 40 min 22 s | | no dev | 7.74 s | 2.86 GB | 2.10 GB | one, on purpose |

The swap rule of section 7 of `TRAINING.md` was active in every run and stopped none. The
three runs took 32.3 hours in all, against about 39 in the plan.
