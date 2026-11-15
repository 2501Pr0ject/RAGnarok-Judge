# Results of RAGnarok Judge

The retained judge (RJ) was evaluated once, on 2026-11-15, on the RAGTruth test set and on the five
out-of-domain datasets of LLM-AggreFact. The protocol is `TARGETS.md` (tag `protocol-v1`) with its
amendment 1, and the training plan is `TRAINING.md` (tag `training-plan-v1`). The judge, its
selection, its dev threshold and its cost were fixed at the tag `selection-v1`, set before this
evaluation. RJ was selected on dev among **3 configurations and 12 checkpoints** (section 12 of
`TARGETS.md`). These results are published whatever the outcome (sections 10 and 14 of
`TARGETS.md`). They are aggregates only, with no text, no label and no score of a sentence.

## Claim allowed (section 5 of `TARGETS.md`)

> no demonstrated gain on contradictions; RJ is not worse than HHEM on unsupported sentences

This is the "P1 not met, P2 met" row of the table of section 5, quoted word for word. As section 5
states, it is a statement about discrimination at a common specificity of 0.96 on the RAGTruth test
set, against HHEM. It is also a statement about agreement with the RAGTruth labels, whose quality is
described in section 2 of `TARGETS.md`. Any statement about a threshold set in advance relies on
the transfer analysis below.

The out-of-domain target of section 7 is met. Its reading, quoted from section 7: "out of domain,
RJ separates supported from not supported sentences no worse than HHEM, within 0.02 of AUC." Also
from section 7: "The target says nothing about a threshold set on RAGTruth remaining usable
elsewhere."

## How to read this file

- **Protocol conclusion**: the outcome of a test written in `TARGETS.md` before any training. There
  are four: P1 and P2 with the Holm procedure, the out-of-domain target and the cost targets. Only
  these support a claim.
- **Descriptive**: every other figure. This covers the secondary measures of sections 6 and 7,
  tests against any other null hypothesis, and every difference to MiniCheck. These figures have no
  pass or fail threshold and support no claim, even when an interval excludes zero. Two of them
  look like tests and are not:
  - the p-value of Delta_U against Delta_U <= 0;
  - the out-of-domain difference whose interval lies above 0.

  The protocol tests no superiority on unsupported sentences and none out of domain.
- MiniCheck is a reference, not a comparator. Its differences are reported next to every claim
  (sections 5 and 7) and enter none.

## Outcomes

| Target | Section | Null hypothesis | Observed | p, one-sided | Level | Outcome |
|---|---|---|---|---|---|---|
| P1, contradictions (superiority) | 5 | Delta_C <= 0 | +0.0719 [-0.0063, +0.1704] | 0.0394 | 0.025 (Holm, second) | **not met** |
| P2, unsupported sentences (non-inferiority) | 5 | Delta_U <= -0.05 | +0.0853 [+0.0434, +0.1477] | 0.0001 | 0.0125 (Holm, first) | **met** |
| Out of domain (non-inferiority) | 7 | Delta_OOD <= -0.02 | +0.0245 [+0.0142, +0.0351] | 0.0001 | 0.025, outside the Holm family | **met** |
| Cost, median latency per call | 8 | | 0.302 s | | < 1 s | **met** |
| Cost, peak memory footprint | 8 | | 2.68 GB | | < 3 GB | **met** |

All differences are RJ minus HHEM. Intervals are percentile intervals at 95% from 10000 resamples
with seed 7, the same resamples for every judge. On RAGTruth the unit of resampling is the source
(300 groups); out of domain it is the document, within each dataset.

## Primary targets on the RAGTruth test set (protocol conclusions)

These are measured at equal specificity on the test set (section 4): each judge's threshold flags
391 of the 9777 SUPPORTED test sentences, and it is re-derived in every resample. The test set
holds 9777 SUPPORTED, 469 UNSUPPORTED and 139 CONTRADICTED sentences from 300 sources.

| Quantity | RJ | HHEM | MiniCheck |
|---|---|---|---|
| Threshold on the full test | 0.35262 | 0.26894 | 0.06499 |
| Specificity | 0.9600 (391 flagged, 1 tied) | 0.9600 (391 flagged, 1 tied) | 0.9600 (391 flagged, 1 tied) |
| Recall, CONTRADICTED (139) | 0.2302 [0.1481, 0.3333] | 0.1583 [0.0976, 0.2222] | 0.2230 [0.1366, 0.2917] |
| Recall, UNSUPPORTED (469) | 0.6951 [0.6163, 0.7717] | 0.6098 [0.5277, 0.6795] | 0.5629 [0.4900, 0.6301] |
| Recall, not supported (608) | 0.5888 [0.5261, 0.6511] | 0.5066 [0.4390, 0.5603] | 0.4852 [0.4202, 0.5378] |
| Precision, not supported | 0.4780 [0.4220, 0.5352] | 0.4406 [0.3774, 0.4987] | 0.4300 [0.3660, 0.4913] |
| F1, not supported | 0.5276 [0.4760, 0.5755] | 0.4713 [0.4125, 0.5181] | 0.4560 [0.3972, 0.5054] |

| Quantity | RJ minus HHEM | p, one-sided | RJ minus MiniCheck (alongside) | p, one-sided |
|---|---|---|---|---|
| Delta_C (P1) | +0.0719 [-0.0063, +0.1704], se 0.0451 | Delta <= 0: 0.0394 | +0.0072 [-0.0853, +0.1364], se 0.0562 | Delta <= 0: 0.3483 |
| Delta_U (P2) | +0.0853 [+0.0434, +0.1477], se 0.0268 | Delta <= -0.05: 0.0001 | +0.1322 [+0.0812, +0.1953], se 0.0290 | Delta <= -0.05: 0.0001 |

**Holm procedure** (section 5), over P1 and P2 at a family-wise one-sided level of 0.025:

1. P2 has the smaller p-value, 0.0001. It is compared with 0.0125 and passes: **P2 is met**.
2. P1 is then compared with 0.025. Its p-value is 0.0394, so **P1 is not met**.

Section 5 gives the resolution of the test set. P1 is rejected from an observed Delta_C of about
+0.100 at 0.025; the observed value is +0.072. The correlation of the two recalls on CONTRADICTED
across resamples is 0.42 for RJ and HHEM; section 5 had found the recalls of the two baselines
uncorrelated.

**Descriptive, not tests of the protocol:**

- Delta_U against Delta_U <= 0: p = 0.0002.
- Recall on the 608 not supported sentences: Delta_NS = +0.0822 [+0.0452, +0.1366] against HHEM
  and +0.1036 [+0.0602, +0.1637] against MiniCheck.
- Precision: +0.0373 [+0.0205, +0.0624] against HHEM and +0.0479 [+0.0276, +0.0762] against
  MiniCheck.
- F1: +0.0563 [+0.0311, +0.0930] against HHEM and +0.0717 [+0.0417, +0.1125] against MiniCheck.

## Out-of-domain target (protocol conclusion)

The metric is the AUC ROC on raw scores per dataset, then the unweighted mean over WiCE, REVEAL,
ClaimVerify, FactCheck-GPT and ExpertQA (section 7). The target is computed on all 8424 inputs. No
threshold enters it. The AUC of each dataset is given in its own section below.

| | Inputs | Mean AUC, RJ | Mean AUC, HHEM | Delta_OOD | Standard error | p (Delta <= -0.02) |
|---|---|---|---|---|---|---|
| **All inputs (the target)** | 8424 | 0.8245 [0.8129, 0.8355] | 0.8000 [0.7868, 0.8128] | **+0.0245 [+0.0142, +0.0351]** | 0.0053 | 0.0001 |
| Common untruncated subset (secondary) | 8114 | 0.8260 [0.8142, 0.8375] | 0.8046 [0.7910, 0.8178] | +0.0215 [+0.0109, +0.0318] | 0.0054 | 0.0001 |

The null hypothesis is rejected at 0.025 from an observed Delta_OOD of -0.0096. The two mean AUCs
have a correlation of 0.64 across resamples; if the two judges were independent, the standard error
would be 0.0088 instead of 0.0053. No resample was excluded.

**Alongside, MiniCheck:** RJ minus MiniCheck = +0.0163 [+0.0044, +0.0281] on all inputs and
+0.0144 [+0.0025, +0.0263] on the common subset.

**Descriptive:** the interval of Delta_OOD lies above 0 on both sets of inputs. The protocol does
not test superiority out of domain, so this is not a claim that RJ separates better than HHEM.

## Cost (protocol conclusion, section 8)

RJ was measured on 2026-11-15, before the evaluation, in its configuration of use: Qwen3-0.6B in
bfloat16 on MLX, with the LoRA adapters kept separate and not merged (declared as section 8
requires). The regime is the 739 sentences, one sentence per call, with the cache released after
each call. The full report is in `results/cost/rj-qwen3-adapter-mlx-739/summary.md`, and the conditions are
in `results/training/README.md`.

| Measure | RJ | Target | HHEM, reference regime | MiniCheck, reference regime |
|---|---|---|---|---|
| Median latency per call | 0.302 s | < 1 s, met | 0.072 s | 0.407 s |
| Peak memory footprint over the 739 inputs | 2.68 GB | < 3 GB, met | 2.54 GB | 4.74 GB |
| p90 latency | 0.776 s | reported | | |
| Input of 4096 tokens: latency, peak with it | 3.011 s, 2.68 GB | reported | | |

The two baselines were measured with PyTorch on MPS (`results/cost/hhem-mps-739/`,
`results/cost/minicheck-mps-739/`). They are given here for context only.

## Secondary measures on the RAGTruth test set (descriptive, section 6)

### Transfer of the operating point

Each judge is read at its dev threshold, a constant set on the dev split (5536 SUPPORTED sentences,
k = 221). RJ's dev threshold, 0.20575958490371704, was derived once, at `selection-v1`, from the dev
scores of its run.

| Quantity | RJ | HHEM | MiniCheck |
|---|---|---|---|
| Dev threshold | 0.20576 | 0.18617 | 0.04438 |
| Specificity on test | 0.9732 [0.9670, 0.9786] | 0.9723 [0.9674, 0.9769] | 0.9770 [0.9723, 0.9814] |
| Recall, CONTRADICTED | 0.1655 [0.0940, 0.2406] | 0.1439 [0.0857, 0.2061] | 0.0935 [0.0496, 0.1429] |
| Recall, UNSUPPORTED | 0.6418 [0.5655, 0.7146] | 0.5458 [0.4773, 0.6167] | 0.4733 [0.4020, 0.5476] |
| Recall, not supported | 0.5329 [0.4690, 0.5930] | 0.4539 [0.3960, 0.5092] | 0.3865 [0.3240, 0.4485] |
| Precision, not supported | 0.5529 [0.4820, 0.6221] | 0.5046 [0.4369, 0.5694] | 0.5109 [0.4372, 0.5832] |
| F1, not supported | 0.5427 [0.4859, 0.5929] | 0.4779 [0.4237, 0.5256] | 0.4401 [0.3797, 0.4933] |
| Balanced accuracy at the dev threshold | 0.7530 [0.7217, 0.7824] | 0.7131 [0.6846, 0.7404] | 0.6817 [0.6512, 0.7117] |
| Balanced accuracy at 0.5 | 0.7860 [0.7544, 0.8163] | 0.7597 [0.7280, 0.7880] | 0.7773 [0.7460, 0.8039] |

| Quantity, at the dev thresholds | RJ minus HHEM | RJ minus MiniCheck |
|---|---|---|
| Specificity on test | +0.0009 [-0.0046, +0.0059] | -0.0038 [-0.0096, +0.0015] |
| Recall, CONTRADICTED | +0.0216 [-0.0442, +0.0903] | +0.0719 [-0.0148, +0.1591] |
| Recall, UNSUPPORTED | +0.0959 [+0.0525, +0.1403] | +0.1684 [+0.1182, +0.2241] |
| Recall, not supported | +0.0789 [+0.0425, +0.1152] | +0.1464 [+0.1044, +0.1919] |
| Precision | +0.0483 [-0.0040, +0.1005] | +0.0420 [-0.0160, +0.1007] |
| F1 | +0.0648 [+0.0284, +0.1000] | +0.1026 [+0.0617, +0.1467] |

The two judges reach nearly the same specificity, so these differences in recall are read together
with a difference in specificity close to 0.

**Shift of specificity between dev and test.** As for both baselines (section 13 of `TARGETS.md`),
RJ's dev threshold lands above 0.96 on test, and the shift comes from QA:

| Judge | Test specificity, dev and test resampled | Resamples at or below 0.96 | QA, dev to test | Summary, dev to test |
|---|---|---|---|---|
| RJ | 0.9732 [0.9603, 0.9817] | 2.2% | 0.947 to 0.964 | 0.977 to 0.983 |
| HHEM | 0.9723 [0.9642, 0.9801] | 0.2% | 0.951 to 0.970 | 0.972 to 0.975 |
| MiniCheck | 0.9770 [0.9649, 0.9859] | 0.3% | 0.941 to 0.968 | 0.984 to 0.987 |

With dev and test resampled together, the difference in test specificity is +0.0009
[-0.0093, +0.0079] for RJ minus HHEM and -0.0038 [-0.0155, +0.0051] for RJ minus MiniCheck.

### Without threshold

Partial AUCs are standardised and cover a false alarm rate of 0 to 0.10 (section 12).

| Quantity | RJ | HHEM | MiniCheck | RJ minus HHEM | RJ minus MiniCheck |
|---|---|---|---|---|---|
| AUC ROC | 0.8889 [0.8621, 0.9118] | 0.8615 [0.8382, 0.8823] | 0.8532 [0.8161, 0.8817] | +0.0274 [+0.0098, +0.0452] | +0.0357 [+0.0172, +0.0562] |
| AUC, CONTRADICTED against SUPPORTED | 0.7846 [0.7373, 0.8269] | 0.7260 [0.6813, 0.7672] | 0.7474 [0.7024, 0.7885] | +0.0586 [+0.0234, +0.0923] | +0.0372 [-0.0064, +0.0815] |
| AUC, UNSUPPORTED against SUPPORTED | 0.9198 [0.8873, 0.9464] | 0.9017 [0.8781, 0.9246] | 0.8846 [0.8378, 0.9182] | +0.0181 [-0.0022, +0.0382] | +0.0352 [+0.0156, +0.0581] |
| Partial AUC, UNSUPPORTED against SUPPORTED | 0.8243 [0.7853, 0.8609] | 0.7804 [0.7459, 0.8155] | 0.7703 [0.7358, 0.8032] | +0.0439 [+0.0256, +0.0637] | +0.0539 [+0.0309, +0.0793] |
| Partial AUC, CONTRADICTED against SUPPORTED | 0.6109 [0.5758, 0.6474] | 0.5645 [0.5353, 0.5962] | 0.5850 [0.5557, 0.6159] | +0.0465 [+0.0183, +0.0758] | +0.0260 [-0.0124, +0.0643] |
| Selection criterion (mean of the two partial AUCs), test | 0.7176 [0.6914, 0.7443] | 0.6724 [0.6499, 0.6965] | 0.6776 [0.6544, 0.7013] | +0.0452 [+0.0279, +0.0634] | +0.0400 [+0.0169, +0.0636] |
| Selection criterion, dev | 0.7112 [0.6751, 0.7470] | 0.6406 [0.6155, 0.6670] | 0.6185 [0.5915, 0.6486] | | |
| Partial AUC, classes pooled, test (not the criterion) | 0.7755 [0.7448, 0.8041] | 0.7310 [0.7024, 0.7585] | 0.7279 [0.6987, 0.7546] | +0.0445 [+0.0285, +0.0615] | +0.0476 [+0.0276, +0.0690] |

RJ's dev criterion is the best of the 12 checkpoints scored on dev (section 12), so it should be
read as the best of 12. Its test value is 0.7176.

### Recall on CONTRADICTED outside the 14 attribution sentences (section 11)

14 of the 139 CONTRADICTED test sentences carry an annotator note that matches the attribution
patterns. They were found by `analysis/attribution_ids.py`, committed at `selection-v1`, which was
run once with `--expect 14`. The figures below cover the 10371 other sentences (9777 SUPPORTED,
469 UNSUPPORTED, 125 CONTRADICTED), at the thresholds of the full test set.

| | RJ | HHEM | MiniCheck |
|---|---|---|---|
| Recall, CONTRADICTED (125) | 0.2560 [0.1656, 0.3696] | 0.1600 [0.0965, 0.2273] | 0.2400 [0.1513, 0.3143] |
| Recall, not supported (594) | 0.6027 [0.5392, 0.6672] | 0.5152 [0.4466, 0.5707] | 0.4949 [0.4298, 0.5483] |

Section 11 puts the ceiling on contradictions near 125 / 139 = 0.90 for agreement with the labels,
and near 0.61 for a judge that reads as the auditor did.

### Outside meta sentences

These are the 10082 sentences outside the frozen meta pattern list (9475 SUPPORTED, 468
UNSUPPORTED, 139 CONTRADICTED), at the thresholds of the full test set.

| | RJ | HHEM | MiniCheck |
|---|---|---|---|
| Specificity | 0.9588 [0.9587, 0.9593] | 0.9654 [0.9633, 0.9682] | 0.9615 [0.9605, 0.9628] |
| Recall, CONTRADICTED | 0.2302 [0.1481, 0.3333] | 0.1583 [0.0976, 0.2222] | 0.2230 [0.1366, 0.2917] |
| Recall, UNSUPPORTED | 0.6966 [0.6178, 0.7723] | 0.6111 [0.5284, 0.6800] | 0.5641 [0.4908, 0.6308] |
| Recall, not supported | 0.5898 [0.5270, 0.6519] | 0.5074 [0.4394, 0.5613] | 0.4860 [0.4205, 0.5391] |

### Abstentions

The test set holds 66 abstentions, all SUPPORTED. The cell is flagged, since its minority class has
fewer than 10 sentences. At the thresholds of the full test set, the specificity is 1.0000 for RJ,
1.0000 for HHEM and 0.7576 [0.6094, 0.8642] for MiniCheck. No judge of this protocol can check an
abstention (section 6): these figures describe behaviour, not verification.

### Inputs beyond L

There are 0 test inputs beyond 4096 tokens, for each of the three judges.

### Breakdowns by task and generator

The figures below are point values at the thresholds of the full test set, read at equal
specificity. Cells whose minority class has fewer than 10 sentences are flagged. The values at the
dev thresholds are in `results/ragtruth/analysis-rj-vs-hhem.json`.

| Cell | SUPPORTED / UNSUPPORTED / CONTRADICTED | Flag | Specificity, RJ / HHEM / MiniCheck | Recall C, RJ / HHEM / MiniCheck | Recall U, RJ / HHEM / MiniCheck | Recall NS, RJ / HHEM / MiniCheck |
|---|---|---|---|---|---|---|
| QA | 5141 / 320 / 30 | | 0.950 / 0.960 / 0.945 | 0.167 / 0.100 / 0.267 | 0.750 / 0.647 / 0.603 | 0.700 / 0.600 / 0.574 |
| Summary | 4636 / 149 / 109 | | 0.972 / 0.960 / 0.977 | 0.248 / 0.174 / 0.211 | 0.577 / 0.530 / 0.477 | 0.438 / 0.380 / 0.364 |
| gpt-3.5-turbo-0613 | 1571 / 14 / 3 | minority below 10 | 0.987 / 0.987 / 0.986 | 0.000 / 0.000 / 0.000 | 0.571 / 0.357 / 0.357 | 0.471 / 0.294 / 0.294 |
| gpt-4-0613 | 1516 / 4 / 5 | minority below 10 | 0.980 / 0.991 / 0.984 | 0.200 / 0.400 / 0.200 | 0.000 / 0.000 / 0.000 | 0.111 / 0.222 / 0.111 |
| llama-2-13b-chat | 1791 / 87 / 22 | | 0.946 / 0.931 / 0.951 | 0.136 / 0.045 / 0.318 | 0.701 / 0.632 / 0.586 | 0.587 / 0.514 / 0.532 |
| llama-2-70b-chat | 1669 / 96 / 18 | | 0.947 / 0.953 / 0.947 | 0.278 / 0.167 / 0.222 | 0.740 / 0.635 / 0.625 | 0.667 / 0.561 / 0.561 |
| llama-2-7b-chat | 1881 / 140 / 44 | | 0.944 / 0.952 / 0.939 | 0.182 / 0.136 / 0.227 | 0.686 / 0.586 / 0.543 | 0.565 / 0.478 / 0.467 |
| mistral-7B-instruct | 1349 / 128 / 47 | | 0.963 / 0.953 / 0.959 | 0.319 / 0.213 / 0.191 | 0.703 / 0.648 / 0.562 | 0.600 / 0.531 / 0.463 |

## Secondary measures out of domain (descriptive, section 7)

No threshold was ever set on the out-of-domain data. Each judge is read at its RAGTruth dev
threshold and at 0.5, and both columns are given for every judge. A claim is flagged when its score
is below the threshold.

### Balanced accuracy at the two thresholds, all inputs

| Dataset | RJ at its dev threshold | HHEM at its dev threshold | MiniCheck at its dev threshold | RJ at 0.5 | HHEM at 0.5 | MiniCheck at 0.5 |
|---|---|---|---|---|---|---|
| WiCE | 0.8067 [0.7614, 0.8488] | 0.7502 [0.7051, 0.7925] | 0.7419 [0.6982, 0.7818] | 0.7345 [0.6842, 0.7838] | 0.7571 [0.7068, 0.8054] | 0.7242 [0.6742, 0.7726] |
| REVEAL | 0.8897 [0.8739, 0.9046] | 0.8323 [0.8171, 0.8469] | 0.7409 [0.7262, 0.7557] | 0.8867 [0.8675, 0.9049] | 0.8665 [0.8498, 0.8821] | 0.8625 [0.8445, 0.8796] |
| ClaimVerify | 0.6592 [0.6273, 0.6918] | 0.6739 [0.6427, 0.7050] | 0.6469 [0.6189, 0.6750] | 0.7017 [0.6695, 0.7336] | 0.7357 [0.7032, 0.7671] | 0.7459 [0.7159, 0.7760] |
| FactCheck-GPT | 0.7690 [0.7431, 0.7942] | 0.7349 [0.7096, 0.7595] | 0.7362 [0.7141, 0.7572] | 0.7553 [0.7291, 0.7817] | 0.7185 [0.6915, 0.7460] | 0.7472 [0.7212, 0.7726] |
| ExpertQA | 0.5816 [0.5611, 0.6024] | 0.5583 [0.5387, 0.5785] | 0.5676 [0.5488, 0.5868] | 0.5853 [0.5649, 0.6062] | 0.5758 [0.5551, 0.5968] | 0.5887 [0.5687, 0.6087] |
| Mean of the five | 0.7412 [0.7280, 0.7539] | 0.7099 [0.6972, 0.7225] | 0.6867 [0.6750, 0.6984] | 0.7327 [0.7186, 0.7467] | 0.7307 [0.7167, 0.7447] | 0.7337 [0.7200, 0.7474] |

### Recall on not supported claims and specificity, all inputs

The differences are point differences, RJ minus HHEM.

At each judge's dev threshold:

| Dataset | Recall, RJ | Recall, HHEM | Recall, MiniCheck | RJ minus HHEM | Specificity, RJ | Specificity, HHEM | Specificity, MiniCheck | RJ minus HHEM |
|---|---|---|---|---|---|---|---|---|
| WiCE | 0.7935 | 0.6356 | 0.5830 | +0.1579 | 0.8198 | 0.8649 | 0.9009 | -0.0450 |
| REVEAL | 0.8344 | 0.6870 | 0.4893 | +0.1473 | 0.9450 | 0.9775 | 0.9925 | -0.0325 |
| ClaimVerify | 0.3679 | 0.3946 | 0.3344 | -0.0268 | 0.9506 | 0.9531 | 0.9594 | -0.0025 |
| FactCheck-GPT | 0.8252 | 0.6958 | 0.5840 | +0.1294 | 0.7128 | 0.7739 | 0.8883 | -0.0612 |
| ExpertQA | 0.5294 | 0.3953 | 0.3611 | +0.1341 | 0.6338 | 0.7213 | 0.7742 | -0.0875 |
| Mean of the five | 0.6701 | 0.5617 | 0.4704 | +0.1084 | 0.8124 | 0.8581 | 0.9031 | -0.0458 |

At 0.5:

| Dataset | Recall, RJ | Recall, HHEM | Recall, MiniCheck | RJ minus HHEM | Specificity, RJ | Specificity, HHEM | Specificity, MiniCheck | RJ minus HHEM |
|---|---|---|---|---|---|---|---|---|
| WiCE | 0.8745 | 0.7935 | 0.8988 | +0.0810 | 0.5946 | 0.7207 | 0.5495 | -0.1261 |
| REVEAL | 0.8908 | 0.7855 | 0.8099 | +0.1053 | 0.8825 | 0.9475 | 0.9150 | -0.0650 |
| ClaimVerify | 0.4783 | 0.5652 | 0.6388 | -0.0870 | 0.9252 | 0.9062 | 0.8530 | +0.0190 |
| FactCheck-GPT | 0.8882 | 0.8252 | 0.8454 | +0.0630 | 0.6223 | 0.6117 | 0.6489 | +0.0106 |
| ExpertQA | 0.6129 | 0.5800 | 0.6594 | +0.0328 | 0.5577 | 0.5715 | 0.5180 | -0.0138 |
| Mean of the five | 0.7489 | 0.7099 | 0.7705 | +0.0390 | 0.7165 | 0.7515 | 0.6969 | -0.0351 |

Out of domain, RJ's dev threshold lands at a specificity between 0.63 and 0.95, below HHEM's on
every dataset, with a higher recall on four of the five.

### Common untruncated subset

The common subset holds the inputs that neither RJ nor HHEM truncates at 4096 tokens (sections 7
and 14). RJ's run marks 282 inputs as truncated (31 WiCE, 141 ClaimVerify, 110 ExpertQA) and HHEM's
marks 307; 3 inputs are truncated by RJ only. This leaves 8114 inputs, as computed at
`selection-v1` (`results/training/ood-untruncated.json`).

| Dataset | RJ at its dev threshold | HHEM at its dev threshold | MiniCheck at its dev threshold | RJ at 0.5 | HHEM at 0.5 | MiniCheck at 0.5 |
|---|---|---|---|---|---|---|
| WiCE | 0.8095 [0.7623, 0.8537] | 0.7514 [0.7050, 0.7957] | 0.7509 [0.7083, 0.7916] | 0.7412 [0.6887, 0.7923] | 0.7731 [0.7228, 0.8205] | 0.7376 [0.6860, 0.7881] |
| REVEAL | 0.8897 [0.8740, 0.9049] | 0.8323 [0.8176, 0.8474] | 0.7409 [0.7261, 0.7557] | 0.8867 [0.8674, 0.9050] | 0.8665 [0.8506, 0.8825] | 0.8625 [0.8444, 0.8800] |
| ClaimVerify | 0.6605 [0.6265, 0.6946] | 0.6818 [0.6496, 0.7148] | 0.6550 [0.6245, 0.6871] | 0.7029 [0.6690, 0.7366] | 0.7510 [0.7173, 0.7846] | 0.7583 [0.7254, 0.7902] |
| FactCheck-GPT | 0.7690 [0.7434, 0.7942] | 0.7349 [0.7093, 0.7591] | 0.7362 [0.7138, 0.7579] | 0.7553 [0.7290, 0.7816] | 0.7185 [0.6911, 0.7451] | 0.7472 [0.7206, 0.7726] |
| ExpertQA | 0.5795 [0.5576, 0.6011] | 0.5577 [0.5372, 0.5784] | 0.5667 [0.5466, 0.5866] | 0.5847 [0.5633, 0.6061] | 0.5783 [0.5564, 0.5996] | 0.5858 [0.5648, 0.6059] |
| Mean of the five | 0.7416 [0.7278, 0.7552] | 0.7116 [0.6983, 0.7248] | 0.6899 [0.6778, 0.7024] | 0.7342 [0.7197, 0.7487] | 0.7375 [0.7229, 0.7516] | 0.7382 [0.7240, 0.7528] |

The AUCs on this subset are in the next section. On the 310 inputs truncated by RJ or HHEM alone,
the AUCs below are a description without intervals:

| Dataset | Inputs | Supported | AUC, RJ | AUC, HHEM | AUC, MiniCheck |
|---|---|---|---|---|---|
| WiCE | 35 | 10 | 0.8200 | 0.7400 | 0.6680 |
| ClaimVerify | 149 | 105 | 0.8094 | 0.7492 | 0.7969 |
| ExpertQA | 126 | 101 | 0.6091 | 0.5731 | 0.6871 |

### By band of document length

The figures below come from the harness reports: point values at 0.5, with the five datasets
pooled. No cell has fewer than 10 sentences of its minority class.

| Band, words of the document | Inputs | Minority class | Balanced accuracy, RJ / HHEM / MiniCheck | F1 not supported, RJ / HHEM / MiniCheck |
|---|---|---|---|---|
| up to 250 | 6412 | 3160 | 0.708 / 0.694 / 0.682 | 0.738 / 0.709 / 0.711 |
| 251 to 1000 | 947 | 313 | 0.735 / 0.732 / 0.742 | 0.646 / 0.642 / 0.655 |
| 1001 to 2500 | 710 | 199 | 0.733 / 0.702 / 0.744 | 0.619 / 0.570 / 0.623 |
| above 2500 | 355 | 105 | 0.680 / 0.618 / 0.675 | 0.550 / 0.489 / 0.550 |

Section 7 asks for the AUC and the balanced accuracies by band, computed per dataset. No committed
script computes them, and they are not computed at this commit.

### Check of the baselines against the published figures

At 0.5 on all inputs, the balanced accuracies of HHEM and MiniCheck are those of section 7: all
within 2 points of the LLM-AggreFact leaderboard. The mean gap is -0.2 point for HHEM and +0.0 point
for MiniCheck (`results/ood/metrics-*-common.json`, key `published_check`).

## AUC per dataset, out of domain

All 8424 inputs. The differences are paired, on the same resamples by document.

| Dataset | Inputs | Supported | Not supported | AUC, RJ | AUC, HHEM | AUC, MiniCheck | RJ minus HHEM | RJ minus MiniCheck |
|---|---|---|---|---|---|---|---|---|
| WiCE | 358 | 111 | 247 | 0.8795 [0.8420, 0.9131] | 0.8322 [0.7874, 0.8746] | 0.8364 [0.7898, 0.8779] | +0.0473 [+0.0109, +0.0846] | +0.0431 [-0.0012, +0.0899] |
| REVEAL | 1710 | 400 | 1310 | 0.9500 [0.9386, 0.9601] | 0.9384 [0.9261, 0.9498] | 0.9373 [0.9252, 0.9485] | +0.0116 [+0.0019, +0.0219] | +0.0127 [+0.0008, +0.0249] |
| ClaimVerify | 1088 | 789 | 299 | 0.8287 [0.8004, 0.8557] | 0.8262 [0.7933, 0.8566] | 0.8084 [0.7771, 0.8394] | +0.0024 [-0.0210, +0.0269] | +0.0202 [-0.0027, +0.0438] |
| FactCheck-GPT | 1566 | 376 | 1190 | 0.8501 [0.8272, 0.8722] | 0.8049 [0.7786, 0.8307] | 0.8412 [0.8185, 0.8635] | +0.0452 [+0.0254, +0.0654] | +0.0089 [-0.0133, +0.0306] |
| ExpertQA | 3702 | 2971 | 731 | 0.6140 [0.5900, 0.6383] | 0.5981 [0.5743, 0.6222] | 0.6177 [0.5944, 0.6415] | +0.0158 [+0.0017, +0.0297] | -0.0037 [-0.0182, +0.0108] |
| Mean of the five | 8424 | | | 0.8245 [0.8129, 0.8355] | 0.8000 [0.7868, 0.8128] | 0.8082 [0.7954, 0.8209] | +0.0245 [+0.0142, +0.0351] | +0.0163 [+0.0044, +0.0281] |

On the common untruncated subset (8114 inputs):

| Dataset | Inputs | Supported | Not supported | AUC, RJ | AUC, HHEM | AUC, MiniCheck | RJ minus HHEM | RJ minus MiniCheck |
|---|---|---|---|---|---|---|---|---|
| WiCE | 323 | 101 | 222 | 0.8855 [0.8476, 0.9191] | 0.8429 [0.7973, 0.8848] | 0.8538 [0.8095, 0.8953] | +0.0426 [+0.0060, +0.0794] | +0.0317 [-0.0130, +0.0768] |
| REVEAL | 1710 | 400 | 1310 | 0.9500 [0.9390, 0.9603] | 0.9384 [0.9265, 0.9498] | 0.9373 [0.9255, 0.9488] | +0.0116 [+0.0016, +0.0219] | +0.0127 [+0.0006, +0.0249] |
| ClaimVerify | 939 | 684 | 255 | 0.8309 [0.8010, 0.8594] | 0.8372 [0.8025, 0.8692] | 0.8106 [0.7777, 0.8432] | -0.0062 [-0.0311, +0.0202] | +0.0203 [-0.0049, +0.0454] |
| FactCheck-GPT | 1566 | 376 | 1190 | 0.8501 [0.8264, 0.8721] | 0.8049 [0.7785, 0.8304] | 0.8412 [0.8178, 0.8633] | +0.0452 [+0.0251, +0.0661] | +0.0089 [-0.0133, +0.0308] |
| ExpertQA | 3576 | 2870 | 706 | 0.6137 [0.5883, 0.6384] | 0.5994 [0.5740, 0.6238] | 0.6156 [0.5906, 0.6395] | +0.0143 [+0.0003, +0.0283] | -0.0019 [-0.0164, +0.0129] |
| Mean of the five | 8114 | | | 0.8260 [0.8142, 0.8375] | 0.8046 [0.7910, 0.8178] | 0.8117 [0.7987, 0.8244] | +0.0215 [+0.0109, +0.0318] | +0.0144 [+0.0025, +0.0263] |

ExpertQA stays near 0.61 for the three judges. Section 7 notes that it is hard for every published
entry.

## The evaluation

| | RAGTruth test | Out of domain |
|---|---|---|
| Data | `test.jsonl`, SHA-256 `5eb4aa53…90608257` | `llm-aggrefact-5.jsonl`, SHA-256 `b01fe54c…db6793e4` |
| Inputs scored | 10385 of 10385, 0 failed | 8424 of 8424, 0 failed |
| Inputs truncated at 4096 tokens | 0 | 282 |
| Duration of the run | 4736 s | 4718 s |
| Started, ended (2026-11-15, CET) | 18:08, 19:27 | 19:27, 20:45 |

- **Judge:** `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`, bfloat16 on MLX.
  It carries the LoRA adapters of `qwen3-lr3e-4/checkpoint-3462` (SHA-256
  `44cdeecac3ae75c9bb53d538a14968667f9af2fa730a9dfae9477b6f2b1bed31`), kept separate. The scoring
  plugin is `training/rj_judge.py`, harness `judge-bench` of ragnarok-ai 1.13.0, and the
  configuration hash is `b2d8fe21e651d30a5aff2ba6d30d81605df50b3fae4da1c03256b31fd5da43a8`, the same
  for the dev scores, the test and out of domain.
- **Selection** (section 12 and amendment 1): 3 configurations and 12 checkpoints were scored on dev.
  The best checkpoint of Pleias-RAG-350M scored 0.6111 and the best of Qwen3-0.6B scored 0.7112. Their
  difference is -0.1001, with a standard error of 0.0151, so Qwen3-0.6B was retained. No run was
  resumed. Details are in `results/training/README.md`.
- **One evaluation.** The tag `selection-v1` was set at 18:05 CET. Each run then scored its data
  once, with no interruption and no resumption, and no intermediate checkpoint was scored on test or
  out of domain. Each analysis was run once: `ragtruth_analysis.py` once, `ood_metrics.py` once per
  judge and `ood_paired.py` once per pair. There was no re-evaluation.
- **Declarations made before the evaluation** are in `results/training/declarations.md`: the pilot, the check
  on training examples, the 50 dev sentences scored again to check the plugin, the configuration of
  use, the environment, and the attribution patterns.

## What is not published

- No text, label or per-sentence score of RAGTruth or LLM-AggreFact, and no run folder.
- The ids of the 14 attribution sentences of the test set. `TARGETS.md` commits the exclusion lists
  of the training set as sentence ids (section 2, Publication of data), and states the number of
  attribution sentences of the test set without listing them (section 11). The 14 ids are derived
  from the annotator notes of the test set and are not committed. The script that finds them and the
  expected count are committed, so anyone with RAGTruth can regenerate the list. Its SHA-256 is
  `cbc7dec977715c4c2d118270a982d36a6b138c53c289f7d8c94dae9f0f2a6f89`.
- The adapters. They are published separately as a Gjallar model under `ragnarok-ai`, after this
  evaluation (section 9 of `TRAINING.md`).

## Files

All paths in this table are relative to `results/`.

| File | Content |
|---|---|
| `ragtruth/analysis-rj-vs-hhem.json` | output of `analysis/ragtruth_analysis.py`: RJ against HHEM, MiniCheck alongside, with `--exclude-ids` |
| `ragtruth/rj-qwen3/report.{json,md}` | report of the harness on the test set (threshold 0.5, not the protocol) |
| `ood/metrics-rj-qwen3.json` | output of `ood/ood_metrics.py` for RJ, at its dev threshold and at 0.5 |
| `ood/metrics-hhem-l4096-mps-common.json`, `ood/metrics-minicheck-bounded-mps-common.json` | the baselines recomputed with the common untruncated subset |
| `ood/paired-rj-hhem.json` | output of `ood/ood_paired.py`: the out-of-domain target |
| `ood/paired-rj-minicheck.json` | the same against MiniCheck, alongside |
| `ood/rj-qwen3/report.{json,md}` | report of the harness out of domain (threshold 0.5, not the protocol) |
| `training/` | training, selection, dev threshold, common subset and declarations, fixed at `selection-v1` |
| `cost/rj-qwen3-adapter-mlx-739/` | cost of RJ |

The files are copied as the scripts wrote them. The paths they record are those of the machine that
produced them, relative to the root of the repository; the data files they name stay local and are
identified by their SHA-256.
