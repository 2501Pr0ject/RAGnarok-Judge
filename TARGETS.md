# RAGnarok Judge: targets and evaluation protocol

Status: FROZEN, 2026-11-10. The frozen version is the one of tag protocol-v1
in this repository; any later change is appended as a dated amendment with
its reason.

RJ stands for RAGnarok Judge throughout.

## 1. Purpose

RJ is a sentence-level faithfulness judge: given a context and one sentence of
an answer, it returns a score in [0, 1] that the sentence is supported by the
context. This file states, before any training, what RJ has to achieve to
support the claims we intend to make, how it will be measured, and what we
already know about the evaluation data.

The project is RAGnarok Judge (repository `ragnarok-judge`). Gjallar is the
name of the models it produces: they are published under the Hugging Face
organisation `ragnarok-ai`, for instance `ragnarok-ai/gjallar-0.6b`, the
suffix following the base selected in section 12. RJ remains the abbreviation
used throughout this file.

## 2. Data

All files are produced by the reference converter from RAGTruth (QA and
Summary tasks, official test split kept, dev carved out of the official train
split by source).

| Split | File | SHA-256 | Sentences | SUPPORTED / UNSUPPORTED / CONTRADICTED | Sources |
|---|---|---|---|---|---|
| test | `test.jsonl` | `5eb4aa53ad08a7dd73529008411735d51ed5589e0e407ab0a9af143c90608257` | 10385 | 9777 / 469 / 139 | 300 |
| dev | `dev.jsonl` | `18a8e0ff0126ba12c8b76ec05391397b7f8d58438b0184ba0bd2d8907f89ac32` | 6011 | 5536 / 363 / 112 | 163 |
| train, before corrections | `train.jsonl` | `10030e836e98758b1e5d4c1f12c0375335199e3377110ad538bb5656df06f614` | 54655 | 49874 / 3699 / 1082 | 1469 |
| train, after training set rule 5 | `train-rule5.jsonl` | `94bb93b320526a9a57f9accca9ac0920a48f9be78ee934c5d5e069bdf5bcacb2` | 54548 | 49874 / 3699 / 975 | 1469 |
| train, after rules 5 and 1 (training file) | `train-rule5-rule1.jsonl` | `bac1d36fa7ce039bee2be92f9dcd042cc67d205f5a8f86a44eae4677d955edba` | 54490 | 49874 / 3699 / 917 | 1469 |

### Provenance

- **RAGTruth.** Repository `ParticleMedia/RAGTruth`, commit
  `c103204b9ce28d6bbad859304bf30de72b8ed8fe` (2024-12-02), licence MIT
  (Copyright 2023 Particle Media). Files read by the converter:
  `dataset/response.jsonl`, 17 790 lines, SHA-256
  `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`;
  `dataset/source_info.jsonl`, 2965 lines, SHA-256
  `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b`. The
  source texts of RAGTruth (MS MARCO passages, CNN/DailyMail and recent news
  articles, Yelp records) remain under their own terms and are never
  redistributed by this project; only RAGTruth's QA and Summary tasks are
  converted.
- **Converter.** `conversion/ragtruth_to_ragnarok.py`, SHA-256
  `979ad4df2b5fb2d8cddac93e7554dc38349cad03d445c21e74f600e4bd521c3e`,
  sentence splitting by pysbd 0.3.4 (Python 3.12.9), run as
  `convert --data-dir RAGTruth/dataset --out-dir ragtruth-out` with its
  default parameters: seed 13, dev share 0.1 of the training sources, minimum
  overlap of 3 alphanumeric characters to attach a span, minimum coverage of
  0.25 for a span crossing into a sentence, minimum of 3 words for a fragment,
  audit sample of 99.
- **Exclusions.** `conversion/train_exclusions.py`, SHA-256
  `2949c93425875e11f23601ec4f1e522412cd15a2a4a6a23527c1c97fbe44473c`, applied
  twice: rule 5 on `train.jsonl`, ids in `train_excluded_ids_rule5.txt` (107
  ids, SHA-256
  `50545e145c759745766bed77ad0661c7de1716e9ecfa923d72a8c6a381924375`), then
  rule 1 on `train-rule5.jsonl`, ids in `train_excluded_ids_rule1.txt` (58
  ids, SHA-256
  `8e4a94d33499235fdf6b284781a5db38a0675650e1c62f2541e99c6646b59f09`). Each
  application writes a manifest with the hashes of its input and output.
- **Reproducibility check, 2026-11-10.** The three splits were regenerated in
  a temporary directory from the RAGTruth files above, with pysbd 0.3.4 and
  the parameters above, and the two exclusion lists were applied. The five
  hashes of the table above (test, dev, train, train after rule 5, train after
  rules 5 and 1) are exactly those of the regenerated files, and the rule 5
  list regenerated from them is identical to the committed one. The
  converter's report differs only by the output path it records. The
  regenerated audit files were deleted unread.
- **Analysis script.** `analysis/ragtruth_analysis.py` (sections 4, 5, 6 and
  9), SHA-256 `d1c2ac2323b903f58952616d1bb97f06e1da3868872ab842a4606b1a23a496b3`
  on 2026-11-10; checked that day on HHEM and MiniCheck, with MiniCheck as the
  candidate: every value of section 13 is reproduced to the fourth decimal,
  including Delta_C, Delta_U and their p-values. Two controls on runs built
  apart, kept as automated tests in `test_ragtruth_analysis.py`: HHEM against
  a copy of itself gives Delta_C = Delta_U = 0 in every resample, P1 not met
  (p = 1) and P2 met (p = 1 / 10001); an oracle candidate scoring 1 on
  SUPPORTED and 0 elsewhere, on test and dev, meets P1 and P2 (both
  p = 1 / 10001), Holm comparing the smaller p-value with 0.0125 first. The
  hash above is that of the frozen file, whose usage lines name the folders
  of this repository; the checks were run again from it on 2026-11-10.
- **Environment.** `requirements-frozen.txt`, SHA-256
  `ad09dcb7cd68a2498b2fe5c207daaa77988c94057476461d63f9dffb9f63da06`: Python 3.12.9,
  numpy 2.5.4, pysbd 0.3.4, torch 2.15.0, transformers 5.22.2, ragnarok-ai
  1.13.0 for the conversion, the analyses, the baselines and the PyTorch runs;
  mlx 0.32.4 and mlx-lm 0.32.2 in a separate environment for the MLX runs.

The three splits share no sentence id. Dev and test are frozen at the hashes
above: the baselines' thresholds come from this dev file and their scores from
this test file. Only the training file is corrected; its final hash is recorded
here at freeze time. The file after rule 5 is an intermediate one: it is the
file before corrections without the 107 sentences of that rule, every other
line unchanged, and the exclusions of rule 1 remain to apply. The 107 excluded
ids are listed in `train_excluded_ids_rule5.txt` (SHA-256
`50545e145c759745766bed77ad0661c7de1716e9ecfa923d72a8c6a381924375`), written
by `train_exclusions.py`.

Human labels are binarised as supported (SUPPORTED) against not supported
(UNSUPPORTED or CONTRADICTED). A sentence is flagged when its score is strictly
below the judge's threshold.

### Label quality

The converted labels were audited on a sample of the training split: 99
sentences, 33 per converted label, judged by a single annotator who read the
context and the sentence without seeing the label.

**First pass, discarded.** In a first pass, about half of the sentences 13 to
98 of the sheet were judged on the previous sentence instead of the sentence
to judge, through a confusion between two columns of the sheet. That pass was
discarded, and these 86 sentences were all judged again on a sheet where the
sentence to judge comes first; the 13 other sentences had been judged on the
right sentence and were kept. Before judging again, the annotator had seen the
agreement score of the discarded pass, about 0.2 on CONTRADICTED and on
UNSUPPORTED, a figure without meaning since it compared labels with judgments
made in large part on other sentences, and three sentences quoted to explain
the shift. The remarks on the annotation
grid were given to the annotator only after the final score. The tables below
are the final pass.

| Converted label | Judged SUPPORTED | Judged UNSUPPORTED | Judged CONTRADICTED | Undecided |
|---|---|---|---|---|
| SUPPORTED (33) | 28 | 2 | 3 | 0 |
| UNSUPPORTED (33) | 10 | 19 | 3 | 1 |
| CONTRADICTED (33) | 13 | 4 | 16 | 0 |

In the binary reading used by the targets:

- not supported labels confirmed as not supported: 42 of 66, 0.64 (Wilson 95%
  interval 0.52 to 0.74), the undecided sentence counting as not confirmed;
- SUPPORTED labels confirmed: 28 of 33, 0.85 (0.69 to 0.93);
- CONTRADICTED labels confirmed as contradictions: 16 of 33, 0.48 (0.33 to
  0.65), and 13 of the 33 were judged supported.

By conversion rule, with the number of sentences judged not supported in the
last column:

| Converted label and rule | Sentences | Judged SUPPORTED | Judged UNSUPPORTED | Judged CONTRADICTED | Undecided | Judged not supported |
|---|---|---|---|---|---|---|
| CONTRADICTED, conflict span on the sentence | 30 | 11 | 3 | 16 | 0 | 19 of 30 |
| CONTRADICTED, inherited from a span covering several sentences | 3 | 2 | 1 | 0 | 0 | 1 of 3 |
| UNSUPPORTED, baseless span on the sentence | 12 | 2 | 8 | 2 | 0 | 10 of 12 |
| UNSUPPORTED, inherited from a span covering several sentences | 8 | 3 | 4 | 1 | 0 | 5 of 8 |
| UNSUPPORTED, baseless span that RAGTruth marks as implicitly true | 13 | 5 | 7 | 0 | 1 | 7 of 13 |
| SUPPORTED, no span | 33 | 28 | 2 | 3 | 0 | 5 of 33 |

Labels inherited from a span covering several sentences were confirmed as not
supported for 6 of 11 sentences (Wilson 95% interval 0.28 to 0.79), against 29
of 42 (0.54 to 0.81) for labels whose span lies on the sentence. The groups
are too small to separate the rules.

**Spans covering several sentences: decision rule.** Fixed on 2026-11-09 at
18:05 CET, once the blind audit of 40 training sentences was complete and
before its key was opened. SHA-256 of the filled sheet at that time:
`f83f73f1de2ca33a07d7a2888aaa350e89f2419ea8ac65ee0038a53f1f7beaf5`.
The 107 CONTRADICTED training sentences that inherit their label from a span
covering several sentences are excluded from the training set, never
relabelled, if fewer than 10 of the 20 inherited sentences audited are
confirmed as contradictions by the auditor. Otherwise they are kept.

**Blind audit of spans covering several sentences.** Scored on 2026-11-09 at
18:09 CET, after the rule above was written. 40 training sentences were judged
by the same single annotator without any label in view: 20 CONTRADICTED
sentences whose label is inherited from a span covering several sentences,
drawn among the 107 of the training split, and 20 controls. None of the 40 is
in the audit of 99. The sheet is complete, with no undecided sentence, and its
texts are identical to those of the sheet as drawn.

| Group, converted label and rule | Sentences | Judged SUPPORTED | Judged UNSUPPORTED | Judged CONTRADICTED | Undecided |
|---|---|---|---|---|---|
| Inherited: CONTRADICTED, span covering several sentences | 20 | 17 | 0 | 3 | 0 |
| Control: SUPPORTED, no span | 18 | 15 | 3 | 0 | 0 |
| Control: CONTRADICTED, conflict span on the sentence | 1 | 0 | 0 | 1 | 0 |
| Control: UNSUPPORTED, inherited from a span covering several sentences | 1 | 1 | 0 | 0 | 0 |

- Inherited labels confirmed as contradictions: 3 of 20, 0.15 (Wilson 95%
  interval 0.05 to 0.36). Confirmed as not supported, contradicted or
  unsupported: the same 3 of 20. Judged supported: 17 of 20, 0.85 (0.64 to
  0.95).
- The 20 sentences come from 10 responses, four of which give 6, 3, 3 and 2
  sentences, and the three confirmed sentences come from three responses
  audited on one sentence each. The Wilson interval treats the sentences as
  independent and is too narrow; resampling the 10 responses gives 0.00 to
  0.43 for the share confirmed as contradictions.
- With the 3 inherited CONTRADICTED sentences of the audit of 99, which are
  other sentences: 3 of 23 confirmed as contradictions (0.05 to 0.32) and 4 of
  23 as not supported (0.07 to 0.37). For comparison, labels whose conflict
  span lies on the sentence were confirmed as contradictions for 16 of 30
  (0.36 to 0.70).
- Controls: 15 of the 18 SUPPORTED labels confirmed, 0.83 (0.61 to 0.94), in
  line with the 28 of 33 of the first audit.

**Application of the rule.** 3 is fewer than 10: the 107 sentences are
excluded from the training set, never relabelled (training set rule 5). The
rule was fixed after the sheet was filled; the sheet held 4 CONTRADICTED
judgments among its 40 sentences, so the outcome did not depend on the key.
The training split goes from 54655 to 54548 sentences and its CONTRADICTED
class from 1082 to 975. Of the 107 sentences, 103 are QA sentences, they come
from 28 responses, and 41 are also on the attribution list of 115 (rule 1).
The 74 other sentences of that list were validated by hand on 2026-11-10: 58
are excluded by rule 1 and 16 are kept, and the training file goes to 54490
sentences, 917 of them CONTRADICTED (`train-rule5-rule1.jsonl`).

Dev and test are frozen and keep such labels: 10 of the 112 CONTRADICTED dev
sentences and 3 of the 139 of the test set. The 3 test sentences are among
the 14 of section 11.

**Limits.** One annotator, one pass, no adjudication. A disagreement can come
from the RAGTruth annotation, from the conversion of annotated spans into
sentence labels, or from the auditor, and the audit cannot tell which. The
sample is stratified by label and drawn from the training split; dev and test
come from the same annotation campaign and the same conversion but were not
audited, so the same rates are assumed there, not measured. With 33 sentences
per label the intervals are wide.

**Consequences.**

- P1 and P2 measure agreement with the RAGTruth labels as converted, not the
  detection of sentences that are really unsupported. A judge that judged
  exactly as the auditor did would reach a recall of 0.61 on CONTRADICTED
  labels (20 of 33, Wilson 95% interval 0.44 to 0.75) and 0.67 on UNSUPPORTED
  labels (22 of 33, 0.50 to 0.80), at a specificity of 0.85 (28 of 33, 0.69 to
  0.93). These are estimates from a single annotator on 33 sentences per
  label. They show that the operating point of section 4 is itself defined
  against the labels.
- A judge trained on these labels can learn what is specific to this
  annotation campaign, and the test set shares it: same annotators, same
  conversion. Part of an in-domain gain may be of that kind. Any claim beyond
  RAGTruth therefore rests on the out-of-domain target of section 7, whose
  labels come from other annotators.
- The CONTRADICTED class is the least confirmed, and it is the class of P1.

### Training set rules

1. Exclusion, never relabelling, of the CONTRADICTED training sentences whose
   annotator note describes a passage attribution error (the content is in the
   context, but in another passage than the one cited). The list of excluded
   ids is validated by hand and committed. Kept as CONTRADICTED: sentences that
   assert something about the context itself (for instance that the passages do
   not mention a fact they do mention); sentences whose note turns out to
   describe a contradiction of content, even when it also describes an
   attribution error; and sentences whose note does not establish that the
   content appears in another passage (a note saying that the cited passage
   does not hold the content, without placing it elsewhere in the context,
   does not describe an attribution error: the sentence keeps its label). The
   115 candidate sentences were found by patterns on the notes. 41 of them
   were already excluded by rule 5; the 74 others were validated by hand on
   2026-11-10: 58 excluded (7 whose sentence names no passage, 51 whose
   sentence cites one), 16 kept (3 assertions about the context, 10
   contradictions of content, 3 whose note places the content nowhere).
   Applied on 2026-11-10 to the file of rule 5: `train-rule5-rule1.jsonl`,
   ids in `train_excluded_ids_rule1.txt`.
2. Meta sentences (answer framing: announcements and courtesy formulas) stay in
   the training set with their SUPPORTED label. No upstream filter.
   Abstentions, sentences saying that no answer can be given, are not meta: an
   abstention asserts that the context does not answer the question, which can
   be true or false; in the audit of 99 one such sentence was judged
   contradicted. Whether it is true depends on the question, which no judge of
   this protocol receives (section 3): no judge can check an abstention here,
   and the category describes how the judges behave on these sentences, not a
   verification. Abstentions stay in the training set with their labels (532
   SUPPORTED, 1 UNSUPPORTED, 1 CONTRADICTED among the 534 matched), and are
   reported as a category of their own in the secondary measures (section 6).
   Decided on 2026-11-10, after the validation of the meta list.
3. Corrections of the converter apply to the training file only.
4. No data from LLM-AggreFact is used for training or for any selection.
5. Exclusion, never relabelling, of the 107 CONTRADICTED training sentences
   whose label is inherited from a span covering several sentences (converter
   rule `conflict_span+multi_sentence_span`), by the decision rule above,
   applied on 2026-11-09. UNSUPPORTED sentences inherited from such spans are
   not concerned. Dev and test keep these sentences.

Verification of passage citations is out of scope for this version.

### Publication of data

No converted data file is ever published, in a repository, on Hugging Face or
anywhere else, even reformatted: neither the files derived from RAGTruth nor
those derived from LLM-AggreFact (section 7). Only metrics are published.
Exclusion lists are committed as sentence ids and pattern lists as patterns,
without any text from the datasets.

## 3. Judges compared

| Judge | Role | Training data |
|---|---|---|
| RJ | candidate | RAGTruth train as defined in section 2 (in-domain for the RAGTruth test) |
| HHEM-2.1-Open (`vectara/hallucination_evaluation_model`) | comparator of every target | not published by Vectara; treated as unknown |
| MiniCheck-Flan-T5-Large (`lytang/MiniCheck-Flan-T5-Large`) | second reference, reported | 21K ANLI examples and 14K synthetic examples, per its authors |

Prometheus 2 7B (Q5_K_M, Ollama) was evaluated on a 739-sentence sample and is
not carried forward: AUC 0.602 [0.556, 0.649], and no threshold of its
five-point scale gives a kappa above 0.17.

RJ is trained on the same generators, tasks and annotation campaign as the
RAGTruth test. HHEM's exposure to RAGTruth is unknown and MiniCheck's published
training data does not include it. An in-domain gain over them is therefore expected and
is not evidence of a better judge in general: section 7 exists for that reason.

### Judge input

Every judge receives the same two fields of a record, and nothing else:

- the whole context as written in the data file (the passages for QA, the
  article for Summary);
- the sentence to judge.

Not transmitted to any judge: `prev_sentence` (kept in the files, never sent),
the question, and the rest of the answer. Each judge applies its own template
and tokenizer to these two fields. RJ is trained on inputs built the same way.

A limit follows. An abstention, a sentence saying that no answer can be given,
is true or false according to the question: without it, no judge of this
protocol can check such a sentence, and the metrics on abstentions (section 6)
describe behaviour, not verification. Passing the question to the judge is a
lead for a later version, not for this one.

Length handling of the baselines, as run:

- HHEM: no truncation. Its declared training length is 512 tokens; 45.1% of
  test inputs are longer by its tokenizer, the longest being 2397 tokens.
- MiniCheck: the context is split into chunks of 500 words, each chunk input
  is limited to 2048 tokens, and the score is the maximum over chunks.

RJ:

- Maximum input length: L = 4096 tokens of the base model's tokenizer, on the
  complete input (template, context and sentence). The same L applies in
  training, on dev, on test and out of domain.
- No windowing: every input is scored in one pass over its whole context. The
  reason is the verification of absence. To judge a sentence such as "the
  passages do not mention X", the judge has to see all of the context: a
  window that does not contain X proves nothing about the rest, and with a
  maximum over windows any window lacking X would support the sentence.
- No RAGTruth input comes near L with either candidate tokenizer (table
  below), so nothing is truncated in-domain. The number of inputs beyond L is
  reported with the results.
- Beyond L, which can only happen out of domain, the sentence is never
  truncated and the context is truncated at its end so that template, context
  and sentence fit in L. Truncation has the same limit as a window for the
  verification of absence, which is why these inputs are counted.
- Out of domain, the inputs beyond L are counted per dataset and the metrics
  are also reported by band of document length (section 7).
- The counts of inputs beyond the baselines' own limits (512 tokens for HHEM,
  2048 for MiniCheck) are reported alongside.

Input lengths in tokens, with the declared templates below, of 84 tokens
(Qwen3-0.6B) and 83 tokens (Pleias-RAG-350M), measured on 2026-11-10 after
the instruction was made explicit on meta sentences:

| Split | Inputs | Qwen3-0.6B: median, longest | Pleias-RAG-350M: median, longest |
|---|---|---|---|
| test | 10385 | 532, 2181 | 543, 2277 |
| dev | 6011 | 547, 1949 | 561, 1990 |
| train | 54655 | 529, 2540 | 543, 2508 |


Pleias-RAG-350M declares a maximum of 4096 positions: L is its limit, with no
headroom. Qwen3-0.6B declares 40960.

### Candidate bases

| Base | Revision | Licence | Parameters | Layers, width | Maximum positions |
|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | `c1899de289a0` | Apache-2.0 | 596 million, embeddings tied | 28, 1024 | 40960 |
| `PleIAs/Pleias-RAG-350M` | `db001b29a335` | Apache-2.0 | 353 million | 26, 1024 | 4096 |

- Declared stack: MLX in bfloat16, for both bases. Both are published in
  bfloat16, so no conversion is needed; a base published in another precision
  would be converted to bfloat16 for the measure, and the report of the run
  would say so. RJ is trained in MLX, and quality is measured in the
  configuration measured for cost.
- The cost of each base is measured before any training, with its pretrained
  weights, on the 739-sentence sample (section 8): on MLX and, for comparison,
  on PyTorch MPS at float16, four runs in all. Elimination is decided on MLX
  only: a base that misses a cost target there is eliminated.
- The choice between the remaining bases is made on the selection criterion of
  the dev set (section 12).
- The prompt template of each base is declared below.

### Prompt templates and output

Declared on 2026-11-09, instruction amended on 2026-11-10 (the clause on
sentences making no claim names what it covers: announcements and courtesy
formulas, not abstentions). Reference implementation: `prompt_templates.py`.

Instruction, common to both bases:

```
Judge whether the sentence is supported by the context. Answer with one label.
Supported: the context states or directly implies the sentence, or the sentence makes no claim (an announcement or a courtesy formula).
Unsupported: the context neither states nor contradicts the sentence.
Contradicted: the context contradicts the sentence.
```

Qwen3-0.6B, its own chat format, built by the tokenizer's chat template with
thinking disabled:

```
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
Context:
{context}

Sentence:
{sentence}<|im_end|>
<|im_start|>assistant
<think>

</think>

{label}<|im_end|>
```

Pleias-RAG-350M, its native markers, the label directly after
`<|answer_start|>`:

```
<|query_start|>{instruction}
Sentence: {sentence}<|query_end|>
<|source_start|><|source_id|>1 {context}<|source_end|>
<|answer_start|>{label}<|answer_end|>
```

Output: one label among three, written with an initial capital. In the data
files the same labels are in upper case. First token of each label, tokenised
right after the prompt, with its identifier:

| Base | Supported | Unsupported | Contradicted |
|---|---|---|---|
| Qwen3-0.6B | `Supported` (34636) | `Unsupported` (41884) | `Contr` (41437) |
| Pleias-RAG-350M | `Supported` (57889) | `Un` (2076) | `Cont` (3099) |

The three first tokens are distinct for both tokenizers. The score of the
judge is the probability of Supported among these three tokens, computed from
their logits at the last position of the prompt, in one forward pass; nothing
is generated. The three logits are converted to float32 before the softmax:
a softmax computed in bfloat16 would leave the score with a few hundred
distinct values and multiply the tied scores at the threshold (section 4).
The instruction states that a sentence making no claim, an announcement or a
courtesy formula, is Supported, which is how such sentences are labelled in
the training set (section 2). It says nothing of abstentions: whether the
context answers the question depends on the question, which the judge does
not receive (Judge input above), so RJ gets no instruction about them and
scores them as it does any sentence; its behaviour on them is described in
section 6.

## 4. Operating points

**Rule.** For a set of n SUPPORTED sentences, let k = floor(0.04 n). The
threshold t is the (k+1)-th smallest score of the judge on these sentences. A
sentence is flagged when score < t, which flags at most 4% of the set
(specificity 0.96). With tied scores at t, fewer than 4% are flagged, which
can only lower that judge's recall.

The rule is applied in two ways.

**Equal specificity on the test set (primary analysis).** The rule is applied,
for each judge, to the 9777 SUPPORTED test sentences (k = 391): every judge
sits at a test specificity of 0.9600. The threshold depends on the judge's
scores for SUPPORTED test sentences only; no information about the not
supported sentences enters it. The rule is written here before training: the
analysis is pre-specified, not post hoc. In the bootstrap, the threshold is
re-derived in every resample, as the largest threshold that flags at most 4%
of the resample's SUPPORTED sentences.

**Dev threshold (transfer of the operating point, and out of domain).** The
rule applied to the 5536 SUPPORTED dev sentences (k = 221) gives the threshold
that could be set in advance. It is a constant in every analysis that uses it.

| Judge | Dev threshold | Dev specificity |
|---|---|---|
| HHEM | 0.18617 | 0.9601 |
| MiniCheck | 0.04438 | 0.9601 |
| RJ | derived once, on the selected checkpoint | by construction |

The primary analysis compares discrimination at a common specificity. It does
not say whether a threshold chosen in advance lands at that specificity: that
is what the dev-threshold analysis measures (section 6).

## 5. Primary targets

Both are measured on the full test set, RJ against HHEM, at the
equal-specificity point of section 4.

**P1, contradictions (superiority).** Recall on the 139 CONTRADICTED test
sentences. Delta_C = recall_C(RJ) - recall_C(HHEM). Null hypothesis:
Delta_C <= 0. P1 is met when the null is rejected, which is the lower bound of
the interval for Delta_C being above 0.

**P2, unsupported sentences (non-inferiority).** Recall on the 469 UNSUPPORTED
test sentences. Delta_U = recall_U(RJ) - recall_U(HHEM). Null hypothesis:
Delta_U <= -0.05. P2 is met when the null is rejected, which is the lower
bound of the interval for Delta_U being above -0.05.

**Why P2 bears on UNSUPPORTED.** P2 has to guarantee that contradictions are
not bought at the expense of the rest. Overall recall on the 608 not supported
sentences mixes the two classes (139 and 469): with a true gain of 0.156 on
contradictions, a non-inferiority test on overall recall with a margin of 0.03
would still allow a true loss of 0.085 on UNSUPPORTED. Measured on UNSUPPORTED
alone, no gain elsewhere can offset a loss. The margin of 0.05 is about 23 of
the 469 sentences. Overall recall is reported as a secondary measure.

**Multiplicity.** One-sided bootstrap p-values (section 9), Holm procedure over
P1 and P2 at a family-wise one-sided level of 0.025: the smaller p-value is
compared with 0.0125; if it passes, the other is compared with 0.025.

**Reading of the outcomes.**

| P1 | P2 | Claim allowed |
|---|---|---|
| met | met | at 96% specificity on the RAGTruth test, RJ detects more contradictions than HHEM and is not worse on unsupported sentences |
| met | not met | RJ detects more contradictions than HHEM; a loss on unsupported sentences is not excluded |
| not met | met | no demonstrated gain on contradictions; RJ is not worse than HHEM on unsupported sentences |
| not met | not met | no claim |

Every claim names HHEM as the comparator. The differences RJ minus MiniCheck
for the P1 and P2 quantities are always reported next to the claim, with their
intervals (section 6); they enter no claim, for the reason given in section 7
(MiniCheck is a reference, not a comparator). These are statements about
discrimination at a common specificity, and about agreement with the RAGTruth
labels, whose quality is described in section 2. Any statement about a
threshold set in advance relies on the transfer analysis of section 6.

**Resolution of the test set (indicative).** Measured on the full test set
with HHEM and MiniCheck as the pair, under the protocol of section 9
(thresholds re-derived in every resample). Powers by normal approximation.

P2. The standard error of the paired difference on UNSUPPORTED is 0.027; the
two recalls have a correlation of 0.73 across resamples.

| True Delta_U | Power at 0.0125 | Power at 0.025 |
|---|---|---|
| 0.00 | 0.34 | 0.45 |
| +0.02 | 0.63 | 0.73 |
| +0.03 | 0.76 | 0.84 |
| +0.05 | 0.92 | 0.96 |

P2 is rejected from an observed Delta_U of about +0.011 (at 0.0125) or +0.003
(at 0.025). The third revision of this file gave a standard error of 0.022 and
powers of 0.49 and 0.61 at a true difference of zero: that estimate came from
the 739-sentence sample with thresholds held fixed, and re-deriving the
thresholds in every resample adds variance. If RJ's errors were independent of
HHEM's, the standard error would be about 0.052 and RJ would need an observed
gain of +0.05 to +0.07.

P1. The standard error of the paired difference on CONTRADICTED is 0.051; the
two recalls are uncorrelated across resamples. A true gain of 0.10 is detected
with a power of 0.39 to 0.50, a gain of 0.15 with 0.76 to 0.84, a gain of 0.20
with 0.95 to 0.98. P1 is rejected from an observed Delta_C of about +0.100 (at
0.025) or +0.114 (at 0.0125).

## 6. Secondary measures

Reported with 95% intervals, on the full test set, for RJ, HHEM and MiniCheck.
None has a pass or fail threshold.

- **Transfer of the operating point.** The P1 and P2 quantities at each
  judge's dev threshold, with the test specificity each judge obtains there
  and the differences RJ minus HHEM in specificity, Delta_C and Delta_U. This
  is the configuration a user can actually deploy. Both baselines' dev
  thresholds land above 0.96 on test (0.9723 for HHEM, 0.9770 for MiniCheck,
  section 13); a difference in recall at this point is read together with the
  difference in specificity.
- Overall recall on the 608 not supported sentences, at both points, and its
  difference RJ minus HHEM.
- AUC ROC on raw scores (Mann-Whitney, ties counted one half), overall and
  CONTRADICTED against SUPPORTED, UNSUPPORTED against SUPPORTED.
- The selection criterion of section 12 and its two per-class components, on
  the test set.
- Precision and F1 of not supported at the test set's natural prevalence (5.9%
  not supported).
- The differences RJ minus MiniCheck for the P1 and P2 quantities.
- The same metrics outside meta sentences, defined by a pattern list frozen
  on 2026-11-10 (`meta_patterns.json`, SHA-256
  `dbe8efab1034931ff6ebfb3566dec389280ae534486f92bc24a1ccae45d25fc7`). A
  meta sentence has no checkable content: it announces a summary, an answer
  or steps, or offers help or closes the answer. A pattern has to cover the
  whole sentence, so a sentence that carries a claim is never meta.
  Announcements that name a subject and fragments without a proposition are
  left out: the list only serves this secondary measure, and precision comes
  first. On the training file the list matches 1535 sentences, 2.8% (1285
  announcements and 250 courtesy formulas, all SUPPORTED). Validated on
  2026-11-10 on 30 matched sentences drawn at random, 30 without checkable
  content, and on 15 sentences left out although they open like a meta
  sentence, all 15 carrying a claim.
- The same metrics on abstentions alone, sentences saying that no answer can
  be given without naming what is missing, defined by a second pattern list
  frozen the same day (`abstention_patterns.json`, SHA-256
  `2fc07ce8e5cade9110b90b81e2b2635c7c090cd8bb5ab5755d15aa367cd2155a`), 534
  training sentences. An abstention is not a meta sentence (section 2, rule
  2), but whether the context answers depends on the question, which no judge
  receives (section 3): no judge of this protocol can check an abstention. The
  metrics on this category describe how each judge behaves on these sentences
  against the RAGTruth labels; they are not a verification. The number of
  abstentions in the test set is reported with the results, and cells with
  fewer than 10 minority-class sentences are flagged, which this category is
  likely to be (on the training split, 2 of 534 are not supported).
- Recall on CONTRADICTED outside the 14 test sentences with an attribution note
  (section 11).
- The number of test inputs beyond L (section 3), expected to be zero.
- Breakdowns by task and generator, cells with fewer than 10 minority-class
  sentences flagged.

## 7. Out-of-domain target

**Benchmark.** LLM-AggreFact, test split, restricted to the five datasets whose
annotations carry an explicit licence: WiCE, REVEAL, ClaimVerify,
FactCheck-GPT and ExpertQA (`llm-aggrefact-licences.md`). The RAGTruth subset
is excluded by construction. Instances whose document overlaps RJ's training
contexts are removed and counted: texts are lower-cased and reduced to their
words, and a document overlaps when at least half of its 13-word shingles
occur in a training context. A document shorter than 13 words has no shingle:
it overlaps only if it is identical to a whole training context. The
documents sharing at least one shingle, and those too short to assess, are
counted and reported. The check reads the documents only.

Check and conversion of 2026-11-09: no document overlaps and no instance is
removed. Two ExpertQA documents share a shingle with a training context, with
a containment of at most 0.002, and 23 documents are too short to assess. The
converted file holds 8424 instances over 6884 documents; SHA-256
`b01fe54c21845d3f0fe529be2e2829f249e9454402553a6223bc8737db6793e4`. An earlier
run, made before the rule on short documents was written, had removed 6
FactCheck-GPT instances by mistake; its file was discarded. Revision
`981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4` of `lytang/LLM-AggreFact`; SHA-256
of the test file:
`ba6976c3da990a9d7b11670efaf4e3c93c1bd0530061ec742afd6b2d5bce698c`.

**Metric.** AUC ROC on raw scores (Mann-Whitney, ties counted one half), per
dataset, then the unweighted mean over the five datasets. No threshold enters
it.

**Target (non-inferiority).** Delta_OOD = mean AUC(RJ) - mean AUC(HHEM). Null
hypothesis: Delta_OOD <= -0.02. One-sided test at 0.025, separate from the
Holm family of section 5. Any claim that RJ is a better judge than HHEM beyond
RAGTruth requires this target.

**Reading.** When the target is met: out of domain, RJ separates supported
from not supported sentences no worse than HHEM, within 0.02 of AUC. The
target says nothing about a threshold set on RAGTruth remaining usable
elsewhere; the balanced accuracies below describe that.

**Secondary: balanced accuracy at two thresholds fixed in advance.** For every
judge (RJ, HHEM, MiniCheck), balanced accuracy per dataset and its unweighted
mean, at each of two thresholds: the judge's RAGTruth dev threshold of
section 4, and 0.5, the midpoint of the score scale, which is the threshold of
the benchmark's published protocol. Both columns are reported for all judges;
no judge is reported at the better of its two thresholds only. No threshold is
ever set on the out-of-domain data, neither on the dev split of LLM-AggreFact,
which is not used at all, nor on its test split. Recall and specificity per
dataset are reported at both thresholds, with the differences RJ minus HHEM.

**Input length.** L = 4096 applies as on RAGTruth (section 3). The inputs
beyond L are counted per dataset for RJ, and the AUC and the balanced
accuracies are also reported by band of document length. The bands are in
words of the document, so that they do not depend on a tokenizer: up to 250,
251 to 1000, 1001 to 2500, above 2500 (6412, 947, 710 and 355 inputs over the
five datasets). A metric by band is computed per dataset; cells with fewer
than 10 sentences of the minority class are flagged, as for the RAGTruth
breakdowns.

Lengths of the test inputs, measured on the documents and claims only, with
each judge's tokenizer and template (the templates proposed for the two bases):

| Dataset | Inputs | Documents | Median words per document | Beyond 4096: Qwen3-0.6B | Pleias-RAG-350M | HHEM | Longest, HHEM tokens |
|---|---|---|---|---|---|---|---|
| WiCE | 358 | 355 | 930 | 31 (8.7%) | 33 (9.2%) | 35 (9.8%) | 23003 |
| REVEAL | 1710 | 1498 | 78 | 0 | 0 | 0 | 741 |
| ClaimVerify | 1088 | 697 | 1028 | 141 (13.0%) | 142 (13.1%) | 147 (13.5%) | 28496 |
| FactCheck-GPT | 1566 | 1199 | 88 | 0 | 0 | 0 | 385 |
| ExpertQA | 3702 | 3135 | 145 | 110 (3.0%) | 118 (3.2%) | 125 (3.4%) | 30640 |
| All five | 8424 | 6884 | 126 | 282 (3.3%) | 293 (3.5%) | 307 (3.6%) | 30640 |

- With L = 4096, 3.3 to 3.5% of the inputs are truncated for RJ, all in three
  datasets. At 2048 tokens it would have been 8.6%. Where truncation applies
  it is heavy: the median input loses 1900 to 2800 tokens of context.
- Documents carry one claim in the median case (13 at most), so resampling by
  document is close to resampling by instance.
- HHEM cannot be run without truncation on these inputs: 108 exceed 8192
  tokens by its tokenizer, and at 30640 tokens one attention matrix alone
  would take 45 GB. On RAGTruth its longest input was 2397 tokens.
- MiniCheck splits documents into chunks of 500 words, up to 40 chunks for a
  document here: 14 374 chunks for the 8424 inputs.

**HHEM's input length.** Out of domain, HHEM is limited to 4096 tokens of its
own tokenizer, the sentence kept whole and the context truncated at its end,
as for RJ: 307 inputs are concerned. If it runs out of memory on MPS, it is
run on CPU for this benchmark; on RAGTruth the two devices gave scores
identical to within 3e-6. A fabricated input of 4096 tokens passes on both
(peak footprint 4.2 GB on MPS, 3.7 GB on CPU). MiniCheck keeps its chunks.
The run of 2026-11-09 completed on MPS: 8424 inputs scored, no failure, 307
truncated, no memory error. The CPU run was not needed.

**MiniCheck's forward passes.** Out of domain, MiniCheck is run through a
local wrapper (`ood/minicheck_bounded.py`), not through the stock judge of the
harness. The stock judge sends the chunks of a document to the model in
batches of 16, each padded to its longest input, and never releases the device
cache; the attention memory of a forward pass grows with the number of rows
times the square of the padded length. Its run of 2026-11-09 was stopped
after 356 inputs, all of WiCE: the process held 14 GB on a machine of 16 GB
and the machine was swapping. Over the five datasets, 32 inputs have a batch
whose attention tensors exceed 1 GiB each, up to 3.8 GiB.

The wrapper keeps the chunks, the texts, the batches of 16, the truncation at
2048 tokens and the padding of the stock judge, so each chunk is given exactly
the same input. It changes two things. The rows of a batch go through the
model in groups such that rows times the square of the padded length stays at
or below 2048 squared, the attention size of one input of maximum length: 7
rows at 760 tokens, 4 at 1024, 1 at 2048. The device cache is released after
each batch of 16. The wrapper's configuration hash differs from the stock
judge's (`cbccef85dc7a` against `d5ab1a32c50a`) and its run does not resume
the stopped one: the 8424 inputs are all scored through the wrapper.

- 305 of the 8424 inputs have at least one batch that the rule splits: 42 in
  WiCE, 149 in ClaimVerify, 114 in ExpertQA. The other 8119 inputs go through
  exactly the forward passes of the stock judge.
- No batch of the RAGTruth dev set is split (3 chunks and 836 tokens at most).
  The dev threshold of MiniCheck, set with the stock judge, is therefore
  applied unchanged. On RAGTruth the stock judge is kept.
- Tolerance, fixed on 2026-11-09 before the measurement on MPS: scores exactly
  equal to those of the stock judge where no batch is split, and within 1e-5
  where one is. Grouping the rows of a batch differently can move float32
  results in their last digits. On CPU, known when the tolerance was fixed,
  the difference was zero when the batch was not split and at most 4.3e-7 on
  the probability of one chunk when it was.
- Measurement on MPS, 2026-11-09, against the scores of the stopped run:
  exactly equal on 20 inputs without a split batch; 4.6e-08 apart on the
  longest document of WiCE (17 356 words, 36 chunks); 5.1e-07 apart on the
  input with the largest batch of WiCE (7 rows padded to 2048 tokens). Tokens
  sent and truncation flags are equal. Peak memory footprint 5.81 GB, for
  3.61 GB after loading.
- Full run of 2026-11-09 through the wrapper: 8424 inputs scored, no failure,
  7348 s of judge time (2 h 02), swap stable around 7.6 GB throughout. Six
  inputs are marked truncated: one chunk longer than 2048 tokens with the
  claim appended, in 3 inputs of ClaimVerify, 2 of ExpertQA and 1 of WiCE;
  4 of them are among the 307 inputs HHEM truncates, 2 are not.
- Check after the full run, done on 2026-11-09 on the 356 inputs of the
  stopped run: the 314 inputs without a split batch have exactly equal
  scores; of the 42 inputs with a split batch, 23 are exactly equal and 19
  differ, by at most 5.1e-07. Tokens sent and truncation flags are equal on
  the 356. No decision changes at 0.04438 or at 0.5. The tolerance holds and
  the values below are used.

**Secondary: common untruncated subset.** The metrics of all judges on the
inputs that neither RJ nor HHEM truncates. HHEM truncates 307 inputs. With the
declared templates, Qwen3-0.6B would truncate 282 inputs, 3 of them outside
HHEM's 307, and Pleias-RAG-350M 293, 2 of them outside. The common subset will
therefore hold 8114 or 8115 inputs, depending on the base selected.

**Before training.** HHEM and MiniCheck are run on the five datasets before
any training of RJ, for the reference values and for the standard error of the
paired difference of mean AUC. The benchmark's conditions were accepted on
2026-11-09. HHEM and MiniCheck were run on 2026-11-09, MiniCheck through the
wrapper described above; their values are under "Reference values out of
domain" below.

**Reference, not comparator.** MiniCheck is reported alongside. It was
introduced together with LLM-AggreFact by the same authors, so its results
there are those of a model developed with this benchmark in view; comparing
RJ with it on this benchmark would not be a fair test. HHEM's training data is
unknown and may include some of these datasets.

**Pipeline check, not a target.** The balanced accuracies of the baselines at
0.5 are compared once with the published figures, to detect a conversion or
scoring error. A gap above 2 points on a dataset is investigated before the
values are used.

The published figures are those of the benchmark's leaderboard
(https://llm-aggrefact.github.io/, read on 2026-11-09), entries HHEM-2.1-open
and MiniCheck-Flan-T5-L, in points of balanced accuracy on the test split.
Its threshold rule: no threshold is tuned. The leaderboard states that systems
are benchmarked in a zero-shot fashion; the benchmark's paper (Tang, Laban and
Durrett, EMNLP 2024, section 4.1) sets the threshold at the midpoint of the
output score range, 0.5 for most fact-checkers; the official evaluation
notebook converts the raw probability into a label with the threshold 0.5.
The leaderboard does not say how HHEM was given its longest inputs.

HHEM, checked on 2026-11-09 (ours: flagged when score < 0.5):

| Dataset | Ours at 0.5 | Published | Gap, points |
|---|---|---|---|
| WiCE | 75.71 | 75.9 | -0.19 |
| REVEAL | 86.65 | 86.6 | +0.05 |
| ClaimVerify | 73.57 | 73.9 | -0.33 |
| FactCheck-GPT | 71.85 | 71.8 | +0.05 |
| ExpertQA | 57.58 | 58.4 | -0.82 |
| Mean of the five | 73.07 | 73.32 | -0.25 |

No gap exceeds 2 points. The two datasets without any truncated input equal
the published figures once rounded to one decimal. The three gaps are on the
datasets where the 4096-token limit applies, and all three are negative. On
WiCE and ClaimVerify the gap amounts to one or two claims. On ExpertQA the
limit could account for the whole gap only if HHEM, given the full documents,
were far better on the 125 truncated inputs than on the others (for instance
a recall of 0.80 and a specificity of 0.68, against 0.58 and 0.58); the cause
is not established.

MiniCheck, checked on 2026-11-09 (ours: flagged when score < 0.5, run through
the wrapper):

| Dataset | Ours at 0.5 | Published | Gap, points |
|---|---|---|---|
| WiCE | 72.42 | 72.2 | +0.22 |
| REVEAL | 86.25 | 86.2 | +0.05 |
| ClaimVerify | 74.59 | 74.6 | -0.01 |
| FactCheck-GPT | 74.72 | 74.7 | +0.02 |
| ExpertQA | 58.87 | 59.0 | -0.13 |
| Mean of the five | 73.37 | 73.34 | +0.03 |

No gap exceeds 0.3 point. Four datasets equal the published figures once
rounded to one decimal; on WiCE the gap is one not supported claim out of
247. MiniCheck reads whole documents through its chunks, so no length limit
enters this comparison. Its agreement on the three datasets where HHEM's gaps
are negative is consistent with the 4096-token limit as the cause of those
gaps, without establishing it. The conversion and the scoring chain are
checked for both baselines.

**Resolution.** Measured on 2026-11-09 with HHEM and MiniCheck as the pair,
under the protocol of section 9: 10000 resamples by document within each
dataset, the same draws for both judges, AUC per dataset in each resample and
then the mean over the five. Computed by `ood/ood_paired.py`, result in
`results/ood/paired-minicheck-hhem.json`.

- Standard error of the paired difference of mean AUC: 0.0064 on all inputs,
  0.0067 on the 8117 inputs HHEM does not truncate. The two mean AUCs have a
  correlation of 0.53 across resamples; if they were independent, the
  standard error would be 0.0093.
- With a margin of 0.02 and the test at 0.025, the null is rejected from an
  observed Delta_OOD of about -0.0075. Powers by normal approximation, with
  the standard error of this pair taken as that of the RJ-HHEM pair:

  | True Delta_OOD | Power at 0.025 |
  |---|---|
  | -0.02 | 0.03 |
  | -0.01 | 0.35 |
  | 0.00 | 0.88 |
  | +0.01 | 1.00 |

  A judge exactly equal to HHEM passes with a probability of 0.88; if its
  errors were independent of HHEM's, 0.58. The eighteenth revision of this
  file estimated the standard error at 0.007 to 0.012 by the Hanley-McNeil
  approximation, without clustering by document; the measured value sits
  just below that range.
- For the pair itself: MiniCheck's mean AUC exceeds HHEM's by +0.0082
  [-0.0040, +0.0209]; the null Delta <= -0.02 has p = 0.0001. By dataset:
  WiCE +0.0042 [-0.0444, +0.0524], REVEAL -0.0011 [-0.0124, +0.0104],
  ClaimVerify -0.0178 [-0.0472, +0.0116], FactCheck-GPT +0.0363 [+0.0156,
  +0.0572], ExpertQA +0.0196 [+0.0039, +0.0350]. These describe the two
  references; MiniCheck is not a comparator of RJ (above).

### Reference values out of domain

HHEM limited to 4096 tokens, run of 2026-11-09 on the converted file above
(config hash `cfd9eb00c7c54380f23057a2635e49d33a5f6359c6caff4388d5ffcb5cae9adf`),
8424 inputs scored. Intervals: 10000 resamples by document within each
dataset, seed 7, percentile at 95%; no resample excluded. A claim is flagged
when score < threshold. Computed by `ood/ood_metrics.py`. MiniCheck's values
follow HHEM's; the standard error of the paired difference is under
"Resolution" above.

All inputs:

| Dataset | Inputs | Supported | Not supported | AUC | Balanced accuracy at 0.18617 | Balanced accuracy at 0.5 |
|---|---|---|---|---|---|---|
| WiCE | 358 | 111 | 247 | 0.8322 [0.7874, 0.8746] | 0.7502 [0.7051, 0.7925] | 0.7571 [0.7068, 0.8054] |
| REVEAL | 1710 | 400 | 1310 | 0.9384 [0.9261, 0.9498] | 0.8323 [0.8171, 0.8469] | 0.8665 [0.8498, 0.8821] |
| ClaimVerify | 1088 | 789 | 299 | 0.8262 [0.7933, 0.8566] | 0.6739 [0.6427, 0.7050] | 0.7357 [0.7032, 0.7671] |
| FactCheck-GPT | 1566 | 376 | 1190 | 0.8049 [0.7786, 0.8307] | 0.7349 [0.7096, 0.7595] | 0.7185 [0.6915, 0.7460] |
| ExpertQA | 3702 | 2971 | 731 | 0.5981 [0.5743, 0.6222] | 0.5583 [0.5387, 0.5785] | 0.5758 [0.5551, 0.5968] |
| Mean of the five | | | | 0.8000 [0.7868, 0.8128] | 0.7099 [0.6972, 0.7225] | 0.7307 [0.7167, 0.7447] |

Recall on not supported claims and specificity, same inputs:

| Dataset | Recall at 0.18617 | Specificity at 0.18617 | Recall at 0.5 | Specificity at 0.5 |
|---|---|---|---|---|
| WiCE | 0.6356 | 0.8649 | 0.7935 | 0.7207 |
| REVEAL | 0.6870 | 0.9775 | 0.7855 | 0.9475 |
| ClaimVerify | 0.3946 | 0.9531 | 0.5652 | 0.9062 |
| FactCheck-GPT | 0.6958 | 0.7739 | 0.8252 | 0.6117 |
| ExpertQA | 0.3953 | 0.7213 | 0.5800 | 0.5715 |
| Mean of the five | 0.5617 | 0.8581 | 0.7099 | 0.7515 |

Inputs that HHEM does not truncate (8117). REVEAL and FactCheck-GPT have no
truncated input and keep the values above:

| Dataset | Inputs | Supported | Not supported | AUC | Balanced accuracy at 0.18617 | Balanced accuracy at 0.5 |
|---|---|---|---|---|---|---|
| WiCE | 323 | 101 | 222 | 0.8429 [0.7973, 0.8848] | 0.7514 [0.7050, 0.7957] | 0.7731 [0.7228, 0.8205] |
| ClaimVerify | 941 | 685 | 256 | 0.8375 [0.8033, 0.8695] | 0.6829 [0.6509, 0.7145] | 0.7519 [0.7177, 0.7857] |
| ExpertQA | 3577 | 2871 | 706 | 0.5995 [0.5749, 0.6241] | 0.5577 [0.5376, 0.5786] | 0.5784 [0.5568, 0.6000] |
| Mean of the five | | | | 0.8046 [0.7914, 0.8176] | 0.7118 [0.6987, 0.7248] | 0.7377 [0.7232, 0.7519] |

These 8117 inputs are not yet the common subset, which also leaves out the 2
or 3 inputs that only RJ truncates; the values are recomputed on it once the
base is selected. On the 307 truncated inputs alone, as a description without
interval: AUC 0.740 on WiCE (35 inputs, 10 supported), 0.745 on ClaimVerify
(147) and 0.569 on ExpertQA (125).

- The mean AUC of HHEM, the figure RJ is compared with, is 0.800, with an
  interval of half-width 0.013 for HHEM alone. The resolution of the target
  comes from the paired difference: standard error 0.0064, measured with
  MiniCheck as the other member of the pair ("Resolution" above).
- On ExpertQA HHEM is close to chance (AUC 0.598), and this dataset counts for
  a fifth of the mean. It is hard for every published entry: none of the 47
  entries of the leaderboard exceeds 61.0 points of balanced accuracy on it.
- The RAGTruth dev threshold, set for a specificity of 0.96 there, lands at a
  specificity between 0.72 and 0.98 out of domain. This is the transfer that
  the balanced accuracies at that threshold are meant to describe.

MiniCheck-Flan-T5-Large through the wrapper of this section, run of
2026-11-09 on the same file (config hash
`cbccef85dc7ad2138c39b0adc01f883056a7aa539e6334d6bbb1c28a01949d30`), 8424
inputs scored. Same resamples, same intervals, same reading of the
thresholds; the RAGTruth dev threshold of MiniCheck is 0.04438 (section 4).
Result in `results/ood/metrics-minicheck-bounded-mps.json`.

All inputs:

| Dataset | Inputs | Supported | Not supported | AUC | Balanced accuracy at 0.04438 | Balanced accuracy at 0.5 |
|---|---|---|---|---|---|---|
| WiCE | 358 | 111 | 247 | 0.8364 [0.7898, 0.8779] | 0.7419 [0.6982, 0.7818] | 0.7242 [0.6742, 0.7726] |
| REVEAL | 1710 | 400 | 1310 | 0.9373 [0.9252, 0.9485] | 0.7409 [0.7262, 0.7557] | 0.8625 [0.8445, 0.8796] |
| ClaimVerify | 1088 | 789 | 299 | 0.8084 [0.7771, 0.8394] | 0.6469 [0.6189, 0.6750] | 0.7459 [0.7159, 0.7760] |
| FactCheck-GPT | 1566 | 376 | 1190 | 0.8412 [0.8185, 0.8635] | 0.7362 [0.7141, 0.7572] | 0.7472 [0.7212, 0.7726] |
| ExpertQA | 3702 | 2971 | 731 | 0.6177 [0.5944, 0.6415] | 0.5676 [0.5488, 0.5868] | 0.5887 [0.5687, 0.6087] |
| Mean of the five | | | | 0.8082 [0.7954, 0.8209] | 0.6867 [0.6750, 0.6984] | 0.7337 [0.7200, 0.7474] |

Recall on not supported claims and specificity, same inputs:

| Dataset | Recall at 0.04438 | Specificity at 0.04438 | Recall at 0.5 | Specificity at 0.5 |
|---|---|---|---|---|
| WiCE | 0.5830 | 0.9009 | 0.8988 | 0.5495 |
| REVEAL | 0.4893 | 0.9925 | 0.8099 | 0.9150 |
| ClaimVerify | 0.3344 | 0.9594 | 0.6388 | 0.8530 |
| FactCheck-GPT | 0.5840 | 0.8883 | 0.8454 | 0.6489 |
| ExpertQA | 0.3611 | 0.7742 | 0.6594 | 0.5180 |
| Mean of the five | 0.4704 | 0.9031 | 0.7705 | 0.6969 |

Inputs that HHEM does not truncate (8117), the same subset as in HHEM's
table, so that the two judges are read on the same inputs. REVEAL and
FactCheck-GPT keep the values above. MiniCheck's own 6 truncated inputs are
not removed here: 4 are among HHEM's 307, the 2 others stay in the subset.

| Dataset | Inputs | Supported | Not supported | AUC | Balanced accuracy at 0.04438 | Balanced accuracy at 0.5 |
|---|---|---|---|---|---|---|
| WiCE | 323 | 101 | 222 | 0.8538 [0.8095, 0.8953] | 0.7509 [0.7083, 0.7916] | 0.7376 [0.6860, 0.7881] |
| ClaimVerify | 941 | 685 | 256 | 0.8110 [0.7774, 0.8433] | 0.6563 [0.6261, 0.6878] | 0.7583 [0.7261, 0.7907] |
| ExpertQA | 3577 | 2871 | 706 | 0.6156 [0.5911, 0.6398] | 0.5667 [0.5471, 0.5864] | 0.5858 [0.5654, 0.6064] |
| Mean of the five | | | | 0.8118 [0.7989, 0.8246] | 0.6902 [0.6781, 0.7024] | 0.7383 [0.7244, 0.7527] |

On the 307 inputs HHEM truncates, as a description without interval: AUC
0.668 on WiCE (35 inputs, 10 supported), 0.796 on ClaimVerify (147) and
0.686 on ExpertQA (125). MiniCheck reads these documents in full through its
chunks; HHEM reads their first 4096 tokens and gets 0.740, 0.745 and 0.569.

- MiniCheck's mean AUC is 0.808, with an interval of half-width 0.013. It is
  reported alongside, not as a comparator (above).
- At 0.5, the threshold of the published protocol, MiniCheck's balanced
  accuracies reproduce the leaderboard within 0.3 point (above).
- MiniCheck's RAGTruth dev threshold, 0.04438, is low on its scale. Out of
  domain it lands at a specificity between 0.77 and 0.99 and a recall between
  0.33 and 0.58: a high-specificity operating point everywhere, as on
  RAGTruth, at the cost of recall. Its mean balanced accuracy at that
  threshold, 0.687, is below its value at 0.5, 0.734, on every dataset but
  WiCE. HHEM's dev threshold moves its mean balanced accuracy less
  (0.710 against 0.731).

**Terms.** The benchmark is gated on Hugging Face, licensed CC BY-ND 4.0, and
its card states that it is permitted for use as an evaluation benchmark and
that its data should not be used for pretraining or fine-tuning. It is used
for this evaluation only. The converted files stay local and are never
published, even reformatted; only metrics are published, as for RAGTruth
(section 2).

## 8. Cost

**Configuration.** The targets bear on the GPU of the benchmark machine (Apple
M2, 8 cores, 16 GB of unified memory, no fan), each judge running on its
native stack (PyTorch MPS or MLX). The numerical precision is declared for
each judge. The configuration measured for cost is the one that produces the
quality results of sections 5 to 7: same stack, same precision.

**Regime.** One process, one judge, one sentence per call, and the device
cache released after each call: `torch.mps.empty_cache()` for the PyTorch
judges, `mx.clear_cache()` for an MLX one. The regime is applied by a local
script that wraps the judges (`cost/measure_cost.py`); the judges themselves
are not modified. HHEM's scores are identical with and without the release.

**Measurement.** Cost is measured on the 739-sentence sample for every judge,
candidate or baseline, not on the full test set.

- Latency: time of the call plus the release of the cache, per sentence. Each
  judge is measured from a cold start on the 739-sentence sample, one run at a
  time, without interleaving. A cold start means no measurement run and no
  other heavy load in the 15 minutes before, on mains power. The start time,
  the end time, the duration and the order of the runs are reported.
- Memory: peak memory footprint of the process over the same run, the figure
  `/usr/bin/time -l` prints. It includes GPU memory and the loading of the
  model.
- 1 GB is 10^9 bytes.

| Constraint | Target |
|---|---|
| Median latency per sentence, 739-sentence sample, cold start | < 1 s |
| Peak memory footprint over the 739 sentences, cache released after each call | < 3 GB |

Both targets are measured on the 739 sentences, as HHEM's reference is. For a
candidate base the run then ends with one input of 4096 tokens: its latency
and the peak footprint including it are reported without threshold, since such
an input concerns the out-of-domain data and no baseline receives one. Also
reported without threshold: p90 latency; latency by input length; latency by
fifth of the run on the inputs of at most 512 tokens; the footprint after
loading; resident memory.

**Candidate bases.** Measured before any training, with the pretrained
weights: one forward pass per sentence, the score read at the last position
only. The cost of a forward pass does not depend on the values of the weights.
Each base is measured on MLX, its declared stack, and on PyTorch MPS at
float16 for comparison. Elimination is decided on MLX only (section 3).

**Final model.** The measurements of the pretrained bases serve the
elimination before training only. RJ, once trained and selected, is measured
again in the same regime on the 739 sentences, before its evaluation on the
test set, in its configuration of use: same stack and precision as for
quality, adapters merged into the weights or kept separate, which is declared
with the result. The cost targets apply to this measurement.

**Baselines: results (2026-11-09).** HHEM and MiniCheck measured the same way,
on the 739 sentences, cache released after each call, each from a cold start.

| Baseline | Stack, precision | Median latency | p90 | Peak footprint, 739 sentences | Run | Targets |
|---|---|---|---|---|---|---|
| HHEM | PyTorch MPS, float32 | 0.072 s | 0.260 s | 2.54 GB | 95 s, started 41 min after the previous run | both met |
| MiniCheck | PyTorch MPS, float32 | 0.407 s | 1.340 s | 4.74 GB | 459 s, started 30 min after the previous run | latency met, memory missed |

- Both reproduce the scores of the harness exactly (maximum difference 0 over
  the 739 sentences).
- The targets bear on RJ. MiniCheck missing the memory target has no
  consequence; its weights alone take 3.13 GB at float32.
- Thermal drift within a run. On the inputs of at most 512 tokens, MiniCheck
  goes from 57.8 ms per 100 tokens in the first fifth of its run to 101.3 in
  the last, although it started cold: a run of several minutes heats the
  machine by itself. It is flat during its first two minutes (57 to 59 ms per
  100 tokens) and slower from then on (86 to 90). HHEM stays flat, 14.9 to
  15.1, but its whole run lasts 95 s, less than the time MiniCheck takes to
  start slowing down: this run does not show that HHEM would stay flat longer.
- MiniCheck's median therefore mixes a cold and a warm state. A cold start is
  not enough to compare the latencies of judges whose runs differ in length;
  such a comparison is to be read on the first fifth of each run.

**Candidate bases: results (2026-11-09).**

| Base | Stack, precision | Median latency | p90 | Peak footprint, 739 sentences | 4096-token input: latency | Peak with that input | Targets |
|---|---|---|---|---|---|---|---|
| Pleias-RAG-350M | MLX, bfloat16 | 0.144 s | 0.369 s | 1.57 GB | 1.62 s | 1.57 GB | both met |
| Qwen3-0.6B | MLX, bfloat16 | 0.294 s | 0.708 s | 2.15 GB | 3.42 s | 2.15 GB | both met |
| Pleias-RAG-350M | PyTorch MPS, float16 | 0.196 s | 0.506 s | 2.15 GB | 2.01 s | 2.15 GB | both met, comparison only |
| Qwen3-0.6B | PyTorch MPS, float16 | 0.396 s | 1.123 s | 2.98 GB | 5.97 s | 2.98 GB | both met, comparison only |

- Both bases meet both targets on MLX: none is eliminated, and the choice
  between them is made on the dev set (section 12).
- No conversion was needed: the reports confirm weights published in bfloat16.
- The four runs were chained in the order of the table, less than a minute
  apart, not from a cold start. The memory figures do not depend on it. The
  latency verdicts hold, since heat can only raise latency and every median is
  below 0.4 s. The latencies are not valid for comparing the stacks or the
  bases with each other.
- The 4096-token input did not raise the peak in any run. On PyTorch MPS the
  peak is reached while loading and on the first call, before any long input;
  for Qwen3-0.6B it sits 23 MB under the threshold. On MLX it is reached on
  the long inputs of the sample.
- Longest of the 739 inputs: 2203 tokens for Pleias-RAG-350M, 2129 for
  Qwen3-0.6B, with the provisional templates.
- On the inputs of at most 512 tokens, the latency per 100 tokens rises by 16
  to 45% between the first and the last fifth of these four chained runs.

**Earlier baseline runs.** Harness regime, cache kept between calls, unless
stated. Thermal state not controlled. They are superseded by the reference
runs above and kept for the record.

| Judge | Device, precision | Sentences | p50 | p90 | Resident memory | Peak footprint |
|---|---|---|---|---|---|---|
| HHEM | CPU, float32 | 10385 | 0.172 s | 0.645 s | 1756 to 2086 MiB, depending on the run | not measured |
| HHEM | CPU, float32 | 739 | 0.178 s | 0.758 s | 1756 MiB | not measured |
| HHEM | MPS, float32 | 739 | 0.073 s | 0.254 s | 1027 MiB | 3.65 GB (3480 MiB) |
| HHEM | MPS, float32, cache released (diagnostic) | 739 | not a reference | not a reference | not recorded | 2.54 GB (2418 MiB) |
| MiniCheck | MPS, float32 | 10385 | 0.350 s | 1.424 s | 531 MiB | not measured |
| MiniCheck | MPS, float32 | 739, first run | 0.360 s | 1.323 s | 563 MiB | not measured |
| MiniCheck | MPS, float32 | 739, second run | 0.398 s | 1.519 s | 568 MiB | 4.84 GB (4612 MiB) |

On the 739 sentences, HHEM's scores on MPS and on CPU differ by at most 3e-6
and no decision changes at any threshold used in this file: the quality
figures of section 13, computed from the CPU run, hold for the MPS
configuration. The two MiniCheck runs on MPS give identical scores.

**What the diagnostic of HHEM showed (2026-11-09).**

- The footprint includes GPU memory. MiniCheck's weights take 2988 MiB at
  float32; they are absent from its resident memory (568 MiB) and present in
  its footprint (4612 MiB).
- Without release, the footprint counts the allocator's heaps, not the judge's
  need. On this machine, PyTorch's MPS allocator takes GPU memory in heaps of
  1 GiB and, with default settings, keeps them once taken.
- HHEM retains nothing between calls. Its allocated tensors return to 420 MiB,
  the weights, after each of the 739 calls, and repeating one input leaves
  every figure flat. During a call, allocated tensors peak at about 1 GiB for
  the longest inputs (986 MiB observed at 2356 tokens, sampled every 0.5 ms).
- One heap serves every input up to about 1300 tokens (648 of the 739); beyond
  about 1670 tokens a second heap is always needed. The third heap of the
  harness run appears at the 202nd sentence, on a 1984-token input, after a
  2356-token input had been served with two: fragmentation of the cached
  buffers, not need.
- With the cache released after each call, the scores are identical, the
  allocator never holds more than two heaps and the peak footprint is 2.54 GB.
  The 739 sentences then pass under an allocator cap of 2.84 GiB, where the
  harness run stops at the 202nd sentence. Under a cap of 1.89 GiB both stop
  at the 19th sentence, the first input that needs a second heap (1498 tokens).
- Releasing the cache costs no latency below 1500 tokens and about 80 ms per
  call above (paired measure on 60 inputs).
- The threshold of 3 GB is two allocator heaps plus the rest of the process.
  HHEM sits at 2.54 GB on inputs of at most 2356 tokens; MiniCheck's weights
  alone take 3.13 GB.

**Latency and thermal state.** The same 739-sentence series, in the same mode,
gave a median of 77 ms on a first pass and 131 ms on a fourth consecutive pass
a few minutes later. Hence the cold start, and in every report the latency by
fifth of the run: a run long enough heats the machine by itself. The 739
inputs are not in random order, their mean length falling from about 1000 to
350 tokens along the run, so the fifths are compared on the inputs of at most
512 tokens only, with the latency divided by the input length.

## 9. Statistical protocol

- Unit of resampling: the source (`source_id`), 300 groups in the test set.
  Groups are drawn with replacement; all sentences of a drawn group enter the
  resample.
- 10000 resamples, seed 7, the same resamples for every judge, so differences
  are paired.
- Primary analysis: each judge's threshold is re-derived in every resample
  with the rule of section 4, on the resample's SUPPORTED sentences.
- Transfer analysis and out-of-domain balanced accuracies: the thresholds are
  constants (the dev thresholds of section 4, and 0.5).
- Intervals: percentile, 95%.
- One-sided p-value of a null "Delta <= d0":
  p = (1 + number of resamples with Delta* <= d0) / (10000 + 1).
- A resample where a denominator is zero is excluded and counted; the count is
  reported.
- Out-of-domain target: resampling by document within each dataset, AUC per
  dataset in each resample, then the mean over the five datasets. No threshold
  is involved. A resample in which a dataset has a single class is excluded
  and counted.
- The analysis script, `analysis/ragtruth_analysis.py`, is committed with this file
  and its hash recorded at freeze time (section 2, Provenance). Options:
  `--test`, `--dev`, `--run-a` (the candidate), `--run-b` (HHEM), `--run-c`
  (MiniCheck, reported alongside), `--out`.

## 10. What has already been seen

The RAGTruth test set is treated as entirely seen.

- Aggregate results of HHEM and MiniCheck on the full test set, and of HHEM,
  MiniCheck and Prometheus on a 739-sentence sample that contains all 139
  CONTRADICTED sentences, were studied by label, task, generator and input
  length.
- About 60 test sentences were read with their labels and annotator notes.
- The scope decisions of section 2, the pattern lists and the targets of this
  file were written after those analyses.
- The input lengths of the three splits were measured with the tokenizers of
  the two candidate bases (section 3). The measure uses no label and no score.
- During the cost measurement of the candidate bases, the pretrained,
  untrained bases scored the 739 test sentences, and these scores were written
  to the files of the four runs. They were deleted on 2026-11-09 without having
  been read: never displayed, never aggregated, never compared with the
  labels. No other file holds them. The script no longer writes scores for a
  candidate base.
- Audits of training labels. The audit of 99 sentences had a first pass,
  discarded, in which about half of 86 sentences were judged on the previous
  sentence (section 2). Before judging those 86 again, the annotator had seen
  the agreement score of that pass, about 0.2 on CONTRADICTED and UNSUPPORTED
  and without meaning, and three sentences quoted to explain the shift; the
  remarks on the annotation grid came only after the final score. The
  annotator then knew the results of the audit of
  99 sentences, including the breakdown by conversion rule (section 2), before
  doing the blind audit of 40 sentences (section 2). The labels and the
  groups of those 40 sentences stayed hidden from the annotator until the
  audit was complete. The decision rule on inherited labels was fixed after
  the sheet was filled and before the key was opened; with 4 CONTRADICTED
  judgments among 40 sentences, its outcome did not depend on the key.
- Out-of-domain benchmark. The documents and claims of the test split were
  tokenised to count input lengths (section 7), without the labels. HHEM was
  then run on the five datasets on 2026-11-09, before any training of RJ: its
  aggregate results per dataset are known (AUC, balanced accuracy, recall and
  specificity, with and without the truncated inputs), and the benchmark's
  published leaderboard was read. The labels were used by the analysis script
  only; no instance was read with its label. The out-of-domain target, its
  margin and the choice of the five datasets were written before this run.
  The dev split was not read. A first run of the stock MiniCheck judge was
  stopped after 356 inputs of WiCE (section 7). Its scores were used for one
  thing only: to compare them with the wrapper's scores on the same inputs.
  No metric was computed from them and they were not set against the labels.
  The wrapper's rule was written from input lengths, latencies and memory.
  MiniCheck was then run in full through the wrapper on 2026-11-09, still
  before any training of RJ; its aggregate results are known in the same
  terms as HHEM's, and so is the paired difference of mean AUC between the
  two baselines, per dataset and overall. The two scripts that computed them
  print aggregates only.

No contradicted test sentence is untouched, so no claim of a blind test is
made. What protects the result is procedural:

- no training-time choice (data, hyperparameters, checkpoint, threshold) uses
  the test set;
- RJ is evaluated on the test set once, after selection is final; no
  intermediate checkpoint is scored on it;
- this file, the analysis script and the exclusion and pattern lists are
  committed before the first training run;
- pattern lists are refined on training examples only from now on;
- results are published whatever the outcome, with any re-evaluation disclosed.

## 11. Known ceiling on contradictions

14 of the 139 CONTRADICTED test sentences carry an annotator note that matches
the attribution patterns. The training set excludes such cases, so RJ is
expected to pass them: its recall on contradictions is capped near
125 / 139 = 0.90. P1 is computed on all 139 for both judges; the figure outside
these 14 is secondary. The cap is approximate: in the training set, the same
patterns also matched sentences that the hand validation of 2026-11-10 kept
as contradictions (16 of the 74 validated: 10 contradictions of content, 3
assertions about the context, 3 notes placing the content nowhere), which RJ
should detect. The 3 CONTRADICTED test
sentences whose label is inherited from a span covering several sentences, a
kind excluded from the training set by rule 5 of section 2, are among these
14: the cap is unchanged.

This cap concerns agreement with the labels. The audit of section 2 points to
a lower level for a judge that reads as the auditor did: 0.61 on CONTRADICTED
labels, an estimate from a single annotator on 33 sentences (Wilson 95%
interval 0.44 to 0.75).

## 12. Model selection

- Checkpoints and hyperparameters are selected on the dev set
  (5536 / 363 / 112). The same criterion decides between the candidate bases
  that passed the cost targets (section 3).
- Criterion: the mean of two standardised partial AUCs over a false alarm rate
  of 0 to 0.10 (McClish), one for UNSUPPORTED against SUPPORTED, one for
  CONTRADICTED against SUPPORTED. Each class of the primary targets weighs one
  half, whatever its size.
- Definition of each term. Positive class: the not supported class concerned;
  the other not supported class is left out. False alarm rate: share of
  SUPPORTED sentences flagged. The empirical ROC curve is integrated by
  trapezoids from 0 to 0.10, with linear interpolation at 0.10, tied scores
  forming one point. With A that area, the term is
  0.5 * (1 + (A - 0.005) / (0.1 - 0.005)): 0.5 at chance, 1 for a perfect
  judge. It is the value returned by scikit-learn's `roc_auc_score` with
  `max_fpr=0.10`, checked on both baselines.
- Why this criterion. The primary analysis reads the ROC curve at 4% of false
  alarms, a region the whole-curve AUC weighs little. A recall at one point is
  too noisy on dev: at 0.96 specificity, the pooled recall has a standard
  error of 0.035 to 0.045, against 0.013 to 0.015 for the criterion. Pooling
  the two classes would let UNSUPPORTED, three quarters of the not supported
  dev sentences, drive the selection, while P1 bears on contradictions.
- Reference values on dev: HHEM 0.641 [0.616, 0.667], from 0.744 on
  UNSUPPORTED and 0.537 on CONTRADICTED; MiniCheck 0.619 [0.592, 0.649], from
  0.694 and 0.543. Both baselines are close to chance on contradictions in
  this region. The paired difference between the two has a standard error of
  0.012.
- The number of configurations (base, hyperparameters) and of checkpoints
  evaluated on the dev set is reported with the results, so that the dev
  values of the selected model are read for what they are, the best of that
  many.
- The dev threshold of section 4 is derived once, after selection.
- Dev is never used as training data.

## 13. Reference values

Full test set, 10000 grouped resamples, both baselines scored on all 10385
sentences. Computed by `analysis/ragtruth_analysis.py`; result in
`results/ragtruth/analysis-minicheck-vs-hhem.json`.

Primary analysis, equal specificity on test (thresholds re-derived in every
resample):

| Quantity | HHEM | MiniCheck |
|---|---|---|
| Threshold on the full test | 0.26894 | 0.06499 |
| Specificity | 0.9600 by construction | 0.9600 by construction |
| Recall, CONTRADICTED (139) | 0.1583 [0.0976, 0.2222] | 0.2230 [0.1366, 0.2917] |
| Recall, UNSUPPORTED (469) | 0.6098 [0.5277, 0.6795] | 0.5629 [0.4900, 0.6301] |
| Recall, not supported (608) | 0.5066 [0.4390, 0.5603] | 0.4852 [0.4202, 0.5378] |

The protocol applied to MiniCheck as if it were the candidate: Delta_C =
+0.065 [-0.047, 0.153], p = 0.16 against the null of P1; Delta_U = -0.047
[-0.097, 0.010], p = 0.41 against the null of P2. Neither target would be met.

Transfer of the operating point, dev thresholds:

| Quantity | HHEM | MiniCheck |
|---|---|---|
| Dev threshold | 0.18617 | 0.04438 |
| Specificity on test | 0.9723 [0.9674, 0.9769] | 0.9770 [0.9723, 0.9814] |
| Recall, CONTRADICTED (139) | 0.1439 [0.0857, 0.2061] | 0.0935 [0.0496, 0.1429] |
| Recall, UNSUPPORTED (469) | 0.5458 [0.4773, 0.6167] | 0.4733 [0.4020, 0.5476] |
| Recall, not supported (608) | 0.4539 [0.3960, 0.5092] | 0.3865 [0.3240, 0.4485] |
| Balanced accuracy at the dev threshold | 0.7131 | 0.6817 |
| Balanced accuracy at 0.5 | 0.7597 | 0.7773 |

Without threshold:

| Quantity | HHEM | MiniCheck |
|---|---|---|
| AUC ROC | 0.8615 [0.8382, 0.8823] | 0.8532 [0.8161, 0.8817] |
| AUC, CONTRADICTED against SUPPORTED | 0.7260 [0.6813, 0.7672] | 0.7474 [0.7024, 0.7885] |
| AUC, UNSUPPORTED against SUPPORTED | 0.9017 [0.8781, 0.9246] | 0.8846 [0.8378, 0.9182] |
| Partial AUC, UNSUPPORTED against SUPPORTED, test | 0.7804 [0.7459, 0.8155] | 0.7703 [0.7358, 0.8032] |
| Partial AUC, CONTRADICTED against SUPPORTED, test | 0.5645 [0.5353, 0.5962] | 0.5850 [0.5557, 0.6159] |
| Selection criterion (mean of the two), test | 0.6724 [0.6499, 0.6965] | 0.6776 [0.6544, 0.7013] |
| Selection criterion, dev | 0.6406 [0.6155, 0.6670] | 0.6185 [0.5915, 0.6486] |
| Partial AUC, classes pooled, test (not the criterion) | 0.7310 [0.7024, 0.7585] | 0.7279 [0.6987, 0.7546] |

Partial AUCs are standardised and cover a false alarm rate of 0 to 0.10
(section 12).

**Shift of specificity between dev and test.** Both dev thresholds land above
0.96 on test: +0.012 for HHEM and +0.017 for MiniCheck. Resampling the dev
sources (which moves the threshold) together with the test sources, the
intervals for the test specificity are [0.9642, 0.9801] for HHEM and
[0.9649, 0.9859] for MiniCheck, and 0.2% and 0.3% of the resamples fall at or
below 0.96: the sampling of the two splits does not explain the shift. It is
common to the two judges (difference between their test specificities: +0.005
[-0.005, 0.013]) and it comes from QA. At the dev thresholds, specificity on
QA goes from 0.951 on dev to 0.970 on test for HHEM, and from 0.941 to 0.968
for MiniCheck; on Summary it moves by 0.003 for both. A threshold set on this
dev split is therefore likely to be conservative on test for RJ as well, which
is one more reason for the equal-specificity point as primary analysis.

## 14. Decisions before the freeze, and what follows it

Decided on 2026-11-09 and written into the sections above: the primary
analysis at equal specificity on test, without specificity guard; P2 on
UNSUPPORTED with a margin of 0.05, confirmed on the measured power; the
out-of-domain target on the mean AUC over five datasets against HHEM, margin
0.02, with balanced accuracy at two thresholds for all judges, HHEM limited
to 4096 tokens there, the metrics on the common untruncated subset and by
band of document length, and the baselines run on the five datasets before
any training; selection on the mean of the two per-class partial AUCs of the
dev set; two candidate bases with MLX in bfloat16 as declared stack, one
forward pass per sentence with the score read at the last position,
elimination on MLX only; the prompt template of each base, three output
labels with distinct first tokens and the score read from them; L = 4096
everywhere, no windowing because of the verification of absence; cost measured on the 739 sentences for every judge,
cache released after each call, peak footprint below 3 GB, the 4096-token
input reported without threshold, latency from a cold start of 15 minutes;
10000 resamples; no publication of converted data.

Also decided on 2026-11-09: the overlap rule of section 7, 13-word shingles
and removal from half of the document, a document of fewer than 13 words
overlapping only when it equals a whole training context; the scope of the
meta pattern list.

Done on 2026-11-09: the audit of 99 training sentences (section 2, Label
quality); the blind audit of 40 training sentences on spans covering several
sentences, and the application of its decision rule, which excludes the 107
inherited CONTRADICTED sentences from the training set (section 2); the
removal of these 107 sentences, in a new training file written next to the
original (section 2); the overlap check and the conversion of the out-of-domain data
(section 7); the out-of-domain run of HHEM, its reference values and the check
of its balanced accuracies against the published figures, no gap above 2
points (section 7); the wrapper that bounds MiniCheck's forward passes out of
domain, written after its first run was stopped, with its tolerance and its
measurement on MPS (section 7); the out-of-domain run of MiniCheck through
that wrapper, the check of its scores against the 356 inputs of the stopped
run, its reference values, the check of its balanced accuracies against the
published figures, no gap above 0.3 point, and the standard error of the
paired difference of mean AUC between the two baselines, 0.0064, which gives
the resolution of the out-of-domain target (section 7); the cost of the two
baselines in the reference regime (section 8);
the cost of the two candidate bases on both stacks. Both
meet the targets on MLX and neither is eliminated (section 8). The scores
those runs had written for the untrained bases were deleted unread
(section 10).

Decided on 2026-11-10: training set rule 1 keeps the sentences whose note
does not establish that the content appears in another passage (section 2);
abstentions leave the meta category and form a category of their own,
reported separately in the secondary measures, the labels of the training set
unchanged (sections 2 and 6); the instruction of the templates names what its
clause on sentences making no claim covers, announcements and courtesy
formulas (section 3). After a rereading of the same day: no judge of this
protocol can check an abstention, since none receives the question, and the
abstention category describes behaviour (sections 2, 3 and 6); the first,
discarded pass of the audit of 99 is declared (sections 2 and 10); the three
logits of RJ are converted to float32 before the softmax (section 3); the
final RJ is measured again for cost in its configuration of use, and the
targets apply to that measurement (section 8); the number of configurations
and checkpoints evaluated on dev is reported (section 12); every claim names
HHEM, with the differences to MiniCheck alongside (section 5). Also on
2026-11-10: the models produced by the project are named Gjallar, under the
Hugging Face organisation `ragnarok-ai` (section 1); on that day the Hub held
no model, dataset or space named gjallar, under `ragnarok-ai` or elsewhere,
and one empty personal account with that name.

Hashes replaced before the freeze, 2026-11-10: the hash of `meta_patterns.json`
recorded earlier the same day, `cac3483eab84d2d1…`, is replaced by
`dbe8efab1034931f…` (section 6). The 17 patterns are unchanged, checked
pattern by pattern against the earlier export, and the list matches the same
1535 training sentences; only the accompanying text of the file changed, its
`not_meta` field no longer saying that an abstention is checkable. The same
correction was made in the comments of `meta_match.py`, whose examples
reproducing sentences of the datasets were replaced by descriptions. The
hash of `abstention_patterns.json` is unchanged, its file being byte for byte
the same.

Done on 2026-11-10: the hand validation of the 74 attribution sentences and
the application of rule 1, in a new training file written next to the file
of rule 5 (section 2); the validation of the meta list on 30 matched and 15
left-out sentences, and the freezing of the two pattern lists, meta and
abstention, with their hashes (section 6); the tokenisation checks of the
amended templates: first tokens distinct, no RAGTruth input beyond 4096
tokens, the same 282 and 293 out-of-domain inputs beyond it as before
(section 3); the Provenance subsection of section 2, with the hashes of the
RAGTruth files, of the converter, of the exclusion script and of the two id
lists; the regeneration of the three splits in a temporary directory and the
check of the five hashes (section 2, Provenance); the analysis script
`ragtruth_analysis.py`, checked against section 13 on HHEM and MiniCheck
(sections 2 and 9).

After the freeze:

1. **Freeze commit.** This file, the converter and the exclusion script with
   the two lists of excluded ids, the pattern lists and their script, the
   analysis script and its tests, the out-of-domain and cost scripts, the
   aggregates of `results/` and the pinned versions, with the hashes recorded
   in section 2, are committed together and tagged `protocol-v1`. The
   training file is `train-rule5-rule1.jsonl` (section 2); dev and test are
   not touched.
2. **Base selection.** Not an open decision: both bases passed the cost
   targets (section 8), and the rule of section 12, the selection criterion
   on the dev set, decides between them once they are trained.
3. **Common untruncated subset out of domain.** Not an open decision either:
   the rule of section 7 defines it as the inputs that neither RJ nor HHEM
   truncates, and it is computed with the tokenizer of the selected base.
4. **Then**, in this order: training; the cost of the final RJ in its
   configuration of use (section 8); one evaluation on the test set and on
   the five out-of-domain datasets; publication of the results whatever the
   outcome, with the number of configurations and checkpoints evaluated on
   dev (section 12).

## Amendments after the freeze

### Amendment 1, written 2026-11-10, committed 2026-11-13

Written after the tag `protocol-v1` and before any training run, committed with the training
plan (`TRAINING.md`, tag `training-plan-v1`). No model has been trained, and no dev or test
value of a trained RJ is known: the feasibility and timing measurements of 2026-11-10 ran
LoRA updates on training examples, for 40 minutes at most, and every adapter they produced
was deleted without being scored on dev or test. The sections above are unchanged.

**(a) Tie-break between the candidate bases (sections 3, 8, 12 and 14).** Section 12 gives
one criterion for the choice between the bases and no rule for a small gap. This rule is
added:

- The best dev checkpoint of each base is taken, by the criterion of section 12.
- The standard error of the difference of their criteria is estimated on the dev set by a
  paired bootstrap grouped by source (`source_id`): 10000 resamples, seed 7, the same
  resamples for both models, the criterion recomputed for each model in every resample; the
  standard error is the standard deviation of the 10000 differences.
- If the absolute difference of the two criteria is smaller than that standard error,
  Pleias-RAG-350M, the less costly base, is retained.
- Otherwise the base with the higher criterion is retained.

Why Pleias-RAG-350M is the less costly base:

- Memory, section 8: on MLX in bfloat16 its peak footprint over the 739 sentences is
  1.57 GB, against 2.15 GB for Qwen3-0.6B. Section 8 states that these memory figures do not
  depend on the order of the runs.
- Latency, section 8: its median is 0.144 s against 0.294 s, its p90 0.369 s against
  0.708 s, and the 4096-token input takes 1.62 s against 3.42 s. Section 8 also states that
  these latencies, measured in chained runs and not from a cold start, are not valid for
  comparing the bases with each other. They are therefore not the ground of this rule on
  their own.
- Latency, measured since: on 2026-11-10, on the training machine in the same thermal state
  ("fair") for both, one scoring pass over the same kind of sample, 500 dev sentences with
  the pretrained weights, one sentence per call, took 132 s for Pleias-RAG-350M and 221 s
  for Qwen3-0.6B (136 s and 210 s in a first measurement the same day), and a training step
  took 1.6 to 1.8 s per 1000 tokens against 2.6 s. No score of these passes was kept and no
  metric was computed.
- Size: its weights take 0.71 GB against 1.50 GB.

The asymmetry is declared and intended. The training plan (`TRAINING.md`) gives
Pleias-RAG-350M two learning rates and 8 checkpoints evaluated on dev, against one learning
rate, the better of the two on Pleias-RAG-350M, and 4 checkpoints for Qwen3-0.6B: the best
of 8 is favoured over the best of 4, and the tie-break goes to Pleias-RAG-350M as well. Both
lean the same way on purpose: at a performance that the dev set cannot tell apart, the less
costly base is preferred, and Qwen3-0.6B is retained only if it is ahead by at least one
standard error.

The difference, its standard error and the base retained are reported with the results. The
rule concerns the choice between the two bases only; checkpoints and hyperparameters within a
base are still selected on the highest criterion.

**(b) Erratum, input lengths of Pleias-RAG-350M (section 3, Judge input).** The table of
input lengths gives the medians 561 (dev) and 543 (train) for Pleias-RAG-350M. Counted on
the complete rendered input, they are 558 and 540. The figures of the table came from the
script `prompt_templates.py lengths`, whose median adds up token counts of the template, the
context and the sentence taken separately, three tokens above the exact count for this
tokenizer. The medians of Qwen3-0.6B (547 and 529), the longest inputs of both bases and
the counts beyond L are exact and unchanged. The test median of Pleias-RAG-350M (543) comes
from the same script and is not recounted, the test set not being read before the final
evaluation. The same figures appear in `TEMPLATES.md`. This has no effect on any decision:
no input comes near L, and no rule uses a median.
