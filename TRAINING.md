# RAGnarok Judge: training plan

Status: written on 2026-11-10, fixed on 2026-11-13, before any training run, tag
`training-plan-v1`. No trained model exists and no dev value of a trained RJ is known. The feasibility and timing measurements of that day ran
LoRA updates on training examples, for 40 minutes at most; every adapter they produced was
deleted and none was scored on dev or test. `TARGETS.md` (tag `protocol-v1`) prevails over
this file wherever the two could be read differently.

This file fixes what `TARGETS.md` leaves open: the method of adaptation, the use of the
training file, the loss, the schedule, the configurations tried, and how a run is resumed.
Everything else (inputs, templates, score, selection criterion, cost regime, use of the test
set) is set by `TARGETS.md` and only recalled here. The code is in `training/`.

## 1. What is trained

| | Pleias-RAG-350M | Qwen3-0.6B |
|---|---|---|
| Repository and revision | `PleIAs/Pleias-RAG-350M`, `db001b29a335` | `Qwen/Qwen3-0.6B`, `c1899de289a0` |
| Layers | 26 | 28 |
| Trainable parameters (LoRA) | 3 727 360 | 5 046 272 |

- Stack: MLX with the base weights in bfloat16, as declared in section 3 of `TARGETS.md`;
  mlx 0.32.4 and mlx-lm 0.32.2 (`requirements-frozen.txt`).
- Adaptation: LoRA of rank 8 on every linear module of every transformer block (attention
  and feed-forward projections), scale 2.0, no dropout. The base weights stay frozen.
- Input: the whole context and the sentence, through the template of the base
  (`TEMPLATES.md`, `analysis/prompt_templates.py`), L = 4096 tokens, no windowing. No
  training or dev input exceeds L (longest: 2540 and 2508 tokens in train, 1949 and 1990
  in dev), so nothing is truncated.

## 2. Data

The training file is `train-rule5-rule1.jsonl` (section 2 of `TARGETS.md`, SHA-256
`bac1d36f…955edba`): 54 490 sentences, 49 874 SUPPORTED, 3 699 UNSUPPORTED, 917 CONTRADICTED.
Nothing else enters training: no dev sentence, no LLM-AggreFact data, no relabelling.

Each epoch is a subsample of that file:

| Class | Per epoch | Share of the epoch | Rule |
|---|---|---|---|
| CONTRADICTED | 917 | 6.6% | all, every epoch |
| UNSUPPORTED | 3 699 | 26.7% | all, every epoch |
| SUPPORTED | 9 232 | 66.7% | drawn at random, renewed at every epoch |
| Total | 13 848 | | |

- The SUPPORTED sentences of an epoch are drawn uniformly without replacement among the
  49 874, from their sorted ids, with a generator seeded by (13, epoch number). The 13 848
  examples of the epoch are then shuffled with the same generator. The draws depend on the
  seed and the epoch only: the three runs of section 5 see the same examples in the same order.
- The draw looks at nothing but the label. It is not aimed at meta sentences nor at
  abstentions, in either direction: they enter with their labels in the proportion the draw
  gives them, which keeps training set rule 2 of section 2 (no upstream filter). Their
  numbers in each epoch, counted with the frozen pattern lists by
  `training/epoch_counts.py`, follow from the seed and are known before training:

  | | Sentences | Meta | of which announcement | of which courtesy | Abstentions |
  |---|---|---|---|---|---|
  | Training file | 54 490 | 1 535 | 1 285 | 250 | 534 |
  | Epoch 1 | 13 848 | 301 | 252 | 49 | 110 |
  | Epoch 2 | 13 848 | 294 | 245 | 49 | 98 |

  Every meta sentence is SUPPORTED. The abstentions are SUPPORTED except one UNSUPPORTED and
  one CONTRADICTED, both present in every epoch. Meta sentences are 2.2% and 2.1% of an
  epoch against 2.8% of the file, abstentions 0.8% and 0.7% against 1.0%: both are almost
  all SUPPORTED, the class the draw thins out.
- Over the two epochs a SUPPORTED sentence is seen zero, one or two times; every
  UNSUPPORTED and CONTRADICTED sentence is seen twice.
- Why a subsample: one pass over the whole file takes about 17 hours for Pleias-RAG-350M
  and 26 to 28 hours for Qwen3-0.6B on the training machine (section 7), and 91.5% of it is
  SUPPORTED. The subsample keeps every not supported sentence and 24.6% of the tokens.

## 3. Loss

The loss is the quantity the judge is scored on (section 3 of `TARGETS.md`): one forward pass
over the prompt, the logits read at its last position, restricted to the first tokens of the
three labels.

- For each example, the three logits (`Supported`, `Unsupported`, `Contradicted`: tokens
  34636, 41884, 41437 for Qwen3-0.6B; 57889, 2076, 3099 for Pleias-RAG-350M) are converted to
  float32 and the loss is the cross-entropy of their softmax against the label.
- Nothing else is trained on: no other position, no other token of the vocabulary, no later
  token of a label, no end marker. The label text is not appended to the input.
- No class weights. The class balance comes from the subsample of section 2 only.
- The output head is applied to the last position only, in the loss and in the scoring of
  the dev set: the model body gives the final hidden state of every position, the state of
  the last position is kept, and the head (tied to the embeddings in both bases) is applied
  to that single state; the three label logits are read from its output. This is
  mathematically identical to computing the logits of every position and reading the last
  one, since the head is a linear map applied to each position separately; it only avoids
  computing logits that are never read. In bfloat16 the two computations agree to within
  one rounding step of the logits (checked on both bases by a test of `training/`). It is
  also how `cost/measure_cost.py` reads the score. The cost measurement of section 8 of
  `TARGETS.md` keeps its declared regime, unchanged.

## 4. Optimisation

| Setting | Value |
|---|---|
| Batch size | 1 |
| Gradient accumulation | 8 examples per update |
| Updates | 1 731 per epoch, 3 462 per run |
| Epochs | 2 |
| Optimiser | AdamW, betas 0.9 and 0.999, weight decay 0.01 |
| Learning rate | linear warm-up over the first 173 updates (5%), then cosine decay to 0 |
| Peak learning rate | the one setting that varies (section 5) |
| Gradient checkpointing | on |
| Gradient clipping | global norm clipped at 1.0, on the accumulated gradient of each update |
| Seed | 13 (sampling, order, LoRA initialisation) |

Batch 1 with accumulation rather than batches of 4: measured on 2026-11-10, batches of 4
grouped by length give the same throughput as batch 1 (within 4%) for twice the memory, and
do not fit for Qwen3-0.6B on the longest examples.

## 5. Configurations, checkpoints and selection

Three runs, in this order, after the pilot of section 10:

| Run | Base | Peak learning rate |
|---|---|---|
| 1 | Pleias-RAG-350M | 1e-4 |
| 2 | Pleias-RAG-350M | 3e-4 |
| 3 | Qwen3-0.6B | the winner of runs 1 and 2 |

- The learning rate is the only hyperparameter tried. Everything else in sections 1 to 4 is
  the same for the three runs.
- Checkpoints: each run saves four, at every half epoch, after updates 866, 1 731, 2 597
  and 3 462. Each is scored once on the whole dev set (6 011 sentences), one sentence per
  forward pass, adapters kept separate, and the criterion of section 12 of `TARGETS.md` is
  computed: the mean of the two standardised partial AUCs over a false alarm rate of 0 to
  0.10, UNSUPPORTED against SUPPORTED and CONTRADICTED against SUPPORTED.
- There is no early stopping: the four checkpoints of a run are all scored, whatever their
  values.
- Winner of runs 1 and 2: the run whose best checkpoint has the higher criterion; on an exact
  tie, 1e-4. Run 3 uses its learning rate. Qwen3-0.6B is not tuned separately.
- Selection of the model: the best checkpoint of Pleias-RAG-350M (among 8) and the best of
  Qwen3-0.6B (among 4) are compared with the criterion of section 12 and the tie-break rule
  of the amendment to `TARGETS.md` dated 2026-11-10: if the two differ by less than one
  standard error of their difference, Pleias-RAG-350M is retained; otherwise the higher
  criterion is.
- Reported with the results, as section 12 requires: 3 configurations and 12 checkpoints
  evaluated on dev, with the criterion and its two terms for each of the 12.
- The tie-break is applied by `training/dev_eval.py`: Qwen3-0.6B is retained only if its
  criterion is higher by at least one standard error of the difference; an exact tie goes to
  Pleias-RAG-350M.
- If a run has to be changed or added for any reason, it is declared and counted; the test
  set plays no part in such a decision.

Before the first run, the dev evaluation code is checked on the two baselines: from their dev
scores it has to return the reference values of sections 12 and 13 of `TARGETS.md`, 0.6406
[0.6155, 0.6670] for HHEM (0.7437 on UNSUPPORTED, 0.5374 on CONTRADICTED) and 0.6185 [0.5915,
0.6486] for MiniCheck (0.6939 and 0.5432), and 0.012 for the standard error of their paired
difference. It uses the functions of `analysis/ragtruth_analysis.py` and its resamples (by
source, 10 000, seed 7). This check is `training/test_dev_eval.py`; it passed on 2026-11-10,
the values of `results/ragtruth/analysis-minicheck-vs-hhem.json` being reproduced to 1e-12.

## 6. After selection

The judge is one object from selection to publication: the base in bfloat16 at its pinned
revision, carrying the adapters of the selected checkpoint, kept separate and not merged.
Section 8 of `TARGETS.md` leaves the choice open ("adapters merged into the weights or kept
separate, which is declared with the result"); it is declared here. The reason: merging
rounds the adapted weights to bfloat16 and moves the scores. Measured on 2026-11-13 with
adapters drawn at random on 50 training examples, the merged model differed from the base
with separate adapters by up to 0.002 (Pleias-RAG-350M) and 0.11 (Qwen3-0.6B) on the score,
while the merge itself is exact in float32 (2e-5 at most). A merged judge would not be the
one that was selected.

In the order of section 14 of `TARGETS.md`:

1. The selected checkpoint is the judge. Its dev scores are those of the scoring pass made
   during its run (section 5): the checkpoint is not scored on dev again, and the dev
   threshold of section 4 of `TARGETS.md` is derived once, from these scores.
2. Cost (section 8): the judge is measured in the regime of the candidate bases, on the 739
   sentences, one sentence per call, cache released after each call, from a cold start of 15
   minutes on mains power, by `cost/measure_cost.py --adapter`: the same base and adapters,
   the declared template and the three labels. Targets: median latency below 1 s, peak
   memory footprint below 3 GB. As for the candidate bases after the freeze, this run writes
   no score (section 10).
3. One evaluation on the test set and on the five out-of-domain datasets, with the same
   base and adapters. No intermediate checkpoint is scored on them.

Separate adapters cost more than the bare base: each adapted linear module does two more
products. On 60 training sentences on 2026-11-13, scoring took 1.29 times (Pleias-RAG-350M)
and 1.28 times (Qwen3-0.6B) the time of the base alone. Applied to the medians of section 8
(0.144 s and 0.294 s), this stays far below the target of 1 s; the measurement of point 2
decides.

## 7. Estimated durations

From measurements of 2026-11-10 on the training machine (MacBook Air M2, 16 GB, mains
power), with the code of `training/`: LoRA of rank 8 on all layers, batch 1, accumulation 8,
gradient checkpointing, the three-logit loss with the head on the last position, the examples
of the start of epoch 1.

The machine has no fan. A first measurement, 40 minutes of training on Pleias-RAG-350M with
an earlier loss, showed that after about five minutes of load a step takes 1.18 to 1.24 times
its cold duration and stays there, the thermal state of the system being "fair" throughout.
The figures below are taken in that state.

| | Pleias-RAG-350M | Qwen3-0.6B |
|---|---|---|
| Training, cold machine | 1.33 s per 1000 tokens | not measured |
| Training, warm machine | 1.6 to 1.8 s per 1000 tokens | 2.6 s per 1000 tokens |
| Training throughput, warm | 0.85 to 0.9 examples/s | 0.6 examples/s |
| One epoch of 13 848 examples (8.9 and 8.8 million tokens) | 4.2 to 4.4 h | 6.4 h |
| One scoring pass on dev (6 011 sentences), base alone | 27 min | 44 min |
| The same with separate adapters, estimated (1.29 and 1.28 times) | about 35 min | about 57 min |
| One run: 2 epochs and 4 dev passes | about 11 h | about 17 h |
| Peak memory footprint, training, 5 minutes | 2.8 GB | 3.4 GB |
| Peak memory footprint, dev scoring | 1.5 GB | 2.2 GB |

The three runs take about 39 hours of computation, and the pilot of section 10 about 45
minutes.

Limits of these figures:

- With the code of `training/` each base was timed for 5 minutes of training only: the warm
  figures of Pleias-RAG-350M rest on the last two of them (1.76 and 1.79 s per 1000 tokens)
  and on the 40-minute measurement made with the earlier loss (1.60 to 1.66); those of
  Qwen3-0.6B on 5 minutes on a machine already warm.
- Restricting the head to the last position did not change the training time of
  Pleias-RAG-350M measurably and shortened that of Qwen3-0.6B by about 5%. It lowers
  memory: with a cross-entropy over the whole vocabulary at every position, the footprints
  were 4.2 GB and 5.2 GB.
- The 5 minutes did not contain the longest examples (about 2 500 tokens), which set the
  peak of a whole epoch: the footprints of training are lower bounds. With the earlier loss
  the longest examples took 3.2 GB (Pleias-RAG-350M) and 4.8 GB (Qwen3-0.6B).
- Dev passes were timed on 500 sentences drawn at random, on a warm machine, and scaled to
  6 011, with the base alone. A run scores dev with the adapters in place: the factor of
  section 6, measured on 60 training sentences on 2026-11-13, is applied to get the
  estimate.

Runs are made on mains power, with no other heavy load. A run saves its state and stops by
itself if the swap in use grows by more than 500 MB over its value at launch; it is then
resumed (section 8).

## 8. Interruption and resumption

- At each of the four checkpoints, the run writes the adapter weights, the state of the
  optimiser, the number of updates done and the dev scores of that checkpoint.
- Between checkpoints it also writes a resume state every 200 updates (same content, no dev
  scoring), replacing the previous one. A resume state is never scored on dev and is not a
  checkpoint.
- A run is resumed from its latest state. The examples of an epoch and their order depend on
  the seed and the epoch only (section 2), so a resumed run continues on the examples the
  uninterrupted run would have seen; the learning rate follows from the number of updates.
- A checkpoint is scored on dev once. If the interruption happens during a dev pass, the
  pass is redone from the saved checkpoint.
- Resumptions are reported with the results (run, update of the state used). A resumed run
  is not guaranteed to be bit-identical to an uninterrupted one.

## 9. What is kept and what is published

- Published: the judge retained (section 6), that is the adapters of the selected
  checkpoint with the name and the pinned revision of their base, after the final
  evaluation on the test set and out of domain, whatever the result, under the licence of
  the base. Both bases are under the Apache License 2.0 (model cards of
  `PleIAs/Pleias-RAG-350M` and of `Qwen/Qwen3-0.6B` at the pinned revisions). It is
  published as a Gjallar model under the Hugging Face organisation `ragnarok-ai`
  (`README.md`).
- A merged version, the adapters merged into the weights of the base in bfloat16
  (`training/merge.py`), may be published next to it for convenience. It is never evaluated:
  no dev, test or out-of-domain figure is given for it, and none of the results applies to
  it as such. Its drift from the judge is measured on 50 training examples (largest and mean
  difference of the score, next to what bfloat16 arithmetic alone changes) and declared
  with it.
- Kept private: resume states and every intermediate checkpoint, including the eleven
  checkpoints not retained and the weights of the pilot, which are deleted.
- No data of the evaluation sets is published, with the model or elsewhere: no text, no
  label, no per-sentence score of dev, test or the out-of-domain datasets. The converted
  training data is not published either (section 2 of `TARGETS.md`, Publication of data).
- The repository receives the training and evaluation code (`training/`), this file, and
  aggregates only (criterion and its terms per checkpoint, counts, durations, hashes).
  Run folders, with their logs and per-sentence dev scores, stay outside it.

## 10. Pilot

Before run 1, one pilot checks the code on the real base, not the model.

- 300 updates (2 400 examples) on Pleias-RAG-350M at a peak learning rate of 1e-4, with the
  settings of sections 1 to 4 unchanged: the same examples in the same order as the start of
  run 1, the same schedule (the warm-up of 173 updates, then the beginning of the decay of a
  run of 3 462 updates).
- No evaluation on dev, and no reading of the dev set: the pilot produces no dev value and
  cannot inform the selection.
- Purpose, and nothing else: to verify that the training loss goes down, and that
  resumption works. The pilot is interrupted once on purpose and resumed; the resumed part
  must continue on the examples of the uninterrupted order.
- It is not a configuration and not a checkpoint: it is not counted among the 3
  configurations and 12 checkpoints of section 5. It is declared with the results, with its
  loss curve and its duration (about 45 minutes).
- Before its weights are deleted, the equivalence check of `training/merge.py` is run on
  them: the adapters of the pilot are merged and the scores of the merged model are compared
  with those of the base carrying the same adapters, on 50 training examples, without any
  reading of the dev set. This gives the drift of a merged version with trained adapters,
  which the figures of section 6, obtained with adapters drawn at random, do not. It is
  declared with the results; the merged weights of the pilot are deleted too.
- Its weights are deleted. Run 1 starts again from the pretrained base.
- If the pilot shows a defect, the correction is declared with the results before run 1
  starts; a change of a setting of sections 1 to 4 would be declared the same way.
