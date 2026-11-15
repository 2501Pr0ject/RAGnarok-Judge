# Declarations made with the results of training and selection

Written on 2026-11-15, before the retained judge is scored on the test set or out of domain.
The figures they refer to are in `README.md` and in the JSON files of this folder.

## 1. Pilot (section 10 of `TRAINING.md`)

Declared in the note "Pilot, 2026-11-13" at the end of `TRAINING.md`: the defect found (the
swap rule inactive under a French locale) and its correction, committed before run 1; the
interruption after update 210 instead of 150; `/usr/bin/time -l` added to the command. Its
figures are in `runs.json`, key `pilot`:

- 300 updates on Pleias-RAG-350M at 1e-4, the examples, order and schedule of the start of
  run 1. No dev set read, no dev value. Not counted among the 3 configurations and 12
  checkpoints.
- Loss: from 3.63 (updates 1 to 25) to 0.76 (updates 276 to 300), by tranche of 25 in
  `runs.json`, with the gradient norm before clipping (median from 5.8 to 1.4).
- Resumption: interrupted on purpose after update 210, resumed from the state of update 200;
  updates 201 to 210 replayed bit-identical to the first pass (loss, gradient norm, tokens).
- Duration 40 min 22 s, 7.74 s per update; peak footprint 2.86 GB; no swap.
- Updates 1 to 300 of run 1 are bit-identical to those of the pilot.
- Merge check of `training/merge.py` on 50 training examples, with the adapters of the pilot:
  the merge itself is exact in float32 (difference 0); the merged model in bfloat16 differs
  from the base with separate adapters by up to 0.153 on the score (mean 0.114), against 0.0375
  (mean 0.020) for bfloat16 against float32 with nothing merged. With trained adapters the
  drift is much larger than the 0.002 measured with random adapters in section 6 of
  `TRAINING.md`.
- The weights of the pilot and their merged version were deleted.

## 2. Check on training examples, 2026-11-14

Before run 3, a check on the training file only, no dev or test sentence
(`train-check-2026-11-14.json`):

- The label to token to index mapping was checked in the code, from the data file to the loss
  and to the score, for both bases: consistent everywhere (SUPPORTED, UNSUPPORTED,
  CONTRADICTED as indices 0, 1, 2; tokens 57889, 2076, 3099 for Pleias-RAG-350M and 34636,
  41884, 41437 for Qwen3-0.6B).
- 1 200 examples of epoch 1 (300 CONTRADICTED, 300 UNSUPPORTED, 600 SUPPORTED, seed
  20261114), scored by `pleias-lr3e-4/checkpoint-3462` with separate adapters and by the bare
  base. With the adapters: AUC of P(Supported) 0.950 for UNSUPPORTED against SUPPORTED and
  0.801 for CONTRADICTED against SUPPORTED; 6 of the 300 CONTRADICTED predicted CONTRADICTED by
  the argmax, 257 predicted SUPPORTED; mean P(Contradicted) on them 0.15. The bare base: AUCs
  0.514 and 0.519, P(Supported) about 0.002 whatever the label.
- Reading: no mapping defect (UNSUPPORTED is learnt, in the right direction); CONTRADICTED is
  under-learnt by Pleias-RAG-350M, even on examples seen twice, consistent with its dev term
  near 0.52. Not a defect of the code. Per-sentence probabilities were never written.

## 3. 50 dev sentences scored again, 2026-11-14

Point 1 of section 6 of `TRAINING.md` states that the selected checkpoint is not scored on
dev again. To check the scoring plugin `training/rj_judge.py`, 50 dev sentences drawn with a
fixed seed (20261114; 49 SUPPORTED, 1 UNSUPPORTED) were scored again through
`ragnarok judge-bench` and compared with the `dev_scores.json` of the run: the 50 scores are
identical (largest difference 0). This was a check of the code only. No dev value was
computed from these scores, and they were deleted; the dev scores of the judge, its dev
criterion and its dev threshold are those of the scoring pass made during its run
(`plugin-equivalence-2026-11-14.json`).

The same file holds the two other checks of the plugin: 50 training examples scored through
the harness, identical to `judge.score_records` (largest difference 0); and the dev scores of
the run written by `training/dev_attempts.py` as `dev_attempts.jsonl`, read back identical by
the readers of the analysis, with the configuration hash of the plugin and the criterion of
`dev.json`.

## 4. Configuration of use (section 8 of `TARGETS.md`, section 6 of `TRAINING.md`)

The judge is one object from selection to publication: Qwen3-0.6B at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, weights in bfloat16 on MLX, carrying the LoRA
adapters of `qwen3-lr3e-4/checkpoint-3462` (SHA-256 `44cdeeca…bed31`), kept separate and not
merged. It is the form scored on dev during the run, the form measured for cost on
2026-11-15, and the form that will be scored once on the test set and on the five
out-of-domain datasets. A merged version, if one is published for convenience, is never
evaluated; its drift is measured on 50 training examples and declared with it.

## 5. Environment

On 2026-11-14 the PyTorch environment received MLX at the versions of the MLX environment
(mlx 0.32.4, mlx-lm 0.32.2, mlx-metal 0.32.4, protobuf 7.36.2, sentencepiece 0.2.3), for
the scoring plugin: additions only, recorded in `requirements-frozen.txt`. The dev criteria of
the two baselines reproduced by `training/dev_eval.py` in that environment were byte-identical
before and after (HHEM 0.6406, MiniCheck 0.6185, standard error of their difference 0.0118).

## 6. Attribution patterns (sections 6 and 11 of `TARGETS.md`)

The recall on CONTRADICTED outside the 14 test sentences whose annotator note matches the
attribution patterns needs the ids of these sentences. The patterns were not frozen in any
versioned file. They were reconstructed on 2026-11-15 from the `motifs_note` column of the
local draft of 2026-11-09, which records, for each of the 115 training candidates of training
set rule 1, the patterns that matched its note; the six patterns are taken without any change.
The case flag was not recorded: `re.IGNORECASE` was established on the training split, where
it gives the 115 candidates, no more and no fewer (30 without it), then 74 after rule 5, made
of the 58 excluded ids of `train_excluded_ids_rule1.txt` and the 16 sentences kept, identical
to the hand validation of 2026-11-10. A note is looked for in every annotated span attached to
the sentence; looking only in the conflict spans gives the same 115 training sentences. The
script, `analysis/attribution_ids.py`, and its test were committed before any scoring of the
test set; on the test set it is run once, at the analysis, with `--expect 14`, and stops
without writing anything on any other count.
