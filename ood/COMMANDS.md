# Out of domain: the commands

Every command fits on one line. They are run from the root of the repository, with the
Python environment of the PyTorch judges active (torch, transformers, ragnarok-ai; versions in
`requirements-frozen.txt`). The data folders they read and write, `ragtruth-out/`, `ood-out/`
and `ood-runs/`, are local and never committed: only the aggregates kept in `results/ood/` are
published. The dev split of the benchmark is never read.

## Correction of 2026-11-09, made before the runs used

The first version of `check_overlap.py` mishandled documents of fewer than 13 words: it
declared them overlapping as soon as their words appeared somewhere in a training context.
Four FactCheck-GPT documents of one or two words were removed by mistake, 6 instances. No true
overlap exists: two ExpertQA documents share one sequence of 13 words with the training set,
with a containment of at most 0.002.

The script was corrected: a document of fewer than 13 words overlaps only if it is identical to
a whole training context. The converted file of the first pass, 8418 instances instead of
8424, was discarded with the run started on it, and commands 1 to 5 were run again on the
corrected file. The overlap check reports zero overlapping document and the conversion 1566
records for FactCheck-GPT. (An alternative, keeping the first files and scoring the 6 missing
instances apart, was prepared and not used.)

## 1. Overlap of the documents with the RAGTruth training contexts

Without labels: reads the dataset and document columns only, and the training contexts.
Measures, for each document, the share of its 13-word sequences present in a training
context. A document overlaps from one half. Writes `ood-out/overlap.json` with the counts per
dataset. Duration: under a minute.

```bash
uv run --no-project --python 3.12 --with pyarrow --with huggingface_hub python ood/check_overlap.py --train ragtruth-out/train.jsonl --out ood-out/overlap.json
```

## 2. Conversion of the test split, five datasets

Checks the hash of the test file, keeps WiCE, REVEAL, ClaimVerify, FactCheck-GPT and ExpertQA,
removes the instances whose document overlaps and counts them. Writes
`ood-out/llm-aggrefact-5.jsonl` and a manifest next to it. Duration: under a minute.

```bash
uv run --no-project --python 3.12 --with pyarrow --with huggingface_hub python ood/convert_llm_aggrefact.py --overlap ood-out/overlap.json --out ood-out/llm-aggrefact-5.jsonl
```

## 3. HHEM limited to 4096 tokens, on MPS

The judge is `HHEM4096Mps` of `ood/hhem_l4096.py`: a limit of 4096 tokens of its own tokenizer,
the sentence kept whole, the context truncated at its end, the cache released after each call.
Without `--dev`: no threshold is set on these data, the report is at the threshold 0.5.
Estimated duration: 30 to 60 minutes. Done on 2026-11-09: 8424 inputs scored, no failure, 307
truncations, 41 min 34 s.

```bash
PYTHONPATH=ood caffeinate -i ragnarok judge-bench --judge python:hhem_l4096:HHEM4096Mps --data ood-out/llm-aggrefact-5.jsonl --by dataset,length_band --out ood-runs/hhem-l4096-mps
```

## 4. Fallback: HHEM limited to 4096 tokens, on CPU

To run only if command 3 stops on a memory error. The device is part of the configuration: the
CPU run starts from scratch in a new folder. Estimated duration: 1 h to 1 h 45. Not needed.

```bash
PYTHONPATH=ood caffeinate -i ragnarok judge-bench --judge python:hhem_l4096:HHEM4096Cpu --data ood-out/llm-aggrefact-5.jsonl --by dataset,length_band --out ood-runs/hhem-l4096-cpu
```

## 5. MiniCheck, stock judge, on MPS

Stopped on 2026-11-09 after 356 inputs of WiCE out of 8424: process at 14 GB, swap up to 26 GB,
about one input per minute. Do not run this command again on a 16 GB machine: see section 7.
The folder `ood-runs/minicheck-mps` is kept as it is; it serves as the reference for the scores
of the stock judge.

The stock judge splits documents into chunks of 500 words (14 374 chunks here) and sends them
in batches of 16, padded to the longest, without ever releasing the cache.

```bash
caffeinate -i ragnarok judge-bench --judge minicheck --device mps --data ood-out/llm-aggrefact-5.jsonl --by dataset,length_band --out ood-runs/minicheck-mps
```

## 6. Out-of-domain metrics of a run

Reads the converted file and the `attempts.jsonl` of a run. Computes, per dataset and then as
the mean over the five datasets: the AUC, the balanced accuracy, the recall and the specificity
at the thresholds given, with intervals by resampling the documents within each dataset
(10 000 resamples, seed 7). Also gives these metrics on the untruncated inputs, and compares
the balanced accuracy at 0.5 with the published figures of `ood/published_bacc.json`. Prints
aggregates only. Duration: about ten seconds, no GPU.

HHEM, at the RAGTruth dev threshold and at 0.5 (done on 2026-11-09; copy of the result in
`results/ood/metrics-hhem-l4096-mps.json`):

```bash
python ood/ood_metrics.py --data ood-out/llm-aggrefact-5.jsonl --run ood-runs/hhem-l4096-mps --thresholds 0.18617,0.5 --published ood/published_bacc.json --judge hhem --out ood-runs/metrics-hhem-l4096-mps.json
```

MiniCheck, on the bounded run of section 7 (done on 2026-11-09; copy in
`results/ood/metrics-minicheck-bounded-mps.json`). The untruncated subset is HHEM's, hence the
option `--truncation-run`; the 6 inputs MiniCheck truncates itself (one chunk longer than 2048
tokens) are not removed from it, 4 of them being already among HHEM's 307:

```bash
python ood/ood_metrics.py --data ood-out/llm-aggrefact-5.jsonl --run ood-runs/minicheck-bounded-mps --truncation-run ood-runs/hhem-l4096-mps --thresholds 0.04438,0.5 --published ood/published_bacc.json --judge minicheck --out ood-runs/metrics-minicheck-bounded-mps.json
```

Result: mean AUC 0.8082 [0.7954, 0.8209]; mean balanced accuracy 0.6867 at the dev threshold and
0.7337 at 0.5; against the published figures, no gap above 0.3 point (WiCE +0.22, REVEAL +0.05,
ClaimVerify -0.01, FactCheck-GPT +0.02, ExpertQA -0.13).

## 7. MiniCheck with bounded forward passes, on MPS (validated on 2026-11-09)

The judge is `MiniCheckBoundedMps` of `ood/minicheck_bounded.py`. Chunks, texts, batches of 16,
truncation and padding are those of the stock judge. Only the pass through the model changes:
the rows of a batch go through in groups such that rows x length² does not exceed 2048², that
is 7 rows at 760 tokens, 4 at 1024, 1 at 2048. The cache is released after each batch of 16.
Of the 8424 inputs, 305 have at least one split batch; the 8119 others are computed exactly as
by the stock judge. No batch of the RAGTruth dev set is split.

### 7a. Measurement on WiCE (done on 2026-11-09, criteria met)

Three groups of calls, in this order: 20 controls with no split batch, the longest document of
WiCE (17 356 words, 36 chunks, one sentence), then the call with the largest batch of WiCE (7
chunks padded to 2048 tokens). The scores are compared with those of the stopped run, so the
stock judge is not run again. Expected duration: 2 to 3 minutes. The exit code is 0 when the
four criteria are met.

- controls: scores exactly equal to those of the stopped run;
- longest document and largest batch: difference of at most 1e-5 (split batches can move on
  the last float32 digits);
- tokens sent and truncation flag equal;
- peak memory footprint under 8 GB, expected between 5 and 6 GB.

```bash
PYTHONPATH=ood python ood/minicheck_measure.py --data ood-out/llm-aggrefact-5.jsonl --reference ood-runs/minicheck-mps --judge minicheck_bounded:MiniCheckBoundedMps --dataset WiCE --out ood-runs/measure-minicheck-bounded-wice.json
```

To measure the longest document only, add `--controls 0 --skip-heaviest`.

Result of 2026-11-09, copy in `results/ood/measure-minicheck-bounded-wice.json`: controls 20 of
20 exactly equal; difference of 4.56e-08 on the longest document and 5.07e-07 on the largest
batch; tokens and truncation equal; peak footprint 5.81 GB, for 3.61 GB after loading;
durations 27.3 s, 14.7 s and 10.3 s, against 35.1 s, 29.5 s and 67.1 s for the same calls in
the stopped run. The script refuses to overwrite this file: to redo the measurement, give
another name to `--out`.

### 7b. Full run (done on 2026-11-09)

New folder: the configuration differs, `judge-bench` refuses to resume the stopped run.
Estimated duration: about 2 h.

```bash
PYTHONPATH=ood caffeinate -i ragnarok judge-bench --judge python:minicheck_bounded:MiniCheckBoundedMps --data ood-out/llm-aggrefact-5.jsonl --by dataset,length_band --out ood-runs/minicheck-bounded-mps
```

Result: 8424 inputs scored out of 8424, 0 failure, 6 truncations, 7348 s of computation
(2 h 02), swap stable around 7.6 GB throughout. The folder `ood-runs/minicheck-mps` of the
stopped run is unchanged.

### 7c. After the run: scores compared on the 356 common inputs (done on 2026-11-09)

No model. Counts the exactly equal scores, the largest difference and the decisions that change
at the two thresholds.

```bash
python ood/minicheck_measure.py --reference ood-runs/minicheck-mps --compare-run ood-runs/minicheck-bounded-mps --thresholds 0.04438,0.5
```

Result, exit code 0: 337 scores exactly equal out of 356, largest difference 5.07e-07, tokens
and truncation equal on the 356, no decision changed at 0.04438 or at 0.5. The 314 inputs
without a split batch are all exactly equal; the 19 differences fall within the 42 inputs with
a split batch, 23 of which are exactly equal as well.

The metrics are then computed with the MiniCheck command of section 6.

## 8. Paired difference of mean AUC between two judges (done on 2026-11-09)

`ood/ood_paired.py` reads the converted file and the `attempts.jsonl` of two runs, A and B, and
computes Delta = mean AUC(A) - mean AUC(B) with the draws of `ood_metrics.py`: the same draws
for both judges, so the difference is paired. Gives the point value, the standard error
(standard deviation of the resampled differences), the 95% interval, the correlation of the two
mean AUCs, the p-value of the null "Delta <= -margin" as in section 9 of TARGETS.md, the
observed Delta from which it is rejected at 0.025, and the power by normal approximation at a
few true values of Delta. The untruncated subset is, by default, the one neither run truncates;
`--truncation-run` redefines it. Refuses to overwrite `--out`. Duration: about fifteen seconds,
no GPU. Prints aggregates only.

MiniCheck against HHEM, untruncated subset HHEM's as in section 6 (copy in
`results/ood/paired-minicheck-hhem.json`):

```bash
python ood/ood_paired.py --data ood-out/llm-aggrefact-5.jsonl --run-a ood-runs/minicheck-bounded-mps --run-b ood-runs/hhem-l4096-mps --truncation-run ood-runs/hhem-l4096-mps --margin 0.02 --name-a MiniCheck --name-b HHEM --out ood-runs/paired-minicheck-hhem.json
```

Result: Delta = +0.0082 [-0.0040, +0.0209], standard error 0.0064, correlation 0.53 (standard
error 0.0093 if the two judges were independent); null "Delta <= -0.02": p = 0.0001; rejection
from an observed Delta of -0.0075; power 0.88 at a true Delta of zero, 0.35 at -0.01, 1.00 at
+0.01. On the 8117 untruncated inputs: Delta = +0.0071 [-0.0059, +0.0202], standard error
0.0067, power 0.85 at a true Delta of zero.

For RJ, once trained: `--run-a` the run of RJ, `--run-b ood-runs/hhem-l4096-mps`, without
`--truncation-run` (the common subset is then the one that neither RJ nor HHEM truncates, as
section 7 of TARGETS.md provides).

## Where the durations come from

From the lengths measured on the inputs and latencies already observed, not from a trial on
these data.

- HHEM on MPS: about 26 minutes of computation from a cold start, from the diagnostic on
  RAGTruth and a fabricated input of 4096 tokens (1.5 to 2 s). The upper bound allows for the
  thermal slowdown observed on this machine.
- HHEM on CPU: about 65 minutes of computation, with 4.5 to 5 s for an input of 4096 tokens.
- MiniCheck, stock: about 70 minutes, from its full run on RAGTruth. Estimate contradicted: it
  ignored the batch sizes on long documents.
- MiniCheck with bounded passes: about 2 h, from the throughput of the stopped run while it was
  not swapping (0.7 to 0.8 s per million cells rows x length², 6330 million cells in all) and
  the 3276 short calls of REVEAL and FactCheck-GPT.

## What was checked

- Overlap and conversion: on a synthetic file of the same schema, with an exact copy, a partial
  copy, a long page holding a short passage, very short documents and an unrelated document;
  the result is read back by the harness. The case of very short documents was missing from
  the first test, hence the error of the first pass.
- HHEM wrapper: a fabricated context of 21 632 tokens is brought down to 4096 tokens; it runs
  on MPS (peak footprint 4.2 GB) and on CPU (3.7 GB).
- The three `judge-bench` commands: on a synthetic file of five records.
- MiniCheck wrapper, on CPU only: 21 comparisons with the stock judge on WiCE inputs. Batch not
  split: probabilities exactly equal. Batch split: difference of at most 4.3e-7 per chunk.
  Tokens and truncation always equal. In `judge-bench`: 7 inputs scored, and resume refused on
  a copy of the stopped run. On MPS: the measurement 7a.
- Measurement script, on CPU: the peak footprint it reads (5.76 GB) is that of
  `/usr/bin/time -l`; the comparison of runs detects a modified score and a changed decision
  on an altered copy.
- Out-of-domain metrics: on synthetic data, the values and 600 draws are compared with a naive
  computation, pair by pair, with tied scores, a score equal to the threshold, documents of a
  single class and a dataset where a class can disappear from a draw. On the real run, the
  balanced accuracy at 0.5 per dataset is that of the harness report.
- Paired difference: the mean AUCs and their intervals reproduced by the script for each judge
  are, to the fourth decimal, those of the two metrics files, so the draws are the same; the
  difference, its standard error and the correlation are recovered by a naive computation on
  600 draws, pair by pair, with the same seeds.
