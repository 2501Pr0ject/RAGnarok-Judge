# RAGnarok Judge

Status: protocol frozen on 2026-11-10 (tag `protocol-v1`); judge trained and selected (tag
`selection-v1`) and evaluated once on 2026-11-15, results in `results/RESULTS.md`. The model is
not published yet.

RAGnarok Judge is a sentence-level faithfulness judge for the answers of retrieval-augmented
systems: given a context and one sentence of an answer, it returns a score in [0, 1] that the
sentence is supported by the context. It is trained on the RAGTruth training split and
compared with HHEM-2.1-open and MiniCheck-Flan-T5-Large, on the RAGTruth test split and, out
of domain, on five datasets of LLM-AggreFact.

## Results

The retained judge, Qwen3-0.6B with separate LoRA adapters, was selected on dev among 3
configurations and 12 checkpoints and evaluated once, on 2026-11-15. Full results, with every
secondary measure and the differences to MiniCheck: `results/RESULTS.md`.

- **Claim allowed** (section 5 of `TARGETS.md`): "no demonstrated gain on contradictions; RJ is
  not worse than HHEM on unsupported sentences".
- **P1, contradictions: not met.** Recall at a specificity of 0.96, RJ 0.230 against HHEM 0.158;
  Delta_C = +0.072 [-0.006, +0.170], p = 0.039, above its Holm level of 0.025.
- **P2, unsupported sentences: met.** RJ 0.695 against HHEM 0.610; Delta_U = +0.085
  [+0.043, +0.148], p = 0.0001 against the null Delta_U <= -0.05, below its Holm level of 0.0125.
- **Out of domain: met.** Mean AUC over the five datasets, RJ 0.824 against HHEM 0.800;
  Delta_OOD = +0.024 [+0.014, +0.035], p = 0.0001 against the null Delta_OOD <= -0.02. The
  protocol tests non-inferiority only; that the interval lies above 0 is descriptive.
- **Cost: met.** Median latency 0.302 s per sentence, peak memory footprint 2.68 GB, on the
  training machine (MacBook Air M2) with MLX.

These are statements about agreement with the RAGTruth and LLM-AggreFact labels, against HHEM;
MiniCheck is reported alongside and enters no claim.

## How the results were fixed

- `protocol-v1` (2026-11-10): the evaluation protocol `TARGETS.md`, the converter and the
  exclusion lists, the pattern lists, the analysis, out-of-domain and cost scripts, and the
  reference values of the baselines, before any training.
- `training-plan-v1` (2026-11-13): the training plan `TRAINING.md` and amendment 1 of
  `TARGETS.md`, the tie-break between the two candidate bases, before any training run.
- `selection-v1` (2026-11-15): the results of training and selection, the retained judge and
  the hash of its adapters, its dev threshold and its cost, before any scoring on the test set
  or out of domain.
- `results-v1` (2026-11-15): the results of the single evaluation, `results/RESULTS.md`.

Each step was published in this repository before the next one began.

## Targets in brief

Set in `TARGETS.md` before any training, with HHEM-2.1-open as the comparator; MiniCheck is
reported alongside and enters no claim.

- **P1, contradictions (superiority).** On the RAGTruth test set, both judges read at the
  threshold that gives a specificity of 0.96 on its SUPPORTED sentences; RJ's recall on the
  139 CONTRADICTED sentences has to exceed HHEM's (null hypothesis Delta_C <= 0).
- **P2, unsupported sentences (non-inferiority).** At the same specificity, RJ's recall on the
  469 UNSUPPORTED sentences must not fall more than 0.05 below HHEM's (null hypothesis
  Delta_U <= -0.05). P1 and P2 use one-sided bootstrap p-values by source, with Holm at a
  family-wise level of 0.025.
- **Out of domain (non-inferiority).** On five datasets of LLM-AggreFact, RJ's mean AUC must
  not fall more than 0.02 below HHEM's (null hypothesis Delta_OOD <= -0.02, one-sided at
  0.025), with no threshold involved.

## Models

Gjallar is the name of the models produced by the RAGnarok Judge project. They will be
published under the Hugging Face organisation `ragnarok-ai`, for instance
`ragnarok-ai/gjallar-0.6b`, the suffix following the base model selected. RJ is the
abbreviation used for the judge in the protocol file.

## What this repository holds

- `TARGETS.md`: the evaluation protocol, written and frozen before the first training run.
  It states the targets, the comparators, the statistical protocol, the reference values of
  the baselines, the provenance of every file and what had already been seen of the
  evaluation data. `TEMPLATES.md`: the prompt templates of the candidate bases, with their
  checks. `llm-aggrefact-licences.md`: the licences of the out-of-domain datasets and the
  choice of the five used.
- `TRAINING.md`: the training plan, frozen before any training run, with amendment 1 in
  `TARGETS.md`.
- `conversion/`: `ragtruth_to_ragnarok.py`, the converter that turns RAGTruth's span
  annotations into sentence-level labels; `train_exclusions.py`, which applies the two
  exclusion rules of the training set to its output; the two lists of excluded sentence ids.
- `analysis/`: `ragtruth_analysis.py`, the analysis script of the protocol (sections 4, 5, 6
  and 9 of `TARGETS.md`), with its tests; `meta_match.py` and the two pattern lists that
  define the meta sentences and the abstentions; `prompt_templates.py`, the reference
  implementation of the templates.
- `training/`: the training code, the selection on dev, the scoring plugin of the retained
  judge and their tests; the commands run are in `training/COMMANDS.md`.
- `ood/`: the out-of-domain scripts (overlap check, conversion, the two judge wrappers, the
  measurement, the metrics, the paired difference and the metrics by band of document
  length), the published figures used for the pipeline check, and the commands run.
- `cost/`: the cost measurement script and the commands run.
- `results/`: `RESULTS.md`, the results of the evaluation of the retained judge; the
  aggregates of training and selection (`results/training/`); the aggregates of every run
  recorded in `TARGETS.md`: conversion report and
  manifests, harness reports, metrics, paired differences, the measurement of the MiniCheck
  wrapper, cost summaries, and the analysis of the two baselines on the RAGTruth test set.
  No text, label or score of a sentence appears in them.
- `requirements-frozen.txt`: the exact versions used; `TARGETS.md` records the hashes of every
  data file used, so that the files can be regenerated and checked.

The reference runs of the baselines were produced with the `judge-bench` harness of
ragnarok-ai 1.13.0, the evaluation framework this project belongs to
(https://github.com/2501Pr0ject/RAGnarok-AI, https://pypi.org/project/ragnarok-ai/); the
wrappers of `ood/` and the cost script build the judges the way the harness does.

## Running the checks

The analysis script and its tests need the converted RAGTruth files (see Data below) and a run
of the reference judge on the test and dev splits, which are local and never committed. The
run is produced by the harness, from the repository root, in the environment of
`requirements-frozen.txt`; on CPU it gives the scores of the reference run of `TARGETS.md`:

```bash
ragnarok judge-bench --judge hhem --device cpu --data ragtruth-out/test.jsonl --dev ragtruth-out/dev.jsonl --by task,generator --out runs/hhem
```

The output folder then holds `attempts.jsonl` (test) and `dev_attempts.jsonl` (dev). The tests
take the three paths from environment variables:

```bash
RJ_TEST=ragtruth-out/test.jsonl RJ_DEV=ragtruth-out/dev.jsonl RJ_RUN_B=runs/hhem python -m pytest analysis/test_ragtruth_analysis.py -q
```

Without these variables the two controls are skipped and only the unit test of the Holm
procedure runs. The analysis itself runs the same way, `analysis/ragtruth_analysis.py` taking
`--test`, `--dev`, `--run-a` (the candidate), `--run-b` (the reference) and `--out`.

## Data

**RAGTruth.** The training, dev and test sets are converted from RAGTruth, which is not
redistributed here. Download it from its repository, https://github.com/ParticleMedia/RAGTruth,
and use it under the terms stated there: the repository is under the MIT licence (Copyright
2023 Particle Media). The files read by the converter are `dataset/response.jsonl` and
`dataset/source_info.jsonl`; `TARGETS.md` records the commit and the hashes used. The texts
that RAGTruth annotates come from other corpora, MS MARCO passages, CNN/DailyMail and recent
news articles, and Yelp records, each under its own terms. This project never redistributes
them, in any form: the converted files stay on the machine that produced them, and this
repository holds sentence ids, patterns and metrics only.

To regenerate the files, run `conversion/ragtruth_to_ragnarok.py` on the RAGTruth files with
pysbd 0.3.4 and its default parameters, then apply the two exclusion lists with
`conversion/train_exclusions.py`; the hashes of the five resulting files must be those
recorded in `TARGETS.md` (section 2, Provenance). The dataset's authors ask
to be cited:

```bibtex
@misc{wu2023ragtruth,
      title={RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models},
      author={Yuanhao Wu and Juno Zhu and Siliang Xu and Kashun Shum and Cheng Niu and Randy Zhong and Juntong Song and Tong Zhang},
      year={2023},
      eprint={2401.00396},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

**LLM-AggreFact.** The out-of-domain evaluation uses five datasets of the LLM-AggreFact test
split, read from the Hugging Face Hub where the benchmark is gated and licensed CC BY-ND 4.0.
Its texts are not redistributed either; `TARGETS.md` records the revision, the hash and the
conversion.

## What this repository does not hold

No data file derived from RAGTruth or from LLM-AggreFact is published here or anywhere else,
even reformatted. Exclusion lists are committed as sentence ids and pattern lists as patterns,
without any text from the datasets. Only metrics are published.

## License

AGPL-3.0. See `LICENSE` for the text and `NOTICE` for the copyright holder and the
third-party components.
