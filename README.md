# RAGnarok Judge

Status: protocol frozen on 2026-11-10 (tag `protocol-v1`). No model has been trained or
published yet.

RAGnarok Judge is a sentence-level faithfulness judge for the answers of retrieval-augmented
systems: given a context and one sentence of an answer, it returns a score in [0, 1] that the
sentence is supported by the context. It will be trained on the RAGTruth training split and
will be compared with HHEM-2.1-open and MiniCheck-Flan-T5-Large, on the RAGTruth test split
and, out of domain, on five datasets of LLM-AggreFact.

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
- `conversion/`: `ragtruth_to_ragnarok.py`, the converter that turns RAGTruth's span
  annotations into sentence-level labels; `train_exclusions.py`, which applies the two
  exclusion rules of the training set to its output; the two lists of excluded sentence ids.
- `analysis/`: `ragtruth_analysis.py`, the analysis script of the protocol (sections 4, 5, 6
  and 9 of `TARGETS.md`), with its tests; `meta_match.py` and the two pattern lists that
  define the meta sentences and the abstentions; `prompt_templates.py`, the reference
  implementation of the templates.
- `ood/`: the out-of-domain scripts (overlap check, conversion, the two judge wrappers, the
  measurement, the metrics and the paired difference), the published figures used for the
  pipeline check, and the commands run.
- `cost/`: the cost measurement script and the commands run.
- `results/`: the aggregates of every run recorded in `TARGETS.md`: conversion report and
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

## Results

Results will be published whatever the outcome, with any re-evaluation disclosed. None yet.

## License

AGPL-3.0. See `LICENSE` for the text and `NOTICE` for the copyright holder and the
third-party components.
